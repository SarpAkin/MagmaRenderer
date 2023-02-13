use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::hash::Hash;
use std::hash::Hasher;

use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::mpsc::channel;
use std::sync::Mutex;
use std::sync::RwLock;

use bytemuck::cast_slice;
use notify::EventHandler;
use notify::Watcher;
use serde::Deserialize;
use serde::Serialize;
use spirv_reflect::types::op;

use crate::core::*;
use crate::prelude::*;

use super::mesh;
use super::mesh::VertexInputTypes;
use super::sync_cache::SyncCache;

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub struct MaterialID(u32);

impl MaterialID {
    pub const NULL: MaterialID = MaterialID(0);
}

struct MaterialData {
    textures: Box<[Arc<Image>]>,
    buffers: Box<[Arc<Buffer<u8>>]>,
    pipeline: Arc<Pipeline>,
    id: MaterialID,
    material_set: vk::DescriptorSet,
    material_set_index: u32,
    vertex_input_types: Option<Box<[mesh::VertexInputTypes]>>,
}

pub struct Material<'a> {
    data: &'a MaterialData,
    manager: &'a MaterialManager,
}

impl<'a> Material<'a> {
    pub fn pipeline(&self) -> &Arc<Pipeline> { &self.data.pipeline }
    pub fn id(&self) -> MaterialID { self.data.id }
    pub fn vertex_input_types(&self) -> Option<&Box<[VertexInputTypes]>> { self.data.vertex_input_types.as_ref() }
}

impl CommandBuffer {
    pub fn bind_material(&mut self, material: &Material) {
        self.bind_pipeline(material.pipeline());
        if material.data.material_set_index != u32::MAX {
            self.bind_descriptor_set(material.data.material_set_index, material.data.material_set);
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct PipelineArguments {
    shaders: Vec<String>,
    vertex: Vertex,
    subpass: String,
}

pub struct MaterialManager {
    core: Arc<Core>,
    materials: HashMap<MaterialID, MaterialData>,
    shader_to_material: Arc<RwLock<HashMap<PathBuf, HashMap<MaterialID, PipelineArguments>>>>,
    textures: HashMap<String, Arc<Image>>,
    default_sampler: Sampler,
    pool: DescriptorPool,
    mat_id_counter: u32,
    pipeline_loader: Arc<RwLock<PipelineLoader>>,
    shader_hot_reload: bool,
    shader_watcher: notify::INotifyWatcher,
    pipeline_rv: Mutex<std::sync::mpsc::Receiver<(MaterialID, std::result::Result<Arc<PipelineDetails>, eyre::ErrReport>)>>,
}

struct CompiledShader {
    spirv: Vec<u32>,
    stype: vk::ShaderStageFlags,
    source: Option<String>,
    source_hash: u64,
}

struct BindingReflection {
    set: u32,
    binding: u32,
    count: u32,
    stages: vk::ShaderStageFlags,
    descriptor_type: spirv_reflect::types::ReflectDescriptorType,
    // data: spirv_reflect::types::ReflectDescriptorBinding,
}

struct SetReflection {
    bindings: Vec<BindingReflection>,
}

pub struct PipelineReflection {
    sets: Vec<SetReflection>,
    push_details: Option<PushDetails>,
}

impl fmt::Debug for PipelineReflection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { Ok(()) }
}

#[derive(Debug)]
struct PipelineDetails {
    pipeline: Arc<Pipeline>,
    reflection_data: PipelineReflection,
}

struct PushDetails {
    size: u32,
    stages: vk::ShaderStageFlags,
}

impl SetReflection {
    pub fn create_descriptor_set_layout(&self, core: &Arc<Core>) -> Result<vk::DescriptorSetLayout, vk::Result> {
        let mut builder = DescriptorSetLayoutBuilder::new();
        for binding in &self.bindings {
            use spirv_reflect::types::ReflectDescriptorType;
            match binding.descriptor_type {
                ReflectDescriptorType::CombinedImageSampler
                | ReflectDescriptorType::Sampler
                | ReflectDescriptorType::SampledImage => builder.add_sampler(binding.stages, binding.count),
                ReflectDescriptorType::UniformBuffer => builder.add_ubo(binding.stages, binding.count),
                ReflectDescriptorType::StorageBuffer => builder.add_ssbo(binding.stages, binding.count),
                // ReflectDescriptorType::UniformBufferDynamic => todo!(),
                // ReflectDescriptorType::StorageBufferDynamic => todo!(),
                ReflectDescriptorType::InputAttachment => builder.add_input_attachement(binding.stages, binding.count),
                _ => panic!("unhandeled"),
            };
        }

        builder.build(core)
    }
}

impl PipelineReflection {
    pub fn create_pipeline_layout(&self, core: &Arc<Core>) -> Result<(vk::PipelineLayout, Vec<vk::DescriptorSetLayout>)> {
        let mut layout_builder = PipelineLayoutBuilder::new();

        if let Some(push) = &self.push_details {
            layout_builder.add_push_with_size(push.stages, 0, push.size);
        }

        let dset_layouts: Vec<_> = self
            .sets
            .iter()
            .map(|set_reflection| set_reflection.create_descriptor_set_layout(core))
            .collect::<Result<_, _>>()?;

        for set in &dset_layouts {
            layout_builder.add_set(*set);
        }

        Ok((layout_builder.build(core)?, dset_layouts))
    }
}

impl MaterialManager {
    pub fn set_subpass(&mut self, subpass_name: String, subpass: &Subpass) {
        self.pipeline_loader.write().unwrap().render_targets.insert(subpass_name, subpass.get_render_target().clone());
    }

    pub fn get_material(&self, id: MaterialID) -> Option<Material> {
        self.materials.get(&id).map(|m| Material { data: m, manager: self })
    }

    fn load_pipeline(&mut self, arguments: &PipelineArguments) -> Result<Arc<PipelineDetails>> {
        self.pipeline_loader.read().unwrap().load_pipeline(&arguments)
    }

    pub fn update(&mut self) {
        for (mid, p) in self.pipeline_rv.get_mut().unwrap().try_iter() {
            let Ok(pipeline) = p else{
                eprintln!("failed compile pipeline due to {}",p.unwrap_err().root_cause());
                return;
            };

            self.materials.get_mut(&mid).unwrap().pipeline = pipeline.pipeline.clone();
        }
    }

    pub fn load_material(
        &mut self,
        cmd: &mut CommandBuffer,
        path: String,
        // renderpass: &dyn Renderpass,
        // subpass_index: u32,
    ) -> Result<MaterialID> {
        let mat_desc: MaterialDescripton = serde_yaml::from_str(&std::fs::read_to_string(&path)?)?;

        let arguments = PipelineArguments {
            shaders: mat_desc.shaders.clone(),
            vertex: mat_desc.vertex.clone(),
            subpass: mat_desc.subpass.clone(),
        };
        let pipeline = self.load_pipeline(&arguments)?;

        let textures: Box<[_]> =
            mat_desc.textures.iter().map(|tfile| self.load_texture(cmd, tfile.clone())).collect::<Result<_>>()?;

        let (material_set, material_set_index) = if let Some(material_set_index) = mat_desc.material_set {
            let mut dset_builder = DescriptorSetBuilder::new();
            dset_builder
                .add_sampled_images(&textures.iter().map(|t| (t.as_ref(), *self.default_sampler)).collect::<Vec<_>>());
            (
                dset_builder
                    .build(pipeline.pipeline.get_descriptor_set_layout(material_set_index).unwrap(), &mut self.pool)?,
                material_set_index,
            )
        } else {
            (vk::DescriptorSet::null(), u32::MAX)
        };

        let material_id = self.new_material_id();

        let material = MaterialData {
            textures,
            pipeline: pipeline.pipeline.clone(),
            buffers: Box::new([]),
            id: material_id,
            material_set,
            material_set_index,
            vertex_input_types: match mat_desc.vertex {
                Vertex::Inputs(inputs) => Some(inputs.into_boxed_slice()),
                Vertex::Name(_) => None,
            },
        };

        self.materials.insert(material_id, material);

        if self.shader_hot_reload {
            let mut shader_to_material = self.shader_to_material.write().unwrap();

            for shader in &arguments.shaders {
                let path = std::fs::canonicalize(PathBuf::from_str(shader)?)?;

                self.shader_watcher.watch(&path, notify::RecursiveMode::NonRecursive)?;
                let map = shader_to_material.entry(path).or_default();

                map.insert(material_id, arguments.clone());
            }
        }

        Ok(material_id)
    }

    pub fn set_vertex_layout(&mut self, name: String, vertex_description: VertexInputDescriptionBuilder) {
        self.pipeline_loader.write().unwrap().vertex_descriptions.insert(name, vertex_description);
    }

    pub fn compile_compute_shader(&self, filename: &str) -> Result<(Arc<Pipeline>, PipelineReflection)> {
        self.pipeline_loader.read().unwrap().compile_compute_shader(filename)
    }

    pub fn new(core: &Arc<Core>) -> Result<Self> {
        let (sr, rv) = channel();

        let shader_to_material = Arc::new(RwLock::new(HashMap::new()));
        let ploader = Arc::new(RwLock::new(PipelineLoader::new(core)));

        Ok(Self {
            core: core.clone(),
            materials: HashMap::new(),
            textures: HashMap::new(),
            default_sampler: core.create_sampler(vk::Filter::NEAREST, None),
            pool: DescriptorPool::new(core),
            mat_id_counter: 1, //0 is reserved for null
            pipeline_loader: ploader.clone(),
            shader_to_material: shader_to_material.clone(),
            shader_hot_reload: true,
            pipeline_rv: Mutex::new(rv),
            shader_watcher: notify::recommended_watcher(move |e: std::result::Result<notify::Event, _>| {
                // let sr = sr;

                let Ok(e) = e else{eprintln!("{}",e.unwrap_err());return};
                if !e.kind.is_modify() {
                    return;
                }

                let sr1 = sr.clone();
                let ploader1 = ploader.clone();
                let shader_to_material1 = shader_to_material.clone();

                let paths = e.paths;

                rayon::spawn(move || {
                    let stm = shader_to_material1.read().unwrap();
                    let loader = ploader1.read().unwrap();

                    for p in paths {
                        let Some(map) = stm.get(&p) else {
                            eprintln!("failed to find shader_to_material with the path {p:?}");
                            continue;
                        };
                        for (m, arg) in map {
                            let pipline = loader.reload_pipeline(arg);
                            sr1.send((*m, pipline)).unwrap();
                        }
                    }
                })
            })?,
        })
    }

    fn load_texture(&mut self, cmd: &mut CommandBuffer, path: String) -> Result<Arc<Image>> {
        let Some(texture) = self.textures.get(&path) else {
            let texture = Arc::new(cmd.load_image_from_file(&path, vk::ImageUsageFlags::SAMPLED)?);
            self.textures.insert(path, texture.clone());
            return Ok(texture);
        };
        Ok(texture.clone())
    }

    fn new_material_id(&mut self) -> MaterialID {
        let id = MaterialID(self.mat_id_counter);
        self.mat_id_counter += 1;
        id
    }
}

#[derive(Serialize, Deserialize)]
pub struct MaterialDescripton {
    textures: Vec<String>,
    shaders: Vec<String>,
    vertex: Vertex,
    material_set: Option<u32>,
    subpass: String,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum Vertex {
    Inputs(Vec<mesh::VertexInputTypes>),
    Name(String),
}

fn get_shader_type(filename: &str) -> Option<vk::ShaderStageFlags> {
    let Some(a) = filename.rsplit('.').next() else {return None};
    let shader_type = match a {
        "vert" | "vs" => vk::ShaderStageFlags::VERTEX,
        "frag" | "fs" => vk::ShaderStageFlags::FRAGMENT,
        "comp" => vk::ShaderStageFlags::COMPUTE,
        _ => return None,
    };

    Some(shader_type)
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ShaderArguments {
    filename: String,
}

pub struct ShaderCompiler {
    cached_shaders: RwLock<HashMap<String, Arc<CompiledShader>>>,
}

impl ShaderCompiler {
    fn compile_shader_uncached(filename: &str, source: String, hash: u64) -> Result<CompiledShader> {
        use shaderc::*;

        let mut compiler = shaderc::Compiler::new().unwrap();
        let options = shaderc::CompileOptions::new().unwrap();

        // let source = std::fs::read_to_string(filename)?;
        let shader_type = get_shader_type(filename).unwrap();
        let compile_shader_type = match shader_type {
            vk::ShaderStageFlags::VERTEX => ShaderKind::Vertex,
            vk::ShaderStageFlags::FRAGMENT => ShaderKind::Fragment,
            vk::ShaderStageFlags::COMPUTE => ShaderKind::Compute,
            _ => panic!("unknown shader type"),
        };

        let bin = compiler.compile_into_spirv(&source, compile_shader_type, filename, "main", Some(&options))?;

        let mut spirv_code = Vec::new();
        bin.as_binary().clone_into(&mut spirv_code);

        let shader = CompiledShader { spirv: spirv_code, stype: shader_type, source_hash: hash, source: Some(source) };

        Ok(shader)
    }

    fn compile_shader(&self, filename: &str) -> Result<Arc<CompiledShader>> {
        let source = std::fs::read_to_string(filename)?;

        let hash = {
            let mut hasher = DefaultHasher::new();
            source.hash(&mut hasher);
            hasher.finish()
        };

        let Some(mut shader) = self.cached_shaders.read().unwrap().get(filename).map(|a|a.clone()) else {
            let shader = Arc::new(Self::compile_shader_uncached(filename, source,hash)?);
            self.cached_shaders.write().unwrap().insert(filename.to_string(), shader.clone());
            return Ok(shader);
        };

        if shader.source_hash != hash {
            shader = Arc::new(Self::compile_shader_uncached(filename, source, hash)?);
            self.cached_shaders.write().unwrap().insert(filename.to_string(), shader.clone());
        }

        Ok(shader)
    }

    fn new() -> Self { Self { cached_shaders: RwLock::new(HashMap::new()) } }
}

fn foo() {
    let mut watcher = notify::recommended_watcher(|e| {}).unwrap();

    // watcher.watch(path, recursive_mode);
}

// type Key = Hash + PartialEq + Eq;

pub struct PipelineLoader {
    core: Arc<Core>,
    cached_pipelines: SyncCache<PipelineArguments, Arc<PipelineDetails>>,
    shader_compiler: ShaderCompiler,

    pub vertex_descriptions: HashMap<String, VertexInputDescriptionBuilder>,
    pub render_targets: HashMap<String, RenderTargetInfo>,
}

impl PipelineLoader {
    pub fn new(core: &Arc<Core>) -> PipelineLoader {
        Self {
            core: core.clone(),
            cached_pipelines: SyncCache::new(),
            shader_compiler: ShaderCompiler::new(),
            vertex_descriptions: HashMap::new(),
            render_targets: HashMap::new(),
        }
    }

    pub fn compile_compute_shader(&self, filename: &str) -> Result<(Arc<Pipeline>, PipelineReflection)> {
        let spirv = [self.shader_compiler.compile_shader(filename)?];
        let reflection = Self::reflect_shaders(&spirv)?;
        let (playout, descriptro_layouts) = reflection.create_pipeline_layout(&self.core)?;

        let pipeline = self.core.create_compute(
            playout,
            &ShaderModule::new(&self.core, &spirv[0].spirv)?,
            descriptro_layouts.into_boxed_slice(),
        )?;

        Ok((pipeline, reflection))
    }

    fn load_pipeline(&self, arguments: &PipelineArguments) -> Result<Arc<PipelineDetails>> {
        self.cached_pipelines.get_or_try_insert_with(&arguments, || self.load_pipeline_uncached(&arguments))
    }


    fn reload_pipeline(&self, arguments: &PipelineArguments) -> Result<Arc<PipelineDetails>> {
        self.cached_pipelines.try_insert_with(&arguments, || self.load_pipeline_uncached(&arguments))
    }


    fn load_pipeline_uncached(&self, arguments: &PipelineArguments) -> Result<Arc<PipelineDetails>> {
        let compiled_shaders = arguments
            .shaders //
            .iter()
            .map(|s| self.shader_compiler.compile_shader(s))
            .collect::<Result<Vec<_>>>()?;

        let reflection_data = Self::reflect_shaders(&compiled_shaders)?;

        let (layout, dset_layouts) = reflection_data.create_pipeline_layout(&self.core)?;

        let mut pipeline_builder = GPipelineBuilder::new();
        let modules = compiled_shaders
            .iter()
            .map(|shader| ShaderModule::new(&self.core, &shader.spirv))
            .collect::<Result<Vec<_>, _>>()?;
        for (module, shader) in modules.iter().zip(&compiled_shaders) {
            pipeline_builder.add_shader_stage(shader.stype, &module.module());
        }
        pipeline_builder.set_pipeline_layout(layout);
        pipeline_builder.set_depth_testing(true);
        pipeline_builder.set_rasterization(vk::PolygonMode::FILL, vk::CullModeFlags::BACK);
        pipeline_builder.set_topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        pipeline_builder.set_descriptor_set_layouts(dset_layouts.into_boxed_slice());
        pipeline_builder.set_render_target(&self.render_targets[&arguments.subpass]);

        match &arguments.vertex {
            Vertex::Inputs(input_types) => {
                pipeline_builder.set_vertex_description(mesh::Mesh::get_vertex_input_description(input_types));
            }
            Vertex::Name(name) => {
                if let Some(vertex_description) = self.vertex_descriptions.get(name) {
                    pipeline_builder.set_vertex_description(vertex_description.clone());
                }
            }
        }

        let pipeline = pipeline_builder.build(&self.core)?;

        let details = Arc::new(PipelineDetails { pipeline, reflection_data });

        Ok(details)
    }

    fn reflect_shaders(compiled_shaders: &[Arc<CompiledShader>]) -> Result<PipelineReflection> {
        use spirv_reflect::*;

        let mut sets = Vec::new();
        let mut push_details = Option::<PushDetails>::None;

        for shader in compiled_shaders {
            let module = create_shader_module(cast_slice(&shader.spirv)).unwrap();
            for binding in module.enumerate_descriptor_bindings(None).unwrap() {
                while sets.len() <= binding.set as usize {
                    sets.push(HashMap::new());
                }

                sets[binding.set as usize]
                    .entry(binding.binding)
                    .or_insert_with(|| BindingReflection {
                        set: binding.set,
                        binding: binding.binding,
                        count: binding.count,
                        stages: shader.stype,
                        descriptor_type: binding.descriptor_type,
                        // data: binding,
                    })
                    .stages |= shader.stype;
            }

            let push_blocks = module.enumerate_push_constant_blocks(None).unwrap();
            let Some(pushblock) = push_blocks.get(0) else {continue};

            if let Some(_push_details) = &mut push_details {
                _push_details.stages |= shader.stype;
            } else {
                push_details = Some(PushDetails { size: pushblock.size, stages: shader.stype });
                continue;
            };
        }

        let set_bindings: Vec<_> = sets
            .iter_mut()
            .map(|set| {
                let max_binding = set.iter().map(|(k, _)| *k).max().unwrap();
                SetReflection { bindings: (0..=max_binding).map(|i| set.remove(&i).unwrap()).collect() }
            })
            .collect();

        Ok(PipelineReflection { sets: set_bindings, push_details })
    }
}
