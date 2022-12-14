use std::collections::HashMap;
use std::fs;

use bytemuck::cast_slice;
use serde::Deserialize;
use serde::Serialize;
use spirv_reflect::types::op;

use crate::core::*;
use crate::prelude::*;

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
}

pub struct Material<'a> {
    data: &'a MaterialData,
    manager: &'a MaterialManager,
}

impl<'a> Material<'a> {
    pub fn pipeline(&self) -> &Arc<Pipeline> { &self.data.pipeline }
    pub fn id(&self) -> MaterialID { self.data.id }
}

impl CommandBuffer {
    pub fn bind_material(&mut self, material: &Material) {
        self.bind_pipeline(material.pipeline());
        if material.data.material_set_index != u32::MAX {
            self.bind_descriptor_set(material.data.material_set_index, material.data.material_set);
        }
    }
}

pub struct MaterialManager {
    core: Arc<Core>,
    materials: HashMap<MaterialID, MaterialData>,
    textures: HashMap<String, Arc<Image>>,
    vertex_descriptions: HashMap<String, VertexInputDescriptionBuilder>,
    default_sampler: Sampler,
    pool: DescriptorPool,
    mat_id_counter: u32,
}

struct CompiledShader {
    spirv: Vec<u32>,
    stype: vk::ShaderStageFlags,
    source: Option<String>,
}

struct BindingReflection {
    set: u32,
    binding: u32,
    count: u32,
    stages: vk::ShaderStageFlags,
    data: spirv_reflect::types::ReflectDescriptorBinding,
}

struct SetReflection {
    bindings: Vec<BindingReflection>,
}

pub struct PipelineReflection {
    sets: Vec<SetReflection>,
    push_details: Option<PushDetails>,
}

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
            match binding.data.descriptor_type {
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
    pub fn get_material(&self, id: MaterialID) -> Option<Material> {
        self.materials.get(&id).map(|m| Material { data: m, manager: self })
    }

    pub fn load_material(
        &mut self,
        cmd: &mut CommandBuffer,
        path: String,
        renderpass: &dyn Renderpass,
        subpass_index: u32,
    ) -> Result<MaterialID> {
        let mat_desc: MaterialDescripton = serde_yaml::from_str(&std::fs::read_to_string(&path)?)?;

        let pipeline = self.load_pipeline(&mat_desc, renderpass, subpass_index)?;

        let textures: Box<[_]> =
            mat_desc.textures.iter().map(|tfile| self.load_texture(cmd, tfile.clone())).collect::<Result<_>>()?;

        let (material_set, material_set_index) = if let Some(material_set_index) = mat_desc.material_set {
            let mut dset_builder = DescriptorSetBuilder::new();
            dset_builder
                .add_sampled_images(&textures.iter().map(|t| (t.as_ref(), *self.default_sampler)).collect::<Vec<_>>());
            (dset_builder.build(pipeline.pipeline.get_descriptor_set_layout(material_set_index).unwrap(), &mut self.pool)?, material_set_index)
        } else {
            (vk::DescriptorSet::null(), u32::MAX)
        };

        let material_id = self.new_material_id();

        let material = MaterialData {
            textures,
            pipeline: pipeline.pipeline,
            buffers: Box::new([]),
            id: material_id,
            material_set,
            material_set_index,
        };

        self.materials.insert(material_id, material);

        Ok(material_id)
    }

    pub fn set_vertex_layout(&mut self, name: String, vertex_description: VertexInputDescriptionBuilder) {
        self.vertex_descriptions.insert(name, vertex_description);
    }

    pub fn compile_compute_shader(&mut self, filename: &str) -> Result<(Arc<Pipeline>, PipelineReflection)> {
        let spirv = [Self::compile_shader(filename)?];
        let reflection = Self::reflect_shaders(&spirv)?;
        let (playout, descriptro_layouts) = reflection.create_pipeline_layout(&self.core)?;

        let pipeline = self.core.create_compute(
            playout,
            &ShaderModule::new(&self.core, &spirv[0].spirv)?,
            descriptro_layouts.into_boxed_slice(),
        )?;

        Ok((pipeline, reflection))
    }

    pub fn new(core: &Arc<Core>) -> Result<Self> {
        Ok(Self {
            core: core.clone(),
            materials: HashMap::new(),
            textures: HashMap::new(),
            vertex_descriptions: HashMap::new(),
            default_sampler: core.create_sampler(vk::Filter::NEAREST, None),
            pool: DescriptorPool::new(core),
            mat_id_counter: 1,//0 is reserved for null
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

    fn compile_shader(filename: &str) -> Result<CompiledShader> {
        use shaderc::*;

        let mut compiler = shaderc::Compiler::new().unwrap();
        let options = shaderc::CompileOptions::new().unwrap();

        let source = std::fs::read_to_string(filename)?;
        let shader_type = Self::get_shader_type(filename).unwrap();
        let compile_shader_type = match shader_type {
            vk::ShaderStageFlags::VERTEX => ShaderKind::Vertex,
            vk::ShaderStageFlags::FRAGMENT => ShaderKind::Fragment,
            vk::ShaderStageFlags::COMPUTE => ShaderKind::Compute,
            _ => panic!("unknown shader type"),
        };

        let bin = compiler.compile_into_spirv(&source, compile_shader_type, filename, "main", Some(&options))?;

        let mut spirv_code = Vec::new();
        bin.as_binary().clone_into(&mut spirv_code);

        Ok(CompiledShader { spirv: spirv_code, stype: shader_type, source: Some(source) })
    }

    fn load_pipeline(
        &mut self,
        mat_desc: &MaterialDescripton,
        renderpass: &dyn Renderpass,
        subpass_index: u32,
    ) -> Result<PipelineDetails> {
        let compiled_shaders = mat_desc.shaders.iter().map(|s| Self::compile_shader(s)).collect::<Result<Vec<_>>>()?;
        let reflection_data = Self::reflect_shaders(&compiled_shaders)?;

        let (layout, dset_layouts) = reflection_data.create_pipeline_layout(&self.core)?;

        let mut pipeline_builder = GPipelineBuilder::new();
        let mut modules = Vec::new();
        for shader in compiled_shaders {
            let module = ShaderModule::new(&self.core, &shader.spirv)?;
            pipeline_builder.add_shader_stage(shader.stype, &module.module());
            modules.push(module);
        }
        pipeline_builder.set_pipeline_layout(layout);
        pipeline_builder.set_depth_testing(true);
        pipeline_builder.set_rasterization(vk::PolygonMode::FILL, vk::CullModeFlags::BACK);
        pipeline_builder.set_topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        pipeline_builder.set_descriptor_set_layouts(dset_layouts.into_boxed_slice());
        if let Some(vertex_description) = self.vertex_descriptions.get(&mat_desc.vertex) {
            pipeline_builder.set_vertex_description(vertex_description.clone());
        }

        let pipeline = pipeline_builder.build(&self.core, renderpass, subpass_index)?;

        Ok(PipelineDetails { pipeline, reflection_data })
    }

    fn reflect_shaders(compiled_shaders: &[CompiledShader]) -> Result<PipelineReflection> {
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
                        data: binding,
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
    vertex: String,
    material_set: Option<u32>,
}

// mod test {
//     use super::*;
//     #[derive(Serialize, Deserialize)]
//     struct AA {
//         materials: Vec<MaterialDescripton>,
//     }

//     pub fn test_serialization() -> Result<()> {
//         let a = AA {
//             materials: (0..3)
//                 .map(|i| MaterialDescripton {
//                     textures: vec![format!("{i}.png"), String::from("gg.png")],
//                     shaders: vec![String::from("ex.vert"), String::from("tri.frag")],
//                     vertex: String::new(),

//                 })
//                 .collect(),
//         };

//         let file = std::fs::File::create("a.yaml")?;

//         serde_yaml::to_writer(file, &a)?;

//         Ok(())
//     }
// }
