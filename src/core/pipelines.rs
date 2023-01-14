use std::ffi::CStr;
use std::sync::{Arc, Mutex};

use super::core::Core;
use super::renderpass::{RenderTargetInfo, Renderpass};

use ash::vk::{self, DescriptorSetLayout};
use bytemuck::{Pod, Zeroable};

pub struct PipelineManager {
    pipeline_layouts: Mutex<Vec<vk::PipelineLayout>>,
}

impl PipelineManager {
    pub fn cleanup(&self, core: &Core) {
        let pipeline_layouts = self.pipeline_layouts.lock().unwrap();
        let device = core.device();

        for pl in pipeline_layouts.iter() {
            unsafe {
                device.destroy_pipeline_layout(*pl, None);
            }
        }
    }
    pub fn new(_device: &ash::Device) -> Self { Self { pipeline_layouts: Mutex::new(vec![]) } }
}

impl Core {
    fn create_pipeline_layout(&self, builder: &PipelineLayoutBuilder) -> Result<vk::PipelineLayout, vk::Result> {
        let info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(&builder.push_constants)
            .set_layouts(&builder.descriptor_layouts)
            .build();

        let layout = unsafe { self.device().create_pipeline_layout(&info, None) }?;

        self.pipeline_manager.pipeline_layouts.lock().unwrap().push(layout);

        Ok(layout)
    }

    fn pipeline_cache(&self) -> vk::PipelineCache { vk::PipelineCache::null() }
}

pub struct PipelineLayoutBuilder {
    push_constants: Vec<vk::PushConstantRange>,
    descriptor_layouts: Vec<vk::DescriptorSetLayout>,
}

impl PipelineLayoutBuilder {
    pub fn new() -> Self { Self { push_constants: vec![], descriptor_layouts: vec![] } }

    pub fn build(&self, core: &Core) -> Result<vk::PipelineLayout, vk::Result> { core.create_pipeline_layout(self) }

    pub fn add_push_with_size(&mut self, stage: vk::ShaderStageFlags, offset: u32, size: u32) -> &mut PipelineLayoutBuilder {
        self.push_constants.push(vk::PushConstantRange { offset, size, stage_flags: stage });
        self
    }

    pub fn add_push<T>(&mut self, stage: vk::ShaderStageFlags, offset: u32) -> &mut PipelineLayoutBuilder {
        self.push_constants.push(vk::PushConstantRange {
            offset,
            size: std::mem::size_of::<T>() as u32,
            stage_flags: stage,
        });
        self
    }

    pub fn add_set(&mut self, set: DescriptorSetLayout) -> &mut PipelineLayoutBuilder {
        self.descriptor_layouts.push(set);
        self
    }
}

pub struct ShaderModule {
    core: Arc<Core>,
    module: vk::ShaderModule,
}

impl ShaderModule {
    pub fn new<'a>(core: &Arc<Core>, spirv_code: &[u32]) -> Result<ShaderModule, vk::Result> {
        unsafe {
            let module = core.device().create_shader_module(
                &vk::ShaderModuleCreateInfo {
                    code_size: spirv_code.len() * 4,
                    p_code: spirv_code.as_ptr(),
                    ..Default::default()
                },
                None,
            )?;
            Ok(ShaderModule { module, core: core.clone() })
        }
    }

    pub fn module(&self) -> vk::ShaderModule { self.module }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.core.device().destroy_shader_module(self.module, None);
        }
    }
}

#[derive(Default)]
pub struct GPipelineBuilder<'a> {
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo>,
    pipeline_layout: vk::PipelineLayout,
    depth_stencil: vk::PipelineDepthStencilStateCreateInfo,
    input_assembly: vk::PipelineInputAssemblyStateCreateInfo,
    rasterizer: vk::PipelineRasterizationStateCreateInfo,
    multisampling: vk::PipelineMultisampleStateCreateInfo,
    vertex_description: Option<VertexInputDescriptionBuilder>,
    descriptor_set_layouts: Option<Box<[vk::DescriptorSetLayout]>>,
    render_target: Option<&'a RenderTargetInfo>,
}

impl<'a> GPipelineBuilder<'a> {
    pub fn new() -> Self {
        Self {
            multisampling: vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                min_sample_shading: 1.0,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    pub fn set_pipeline_layout(&mut self, layout: vk::PipelineLayout) -> &mut Self {
        self.pipeline_layout = layout;
        self
    }

    pub fn add_shader_stage(&mut self, stage: vk::ShaderStageFlags, module: &vk::ShaderModule) -> &mut Self {
        self.shader_stages.push(vk::PipelineShaderStageCreateInfo {
            stage,
            module: *module,
            p_name: b"main\0".as_ptr() as *const i8,

            ..Default::default()
        });
        self
        // self.marker.push(module);
    }

    pub fn set_render_target(&mut self, render_target: &'a RenderTargetInfo) -> &mut Self {
        self.render_target = Some(render_target);
        self
    }

    pub fn set_topology(&mut self, topology: vk::PrimitiveTopology) -> &mut Self {
        self.input_assembly.topology = topology;
        self
    }
    pub fn set_rasterization(&mut self, polygon_mode: vk::PolygonMode, cull_mode: vk::CullModeFlags) -> &mut Self {
        self.rasterizer =
            vk::PipelineRasterizationStateCreateInfo { polygon_mode, cull_mode, line_width: 1.0, ..Default::default() };
        self
    }
    pub fn set_depth_testing(&mut self, depth_testing: bool) -> &mut Self {
        self.depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: depth_testing as u32,
            depth_write_enable: depth_testing as u32,
            depth_compare_op: if depth_testing { vk::CompareOp::LESS_OR_EQUAL } else { vk::CompareOp::ALWAYS },
            ..Default::default()
        };
        self
    }

    pub fn set_vertex_description(&mut self, desc: VertexInputDescriptionBuilder) -> &mut Self {
        self.vertex_description = Some(desc);
        self
    }

    pub fn set_descriptor_set_layouts(&mut self, layouts: Box<[DescriptorSetLayout]>) {
        self.descriptor_set_layouts = Some(layouts);
    }

    pub fn build(&self, core: &Arc<Core>) -> eyre::Result<Arc<Pipeline>, vk::Result> {
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder().scissor_count(1).viewport_count(1).build();

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let ds_info = vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states).build();

        let render_target = self.render_target.unwrap();

        let attachments = render_target
            .color_attachments
            .iter()
            .map(|_| vk::PipelineColorBlendAttachmentState {
                blend_enable: false as u32,
                color_write_mask: vk::ColorComponentFlags::RGBA,
                ..Default::default()
            })
            .collect::<Box<_>>();

        let color_blend = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(attachments.as_ref())
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .build();

        let empty_vinfo = VertexInputDescriptionBuilder::new();

        // let mut pipeline_render_create_info = vk::PipelineRenderingCreateInfo::builder()
        //     .color_attachment_formats(&render_target.color_attachments)
        //     .depth_attachment_format(render_target.depth_attachment.unwrap_or(vk::Format::UNDEFINED))
        //     .stencil_attachment_format(render_target.stencil_attachment.unwrap_or(vk::Format::UNDEFINED))
        //     .build();

        let create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&self.shader_stages)
            .vertex_input_state(&self.vertex_description.as_ref().unwrap_or(&empty_vinfo).info())
            .input_assembly_state(&self.input_assembly)
            .multisample_state(&self.multisampling)
            .depth_stencil_state(&self.depth_stencil)
            .dynamic_state(&ds_info)
            .layout(self.pipeline_layout)
            .rasterization_state(&self.rasterizer)
            .viewport_state(&viewport_state)
            .color_blend_state(&color_blend)
            .render_pass(render_target.get_render_pass())
            .subpass(render_target.get_subpass_index())
            // .push_next(&mut pipeline_render_create_info)
            .build();

        let pipeline = match unsafe { core.device().create_graphics_pipelines(core.pipeline_cache(), &[create_info], None) }
        {
            Ok(p) => p[0],
            Err((_, e)) => return Err(e),
        };

        Ok(Arc::new(Pipeline {
            core: core.clone(),
            pipeline,
            layout: self.pipeline_layout,
            ptype: vk::PipelineBindPoint::GRAPHICS,
            descriptor_set_layouts: self.descriptor_set_layouts.clone(),
        }))
    }
}

pub struct Pipeline {
    core: Arc<Core>,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub ptype: vk::PipelineBindPoint,
    descriptor_set_layouts: Option<Box<[vk::DescriptorSetLayout]>>,
}

impl Core {
    pub fn create_compute(
        self: &Arc<Core>,
        pipeline_layout: vk::PipelineLayout,
        shader_module: &ShaderModule,
        descriiptor_set_layouts: Box<[vk::DescriptorSetLayout]>,
    ) -> eyre::Result<Arc<Pipeline>, vk::Result> {
        Pipeline::new_compute(self, pipeline_layout, shader_module, descriiptor_set_layouts)
    }
}

impl Pipeline {
    pub fn new_compute(
        core: &Arc<Core>,
        pipeline_layout: vk::PipelineLayout,
        shader_module: &ShaderModule,
        descriiptor_set_layouts: Box<[vk::DescriptorSetLayout]>,
    ) -> eyre::Result<Arc<Pipeline>, vk::Result> {
        let pipeline = match unsafe {
            let shader_stage = vk::PipelineShaderStageCreateInfo::builder()
                .module(shader_module.module())
                .stage(vk::ShaderStageFlags::COMPUTE)
                .name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
                .build();

            core.device().create_compute_pipelines(
                core.pipeline_cache(),
                &[vk::ComputePipelineCreateInfo::builder().stage(shader_stage).layout(pipeline_layout).build()],
                None,
            )
        } {
            Ok(pipelines) => pipelines[0],
            Err((_, e)) => return Err(e),
        };

        Ok(Arc::new(Pipeline {
            core: core.clone(),
            pipeline,
            layout: pipeline_layout,
            ptype: vk::PipelineBindPoint::COMPUTE,
            descriptor_set_layouts: Some(descriiptor_set_layouts),
        }))
    }

    pub fn bind(self: &Arc<Self>, cmd: vk::CommandBuffer) {
        let device = self.core.device();
        unsafe {
            device.cmd_bind_pipeline(cmd, self.ptype, self.pipeline);
        }
    }

    pub fn push_constant<T>(&self, cmd: vk::CommandBuffer, push: &T, shader_stage: vk::ShaderStageFlags)
    where
        T: Pod,
    {
        let device = self.core.device();
        unsafe {
            device.cmd_push_constants(cmd, self.layout, shader_stage, 0, bytemuck::bytes_of(push));
        }
    }

    pub fn get_descriptor_set_layout(&self, index: u32) -> Option<DescriptorSetLayout> {
        self.descriptor_set_layouts.as_ref().and_then(|sls| sls.get(index as usize).map(|l| l.clone()))
    }

    // pub fn bind_descriptor(&self,cmd:vk::CommandBuffer){

    // }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.core.device().destroy_pipeline(self.pipeline, None);
        }
    }
}

#[derive(Debug, Clone)]
pub struct VertexInputDescriptionBuilder {
    bindings: Vec<vk::VertexInputBindingDescription>,
    attribures: Vec<vk::VertexInputAttributeDescription>,
}

pub trait GPUFormat {
    fn get_format() -> vk::Format;
}

impl VertexInputDescriptionBuilder {
    pub fn info(&self) -> vk::PipelineVertexInputStateCreateInfo {
        vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&self.bindings)
            .vertex_attribute_descriptions(&self.attribures)
            .build()
    }

    pub fn new() -> Self { Self { bindings: vec![], attribures: vec![] } }

    pub fn push_binding<T>(&mut self, input_rate: vk::VertexInputRate)
    where
        T: Pod,
    {
        self.bindings.push(vk::VertexInputBindingDescription {
            binding: self.bindings.len() as u32,
            input_rate,
            stride: std::mem::size_of::<T>() as u32,
        });
    }

    pub fn push_attribure<T>(&mut self, _: &T, offset: u32)
    where
        T: GPUFormat,
    {
        self.attribures.push(vk::VertexInputAttributeDescription {
            format: T::get_format(),
            binding: self.bindings.last().unwrap().binding,
            offset,
            location: self.attribures.len() as u32,
        });
    }
}

impl GPUFormat for u32 {
    fn get_format() -> vk::Format { vk::Format::R32_UINT }
}

impl GPUFormat for f32 {
    fn get_format() -> vk::Format { vk::Format::R32_SFLOAT }
}

impl GPUFormat for [f32; 2] {
    fn get_format() -> vk::Format { vk::Format::R32G32_SFLOAT }
}

impl GPUFormat for [f32; 3] {
    fn get_format() -> vk::Format { vk::Format::R32G32B32_SFLOAT }
}

impl GPUFormat for [f32; 4] {
    fn get_format() -> vk::Format { vk::Format::R32G32B32A32_SFLOAT }
}

pub trait VertexDescription: Zeroable {
    fn get_desciption() -> VertexInputDescriptionBuilder;
}

#[macro_export]
macro_rules! auto_description {
    (
        $( #[$meta:meta] )*
        $(pub)? struct $type_name:ident {
            $($field:ident : $type:ty),* $(,)?
        }
    ) => {
        $( #[$meta] )*
        pub struct $type_name{
            $($field : $type,)*
        }

        impl magma_renderer::core::VertexDescription for $type_name {
            fn get_desciption() -> magma_renderer::core::VertexInputDescriptionBuilder {
                use ash::vk;
                use bytemuck::offset_of;

                let mut builder = magma_renderer::core::VertexInputDescriptionBuilder::new();
                let vert = $type_name::zeroed();
                builder.push_binding::<$type_name>(vk::VertexInputRate::VERTEX);
                $(
                    builder.push_attribure(&vert.$field,offset_of!(vert,$type_name,$field) as u32);
                )*
                builder
            }
        }
    };
}

// pub(crate) use auto_description;
