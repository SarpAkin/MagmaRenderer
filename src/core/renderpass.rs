use std::{
    any::Any,
    cell::{Cell, RefCell},
    error,
    ptr::null,
    sync::Arc,
};

use ash::vk::{self};
use eyre::Result;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RenderTargetInfo {
    pub color_attachments: Box<[vk::Format]>,
    pub depth_attachment: Option<vk::Format>,
    pub stencil_attachment: Option<vk::Format>,
    renderpass:vk::RenderPass,
    subpass_index:u32,
}

impl RenderTargetInfo{
    pub fn get_subpass_index(&self) -> u32 {self.subpass_index}
    pub fn get_render_pass(&self) -> vk::RenderPass {self.renderpass}
}

use super::{
    core::{Core, Surface},
    is_depth_format, Image,
};

enum AttachmentType {
    External,
}

struct Attachment {}

pub struct Subpass {
    // attachment_formats: Box<[vk::Format]>,
    render_target: RenderTargetInfo,
}

impl Subpass {
    pub fn get_attachments(&self) -> &[vk::Format] { &self.render_target.color_attachments }
    pub fn get_render_target(&self) -> &RenderTargetInfo { &self.render_target }
}

pub trait Renderpass: Any {
    fn get_subpasses(&self) -> &[Subpass];
    fn vk_renderpas(&self) -> vk::RenderPass;
    fn extends(&self) -> (u32, u32);
    fn core(&self) -> &Arc<Core>;
    fn framebuffer(&self) -> vk::Framebuffer;
    fn clear_values(&self) -> &[vk::ClearValue];
    fn get_attachment<'a>(&'a self, index: AttachmentIndex) -> &'a Image;
    fn resize(&mut self, width: u32, height: u32) -> eyre::Result<()>;

    fn set_scissor_and_viewport(&self, cmd: vk::CommandBuffer) {
        let device = self.core().device();

        let (width, height) = self.extends();
        let render_area = vk::Rect2D { extent: vk::Extent2D { width, height }, offset: vk::Offset2D { x: 0, y: 0 } };

        unsafe {
            device.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: width as f32,
                    height: height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );

            device.cmd_set_scissor(cmd, 0, &[render_area]);
        }
    }

    fn begin(&self, cmd: vk::CommandBuffer, inline: bool) {
        let device = self.core().device();

        let (width, height) = self.extends();
        let render_area = vk::Rect2D { extent: vk::Extent2D { width, height }, offset: vk::Offset2D { x: 0, y: 0 } };

        unsafe {
            device.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(self.vk_renderpas())
                    .framebuffer(self.framebuffer())
                    .clear_values(self.clear_values())
                    .render_area(render_area)
                    .build(),
                inline.then_some(vk::SubpassContents::INLINE).unwrap_or(vk::SubpassContents::SECONDARY_COMMAND_BUFFERS),
            );
        }

        if inline {
            self.set_scissor_and_viewport(cmd);
        }
    }

    fn next(&self, cmd: vk::CommandBuffer, inline: bool) {
        let device = self.core().device();

        unsafe {
            device.cmd_next_subpass(
                cmd,
                inline.then_some(vk::SubpassContents::INLINE).unwrap_or(vk::SubpassContents::SECONDARY_COMMAND_BUFFERS),
            );
        }
    }

    fn end(&self, cmd: vk::CommandBuffer) {
        let device = self.core().device();

        unsafe {
            device.cmd_end_render_pass(cmd);
        }
    }
}

pub struct SurfaceRenderpass {
    surface: Surface,
    core: Arc<Core>,
    renderpass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    framebuffer_index: Cell<u32>,
    clear_values: Vec<vk::ClearValue>,
    depth_image: Image,
    subpasses: Box<[Subpass]>,
}

impl SurfaceRenderpass {
    fn create_framebuffers(&mut self) -> Result<(), vk::Result> {
        self.framebuffers = self
            .surface
            .swapchain_images
            .iter()
            .map(|(_, view)| unsafe {
                let views = [*view, self.depth_image.view()];
                self.core.device().create_framebuffer(
                    &vk::FramebufferCreateInfo::builder()
                        .render_pass(self.renderpass)
                        .attachments(&views)
                        .width(self.surface.width)
                        .height(self.surface.height)
                        .layers(1)
                        .build(),
                    None,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(())
    }

    pub fn new(core: Arc<Core>, surface: Surface) -> eyre::Result<SurfaceRenderpass> {
        let depth_image = core.new_image(
            vk::Format::D16_UNORM,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            surface.width,
            surface.height,
            1,
        )?;

        let attachments = [
            //
            vk::AttachmentDescription::builder()
                .format(surface.format())
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                // .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                // .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .samples(vk::SampleCountFlags::TYPE_1)
                .build(),
            vk::AttachmentDescription::builder()
                .format(vk::Format::D16_UNORM)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                // .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                // .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .samples(vk::SampleCountFlags::TYPE_1)
                .build(),
        ];

        let color_attachment_references =
            [vk::AttachmentReference { attachment: 0, layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL }];

        let depth_attachment_reference =
            vk::AttachmentReference { attachment: 1, layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

        let subpasses = [vk::SubpassDescription::builder()
            .color_attachments(&color_attachment_references)
            .depth_stencil_attachment(&depth_attachment_reference)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .build()];

        let renderpass_info = vk::RenderPassCreateInfo::builder().attachments(&attachments).subpasses(&subpasses).build();

        let renderpass = unsafe { core.device().create_render_pass(&renderpass_info, None)? };

        // let clear_values = surface
        //     .swapchain_images
        //     .iter()
        //     .map(|_| vk::ClearValue {
        //         color: vk::ClearColorValue {
        //             float32: [0.0, 0.0, 0.0, 0.0],
        //         },
        //     })
        //     .collect::<Vec<_>>();

        let mut rp = SurfaceRenderpass {
            depth_image,
            core,
            surface,
            renderpass,
            framebuffers: vec![],
            framebuffer_index: Cell::new(0),
            clear_values: vec![
                vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] } },
                vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
            ],
            subpasses: subpasses
                .iter().enumerate()
                .map(|(subpass_index,s)| Subpass {
                    render_target: RenderTargetInfo {
                        //
                        color_attachments: unsafe {
                            std::slice::from_raw_parts(s.p_color_attachments, s.color_attachment_count as usize)
                        }
                        .iter()
                        .map(|aref| attachments[aref.attachment as usize].format)
                        .collect(),
                        depth_attachment: unsafe { s.p_depth_stencil_attachment.as_ref() }
                            .map(|aref| attachments[aref.attachment as usize].format),
                        stencil_attachment: None,
                        renderpass,
                        subpass_index:subpass_index as u32,
                    },
                })
                .collect(),
        };

        rp.create_framebuffers()?;

        Ok(rp)
    }

    fn destroy_framebuffers(&mut self) {
        let device = self.core.device();
        unsafe {
            for fb in &mut self.framebuffers {
                device.destroy_framebuffer(*fb, None);
            }
        }
    }
}

impl Drop for SurfaceRenderpass {
    fn drop(&mut self) {
        self.destroy_framebuffers();
        let device = self.core.device();
        unsafe {
            device.destroy_render_pass(self.renderpass, None);
        }
    }
}

impl Renderpass for SurfaceRenderpass {
    fn vk_renderpas(&self) -> vk::RenderPass { self.renderpass }
    fn extends(&self) -> (u32, u32) { (self.surface.width, self.surface.height) }
    fn core(&self) -> &Arc<Core> { &self.core }
    fn framebuffer(&self) -> vk::Framebuffer {
        let index = self.framebuffer_index.get();

        // self.framebuffer_index.set((index + 1) % self.framebuffers.len() as u32);

        self.framebuffers[index as usize]
    }
    fn clear_values(&self) -> &[vk::ClearValue] { self.clear_values.as_slice() }

    fn get_subpasses(&self) -> &[Subpass] { &self.subpasses }

    fn get_attachment(&self, index: AttachmentIndex) -> &Image { todo!() }

    fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        self.surface.recrate_swapchain();
        self.surface.width = width;
        self.surface.height = height;

        self.depth_image = self.core.new_image(
            vk::Format::D16_UNORM,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            self.surface.width,
            self.surface.height,
            1,
        )?;

        self.destroy_framebuffers();
        self.create_framebuffers()?;

        Ok(())
    }
}

impl SurfaceRenderpass {
    pub fn prepare(&self, semaphore: vk::Semaphore) -> Result<(), vk::Result> {
        let (index, is_suboptimal) = unsafe {
            self.surface.swapchain_loader.acquire_next_image(
                self.surface.swapchain,
                10_000_000_000,
                semaphore,
                vk::Fence::null(),
            )
        }?;

        if is_suboptimal {
            println!("suboptimal");
        }

        self.framebuffer_index.set(index);

        Ok(())
    }

    pub fn present(&self, semaphore: vk::Semaphore) -> Result<(), vk::Result> {
        unsafe {
            self.surface.swapchain_loader.queue_present(
                self.core.queue(),
                &vk::PresentInfoKHR::builder()
                    .wait_semaphores(&[semaphore])
                    .swapchains(&[self.surface.swapchain])
                    .image_indices(&[self.framebuffer_index.get()])
                    .build(),
            )
        }?;

        Ok(())
    }
}

// impl dyn Renderpass {
//     fn new_singlepass(core: Arc<Core>, attachments: Vec<Attachment>, depth_attachment: Option<Attachment>) {}
// }

struct AttachmentInfo {
    description: vk::AttachmentDescription,
    sampled: bool,
    is_input_attachment: bool,
}

struct SubpassDescription {
    attachments: Vec<vk::AttachmentReference>,
    depth_attachment: Option<vk::AttachmentReference>,
    input_attachments: Vec<vk::AttachmentReference>,
}

impl SubpassDescription {
    fn get_description(&self) -> vk::SubpassDescription {
        let builder = vk::SubpassDescription::builder()
            .color_attachments(&self.attachments)
            .input_attachments(&self.input_attachments);

        if let Some(d) = &self.depth_attachment { builder.depth_stencil_attachment(d).build() } else { builder.build() }
    }
}

pub struct RenderPassBuilder {
    attachments: Vec<AttachmentInfo>,
    subpasses: Vec<SubpassDescription>,
    clear_values: Vec<vk::ClearValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttachmentIndex(u32);

impl RenderPassBuilder {
    pub fn new() -> RenderPassBuilder { Self { attachments: vec![], subpasses: vec![], clear_values: vec![] } }

    pub fn add_attachment(
        &mut self,
        format: vk::Format,
        clear_value: Option<vk::ClearValue>,
        is_sampled: bool,
    ) -> AttachmentIndex {
        self.attachments.push(AttachmentInfo {
            description: vk::AttachmentDescription {
                flags: vk::AttachmentDescriptionFlags::empty(),
                format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: if let Some(_) = clear_value {
                    vk::AttachmentLoadOp::CLEAR
                } else {
                    vk::AttachmentLoadOp::DONT_CARE
                },
                store_op: vk::AttachmentStoreOp::STORE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: if is_sampled {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                } else {
                    if is_depth_format(format) {
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                    } else {
                        vk::ImageLayout::ATTACHMENT_OPTIMAL
                    }
                },
            },
            sampled: is_sampled,
            is_input_attachment: false,
        });
        self.clear_values.push(clear_value.unwrap_or_default());

        AttachmentIndex((self.attachments.len() - 1) as u32)
    }

    pub fn add_subpass(
        &mut self,
        attachments: &[AttachmentIndex],
        depth_attachment: Option<AttachmentIndex>,
        input_attachments: &[AttachmentIndex],
    ) -> u32 {
        self.subpasses.push(SubpassDescription {
            attachments: attachments
                .iter()
                .map(|i| vk::AttachmentReference { attachment: i.0, layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL })
                .collect(),
            depth_attachment: depth_attachment.map(|i| vk::AttachmentReference {
                attachment: i.0,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            }),
            input_attachments: input_attachments
                .iter()
                .map(|i| vk::AttachmentReference { attachment: i.0, layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL })
                .collect(),
        });

        for iatt in input_attachments {
            self.attachments[iatt.0 as usize].is_input_attachment = true;
        }

        self.subpasses.len() as u32 - 1
    }

    pub fn build(self, core: &Arc<Core>, width: u32, height: u32) -> eyre::Result<MultiPassRenderPass> {
        let dependencies = self.create_subpass_dependencies();

        let subpasses: Vec<_> = self.subpasses.iter().map(|s| s.get_description()).collect();
        let attachments: Vec<_> = self.attachments.iter().map(|a| a.description).collect();

        let render_pass = unsafe {
            core.device().create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&attachments)
                    .subpasses(&subpasses)
                    .dependencies(&dependencies)
                    .build(),
                None,
            )
        }?;

        let mut mrender_pass = MultiPassRenderPass {
            core: core.clone(),
            render_pass,
            clear_values: self.clear_values,
            width,
            height,
            framebuffer: None,
            subpasses: subpasses
                .iter().enumerate()
                .map(|(subpass_index,s)| Subpass {
                    render_target: RenderTargetInfo {
                        //
                        color_attachments: unsafe {
                            std::slice::from_raw_parts(s.p_color_attachments, s.color_attachment_count as usize)
                        }
                        .iter()
                        .map(|aref| attachments[aref.attachment as usize].format)
                        .collect(),
                        depth_attachment: unsafe { s.p_depth_stencil_attachment.as_ref() }
                            .map(|aref| attachments[aref.attachment as usize].format),
                        stencil_attachment: None,
                        renderpass:render_pass,
                        subpass_index:subpass_index as u32,
                    },
                })
                .collect(),
            attachments: self.attachments,
        };

        mrender_pass.init()?;

        Ok(mrender_pass)
    }
}

pub struct MultiPassRenderPass {
    core: Arc<Core>,
    render_pass: vk::RenderPass,
    clear_values: Vec<vk::ClearValue>,
    attachments: Vec<AttachmentInfo>,
    width: u32,
    height: u32,
    framebuffer: Option<Framebuffer>,
    subpasses: Box<[Subpass]>,
}

struct Framebuffer {
    framebuffer: vk::Framebuffer,
    images: Box<[Image]>,
    views: Box<[vk::ImageView]>,
    core: Arc<Core>,
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe {
            self.core.device().destroy_framebuffer(self.framebuffer, None);
        }
    }
}

impl MultiPassRenderPass {
    fn init(&mut self) -> eyre::Result<()> {
        self.create_framebuffers()?;
        Ok(())
    }

    fn create_framebuffers(&mut self) -> eyre::Result<()> {
        let images = self
            .attachments
            .iter()
            .map(|att| {
                let is_depth = is_depth_format(att.description.format);

                let mut flags = if is_depth {
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                } else {
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                };
                if att.is_input_attachment {
                    flags |= vk::ImageUsageFlags::INPUT_ATTACHMENT
                }
                if att.sampled {
                    flags |= vk::ImageUsageFlags::SAMPLED;
                }

                let image = self.core.new_image(att.description.format, flags, self.width, self.height, 1)?;

                Ok(image)
            })
            .collect::<Result<Box<[_]>>>()?;

        let views: Box<[_]> = images.iter().map(|i| i.view()).collect();

        let framebuffer = unsafe {
            self.core.device().create_framebuffer(
                &vk::FramebufferCreateInfo::builder()
                    .attachments(&views)
                    .render_pass(self.render_pass)
                    .width(self.width)
                    .height(self.height)
                    .layers(1)
                    .build(),
                None,
            )
        }?;

        self.framebuffer = Some(Framebuffer { framebuffer, images, views, core: self.core.clone() });

        Ok(())
    }
}

impl Renderpass for MultiPassRenderPass {
    fn get_subpasses(&self) -> &[Subpass] { &self.subpasses }

    fn vk_renderpas(&self) -> vk::RenderPass { self.render_pass }

    fn extends(&self) -> (u32, u32) { (self.width, self.height) }

    fn core(&self) -> &Arc<Core> { &self.core }

    fn framebuffer(&self) -> vk::Framebuffer { self.framebuffer.as_ref().unwrap().framebuffer }

    fn clear_values(&self) -> &[vk::ClearValue] { &self.clear_values }

    fn get_attachment(&self, index: AttachmentIndex) -> &Image {
        &self.framebuffer.as_ref().unwrap().images[index.0 as usize]
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        self.width = width;
        self.height = height;
        self.create_framebuffers()?;

        Ok(())
    }
}

impl Drop for MultiPassRenderPass {
    fn drop(&mut self) {
        unsafe {
            self.core.device().destroy_render_pass(self.render_pass, None);
        }
    }
}

impl RenderPassBuilder {
    fn create_subpass_dependencies(&self) -> Vec<vk::SubpassDependency> {
        let mut attachment_uses = Vec::new();
        attachment_uses.resize(self.attachments.len(), None);

        //create subpass dependencies
        let mut dependencies = Vec::new();
        for (i, subpass) in self.subpasses.iter().enumerate() {
            for att in &subpass.attachments {
                let att_use = &mut attachment_uses[att.attachment as usize];
                if let Some(usage) = att_use {
                    dependencies.push(vk::SubpassDependency {
                        src_subpass: *usage,
                        dst_subpass: i as u32,
                        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        dependency_flags: vk::DependencyFlags::BY_REGION,
                    });
                } else {
                    *att_use = Some(i as u32);
                }
            }

            if let Some(att) = &subpass.depth_attachment {
                let att_use = &mut attachment_uses[att.attachment as usize];
                if let Some(usage) = att_use {
                    dependencies.push(vk::SubpassDependency {
                        src_subpass: *usage,
                        dst_subpass: i as u32,
                        src_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dependency_flags: vk::DependencyFlags::BY_REGION,
                    });
                } else {
                    *att_use = Some(i as u32);
                }
            }

            for att in &subpass.input_attachments {
                let att_use = &mut attachment_uses[att.attachment as usize];
                let att_desc = &self.attachments[att.attachment as usize];
                let is_depth = is_depth_format(att_desc.description.format);

                if let Some(usage) = att_use {
                    dependencies.push(vk::SubpassDependency {
                        src_subpass: *usage,
                        dst_subpass: i as u32,
                        src_stage_mask: if is_depth {
                            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                        } else {
                            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        },
                        dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                        src_access_mask: if is_depth {
                            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                        } else {
                            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        },
                        dst_access_mask: vk::AccessFlags::INPUT_ATTACHMENT_READ,
                        dependency_flags: vk::DependencyFlags::BY_REGION,
                    });
                } else {
                    *att_use = Some(i as u32);
                }
            }
        }

        dependencies
    }
}
