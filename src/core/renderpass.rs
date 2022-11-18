use std::{
    cell::{Cell, RefCell},
    error,
    ptr::null,
    sync::Arc,
};

use ash::vk::{self};
use eyre::Result;

use super::{
    core::{Core, Surface},
    Image,
};

enum AttachmentType {
    External,
}

struct Attachment {}

pub struct Subpass {
    attachment_formats: Box<[vk::Format]>,
}

impl Subpass {
    pub fn get_attachments(&self) -> &[vk::Format] { &self.attachment_formats }
}

pub trait Renderpass {
    fn get_subpasses(&self) -> &[Subpass];
    fn vk_renderpas(&self) -> vk::RenderPass;
    fn extends(&self) -> (u32, u32);
    fn core(&self) -> &Arc<Core>;
    fn framebuffer(&self) -> vk::Framebuffer;
    fn clear_values(&self) -> &[vk::ClearValue];
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
                .iter()
                .map(|s| Subpass {
                    attachment_formats: unsafe {
                        std::slice::from_raw_parts(s.p_color_attachments, s.color_attachment_count as usize)
                    }
                    .iter()
                    .map(|aref| attachments[aref.attachment as usize].format)
                    .collect(),
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

    // fn begin(&self, cmd: vk::CommandBuffer) { SurfaceRenderpass::begin(&self, cmd); }
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

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
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

// impl dyn Renderpass {
//     fn new_singlepass(core: Arc<Core>, attachments: Vec<Attachment>, depth_attachment: Option<Attachment>) {}
// }
