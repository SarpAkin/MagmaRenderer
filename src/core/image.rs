use crate::prelude::*;
use bytemuck::{Pod, cast_slice, cast_slice_mut};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc};
use std::{marker::PhantomData, mem::ManuallyDrop, fmt};

use super::Core;

pub struct Image {
    core: Arc<Core>,
    allocation: ManuallyDrop<Allocation>,
    image: vk::Image,
    view: vk::ImageView,
    format: vk::Format,
    width: u32,
    height: u32,
}

pub fn is_depth_format(format: vk::Format) -> bool {
    match format {
        vk::Format::D16_UNORM
        | vk::Format::D32_SFLOAT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => true,
        _ => false,
    }
}

impl Core {
    pub fn new_image(
        self: &Arc<Core>,
        format: vk::Format,
        flags: vk::ImageUsageFlags,
        width: u32,
        height: u32,
        layers: u32,
    ) -> eyre::Result<Image> {
        let image = unsafe {
            self.device().create_image(
                &vk::ImageCreateInfo::builder()
                    .array_layers(layers)
                    .format(format)
                    .image_type(vk::ImageType::TYPE_2D)
                    .extent(vk::Extent3D { width, height, depth: 1 })
                    .usage(flags)
                    .mip_levels(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .build(),
                None,
            )
        }?;

        let allocation = unsafe {
            let requirements = self.device().get_image_memory_requirements(image);

            let allocation = self.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
                name: "image alloc",
                requirements,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: false,
            })?;

            self.device().bind_image_memory(image, allocation.memory(), allocation.offset())?;

            allocation
        };

        let view = unsafe {
            self.device().create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .view_type(if layers >= 1 { vk::ImageViewType::TYPE_2D } else { vk::ImageViewType::TYPE_2D_ARRAY })
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: if is_depth_format(format) {
                            vk::ImageAspectFlags::DEPTH
                        } else {
                            vk::ImageAspectFlags::COLOR
                        },
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: layers,
                    })
                    .build(),
                None,
            )
        }?;

        Ok(Image { core: self.clone(), allocation: ManuallyDrop::new(allocation), image, view, format, width, height })
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        let device = self.core.device();
        // let allocation = std::mem::replace(&mut self.allocation, None).unwrap();
        unsafe {
            self.core.allocator.lock().unwrap().free(ManuallyDrop::take(&mut self.allocation)).unwrap();
            device.destroy_image(self.image, None);
            device.destroy_image_view(self.view, None);
        }
    }
}

impl Image {
    pub fn image(&self) -> vk::Image { self.image }
    pub fn view(&self) -> vk::ImageView { self.view }
    pub fn format(&self) -> vk::Format { self.format }
    pub fn extends(&self) -> (u32, u32) { (self.width, self.height) }
}
