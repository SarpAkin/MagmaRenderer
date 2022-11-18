use std::{fs::File, sync::Arc};

use super::*;

impl CommandBuffer {
    pub fn load_image_from_file(&mut self, file_path: &str,flags: vk::ImageUsageFlags) -> eyre::Result<Image> {
        let decoder = png::Decoder::new(File::open(file_path)?);
        let mut reader = decoder.read_info()?;
        let mut buffer =
            self.core().create_buffer(vk::BufferUsageFlags::TRANSFER_SRC, reader.output_buffer_size() as u32, true)?;
        let bytes: &mut [u8] = buffer.get_data_mut().unwrap();
        let out = reader.next_frame(bytes)?;
        
        println!("size {},type {:?}",bytes.len(),out.color_type);

        let format = match out.color_type {
            png::ColorType::Grayscale => todo!(),
            png::ColorType::Rgb => vk::Format::R8G8B8A8_SRGB,
            png::ColorType::Indexed => todo!(),
            png::ColorType::GrayscaleAlpha => todo!(),
            png::ColorType::Rgba => vk::Format::R8G8B8A8_UNORM,
        };

        let image = self.load_image_from_buffer(&buffer, format, out.width, out.height, flags)?;
        
        self.add_dependency(&Arc::new(buffer));

        Ok(image)
    }

    pub fn load_image_from_png_bytes(&mut self, bytes: &[u8]) -> eyre::Result<Image> { todo!() }

    pub fn load_image_from_buffer(
        &mut self,
        buffer: &dyn RawBufferSlice,
        format: vk::Format,
        width: u32,
        height: u32,
        flags: vk::ImageUsageFlags,
    ) -> eyre::Result<Image> {
        let core = self.core();
        let device = core.device();
        let image = core.new_image(format, vk::ImageUsageFlags::TRANSFER_DST | flags, width, height, 1)?;

        let mut barrier = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            image: image.image(),
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_queue_family_index:self.core().queue_index(),
            dst_queue_family_index:self.core().queue_index(),
            ..Default::default()
        };

        unsafe {
            device.cmd_pipeline_barrier(
                self.inner(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        let copy = vk::BufferImageCopy {
            buffer_offset: buffer.byte_offset(),
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D { width, height, depth: 1 },
        
        };

        unsafe {
            device.cmd_copy_buffer_to_image(
                self.inner(),
                buffer.raw_buffer(),
                image.image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
        }

        barrier.old_layout = barrier.new_layout;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = barrier.dst_access_mask;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        unsafe {
            device.cmd_pipeline_barrier(
                self.inner(),
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        Ok(image)
    }
}

impl Core {
    pub fn new_cmd(self: &Arc<Core>) -> CommandBuffer { CommandBuffer::new(self) }
    pub fn new_secondry_cmd(self: &Arc<Core>) -> CommandBuffer { CommandBuffer::new_secondry(self) }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::core::*;

    fn init_core() -> eyre::Result<Arc<Core>> {
        let (core, _) = unsafe { Core::new(None) };

        Ok(core)
    }

    #[test]
    fn load_image_test() -> eyre::Result<()> {
        let core = init_core()?;
        let mut cmd = core.new_cmd();
        cmd.begin()?;
        let image = cmd.load_image_from_file("res/a.png",vk::ImageUsageFlags::SAMPLED)?;

        cmd.end()?;
        cmd.immediate_submit()?;
        Ok(())
    }
}
