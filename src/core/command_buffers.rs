use std::any::Any;

use crate::prelude::*;

use super::{buffer::*, cast_to_static_lifetime, core::*, renderpass, Pipeline};

use bytemuck::Pod;
use smallvec::SmallVec;

impl Core {
    pub fn create_command_pool(self: &Arc<Self>) -> Arc<CommandPool> { CommandPool::new(&self) }
}

pub struct CommandPool {
    core: Arc<Core>,
    pool: vk::CommandPool,
}

impl CommandPool {
    fn new(core: &Arc<Core>) -> Arc<CommandPool> {
        let device = core.device();

        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(core.queue_index())
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .build(), //
                None,
            )
        }
        .expect("failed to allocate command pool");

        Arc::new(CommandPool { core: core.clone(), pool })
    }

    fn allocate_cmd_internal(self: &Arc<Self>, level: vk::CommandBufferLevel, count: u32) -> Vec<vk::CommandBuffer> {
        let device = self.core.device();
        unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(self.pool)
                    .command_buffer_count(count)
                    .level(level)
                    .build(),
            )
        }
        .expect("failed to allocate command buffers")
    }

    pub fn allocate_cmd(self: &Arc<Self>) -> CommandBuffer {
        CommandBuffer {
            cmd: self.allocate_cmd_internal(vk::CommandBufferLevel::PRIMARY, 1)[0],
            pool: self.clone(),
            dependencies: vec![],
            bound_pipeline: None,
            device: unsafe { cast_to_static_lifetime(self.core.device()) },
            bound_descriptor_sets: [vk::DescriptorSet::null(); 4],
        }
    }

    pub fn allocate_secondry_cmd(self: &Arc<Self>) -> CommandBuffer {
        CommandBuffer {
            cmd: self.allocate_cmd_internal(vk::CommandBufferLevel::SECONDARY, 1)[0],
            pool: self.clone(),
            dependencies: vec![],
            bound_pipeline: None,
            device: unsafe { cast_to_static_lifetime(self.core.device()) },
            bound_descriptor_sets: [vk::DescriptorSet::null(); 4],
        }
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        let device = self.core.device();
        unsafe {
            device.destroy_command_pool(self.pool, None);
        }
    }
}

pub struct CommandBuffer {
    pool: Arc<CommandPool>,
    cmd: vk::CommandBuffer,
    dependencies: Vec<Box<dyn Any>>,
    bound_descriptor_sets: [vk::DescriptorSet; 4],
    bound_pipeline: Option<Arc<Pipeline>>,
    device: &'static ash::Device,
}

unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

pub trait IndexType {
    fn index_type() -> vk::IndexType;
}

impl IndexType for u16 {
    fn index_type() -> vk::IndexType { vk::IndexType::UINT16 }
}
impl IndexType for u32 {
    fn index_type() -> vk::IndexType { vk::IndexType::UINT32 }
}

impl CommandBuffer {
    pub fn new_secondry(core: &Arc<Core>) -> CommandBuffer { core.create_command_pool().allocate_secondry_cmd() }
    pub fn new(core: &Arc<Core>) -> CommandBuffer { core.create_command_pool().allocate_cmd() }
    pub fn reset(&mut self) { todo!() }

    pub fn device(&self) -> &ash::Device { self.device }
    pub fn core(&self) -> &Arc<Core> { &self.pool.core }

    pub fn inner(&self) -> ash::vk::CommandBuffer { self.cmd }
    pub fn add_dependency<T>(&mut self, d: T)
    where
        T: Any + 'static,
    {
        self.dependencies.push(Box::new(d));
    }

    pub fn begin(&self) -> Result<(), ash::vk::Result> {
        unsafe {
            self.device().begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT).build(),
            )
        }
    }

    pub fn begin_secondry(&mut self, subpass: Option<(&dyn renderpass::Renderpass, u32)>) -> Result<(), vk::Result> {
        let intheritence_info = if let Some((rp, spindex)) = subpass {
            vk::CommandBufferInheritanceInfo::builder().render_pass(rp.vk_renderpas()).subpass(spindex)
        } else {
            vk::CommandBufferInheritanceInfo::builder()
        }
        .build();

        unsafe {
            self.device().begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(
                        vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
                            | subpass.map_or(vk::CommandBufferUsageFlags::default(), |_| {
                                vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE
                            }),
                    )
                    .inheritance_info(&intheritence_info)
                    .build(),
            )?
        };

        if let Some((rp, _)) = subpass {
            rp.set_scissor_and_viewport(self.cmd);
        };

        Ok(())
    }

    pub fn exectue_secondry(&mut self, cmd: CommandBuffer) {
        unsafe { self.device().cmd_execute_commands(self.inner(), &[cmd.inner()]) };
        self.add_dependency(cmd);
    }

    pub fn exectue_secondries(&mut self, cmds: Vec<CommandBuffer>) {
        if cmds.len() == 0 {
            return;
        }
        let inner_cmds: SmallVec<[_; 16]> = cmds.iter().map(|c| c.inner()).collect();
        unsafe { self.device().cmd_execute_commands(self.inner(), &inner_cmds) };

        self.add_dependency(cmds);
    }

    pub fn end(&self) -> Result<(), ash::vk::Result> { unsafe { self.pool.core.device().end_command_buffer(self.cmd) } }

    pub fn bind_vertex_buffers(&mut self, buffers: &[&dyn RawBufferSlice]) {
        let vbuffers: SmallVec<[_; 6]> = buffers.iter().map(|rb| rb.raw_buffer()).collect();
        let offsets: SmallVec<[u64; 6]> = buffers.iter().map(|rb| rb.byte_offset() as u64).collect();

        unsafe {
            self.device().cmd_bind_vertex_buffers(self.inner(), 0, &vbuffers, &offsets);
        }
    }

    pub fn bind_index_buffer<T: IndexType + Pod>(&mut self, buffer: BufferSlice<T>) {
        unsafe {
            self.device().cmd_bind_index_buffer(
                self.inner(),
                buffer.raw_buffer(),
                buffer.byte_offset() as u64,
                T::index_type(),
            );
        }
    }

    pub fn bind_pipeline(&mut self, pipeline: &Arc<Pipeline>) {
        if let Some(bound_pipeline) = &self.bound_pipeline {
            if bound_pipeline.pipeline == pipeline.pipeline {
                //do nothing if pipeline is already bound
                return;
            }
        } else {
            //if the pipeline is'nt already bound bind the descriptor sets
            for (i, dset) in self.bound_descriptor_sets.iter().enumerate() {
                if *dset == vk::DescriptorSet::null() {
                    continue;
                }

                unsafe {
                    self.device.cmd_bind_descriptor_sets(
                        self.inner(),
                        pipeline.ptype,
                        pipeline.layout,
                        i as u32,
                        &[*dset],
                        &[],
                    )
                }
            }
        }

        pipeline.bind(self.inner());
        self.bound_pipeline.take().map(|p| self.add_dependency(p));

        self.bound_pipeline = Some(pipeline.clone());
    }

    pub fn bind_descriptor_set(&mut self, index: u32, dset: vk::DescriptorSet) {
        if let Some(bound_set) = self.bound_descriptor_sets.get_mut(index as usize) {
            if *bound_set == dset {
                return;
            } else {
                *bound_set = dset;
            }
        }

        let Some(pipeline) = &self.bound_pipeline else{return};

        unsafe { self.device.cmd_bind_descriptor_sets(self.inner(), pipeline.ptype, pipeline.layout, index, &[dset], &[]) }
    }

    pub fn push_constant<T>(&mut self, push: &T, stage_flags: vk::ShaderStageFlags, offset: u32)
    where
        T: Pod,
    {
        unsafe {
            self.device().cmd_push_constants(
                self.cmd,
                self.get_pipeline().layout,
                stage_flags,
                offset,
                bytemuck::cast_slice(&[*push]),
            );
        }
    }

    pub fn copy_buffers(&mut self, src: &dyn RawBufferSlice, dst: &dyn RawBufferSlice) {
        unsafe {
            assert!(src.byte_size() <= dst.byte_size(), "copy source cannot be bigger than copy destination!");

            self.device().cmd_copy_buffer(
                self.inner(),
                src.raw_buffer(),
                dst.raw_buffer(),
                &[vk::BufferCopy { src_offset: src.byte_offset(), dst_offset: dst.byte_offset(), size: src.byte_size() }],
            );
        }
    }

    pub fn gpu_buffer_from_slice<T: Pod>(&mut self, usage: vk::BufferUsageFlags, data: &[T]) -> eyre::Result<Buffer<T>> {
        let src: Buffer<T> = self.core().create_buffer_from_slice(usage | vk::BufferUsageFlags::TRANSFER_SRC, data)?;
        let dst: Buffer<T> =
            self.core().create_buffer(usage | vk::BufferUsageFlags::TRANSFER_DST, data.len() as u32, false)?;

        self.copy_buffers(&src, &dst);
        self.add_dependency(src);

        Ok(dst)
    }

    pub fn gpu_buffer_from_iter<T: Pod>(
        &mut self,
        usage: vk::BufferUsageFlags,
        iter: impl Iterator<Item = T>,
    ) -> Result<Buffer<T>> {
        let src: Buffer<T> = self.core().create_buffer_from_iterator(usage | vk::BufferUsageFlags::TRANSFER_SRC, iter)?;
        let dst: Buffer<T> = self.core().create_buffer(usage | vk::BufferUsageFlags::TRANSFER_DST, src.size(), false)?;

        self.copy_buffers(&src, &dst);
        self.add_dependency(src);

        Ok(dst)
    }

    fn get_pipeline(&self) -> &Arc<Pipeline> { &self.bound_pipeline.as_ref().expect("pipeline must be bound first") }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        let device = self.pool.core.device();
        unsafe {
            device.free_command_buffers(self.pool.pool, &[self.cmd]);
        }
    }
}

impl CommandBuffer {
    pub fn immediate_submit(self) -> eyre::Result<()> {
        let mut fence = self.core().create_fence();

        unsafe {
            fence.queue_submit(
                self.core().queue(),
                &[vk::SubmitInfo::builder().command_buffers(&[self.inner()]).build()],
                Box::new(self),
            )?;
        }

        fence.try_wait(None)?;

        Ok(())
    }
}

//draw commands and & dispatch
impl CommandBuffer {
    pub unsafe fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
        self.device().cmd_draw(self.inner(), vertex_count, instance_count, first_vertex, first_instance);
    }

    pub unsafe fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        self.device().cmd_draw_indexed(self.inner(), index_count, instance_count, first_index, vertex_offset, first_instance)
    }

    pub unsafe fn draw_indexed_indirect_count(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count_buffer: vk::Buffer,
        count_buffer_offset: vk::DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) {
        self.device().cmd_draw_indexed_indirect_count(
            self.inner(),
            buffer,
            offset,
            count_buffer,
            count_buffer_offset,
            max_draw_count,
            stride,
        )
    }

    pub unsafe fn draw_indirect_count(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count_buffer: vk::Buffer,
        count_buffer_offset: vk::DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) {
        self.device().cmd_draw_indirect_count(
            self.inner(),
            buffer,
            offset,
            count_buffer,
            count_buffer_offset,
            max_draw_count,
            stride,
        )
    }

    pub unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        self.device().cmd_draw_indexed_indirect(self.inner(), buffer, offset, draw_count, stride)
    }

    pub unsafe fn draw_indirect(&mut self, buffer: vk::Buffer, offset: vk::DeviceSize, draw_count: u32, stride: u32) {
        self.device().cmd_draw_indirect(self.inner(), buffer, offset, draw_count, stride)
    }

    pub unsafe fn dispatch(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        self.device().cmd_dispatch(self.inner(), group_count_x, group_count_y, group_count_z)
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCmdDispatchIndirect.html>
    pub unsafe fn dispatch_indirect(&mut self, buffer: vk::Buffer, offset: vk::DeviceSize) {
        self.device().cmd_dispatch_indirect(self.inner(), buffer, offset)
    }

    pub unsafe fn dispatch_base(
        &mut self,
        base_group_x: u32,
        base_group_y: u32,
        base_group_z: u32,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        self.device().cmd_dispatch_base(
            self.inner(),
            base_group_x,
            base_group_y,
            base_group_z,
            group_count_x,
            group_count_y,
            group_count_z,
        )
    }

    pub unsafe fn copy_buffer_reigons(
        &mut self,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        regions: &[vk::BufferCopy],
    ) {
        if regions.len() == 0 {
            return;
        }

        self.device().cmd_copy_buffer(self.inner(), src_buffer, dst_buffer, regions);
    }

    pub unsafe fn pipeline_barrier(
        &mut self,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier],
        buffer_memory_barriers: &[vk::BufferMemoryBarrier],
        image_memory_barriers: &[vk::ImageMemoryBarrier],
    ) {
        self.device().cmd_pipeline_barrier(
            self.inner(),
            src_stage_mask,
            dst_stage_mask,
            dependency_flags,
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
        )
    }
}
