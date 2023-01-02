use crate::prelude::*;
use bytemuck::{Pod, cast_slice, cast_slice_mut};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc};
use std::{marker::PhantomData, mem::ManuallyDrop, fmt};

use super::Core;

pub struct Buffer<T: Pod> {
    buffer: vk::Buffer,
    allocation: ManuallyDrop<Allocation>,
    core: Arc<Core>,
    size_in_bytes: u32,
    size_in_items: u32,
    phantom: PhantomData<T>,
    usage: vk::BufferUsageFlags,
}

impl Core {
    pub fn create_buffer<T>(
        self: &Arc<Self>,
        usage: vk::BufferUsageFlags,
        size_in_items: u32,
        host_visible: bool,
    ) -> eyre::Result<Buffer<T>>
    where
        T: Pod,
    {
        unsafe {
            let device = self.device();

            let size_in_bytes = (std::mem::size_of::<T>() as u32) * size_in_items;

            let binfo = ash::vk::BufferCreateInfo::builder().size(size_in_bytes as u64).usage(usage).build();

            let buffer = device.create_buffer(&binfo, None)?;

            ash::vk::MemoryAllocateInfo::builder();
            let requirements = device.get_buffer_memory_requirements(buffer);

            let allocation = self.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
                name: "buffer alloc",
                requirements,
                location: if host_visible {
                    gpu_allocator::MemoryLocation::CpuToGpu
                } else {
                    gpu_allocator::MemoryLocation::GpuOnly
                },
                linear: true,
            })?;

            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;

            Ok(Buffer {
                buffer,
                allocation: ManuallyDrop::new(allocation),
                core: self.clone(),
                size_in_bytes,
                size_in_items,
                phantom: PhantomData,
                usage,
            })
        }
    }

    pub fn create_buffer_from_slice<T: Pod>(
        self: &Arc<Self>,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> eyre::Result<Buffer<T>> {
        let mut buffer = self.create_buffer(usage, data.len() as u32, true)?;
        buffer.get_data_mut().unwrap()[0..data.len()].copy_from_slice(data);
        Ok(buffer)
    }
}

impl<T: Pod> Buffer<T> {
    pub fn get_usage(&self) -> vk::BufferUsageFlags { self.usage }
    pub fn get_data(&self) -> Option<&[T]> { self.allocation.mapped_slice().and_then(|s| Some(cast_slice::<u8, T>(s))) }

    pub fn get_data_mut(&mut self) -> Option<&mut [T]> {
        self.allocation.mapped_slice_mut().and_then(|s| Some(cast_slice_mut::<u8, T>(s)))
    }

    pub fn inner(&self) -> vk::Buffer { self.buffer }
    pub fn as_slice(&self) -> BufferSlice<T> {
        BufferSlice { buffer: self, offset_items: self.offset(), size_items: self.size() }
    }
    pub fn core(&self) -> &Arc<Core> { &self.core }
    // pub fn size(&self) -> u32 { self.size_in_items }
}

pub struct BufferSlice<'a, T: Pod> {
    buffer: &'a Buffer<T>,
    offset_items: u32,
    size_items: u32,
}

pub trait IBufferSlice<T: Pod> {
    fn buffer(&self) -> &Buffer<T>;
    fn size(&self) -> u32;
    fn offset(&self) -> u32;

    fn slice(&self, begin: u32, end: u32) -> Option<BufferSlice<T>> {
        if self.size() < begin + end {
            return None;
        }

        Some(BufferSlice { buffer: self.buffer(), offset_items: self.offset() + begin, size_items: end - begin })
    }
}

fn foo(buf: Buffer<u32>) { let a = buf.slice(0, 10).unwrap().slice(0, 10).unwrap(); }

impl<T: Pod> IBufferSlice<T> for Buffer<T> {
    fn buffer(&self) -> &Buffer<T> { self }
    fn size(&self) -> u32 { self.size_in_items }
    fn offset(&self) -> u32 { 0 }
}

impl<'a, T: Pod> IBufferSlice<T> for BufferSlice<'a, T> {
    fn buffer(&self) -> &Buffer<T> { self.buffer }
    fn size(&self) -> u32 { self.size_items }
    fn offset(&self) -> u32 { self.offset_items }
}

impl<T: Pod> Drop for Buffer<T> {
    fn drop(&mut self) {
        let device = self.core.device();
        let allocation = unsafe { ManuallyDrop::take(&mut self.allocation) };

        self.core.allocator.lock().unwrap().free(allocation).unwrap();
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
    }
}

pub trait RawBufferSlice {
    fn raw_buffer(&self) -> vk::Buffer;
    fn byte_size(&self) -> u64;
    fn byte_offset(&self) -> u64;
}

impl<T: Pod> RawBufferSlice for Buffer<T> {
    fn raw_buffer(&self) -> vk::Buffer { self.buffer().buffer }
    fn byte_size(&self) -> u64 { self.size() as u64 * std::mem::size_of::<T>() as u64 }
    fn byte_offset(&self) -> u64 { self.offset() as u64 * std::mem::size_of::<T>() as u64 }
}

impl<'a, T: Pod> RawBufferSlice for BufferSlice<'a, T> {
    fn raw_buffer(&self) -> vk::Buffer { self.buffer().buffer }
    fn byte_size(&self) -> u64 { self.size() as u64 * std::mem::size_of::<T>() as u64 }
    fn byte_offset(&self) -> u64 { self.offset() as u64 * std::mem::size_of::<T>() as u64 }
}

#[derive(Debug, Clone)]
pub struct BufferCreateError;

impl fmt::Display for BufferCreateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "failed to create buffer") }
}

impl std::error::Error for BufferCreateError {}
