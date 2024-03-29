use crate::prelude::*;
use bytemuck::{cast_slice, cast_slice_mut, Pod};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc};
use std::{fmt, marker::PhantomData, mem::ManuallyDrop};

use super::Core;

struct ByteBuffer {
    buffer: vk::Buffer,
    allocation: ManuallyDrop<Allocation>,
    core: Arc<Core>,
    size: u64,
    usage: vk::BufferUsageFlags,
}

pub struct Buffer<T: Pod> {
    byte_buffer: ByteBuffer,
    size_in_items: u32,
    phantom: PhantomData<T>,
}

impl Core {
    fn create_byte_buffer(
        self: &Arc<Self>,
        usage: vk::BufferUsageFlags,
        size_in_bytes: u64,
        host_visible: bool,
    ) -> Result<ByteBuffer> {
        unsafe {
            let device = self.device();

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

            Ok(ByteBuffer {
                buffer,
                allocation: ManuallyDrop::new(allocation),
                core: self.clone(),
                size: size_in_bytes,
                usage,
            })
        }
    }

    pub fn create_buffer<T>(
        self: &Arc<Self>,
        usage: vk::BufferUsageFlags,
        size_in_items: u32,
        host_visible: bool,
    ) -> eyre::Result<Buffer<T>>
    where
        T: Pod,
    {
        let size_in_bytes = (std::mem::size_of::<T>() as u32) * size_in_items;

        Ok(Buffer {
            byte_buffer: self.create_byte_buffer(usage, size_in_bytes as u64, host_visible)?,
            size_in_items,
            phantom: PhantomData,
        })
    }

    //creates a host visible buffer from given slice
    pub fn create_buffer_from_slice<T: Pod>(
        self: &Arc<Self>,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> eyre::Result<Buffer<T>> {
        let mut buffer = self.create_buffer(usage, data.len() as u32, true)?;
        buffer.get_data_mut().unwrap()[0..data.len()].copy_from_slice(data);
        Ok(buffer)
    }

    //creates a host visible buffer from given iterator
    pub fn create_buffer_from_iterator<T: Pod>(
        self: &Arc<Self>,
        usage: vk::BufferUsageFlags,
        iter: impl Iterator<Item = T>,
    ) -> eyre::Result<Buffer<T>> {
        let (min_size, max_size) = iter.size_hint();
        if min_size != max_size.unwrap_or(usize::MAX) {
            let vec = iter.collect::<Vec<_>>();
            return self.create_buffer_from_slice(usage, &vec);
        }

        let mut buffer = self.create_buffer(usage, min_size as u32, true)?; //staging buffer

        let buffer_data = buffer.get_data_mut().unwrap();

        for (item, buffer_storage) in iter.zip(buffer_data) {
            *buffer_storage = item;
        }

        Ok(buffer)
    }
}

impl<T: Pod> Buffer<T> {
    pub fn get_usage(&self) -> vk::BufferUsageFlags { self.byte_buffer.usage }
    pub fn get_data(&self) -> Option<&[T]> {
        self.byte_buffer.allocation.mapped_slice().and_then(|s| Some(cast_slice::<u8, T>(s)))
    }

    pub fn get_data_mut(&mut self) -> Option<&mut [T]> {
        self.byte_buffer.allocation.mapped_slice_mut().and_then(|s| Some(cast_slice_mut::<u8, T>(s)))
    }

    pub fn inner(&self) -> vk::Buffer { self.byte_buffer.buffer }
    pub fn as_slice(&self) -> BufferSlice<T> {
        BufferSlice { buffer: self, offset_items: self.offset(), size_items: self.size() }
    }
    pub fn as_raw_buffer_slice(&self) -> &dyn RawBufferSlice {
        self as &dyn RawBufferSlice
    }

    pub fn core(&self) -> &Arc<Core> { &self.byte_buffer.core }

    pub(crate) fn cast<TOther: Pod>(self) -> Buffer<TOther> {
        Buffer {
            size_in_items: (self.byte_buffer.size as usize / std::mem::size_of::<TOther>()) as u32,
            byte_buffer: self.byte_buffer,
            phantom: PhantomData,
        }
    }
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

impl Drop for ByteBuffer {
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
    fn raw_buffer(&self) -> vk::Buffer { self.buffer().inner() }
    fn byte_size(&self) -> u64 { self.size() as u64 * std::mem::size_of::<T>() as u64 }
    fn byte_offset(&self) -> u64 { self.offset() as u64 * std::mem::size_of::<T>() as u64 }
}

impl<'a, T: Pod> RawBufferSlice for BufferSlice<'a, T> {
    fn raw_buffer(&self) -> vk::Buffer { self.buffer().inner() }
    fn byte_size(&self) -> u64 { self.size() as u64 * std::mem::size_of::<T>() as u64 }
    fn byte_offset(&self) -> u64 { self.offset() as u64 * std::mem::size_of::<T>() as u64 }
}

#[derive(Debug, Clone)]
pub struct BufferCreateError;

impl fmt::Display for BufferCreateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "failed to create buffer") }
}

impl std::error::Error for BufferCreateError {}
