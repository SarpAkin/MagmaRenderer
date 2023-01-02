use eyre::Result;
use smallvec::SmallVec;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use ash::vk::{self};

use super::{Buffer, Core, IBufferSlice, Image, RawBufferSlice};

pub struct DescriptorSetManager {
    layouts: Mutex<HashMap<Box<[u32]>, vk::DescriptorSetLayout>>,
}

impl DescriptorSetManager {
    pub fn new() -> DescriptorSetManager { DescriptorSetManager { layouts: Mutex::new(HashMap::new()) } }
    pub fn cleanup(&self, core: &Core) {
        let device = core.device();
        for (_, layout) in core.descriptor_set_manager.layouts.lock().unwrap().iter() {
            unsafe { device.destroy_descriptor_set_layout(*layout, None) };
        }
    }
}

pub struct DescriptorSetLayoutBuilder {
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl DescriptorSetLayoutBuilder {
    pub fn new() -> DescriptorSetLayoutBuilder { DescriptorSetLayoutBuilder { bindings: vec![] } }

    pub fn add_binding(
        &mut self,
        dtype: vk::DescriptorType,
        stage: vk::ShaderStageFlags,
        count: u32,
    ) -> &mut DescriptorSetLayoutBuilder {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::builder() //
                .descriptor_type(dtype)
                .descriptor_count(count)
                .stage_flags(stage)
                .binding(self.bindings.len() as u32)
                .build(),
        );
        self
    }
    pub fn add_ubo(&mut self, stage: vk::ShaderStageFlags, count: u32) -> &mut DescriptorSetLayoutBuilder {
        self.add_binding(vk::DescriptorType::UNIFORM_BUFFER, stage, count)
    }
    pub fn add_ssbo(&mut self, stage: vk::ShaderStageFlags, count: u32) -> &mut DescriptorSetLayoutBuilder {
        self.add_binding(vk::DescriptorType::STORAGE_BUFFER, stage, count)
    }
    pub fn add_sampler(&mut self, stage: vk::ShaderStageFlags, count: u32) -> &mut DescriptorSetLayoutBuilder {
        self.add_binding(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, stage, count)
    }
    pub fn add_input_attachement(&mut self, stage: vk::ShaderStageFlags, count: u32) -> &mut DescriptorSetLayoutBuilder {
        self.add_binding(vk::DescriptorType::UNIFORM_BUFFER, stage, count)
    }

    pub fn build(&self, core: &Arc<Core>) -> eyre::Result<vk::DescriptorSetLayout, vk::Result> {
        // let layout = ?;
        let layout = *core.descriptor_set_manager.layouts.lock().unwrap().entry(self.get_signature()).or_insert_with(|| {
            unsafe {
                core.device().create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&self.bindings).build(),
                    None,
                )
            }
            .unwrap()
        });
        Ok(layout)
    }

    fn get_signature(&self) -> Box<[u32]> {
        self.bindings
            .iter()
            .map(|b| {
                [
                    b.binding,
                    b.descriptor_count,
                    b.descriptor_type.as_raw() as u32,
                    b.stage_flags.as_raw() as u32,
                    (b.p_immutable_samplers as u64 >> 32) as u32,
                    b.p_immutable_samplers as u32,
                ]
            })
            .flatten()
            .collect()
    }
}

struct BufferBinding {
    buffer_infos: Box<[vk::DescriptorBufferInfo]>,
    binding: u32,
    dtype: vk::DescriptorType,
}
struct ImageBinding {
    image_infos: Box<[vk::DescriptorImageInfo]>,
    binding: u32,
    dtype: vk::DescriptorType,
}

pub struct DescriptorSetBuilder {
    buffer_bindings: Vec<BufferBinding>,
    image_bindings: Vec<ImageBinding>,
    binding_counter: u32,
}

impl DescriptorSetBuilder {
    pub fn new() -> DescriptorSetBuilder {
        DescriptorSetBuilder { buffer_bindings: vec![], image_bindings: vec![], binding_counter: 0 }
    }

    pub fn add_buffers(&mut self, dtype: vk::DescriptorType, buffers: &[&dyn RawBufferSlice]) -> &mut Self {
        self.buffer_bindings.push(BufferBinding {
            buffer_infos: buffers
                .iter()
                .map(|bs| vk::DescriptorBufferInfo {
                    buffer: bs.raw_buffer(),
                    offset: bs.byte_offset() as u64,
                    range: bs.byte_size() as u64,
                })
                .collect(),
            binding: self.binding_counter,
            dtype,
        });

        self.binding_counter += 1;
        self
    }

    pub fn add_ubo(&mut self, buffers: &[&dyn RawBufferSlice]) -> &mut DescriptorSetBuilder {
        self.add_buffers(vk::DescriptorType::UNIFORM_BUFFER, buffers)
    }

    pub fn add_ssbo(&mut self, buffers: &[&dyn RawBufferSlice]) -> &mut DescriptorSetBuilder {
        self.add_buffers(vk::DescriptorType::STORAGE_BUFFER, buffers)
    }

    pub fn add_image(
        &mut self,
        view: vk::ImageView,
        layout: vk::ImageLayout,
        sampler: vk::Sampler,
        dtype: vk::DescriptorType,
    ) -> &mut DescriptorSetBuilder {
        self.image_bindings.push(ImageBinding {
            image_infos: Box::new([vk::DescriptorImageInfo { sampler, image_view: view, image_layout: layout }]),
            binding: self.binding_counter,
            dtype,
        });
        self.binding_counter += 1;
        self
    }

    pub fn add_sampled_images(&mut self, images: &[(&Image, vk::Sampler)]) -> &mut DescriptorSetBuilder {
        self.image_bindings.push(ImageBinding {
            image_infos: images.iter().map(|(image, sampler)| vk::DescriptorImageInfo {
                sampler: *sampler,
                image_view: image.view(),
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }).collect(),
            binding: self.binding_counter,
            dtype: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        });
        self.binding_counter += 1;
        self
    }

    pub fn add_sampled_image(&mut self, image: &Image, sampler: vk::Sampler) -> &mut DescriptorSetBuilder {
        self.add_image(
            image.view(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            sampler,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        )
    }

    pub fn build(
        &self,
        layout: vk::DescriptorSetLayout,
        pool: &mut DescriptorPool,
    ) -> Result<vk::DescriptorSet, vk::Result> {
        let set = pool.allocate_set(layout, self)?;

        let writes: SmallVec<[_; 16]> = self
            .buffer_bindings
            .iter()
            .map(|b| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(b.binding)
                    .buffer_info(&b.buffer_infos)
                    .descriptor_type(b.dtype)
                    .build()
            })
            .chain(self.image_bindings.iter().map(|b| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(b.binding)
                    .image_info(&b.image_infos)
                    .descriptor_type(b.dtype)
                    .build()
            }))
            .collect();

        unsafe { pool.core.device().update_descriptor_sets(&writes, &[]) };

        Ok(set)
    }
}

pub struct DescriptorPool {
    free_pools: Vec<vk::DescriptorPool>,
    used_pools: Vec<vk::DescriptorPool>,
    core: Arc<Core>,
    max_sets: u32,
    // pool_sizes:&'static[(vk::DescriptorType,f32)],
}

impl DescriptorPool {
    pub fn new(core: &Arc<Core>) -> DescriptorPool {
        Self { core: core.clone(), free_pools: vec![], used_pools: vec![], max_sets: 100 }
    }

    pub fn reset(&mut self) -> Result<(), vk::Result> {
        self.free_pools.append(&mut self.used_pools);
        let device = self.core.device();
        for p in &self.free_pools {
            unsafe { device.reset_descriptor_pool(*p, vk::DescriptorPoolResetFlags::empty()) }?;
        }
        Ok(())
    }

    fn allocate_set(
        &mut self,
        layout: vk::DescriptorSetLayout,
        _descriptor_builder: &DescriptorSetBuilder,
    ) -> Result<vk::DescriptorSet, vk::Result> {
        if self.free_pools.len() == 0 {
            self.next_pool()?;
        }
        let pool = *self.free_pools.last().unwrap();

        let mut alloc_info = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(pool).set_layouts(&[layout]).build();

        let res = unsafe { self.core.device().allocate_descriptor_sets(&alloc_info) };
        match res {
            Ok(sets) => return Ok(sets[0]),
            Err(err) => match err {
                vk::Result::ERROR_FRAGMENTED_POOL | vk::Result::ERROR_OUT_OF_POOL_MEMORY => {
                    self.next_pool()?;
                    alloc_info.descriptor_pool = *self.free_pools.last().unwrap();
                    //if fails again return the error instead of recursing
                    return Ok(unsafe { self.core.device().allocate_descriptor_sets(&alloc_info) }?[0]);
                }
                other => return Err(other),
            },
        };
    }

    fn get_pool_size_multipliers(&self) -> &[(vk::DescriptorType, f32)] {
        &[
            (vk::DescriptorType::UNIFORM_BUFFER, 1.5),
            (vk::DescriptorType::STORAGE_BUFFER, 1.5),
            (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 2.5),
            (vk::DescriptorType::STORAGE_BUFFER, 1.5),
        ]
    }

    fn next_pool(&mut self) -> Result<(), vk::Result> {
        let new_pool = self.allocate_pool()?;
        if let Some(used) = self.free_pools.pop() {
            self.used_pools.push(used);
        }
        self.free_pools.push(new_pool);
        Ok(())
    }

    fn allocate_pool(&self) -> Result<vk::DescriptorPool, vk::Result> {
        let pool_sizes = self
            .get_pool_size_multipliers()
            .iter()
            .map(|(dt, mul)| -> vk::DescriptorPoolSize {
                vk::DescriptorPoolSize { descriptor_count: (self.max_sets as f32 * *mul) as u32, ty: *dt }
            })
            .collect::<Box<_>>();
        let device = self.core.device();
        unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::builder().max_sets(self.max_sets).pool_sizes(&pool_sizes).build(),
                None,
            )
        }
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        let device = self.core.device();
        unsafe {
            self.free_pools.iter().chain(self.used_pools.iter()).for_each(|p| device.destroy_descriptor_pool(*p, None));
        }
    }
}
