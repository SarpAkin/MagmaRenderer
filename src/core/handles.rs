use ash::{prelude::VkResult, vk};

use super::core::Core;
use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
    u64,
};

// macro_rules! auto_handle {
//     (
//         $( #[$meta:meta] )*
//         pub struct $type_name:ident {
//             $($field:ident : $type:ty),* $(,)?
//         },
//         fn drop(&mut self) {
//             $body:tt
//         }
//     ) => {
//         $( #[$meta] )*
//         pub struct $type_name{
//             $($field : $type,)*
//             core:Arc<Core>,
//         }

//         impl Drop for $type_name{
//             fn drop(&mut self) {
//                 $body
//             }
//         }
//     };
// }

// auto_handle!(
//     pub struct Semaphore {}    ,    fn drop(&mut self) {}
// );

pub struct Handle<T>
where
    T: Clone,
{
    core: Arc<Core>,
    inner: T,
    cleanup: unsafe fn(&ash::Device, T, Option<&vk::AllocationCallbacks>),
}

impl<T> Deref for Handle<T>
where
    T: Clone,
{
    type Target = T;
    fn deref(&self) -> &Self::Target { &self.inner }
}

impl<T> Drop for Handle<T>
where
    T: Clone,
{
    fn drop(&mut self) {
        unsafe {
            (self.cleanup)(self.core.device(), self.inner.clone(), None);
        }
    }
}

pub type Semaphore = Handle<vk::Semaphore>;
// pub type Fence = Handle<vk::Fence>;
pub type Sampler = Handle<vk::Sampler>;

pub struct Fence {
    core: Arc<Core>,
    inner: vk::Fence,
    dependents: Option<Box<dyn Drop>>,
}

impl Fence {
    pub unsafe fn queue_submit(
        &mut self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo],
        dependent: Box<dyn Drop>,
    ) -> Result<(), vk::Result> {
        if let Some(_) = self.dependents {
            panic!("wait on fence before submiting another submit!");
        }

        self.dependents = Some(dependent);
        self.core.device().queue_submit(queue, submits, self.inner)?;

        Ok(())
    }

    //waits if there is somtething to wait
    pub fn try_wait(&mut self, timeout: Option<u64>) -> Result<bool, vk::Result> {
        if self.dependents.is_none() {
            return Ok(false);
        }

        self.wait_and_reset(timeout)?;

        Ok(true)
    }

    pub fn wait_and_reset(&mut self, timeout: Option<u64>) -> Result<(), vk::Result> {
        unsafe {
            self.core.device().wait_for_fences(&[self.inner], true, timeout.unwrap_or(u64::MAX))?;
            self.core.device().reset_fences(&[self.inner])?;
        }
        self.dependents = None;
        Ok(())
    }
}

impl Deref for Fence {
    type Target = vk::Fence;
    fn deref(&self) -> &Self::Target { &self.inner }
}

impl Drop for Fence {
    fn drop(&mut self) {
        if self.dependents.is_some() {
            panic!("can't drop fence that isn't waited")
            // self.wait_and_reset(None).unwrap();
        }

        unsafe {
            self.core.device().destroy_fence(self.inner, None);
        }
    }
}

impl Core {
    pub fn create_semaphore(self: &Arc<Self>) -> Semaphore {
        Semaphore {
            core: self.clone(),
            inner: unsafe {
                self.device()
                    .create_semaphore(&vk::SemaphoreCreateInfo::builder().build(), None)
                    .expect("failed to create semaphore")
            },
            cleanup: ash::Device::destroy_semaphore,
        }
    }

    pub fn create_sampler(
        self: &Arc<Self>,
        filter: vk::Filter,
        address_mode: Option<(vk::SamplerAddressMode, vk::BorderColor)>,
    ) -> Sampler {
        let (address_mode, border_color) =
            address_mode.unwrap_or((vk::SamplerAddressMode::REPEAT, vk::BorderColor::FLOAT_OPAQUE_BLACK));

        let create_info = vk::SamplerCreateInfo::builder()
            .address_mode_u(address_mode)
            .address_mode_v(address_mode)
            .address_mode_w(address_mode)
            .border_color(border_color)
            .mag_filter(filter)
            .min_filter(filter)
            .build();

        Sampler {
            core: self.clone(),
            inner: unsafe { self.device().create_sampler(&create_info, None).expect("failed to create sampler") },
            cleanup: ash::Device::destroy_sampler,
        }
    }

    pub fn create_fence(self: &Arc<Self>) -> Fence {
        Fence {
            core: self.clone(),
            inner: unsafe {
                self.device().create_fence(&vk::FenceCreateInfo::builder().build(), None).expect("failed to create fence")
            },
            dependents: None,
            // cleanup: ash::Device::destroy_fence,
        }
    }
}

// const a: &[u32] = vk_shader_macros::include_glsl!("res/hello_tri.vert");
