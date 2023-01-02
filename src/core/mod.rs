mod command_buffers;
mod core;
mod descriptor_set;
mod handles;
mod pipelines;
mod renderpass;
// mod winit_iwindow;
mod glfw_iwindow;
mod image_load;
mod buffer;
mod image;

unsafe fn cast_to_static_lifetime<T>(val: &T) -> &'static T { std::mem::transmute(val) }

use raw_window_handle::{HasRawWindowHandle};

use ash::vk;

pub use self::command_buffers::*;
pub use self::core::*;
pub use self::descriptor_set::*;
pub use self::handles::*;
pub use self::pipelines::*;
pub use self::renderpass::*;
pub use self::buffer::*;
pub use self::image::*;
pub trait IWindow: HasRawWindowHandle {
    fn extends(&self) -> (u32, u32);
    fn to_has_raw_window_handle(&self) -> &dyn HasRawWindowHandle;
}
