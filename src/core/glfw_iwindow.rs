use super::IWindow;

use glfw::*;
use raw_window_handle::HasRawWindowHandle;

impl IWindow for glfw::Window {
    fn extends(&self) -> (u32, u32) {
        let (w, h) = self.get_size();
        (w as u32, h as u32)
    }

    fn to_has_raw_window_handle(&self) -> &dyn raw_window_handle::HasRawWindowHandle {
        self as &dyn HasRawWindowHandle
    }
}

fn foo(w: &glfw::Window) {}
