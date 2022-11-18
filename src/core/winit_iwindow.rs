use raw_window_handle::HasRawWindowHandle;
use winit::window;

impl super::IWindow for window::Window {
    fn extends(&self) -> (u32, u32) {
        let p = self.inner_size();
        (p.width, p.height)
    }

    fn to_has_raw_window_handle(&self) -> &dyn raw_window_handle::HasRawWindowHandle {
        self as &dyn HasRawWindowHandle
    }
}

