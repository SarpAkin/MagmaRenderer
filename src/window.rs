use std::{collections::HashMap, sync::Arc, time::SystemTime};

use ash::vk;
use glfw::Context;

use crate::core::*;

pub const CPU_BUFFERING_NUM: usize = 2;

struct BufferedResources {
    render_semaphore: Semaphore,
    present_semaphore: Semaphore,
    render_fence: Fence,
}

pub struct Window {
    pub renderpass: SurfaceRenderpass,
    glfw: glfw::Glfw,
    window: glfw::Window,
    buffered_resources: Box<[BufferedResources]>,
    buffer_index: u32,
    pub core: Arc<Core>,
    events: std::sync::mpsc::Receiver<(f64, glfw::WindowEvent)>,
    cursor: Cursor,
    last_time: SystemTime,
    delta_time: f64,
    key_states: HashMap<Key, InputState>,
    mbutton_states: HashMap<MouseButton, InputState>,
}

#[derive(Default)]
pub struct Cursor {
    pub mousex: f32,
    pub mousey: f32,
    pub mouserelx: f32,
    pub mouserely: f32,
    oldmousex: f32,
    oldmousey: f32,
}

impl Window {
    pub fn frame_index(&self) -> usize { self.buffer_index as usize }

    pub fn prepare_and_poll_events(&mut self) -> eyre::Result<bool> {
        if self.window.should_close() {
            self.wait_for_fences();
            return Ok(false);
        }

        let now = SystemTime::now();
        self.delta_time = now.duration_since(self.last_time).map(|d| d.as_secs_f64()).unwrap_or_else(|err| {
            eprintln!("error when getting delta time: {err}");
            0.1//return a non zero value
        });
        self.last_time = now;

        self.poll_events();

        self.cursor.mouserelx = self.cursor.mousex - self.cursor.oldmousex;
        self.cursor.mouserely = self.cursor.mousey - self.cursor.oldmousey;
        self.cursor.oldmousex = self.cursor.mousex;
        self.cursor.oldmousey = self.cursor.mousey;

        self.current_framedata_mut().render_fence.try_wait(None)?;

        match self.renderpass.prepare(*self.current_framedata().present_semaphore) {
            Ok(_) => {}
            Err(err) => match err {
                vk::Result::ERROR_OUT_OF_DATE_KHR => {
                    self.wait_for_fences();
                    let (width, height) = self.window.extends();
                    self.renderpass.resize(width, height)?;
                    return Ok(true);
                }
                _ => return Err(eyre::ErrReport::new(err)),
            },
        };

        Ok(true)
    }

    pub fn submit_and_present(&mut self, cmd: CommandBuffer) -> eyre::Result<()> {
        let core = self.core.clone();
        let fdata = self.current_framedata_mut();

        unsafe {
            fdata.render_fence.queue_submit(
                core.queue(),
                &[vk::SubmitInfo::builder()
                    .wait_semaphores(&[*fdata.present_semaphore])
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                    .command_buffers(&[cmd.inner()])
                    .signal_semaphores(&[*fdata.render_semaphore])
                    .build()],
                Box::new(cmd),
            )?;
        }

        match self.renderpass.present(*self.current_framedata().render_semaphore) {
            Ok(_) => {}
            Err(err) => match err {
                vk::Result::ERROR_OUT_OF_DATE_KHR => {
                    self.wait_for_fences();
                    let (width, height) = self.window.extends();
                    self.renderpass.resize(width, height)?;
                    return Ok(());
                }
                _ => return Err(eyre::ErrReport::new(err)),
            },
        };

        self.next_frame();

        Ok(())
    }

    pub fn new() -> eyre::Result<Self> {
        let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
        glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));

        let (mut window, events) = glfw.create_window(1280, 720, "title", glfw::WindowMode::Windowed).unwrap();

        window.set_key_polling(true);
        window.set_mouse_button_polling(true);
        window.set_cursor_pos_polling(true);

        // window.make_current();

        let (core, surface) = unsafe { Core::new(Some(&window)) };
        let surface = surface.unwrap();

        let renderpass = SurfaceRenderpass::new(core.clone(), surface)?;

        let window = Window {
            renderpass,
            glfw,
            window,
            events,
            buffered_resources: Self::init_buffered_resources(&core),
            buffer_index: 0,
            core,
            cursor: Cursor { ..Default::default() },
            last_time: SystemTime::now(),
            key_states: HashMap::new(),
            mbutton_states: HashMap::new(),
            delta_time: 0.0,
        };

        Ok(window)
    }

    fn init_buffered_resources(core: &Arc<Core>) -> Box<[BufferedResources]> {
        (0..CPU_BUFFERING_NUM)
            .map(|_| {
                let render_fence = core.create_fence();
                let present_semaphore = core.create_semaphore();
                let render_semaphore = core.create_semaphore();

                BufferedResources { render_fence, present_semaphore, render_semaphore }
            })
            .collect()
    }

    fn poll_events(&mut self) {
        // self.window.make_current();

        self.glfw.poll_events();

        for (_, e) in glfw::flush_messages(&self.events) {
            // println!("{e:?}");

            match e {
                glfw::WindowEvent::CursorPos(x, y) => (self.cursor.mousex, self.cursor.mousey) = (x as f32, y as f32),
                // glfw::WindowEvent::Size(_, _) => todo!(),
                // glfw::WindowEvent::Close => todo!(),
                // glfw::WindowEvent::Refresh => todo!(),
                // glfw::WindowEvent::Focus(_) => todo!(),
                // glfw::WindowEvent::Iconify(_) => todo!(),
                // glfw::WindowEvent::FramebufferSize(_, _) => todo!(),
                glfw::WindowEvent::MouseButton(button, action, _m) => {
                    self.mbutton_states.insert(button, InputState::from(action));
                }
                // glfw::WindowEvent::Pos(_, _) => todo!(),
                // glfw::WindowEvent::CursorEnter(_) => todo!(),
                // glfw::WindowEvent::Scroll(_, _) => todo!(),
                glfw::WindowEvent::Key(key, _scandcode, action, _modifiers) => {
                    self.key_states.insert(key, InputState::from(action));
                }
                // glfw::WindowEvent::Char(_) => todo!(),
                // glfw::WindowEvent::CharModifiers(_, _) => todo!(),
                // glfw::WindowEvent::FileDrop(_) => todo!(),
                // glfw::WindowEvent::Maximize(_) => todo!(),
                // glfw::WindowEvent::ContentScale(_, _) => todo!(),
                _ => {}
            }
        }
    }

    fn wait_for_fences(&mut self) {
        for r in (&mut self.buffered_resources).as_mut().into_iter() {
            r.render_fence.try_wait(None).unwrap();
        }
    }

    fn current_framedata(&self) -> &BufferedResources { &self.buffered_resources[self.buffer_index as usize] }
    fn current_framedata_mut(&mut self) -> &mut BufferedResources {
        &mut self.buffered_resources[self.buffer_index as usize]
    }

    fn next_frame(&mut self) { self.buffer_index = (self.buffer_index + 1) % CPU_BUFFERING_NUM as u32; }
}

//Input
pub type Key = glfw::Key;
pub type MouseButton = glfw::MouseButton;
// pub type InputState = glfw::Action;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputState {
    Released,
    Pressed,
}

impl From<glfw::Action> for InputState {
    fn from(a: glfw::Action) -> Self {
        match a {
            glfw::Action::Release => Self::Released,
            glfw::Action::Press => Self::Pressed,
            glfw::Action::Repeat => Self::Pressed,
        }
    }
}

impl Window {
    pub fn get_key(&self, key: Key) -> InputState {
        self.key_states.get(&key).and_then(|a| Some(*a)).unwrap_or(InputState::Released)
    }
    pub fn get_mouse_movement(&self) -> (f32, f32) { (self.cursor.mouserelx, self.cursor.mouserely) }
    pub fn get_mouse_pos(&self) -> (f32, f32) { (self.cursor.mousex, self.cursor.mousey) }
    pub fn get_mouse_button(&self, button: MouseButton) -> InputState {
        self.mbutton_states.get(&button).and_then(|a| Some(*a)).unwrap_or(InputState::Released)
    }
    pub fn delta_time(&self) -> f64 { self.delta_time }
    pub fn lock_cursor(&mut self) { self.window.set_cursor_mode(glfw::CursorMode::Disabled); }
    pub fn unlock_cursor(&mut self) { self.window.set_cursor_mode(glfw::CursorMode::Normal); }
}

impl Drop for Window {
    fn drop(&mut self) { self.wait_for_fences(); }
}
