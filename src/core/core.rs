use crate::prelude::*;

use std::{
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use gpu_allocator::vulkan::*;

use super::{descriptor_set::DescriptorSetManager, pipelines::*};
use ash::{Entry, Instance};

use super::IWindow;

pub struct Surface {
    pub surface: vk::SurfaceKHR,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<(vk::Image, vk::ImageView)>,
    pub swapchain_loader: ash::extensions::khr::Swapchain,
    core: Arc<Core>,
    surface_loader: ash::extensions::khr::Surface,
    format: vk::Format,
    pub width: u32,
    pub height: u32,
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            let instance = self.core.instance();
            let device = self.core.device();

            for (_, view) in &self.swapchain_images {
                device.destroy_image_view(*view, None);
            }

            ash::extensions::khr::Swapchain::new(instance, device).destroy_swapchain(self.swapchain, None);

            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

impl Surface {
    unsafe fn create_swapchain(
        core: &Arc<Core>,
        sw_loader: &ash::extensions::khr::Swapchain,
        surface: vk::SurfaceKHR,
        surface_loader: &ash::extensions::khr::Surface,
    ) -> (vk::SwapchainKHR, Vec<(vk::Image, vk::ImageView)>) {
        let pdevice = &core.physical_device;
        let device = core.device();

        let surface_capabilities = surface_loader.get_physical_device_surface_capabilities(*pdevice, surface).unwrap();

        let surface_formats = surface_loader.get_physical_device_surface_formats(*pdevice, surface).unwrap();

        let surface_format = surface_formats.first().unwrap();

        let queue_family_indicies = [core.graphics_queue_index];

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(3.max(surface_capabilities.min_image_count).min(surface_capabilities.max_image_count))
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(surface_capabilities.current_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_family_indicies)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO);

        let swapchain = sw_loader.create_swapchain(&swapchain_create_info, None).unwrap();

        let swapchain_images: Vec<_> = sw_loader
            .get_swapchain_images(swapchain)
            .unwrap()
            .into_iter()
            .map(|image| {
                let subresource_range = vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);
                let imageview_create_info = vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .subresource_range(*subresource_range);
                (image, device.create_image_view(&imageview_create_info, None).unwrap())
            })
            .collect();

        (swapchain, swapchain_images)
    }

    pub fn recrate_swapchain(&mut self) {
        let device = self.core.device();
        unsafe {
            for (_, view) in &self.swapchain_images {
                device.destroy_image_view(*view, None);
            }

            ash::extensions::khr::Swapchain::new(self.core.instance(), device).destroy_swapchain(self.swapchain, None);
        }

        (self.swapchain, self.swapchain_images) =
            unsafe { Self::create_swapchain(&self.core, &self.swapchain_loader, self.surface, &self.surface_loader) };
    }

    unsafe fn new(core: Arc<Core>, graphics_queue_index: u32, window: &dyn IWindow) -> Surface {
        let entry = &core.entry;
        let instance = core.instance();
        let pdevice = &core.physical_device;
        let device = core.device();

        let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);
        let surface = ash_window::create_surface(
            &entry, //
            &instance,
            // window.raw_display_handle(),
            window.to_has_raw_window_handle(),
            None,
        )
        .unwrap();

        // let surface_present_modes = surface_loader.get_physical_device_surface_present_modes(*pdevice, surface).unwrap();
        let surface_formats = surface_loader.get_physical_device_surface_formats(*pdevice, surface).unwrap();
        let surface_format = surface_formats.first().unwrap();
        println!("swapchain formats {:?}", surface_formats);

        let swapchain_loader = ash::extensions::khr::Swapchain::new(&instance, &device);

        let (swapchain, swapchain_images) = Self::create_swapchain(&core, &swapchain_loader, surface, &surface_loader);
        let (width, height) = window.extends();
        // let width = extend.width;
        // let height = extend.height;

        Self {
            surface,
            swapchain,
            swapchain_images,
            swapchain_loader,
            core,
            surface_loader,
            format: surface_format.format,
            width,
            height,
        }
    }

    pub fn format(&self) -> vk::Format { self.format }
}

struct CoreDebugLayer {
    loader: ash::extensions::ext::DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT,
}

pub struct Core {
    entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    debug_layer: Option<CoreDebugLayer>,
    pub(super) allocator: ManuallyDrop<Mutex<Allocator>>,
    pub pipeline_manager: PipelineManager,
    pub descriptor_set_manager: DescriptorSetManager,
    graphics_queue: ash::vk::Queue,
    graphics_queue_index: u32,
}

unsafe fn as_cstr(s: &[u8]) -> *const i8 { s.as_ptr() as *const i8 }

impl Core {
    unsafe fn pick_physical_device(instance: &Instance) -> vk::PhysicalDevice {
        let mut vec: Vec<_> = instance
            .enumerate_physical_devices()
            .unwrap()
            .into_iter()
            .map(|p| {
                let prop = instance.get_physical_device_properties(p);

                let mut score = 0.0;
                if prop.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                    score += 1000.0
                }
                (p, score)
            })
            .collect();

        vec.sort_by(|(_, score_a), (_, score_b)| score_a.partial_cmp(score_b).unwrap());

        vec[0].0
    }

    unsafe fn create_debug_layer(entry: &Entry, instance: &Instance) -> CoreDebugLayer {
        let debug_utils = ash::extensions::ext::DebugUtils::new(&entry, &instance);
        let debugcreateinfo = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    // | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(vulkan_debug_utils_callback));

        let utils_messenger = debug_utils.create_debug_utils_messenger(&debugcreateinfo, None).unwrap();

        CoreDebugLayer { loader: debug_utils, messenger: utils_messenger }
    }

    unsafe fn init_instance(
        entry: &Entry,
        layer_names: &Vec<*const i8>,
        debug_layer_enabled: bool,
    ) -> (Instance, Option<CoreDebugLayer>) {
        let app_info = vk::ApplicationInfo {
            p_application_name: as_cstr(b"Unnamed Application\0"),
            p_engine_name: b"Unnamed Engine\0".as_ptr() as *const i8,
            api_version: vk::make_api_version(0, 1, 3, 0),
            ..Default::default()
        };

        let extension_names = vec![
            ash::extensions::ext::DebugUtils::name().as_ptr(),
            ash::extensions::khr::Surface::name().as_ptr(),
            ash::extensions::khr::XlibSurface::name().as_ptr(),
            // ash::extensions::khr::DynamicRendering::name().as_ptr()
        ];

        let create_info = vk::InstanceCreateInfo::builder()
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&extension_names)
            .application_info(&app_info);

        let instance = entry.create_instance(&create_info, None).unwrap();

        let debug = if debug_layer_enabled { Some(Self::create_debug_layer(entry, &instance)) } else { None };

        (instance, debug)
    }

    unsafe fn create_device(
        instance: &Instance,
        pdevice: vk::PhysicalDevice,
        layer_names: &Vec<*const i8>,
    ) -> (ash::Device, vk::Queue, u32) {
        let queuefamilyproperties = instance.get_physical_device_queue_family_properties(pdevice);

        let queue_index = queuefamilyproperties
            .iter()
            .enumerate()
            .find(|(_, qf)| qf.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .unwrap()
            .0 as u32;

        let priorities = [1.0f32];
        let queue_infos =
            [vk::DeviceQueueCreateInfo::builder().queue_family_index(queue_index).queue_priorities(&priorities).build()];

        let device_extension_names = vec![ash::extensions::khr::Swapchain::name().as_ptr()];

        let mut features = vk::PhysicalDeviceFeatures2 {
            features: vk::PhysicalDeviceFeatures {
                draw_indirect_first_instance: vk::TRUE,
                ..Default::default() //
            },
            ..Default::default()
        };

        let mut features12 = vk::PhysicalDeviceVulkan12Features {
            draw_indirect_count: vk::TRUE,
            ..Default::default() //
        };

        let mut features13 = vk::PhysicalDeviceVulkan13Features {
            // dynamic_rendering: vk::TRUE,
            ..Default::default() //
        };

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extension_names)
            .enabled_layer_names(&layer_names)
            .push_next(&mut features13)
            .push_next(&mut features12)
            .push_next(&mut features)
            .build();

        let device = instance.create_device(pdevice, &device_create_info, None).unwrap();
        let graphics_queue = device.get_device_queue(queue_index, 0);
        (device, graphics_queue, queue_index)
    }

    pub unsafe fn new(window: Option<&dyn IWindow>) -> (Arc<Self>, Option<Surface>) {
        let entry = Entry::load().unwrap();

        let layer_names = vec![as_cstr(b"VK_LAYER_KHRONOS_validation\0")];

        let (instance, debug_layer) = Self::init_instance(&entry, &layer_names, true);

        let pdevice = Self::pick_physical_device(&instance);
        let (device, graphics_queue, graphics_queue_index) = Self::create_device(&instance, pdevice, &layer_names);

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: pdevice,
            debug_settings: Default::default(),
            buffer_device_address: false, // Ideally, check the BufferDeviceAddressFeatures struct.
        })
        .unwrap();

        let core = Arc::new(Self {
            entry,
            instance,
            physical_device: pdevice,
            debug_layer,
            allocator: ManuallyDrop::new(Mutex::new(allocator)),
            pipeline_manager: PipelineManager::new(&device),
            device,
            graphics_queue,
            graphics_queue_index,
            descriptor_set_manager: DescriptorSetManager::new(),
            //debug_layer:None
        });

        let core_surface = window.and_then(|window| Some(Surface::new(core.clone(), graphics_queue_index, window)));

        (core, core_surface)
    }

    pub fn queue_index(&self) -> u32 { self.graphics_queue_index }
    pub fn queue(&self) -> ash::vk::Queue { self.graphics_queue }

    pub fn device(&self) -> &ash::Device { &self.device }

    pub fn instance(&self) -> &ash::Instance { &self.instance }
}

impl Drop for Core {
    fn drop(&mut self) {
        self.pipeline_manager.cleanup(&self);
        self.descriptor_set_manager.cleanup(&self);
        unsafe {
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);

            if let Some(debug_layer) = &self.debug_layer {
                debug_layer.loader.destroy_debug_utils_messenger(debug_layer.messenger, None);
            }

            self.instance.destroy_instance(None);
        }
    }
}

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    let severity = format!("{:?}", message_severity).to_lowercase();
    let ty = format!("{:?}", message_type).to_lowercase();
    println!("[Debug][{}][{}] {:?}", severity, ty, message);

    if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        // use backtrace::Backtrace;
        // let bt = Backtrace::new();
        // println!("{bt:?}");
        // panic!("{bt:?}")
    }

    vk::FALSE
}
