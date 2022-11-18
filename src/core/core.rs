use std::{
    fmt,
    marker::PhantomData,
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use gpu_allocator::vulkan::*;

use super::{descriptor_set::DescriptorSetManager, pipelines::*, ref_or_arc::RefOrArc};

use ash::{
    vk::{self, PhysicalDevice, PhysicalDeviceType},
    Entry, Instance,
};

use super::IWindow;
use bytemuck::{cast_slice_mut, checked::cast_slice, Pod};

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
    allocator: ManuallyDrop<Mutex<Allocator>>,
    pub pipeline_manager: PipelineManager,
    pub descriptor_set_manager: DescriptorSetManager,
    graphics_queue: ash::vk::Queue,
    graphics_queue_index: u32,
}

unsafe fn as_cstr(s: &[u8]) -> *const i8 { s.as_ptr() as *const i8 }

impl Core {
    unsafe fn pick_physical_device(instance: &Instance) -> PhysicalDevice {
        let mut vec: Vec<_> = instance
            .enumerate_physical_devices()
            .unwrap()
            .into_iter()
            .map(|p| {
                let prop = instance.get_physical_device_properties(p);

                let mut score = 0.0;
                if prop.device_type == PhysicalDeviceType::DISCRETE_GPU {
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
        pdevice: PhysicalDevice,
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

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extension_names)
            .enabled_layer_names(&layer_names);

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

        let mut allocator = Allocator::new(&AllocatorCreateDesc {
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

    pub fn ref_or_arc<'a>(self: &'a Arc<Self>) -> RefOrArc<'a, Core> { RefOrArc::new_ref(self) }
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
    use backtrace::Backtrace;

    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    let severity = format!("{:?}", message_severity).to_lowercase();
    let ty = format!("{:?}", message_type).to_lowercase();
    println!("[Debug][{}][{}] {:?}", severity, ty, message);

    if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        let bt = Backtrace::new();
        println!("{bt:?}");

        // panic!("{bt:?}")
    }

    vk::FALSE
}

pub struct Buffer<T: Pod> {
    buffer: vk::Buffer,
    allocation: ManuallyDrop<Allocation>,
    core: Arc<Core>,
    size_in_bytes: u32,
    size_in_items: u32,
    phantom: PhantomData<T>,
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
    pub fn get_data(&self) -> Option<&[T]> { self.allocation.mapped_slice().and_then(|s| Some(cast_slice::<u8, T>(s))) }

    pub fn get_data_mut(&mut self) -> Option<&mut [T]> {
        self.allocation.mapped_slice_mut().and_then(|s| Some(cast_slice_mut::<u8, T>(s)))
    }

    pub fn inner(&self) -> vk::Buffer { self.buffer }
    pub fn as_slice(&self) -> BufferSlice<T> {
        BufferSlice { buffer: self, offset_items: self.offset(), size_items: self.size() }
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

pub struct Image {
    core: Arc<Core>,
    allocation: ManuallyDrop<Allocation>,
    image: vk::Image,
    view: vk::ImageView,
    format: vk::Format,
    width: u32,
    height: u32,
}

pub fn is_depth_format(format: vk::Format) -> bool {
    match format {
        vk::Format::D16_UNORM
        | vk::Format::D32_SFLOAT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => true,
        _ => false,
    }
}

impl Core {
    pub fn new_image(
        self: &Arc<Core>,
        format: vk::Format,
        flags: vk::ImageUsageFlags,
        width: u32,
        height: u32,
        layers: u32,
    ) -> eyre::Result<Image> {
        let image = unsafe {
            self.device().create_image(
                &vk::ImageCreateInfo::builder()
                    .array_layers(layers)
                    .format(format)
                    .image_type(vk::ImageType::TYPE_2D)
                    .extent(vk::Extent3D { width, height, depth: 1 })
                    .usage(flags)
                    .mip_levels(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .build(),
                None,
            )
        }?;

        let allocation = unsafe {
            let requirements = self.device().get_image_memory_requirements(image);

            let allocation = self.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
                name: "image alloc",
                requirements,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: false,
            })?;

            self.device().bind_image_memory(image, allocation.memory(), allocation.offset())?;

            allocation
        };

        let view = unsafe {
            self.device().create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .view_type(if layers >= 1 { vk::ImageViewType::TYPE_2D } else { vk::ImageViewType::TYPE_2D_ARRAY })
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: if is_depth_format(format) {
                            vk::ImageAspectFlags::DEPTH
                        } else {
                            vk::ImageAspectFlags::COLOR
                        },
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: layers,
                    })
                    .build(),
                None,
            )
        }?;

        Ok(Image { core: self.clone(), allocation: ManuallyDrop::new(allocation), image, view, format, width, height })
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        let device = self.core.device();
        // let allocation = std::mem::replace(&mut self.allocation, None).unwrap();
        unsafe {
            self.core.allocator.lock().unwrap().free(ManuallyDrop::take(&mut self.allocation)).unwrap();
            device.destroy_image(self.image, None);
            device.destroy_image_view(self.view, None);
        }
    }
}

impl Image {
    pub fn image(&self) -> vk::Image { self.image }
    pub fn view(&self) -> vk::ImageView { self.view }
    pub fn format(&self) -> vk::Format { self.format }
    pub fn extends(&self) -> (u32, u32) { (self.width, self.height) }
}
