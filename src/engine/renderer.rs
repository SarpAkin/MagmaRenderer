use crate::core::*;
use std::collections::HashMap;

use crate::prelude::*;

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use super::material::MaterialManager;
use super::mesh_manager::MeshManager;
use super::{material::MaterialID, mesh_manager::MeshID};

pub struct MeshPass {
    pub proj_view:Mat4,
    pub scene_buffer:vk::DescriptorSet,
}

struct PerMaterialRenderTask {
    tasks: Vec<(MeshID, Mat4)>,
}

pub struct BatchRenderer {
    material_render_tasks: HashMap<MaterialID, PerMaterialRenderTask>,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct Push {
    mvp: [f32; 16],
}

impl BatchRenderer {
    pub fn add_entity(&mut self, transform: Mat4, meshid: MeshID, material_id: MaterialID) {
        self.get_materail_render_task(material_id).tasks.push((meshid, transform));
    }

    pub fn flush_and_draw(
        &mut self,
        cmd: &mut CommandBuffer,
        meshpass: &MeshPass,
        material_man: &MaterialManager,
        mesh_manager: &MeshManager,
    ) {
        cmd.bind_descriptor_set(0, meshpass.scene_buffer);

        for (material_id, material_task) in &self.material_render_tasks {
            let Some(material) = material_man.get_material(*material_id) else {continue};

            cmd.bind_material(&material);

            let mut previous_mesh_id = None;

            let mut mesh_ref = None;

            for (meshid, transform) in &material_task.tasks {
                if previous_mesh_id != Some(*meshid) {
                    let mesh = mesh_manager.get_mesh(*meshid);
                    mesh.bind_index_buffer(cmd);
                    mesh.bind_vertex_buffers(
                        cmd,
                        material
                            .vertex_input_types()
                            .expect("this material requires vertex input types in its descriptoion!"),
                    );

                    mesh_ref = Some(mesh);
                    previous_mesh_id = Some(*meshid);
                }

                let mesh = mesh_ref.unwrap();

                let push = Push { mvp: (meshpass.proj_view * *transform).to_cols_array() };

                cmd.push_constant(&push, vk::ShaderStageFlags::VERTEX, 0);

                if mesh.has_index_buffer() {
                    unsafe {
                        cmd.draw_indexed(mesh.index_count(), 1, 0, 0, 0);
                    }
                } else {
                    unsafe {
                        cmd.draw(mesh.vertex_count(), 1, 0, 0);
                    }
                }
            }
        }
    }

    pub fn reset(&mut self) {
        for (_, task) in &mut self.material_render_tasks {
            task.tasks.clear();
        }
    }

    pub fn new() -> BatchRenderer {
        Self{
            material_render_tasks: HashMap::new(),
        }
    }

    fn get_materail_render_task(&mut self, material_id: MaterialID) -> &mut PerMaterialRenderTask {
        self.material_render_tasks.entry(material_id).or_insert_with(PerMaterialRenderTask::new)
    }
}

impl PerMaterialRenderTask {
    fn new() -> PerMaterialRenderTask { Self { tasks: Vec::new() } }
}
