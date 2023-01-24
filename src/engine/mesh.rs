use glam::Vec2;
use glam::Vec3;
use serde::Deserialize;
use serde::Serialize;
use smallvec::SmallVec;

use crate::core::*;
use crate::prelude::*;

use super::pack_RG16_unorm;
use super::pack_RGB10_A2_snorm;

#[derive(Serialize,Deserialize)]
pub enum VertexInputTypes {
    Position,
    UvCoord,
    Normal,
    Tangent,
    BoneID, //todo bones
}

enum IndexBuffer {
    U16(Buffer<u16>),
    U32(Buffer<u32>),
    None,
}

const UV_FORMAT: vk::Format = vk::Format::R16G16_USCALED;
const NORMAL_FORMAT: vk::Format = vk::Format::A2B10G10R10_SNORM_PACK32;

pub struct Mesh {
    index_buffer: IndexBuffer,
    vpositions: Buffer<[f32; 3]>,
    uv_coords: Option<Buffer<u8>>, //
    vnormals: Buffer<u8>,
    vtangents: Option<Buffer<u8>>,
    index_count:u32,
    vertex_count:u32,
}

impl Mesh {
    pub fn index_count(&self) -> u32{
        self.index_count
    }

    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    pub fn has_index_buffer(&self) -> bool {
        match &self.index_buffer {
            IndexBuffer::None => false,
            _ => true,
        }
    }

    pub fn bind_index_buffer(&self, cmd: &mut CommandBuffer) {
        match &self.index_buffer {
            IndexBuffer::U16(buffer) => cmd.bind_index_buffer(buffer.as_slice()),
            IndexBuffer::U32(buffer) => cmd.bind_index_buffer(buffer.as_slice()),
            IndexBuffer::None => {}
        }
    }

    pub fn bind_vertex_buffers(&self, cmd: &mut CommandBuffer, input_types: &[VertexInputTypes]) {
        let buffers = input_types
            .iter()
            .map(|input_type| match input_type {
                VertexInputTypes::Position => self.vpositions.as_raw_buffer_slice(),
                VertexInputTypes::UvCoord => self.uv_coords.as_ref().unwrap().as_raw_buffer_slice(),
                VertexInputTypes::Normal => self.vnormals.as_raw_buffer_slice(),
                VertexInputTypes::Tangent => self.vtangents.as_ref().unwrap().as_raw_buffer_slice(),
                VertexInputTypes::BoneID => todo!(),
            })
            .collect::<SmallVec<[_; 5]>>();

        cmd.bind_vertex_buffers(&buffers);
    }

    pub fn get_vertex_input_description(input_types: &[VertexInputTypes]) -> VertexInputDescriptionBuilder {
        let mut builder = VertexInputDescriptionBuilder::new();

        for input_type in input_types {
            match input_type {
                VertexInputTypes::Position => {
                    builder.push_binding::<[f32; 3]>(vk::VertexInputRate::VERTEX);
                    builder.push_attribure(&[0.0f32; 3], 0);
                }
                VertexInputTypes::UvCoord => {
                    builder.push_binding::<[u32; 1]>(vk::VertexInputRate::VERTEX);
                    builder.push_attribure_with_format(UV_FORMAT, 0);
                }
                VertexInputTypes::Normal => {
                    builder.push_binding::<[u32; 1]>(vk::VertexInputRate::VERTEX);
                    builder.push_attribure_with_format(NORMAL_FORMAT, 0);
                }
                VertexInputTypes::Tangent => todo!(),
                VertexInputTypes::BoneID => todo!(),
            }
        }

        builder
    }
}

pub struct MeshBuilder<'a> {
    pub indicies: Option<&'a [u32]>,
    pub positions: Option<&'a [[f32; 3]]>,
    pub uvs: Option<&'a [[f32; 2]]>,
    pub normals: Option<Vec<[f32; 3]>>,
    pub tangents: Option<Vec<[f32; 3]>>,
}

impl<'a> MeshBuilder<'a> {
    pub fn new() -> Self { Self { indicies: None, positions: None, normals: None, uvs: None, tangents: None } }
    pub fn set_indicies(&mut self, indicies: &'a [u32]) -> &mut Self {
        self.indicies = Some(indicies);
        self
    }

    pub fn set_positions(&mut self, positions: &'a [[f32; 3]]) -> &mut Self {
        self.positions = Some(positions);
        self
    }

    pub fn set_uv_coords(&mut self, uv_coords: &'a [[f32; 2]]) -> &mut Self {
        self.uvs = Some(uv_coords);
        self
    }

    pub fn build(&mut self, core: &Arc<Core>, cmd: &mut CommandBuffer) -> Result<Mesh> {
        // let indicies = self.indicies.as_ref().expect("requires an index buffer to be passed before build is called");

        if self.normals == None {
            self.construct_normals();
        }

        let vpositions = self.positions.expect("vertex positions are required!");
        let vnormals = self.normals.as_ref().unwrap();

        let mesh = Mesh {
            index_buffer: match self.indicies {
                None => IndexBuffer::None,
                Some(indicies) => {
                    let usage = vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER;

                    if vpositions.len() > u16::MAX as usize {
                        IndexBuffer::U32(cmd.gpu_buffer_from_slice(usage, &indicies)?)
                    } else {
                        IndexBuffer::U16(cmd.gpu_buffer_from_iter(usage, indicies.iter().map(|i| *i as u16))?)
                    }
                }
            },
            vpositions: cmd.gpu_buffer_from_slice(
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
                &vpositions,
            )?,
            uv_coords: if let Some(buffer) = self.uvs {
                Some(
                    cmd.gpu_buffer_from_iter(
                        vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
                        buffer.iter().map(|arr| pack_RG16_unorm(Vec2::from_array(*arr))),
                    )?
                    .cast(),
                )
            } else {
                None
            },
            vnormals: cmd
                .gpu_buffer_from_iter(
                    vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
                    vnormals.iter().map(|arr| pack_RGB10_A2_snorm(Vec3::from_array(*arr))),
                )?
                .cast(),
            vtangents: None,
            index_count: self.indicies.map(|b|b.len()).unwrap_or(0) as u32,
            vertex_count: vpositions.len() as u32,
        };

        Ok(mesh)
    }

    fn normal_from_positions(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
        let ab = a - b;
        let ac = a - c;

        let normal = ab.cross(ac).normalize_or_zero();

        normal
    }

    fn construct_normals(&mut self) {
        let vpositions = self.positions.as_ref().expect("vertex positions are required!");
        let mut normals = Vec::new();
        normals.resize(vpositions.len(), [0.0; 3]);

        if let Some(indicies) = &self.indicies {
            for c in indicies.chunks_exact(3) {
                let [ai,bi,ci] = *c else {return}; //shouldn't fail and return

                let a = Vec3::from_array(vpositions[ai as usize]);
                let b = Vec3::from_array(vpositions[bi as usize]);
                let c = Vec3::from_array(vpositions[ci as usize]);

                let normal = Self::normal_from_positions(a, b, c).to_array();
                normals[ai as usize] = normal;
                normals[bi as usize] = normal;
                normals[ci as usize] = normal;
            }
        }

        self.normals = Some(normals);
    }
}
