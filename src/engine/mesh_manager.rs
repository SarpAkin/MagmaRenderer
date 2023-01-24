use std::{collections::HashMap, num::NonZeroU32};

use crate::prelude::*;

use super::mesh::Mesh;

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub struct MeshID(NonZeroU32);
pub struct MeshManager {
    meshes: HashMap<MeshID, Box<Mesh>>,
    id_counter: u32,
}

impl MeshManager {
    pub fn register_mesh(&mut self, mesh: Mesh) -> MeshID {
        let id = MeshID(NonZeroU32::new(self.id_counter).unwrap());
        self.id_counter += 1;

        if self.id_counter == 0 {
            panic!("mesh id overflow!");
        }

        self.meshes.insert(id, Box::new(mesh));

        id
    }

    pub fn remove_mesh(&mut self,id:MeshID) -> Option<Box<Mesh>> {
        self.meshes.remove(&id)
    }

    pub fn get_mesh(&self,id:MeshID) -> &Mesh {
        self.meshes.get(&id).expect("trying to get removed mesh!")
    }

    pub fn new() -> MeshManager {
        Self{
            meshes: HashMap::new(),
            id_counter: 1,
        }
    }
}
