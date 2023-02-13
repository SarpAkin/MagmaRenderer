use std::{sync::RwLock, collections::HashMap};
use std::hash::Hash;


pub struct SyncCache<K:Hash + PartialEq + Eq + Clone, T: Clone> {
    cache: RwLock<HashMap<K, T>>,
}

impl<K:Hash + PartialEq + Eq + Clone,T: Clone> SyncCache<K,T> {
    pub fn get_or_insert_with<F: FnOnce() -> T>(&self, key: &K, f: F) -> T {
        let Some(item) = self.cache.read().unwrap().get(key).map(|i|i.clone()) else {
            let item = f();

            self.cache.write().unwrap().insert(key.clone(), item.clone());

            return item;
        };

        item
    }

    pub fn get_or_try_insert_with<E,F: FnOnce() -> Result<T,E>>(&self, key: &K, f: F) -> Result<T,E> {
        let Some(item) = self.cache.read().unwrap().get(key).map(|i|i.clone()) else {
            let item = f()?;

            self.cache.write().unwrap().insert(key.clone(), item.clone());

            return Ok(item);
        };

        Ok(item)
    }

    pub fn try_insert_with<E,F: FnOnce() -> Result<T,E>>(&self, key: &K, f: F) -> Result<T,E>{
        let item = f()?;

        self.cache.write().unwrap().insert(key.clone(), item.clone());

        Ok(item)
    }

    pub(crate) fn new() -> Self {
        Self{
            cache: RwLock::new(HashMap::new()),
        }
    }

}