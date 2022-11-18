use std::{ops::Deref, sync::Arc};

pub enum RefOrArc<'a, T> {
    Arc(Arc<T>),
    Ref(&'a Arc<T>),
}

impl<'a, T> RefOrArc<'a, T> {
    pub fn new_arc(arc: Arc<T>) -> RefOrArc<'static, T> { RefOrArc::Arc(arc) }

    pub fn new_ref(reference: &'a Arc<T>) -> RefOrArc<'a, T> { RefOrArc::Ref(reference) }

    pub fn owned(&self) -> RefOrArc<'static, T> { Self::new_arc(self.get_arc()) }

    pub fn get_arc(&self) -> Arc<T> {
        match self {
            RefOrArc::Arc(a) => a.clone(),
            RefOrArc::Ref(r) => r.deref().clone(),
        }
    }
}



impl<'a, T> Deref for RefOrArc<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match &self {
            RefOrArc::Arc(arc) => arc.as_ref(),
            RefOrArc::Ref(r) => r.as_ref(),
        }
    }
}
