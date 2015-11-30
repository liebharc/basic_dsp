use basic_dsp::{
    DataVector32, 
    RealTimeVector32, 
    ComplexTimeVector32};
use std::boxed::Box;

pub const DEFAULT_DATA_SIZE: usize = 10000;

pub struct VectorBox<T>
{
    vector: *mut T
}

#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum Size {
	Small,
    Medium,
    Large
}

fn translate_size(size: Size) -> usize {
    match size {
        Size::Small => DEFAULT_DATA_SIZE,
        Size::Medium => 100000,
        Size::Large => 500000
    }
}

impl VectorBox<DataVector32>
{
    pub fn with_size(is_complex: bool, size: usize) -> VectorBox<DataVector32>
    {
        let data = vec![0.0; size];
        let vector = 
            if is_complex {
                DataVector32::from_interleaved_no_copy(data)
            } else {
                DataVector32::from_array_no_copy(data)
            };
        VectorBox
        {
            vector: Box::into_raw(Box::new(vector))
        }
    }
    
    pub fn new(size: Size, is_complex: bool) -> VectorBox<DataVector32>
    {
        let size = translate_size(size);
        let data = vec![0.0; size];
        let vector = 
            if is_complex {
                DataVector32::from_interleaved_no_copy(data)
            } else {
                DataVector32::from_array_no_copy(data)
            };
        VectorBox
        {
            vector: Box::into_raw(Box::new(vector))
        }
    }
}

impl VectorBox<RealTimeVector32>
{
    pub fn new(size: Size) -> VectorBox<RealTimeVector32>
    {
        let size = translate_size(size);
        let data = vec![0.0; size];
        let vector = RealTimeVector32::from_array_no_copy(data);
        VectorBox
        {
            vector: Box::into_raw(Box::new(vector))
        }
    }
}

impl VectorBox<ComplexTimeVector32>
{
    pub fn new(size: Size) -> VectorBox<ComplexTimeVector32>
    {
        let size = translate_size(size);
        let data = vec![0.0; size];
        let vector = ComplexTimeVector32::from_interleaved_no_copy(data);
        VectorBox
        {
            vector: Box::into_raw(Box::new(vector))
        }
    }
}

#[allow(dead_code)]
impl<T> VectorBox<T>
{
    pub fn execute<F>(&mut self, function: F) -> bool
        where F: Fn(T) -> T + 'static + Sync
    {
        unsafe {
            let vector = Box::from_raw(self.vector);
            let result = function(*vector);
            self.vector = Box::into_raw(Box::new(result));
        }
        
        true
    }
}

impl<T> Drop for VectorBox<T>
{
    fn drop(&mut self) {
        unsafe {
            let _ = Box::from_raw(self.vector); // make sure that the vector is deleted
        }
    }
}