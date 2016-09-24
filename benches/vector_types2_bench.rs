#![feature(test)]
#![feature(box_syntax)]
extern crate test;
extern crate basic_dsp;
extern crate num;

use basic_dsp::RealNumber;
use basic_dsp::vector_types2::*;

pub const DEFAULT_DATA_SIZE: usize = 10000;

pub struct VectorBox<B, T>
    where T: RealNumber
{
    pub vector: *mut B,
    pub size: usize,
    pub buffer: SingleBuffer<T>
}

#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum Size {
    Tiny,
    Small,
    Medium,
    Large
}

pub fn translate_size(size: Size) -> usize {
    match size {
        Size::Tiny => 1000,
        Size::Small => DEFAULT_DATA_SIZE,
        Size::Medium => 100000,
        Size::Large => 1000000
    }
}

impl VectorBox<RealTimeVec<Vec<f32>, f32>, f32> {
    pub fn new(size: Size) -> VectorBox<RealTimeVec<Vec<f32>, f32>, f32>
    {
        let size = translate_size(size);
        let data = vec![10.0; size];
        VectorBox
        {
            vector: Box::into_raw(Box::new(data.to_real_time_vec())),
            buffer: SingleBuffer::with_capacity(size),
            size: size
        }
    }
}

impl VectorBox<ComplexTimeVec<Vec<f32>, f32>, f32> {
    pub fn new(size: Size) -> VectorBox<ComplexTimeVec<Vec<f32>, f32>, f32>
    {
        let size = translate_size(size);
        let data = vec![10.0; size];
        VectorBox
        {
            vector: Box::into_raw(Box::new(data.to_complex_time_vec())),
            buffer: SingleBuffer::with_capacity(size),
            size: size
        }
    }
}

impl VectorBox<RealTimeVec<Vec<f64>, f64>, f64> {
    pub fn new(size: Size) -> VectorBox<RealTimeVec<Vec<f64>, f64>, f64>
    {
        let size = translate_size(size);
        let data = vec![10.0; size];
        VectorBox
        {
            vector: Box::into_raw(Box::new(data.to_real_time_vec())),
            buffer: SingleBuffer::with_capacity(size),
            size: size
        }
    }
}

#[allow(dead_code)]
impl<B, T> VectorBox<B, T>
    where T: RealNumber
{
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn execute<F>(&mut self, function: F) -> bool
        where F: Fn(B, &mut SingleBuffer<T>) -> B + 'static + Sync
    {
        unsafe {
            let vector = Box::from_raw(self.vector);
            let result = function(*vector, &mut self.buffer);
            self.vector = Box::into_raw(Box::new(result));
        }

        true
    }
}

impl<B, T> Drop for VectorBox<B, T>
    where T: RealNumber
{
    fn drop(&mut self) {
        unsafe {
            let _ = Box::from_raw(self.vector); // make sure that the vector is deleted
        }
    }
}

#[cfg(test)]
mod vec2 {
    use test::Bencher;
    use basic_dsp::vector_types2::*;
    use super::*;

    #[bench]
    fn real_sin_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVec<Vec<f32>, f32>, f32>::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _b|  {
                assert!(v.len() == 10000);
                v.sin();
                v
            } )
        });
    }

    #[bench]
    fn real_sin_32m_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVec<Vec<f32>, f32>, f32>::new(Size::Medium);
        b.iter(|| {
            vector.execute(|mut v, _b|  {
                v.sin();
                v
            } )
        });
    }

    #[bench]
    fn real_sin_64s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVec<Vec<f64>, f64>, f64>::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _b|  {
                v.sin();
                v
            } )
        });
    }

    #[bench]
    fn complex_sin_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<ComplexTimeVec<Vec<f32>, f32>, f32>::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _b|  {
                v.sin();
                v
            } )
        });
    }

    #[bench]
    fn swap_halves_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<ComplexTimeVec<Vec<f32>, f32>, f32>::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, buffer|  {
                v.swap_halves_b(buffer);
                v
            } )
        });
    }

    #[bench]
    fn reverse_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<ComplexTimeVec<Vec<f32>, f32>, f32>::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _b|  {
                v.reverse();
                v
            } )
        });
    }
}
