#![feature(test)]
#![feature(box_syntax)] 
extern crate test;
extern crate basic_dsp;
extern crate num;

pub mod tools;
use tools::{Size, VectorBox, translate_size};
use basic_dsp::vector_types2::*;

impl VectorBox<RealTimeVec<Vec<f32>, f32>> {
    pub fn new(size: Size) -> VectorBox<RealTimeVec<Vec<f32>, f32>>
    {
        let size = translate_size(size);
        let data = vec![0.0; size];
        VectorBox
        {
            vector: Box::into_raw(Box::new(data.to_real_time_vec())),
            size: size
        }
    }
}

#[cfg(test)]
mod vec2 {
    use test::Bencher;
    use basic_dsp::vector_types2::*;
    use tools::{VectorBox, Size};
    
    #[bench]
    fn real_sin_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVec<Vec<f32>, f32>>::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v|  { 
                v.sin();
                v
            } )
        });
    }
}