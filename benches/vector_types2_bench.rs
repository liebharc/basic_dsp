#![feature(test)]
#![feature(box_syntax)]
extern crate test;
extern crate basic_dsp;
extern crate num;
pub mod tools;

use basic_dsp::RealNumber;
use basic_dsp::vector_types2::*;

#[cfg(test)]
mod vec2 {
    use test::Bencher;
    use basic_dsp::vector_types2::*;
    use super::*;
    use tools::*;

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
