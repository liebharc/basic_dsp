#![feature(test)]
#![feature(box_syntax)]
extern crate test;
extern crate basic_dsp;
extern crate num;

pub mod tools;

#[cfg(test)]
mod real {
    use test::Bencher;
    use basic_dsp::*;
    use basic_dsp::combined_ops::*;
    use tools::*;

    #[bench]
    fn multi_operations_2ops1_vector_32_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|v, mut buffer| {
                let ops = multi_ops1(v);
                let ops = ops.add_ops(|mut x| {
                    x.log(10.0);
                    x.scale(10.0);
                    x
                });
                ops.get(buffer).unwrap()
            })
        });
    }

    #[bench]
    fn multi_operations_2ops1_vector_32_reference(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _| {
                v.log(10.0);
                v.scale(10.0);
                v
            })
        });
    }

    #[bench]
    fn multi_operations_3ops1_vector_32_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|v, mut buffer| {
                let ops = multi_ops1(v);
                let ops = ops.add_ops(|mut x| {
                    x.log(10.0);
                    x.scale(10.0);
                    x.sqrt();
                    x
                });
                ops.get(buffer).unwrap()
            })
        });
    }

    #[bench]
    fn multi_operations_3ops1_vector_32_reference(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _| {
                v.log(10.0);
                v.scale(10.0);
                v.sqrt();
                v
            })
        });
    }

    #[bench]
    fn multi_operations_3ops2_vector_32_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|v, mut buffer| {
                let len = v.len();
                let operand = vec!(6.0; len).to_complex_time_vec();
                let ops = multi_ops2(v, operand);
                let ops = ops.add_ops(|mut v, o| {
                    v.log(10.0);
                    v.mul(&o).unwrap();
                    v.sin();
                    (v, o)
                });
                let (v, _) = ops.get(buffer).unwrap();
                v
            })
        });
    }

    #[bench]
    fn multi_operations_6ops1_vector_32_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|v, mut buffer| {
                let ops = multi_ops1(v);
                let ops = ops.add_ops(|mut x| {
                    x.square();
                    x.scale(6.0);
                    x.sin();
                    x.log(10.0);
                    x.scale(10.0);
                    x.sqrt();
                    x
                });
                ops.get(buffer).unwrap()
            })
        });
    }

    #[bench]
    fn multi_operations_6ops2_vector_32_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|v, mut buffer| {
                let len = v.len();
                let operand = vec!(6.0; len).to_complex_time_vec();
                let ops = multi_ops2(v, operand);
                let ops = ops.add_ops(|mut v, o| {
                    v.square();
                    v.mul(&o).unwrap();
                    v.sin();
                    v.log(10.0);
                    v.scale(10.0);
                    v.sqrt();
                    (v, o)
                });
                let (v, _) = ops.get(buffer).unwrap();
                v
            })
        });
    }

    #[bench]
    fn real_scale_with_multi_ops_mapping_32m_benchmark(b: &mut Bencher) {
        let mut vector = RealTime32Box::new(Size::Medium);
        b.iter(|| {
            vector.execute(|v, mut buffer| {
                let len = v.len();
                let operand = vec!(6.0; len).to_real_time_vec();
                let ops = multi_ops2(v, operand);
                let ops = ops.add_ops(|mut v, o| {
                    v.map_inplace(|v, _| 2.0 * v);
                    (v, o)
                });
                let (v, _) = ops.get(buffer).unwrap();
                v
            })
        });
    }
}
