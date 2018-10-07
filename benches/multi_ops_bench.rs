#![feature(test)]
#![feature(box_syntax)]
extern crate basic_dsp;
extern crate test;

pub mod tools;

#[cfg(test)]
mod real {
    use basic_dsp::combined_ops::*;
    use basic_dsp::*;
    use test::Bencher;
    use tools::*;

    #[bench]
    fn multi_operations_2ops_1vector_32_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Medium);
        b.iter(|| {
            vector.execute(|v, buffer| {
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
    fn multi_operations_2ops_1vector_32_reference(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Medium);
        b.iter(|| {
            vector.execute(|mut v, _| {
                v.log(10.0);
                v.scale(10.0);
                v
            })
        });
    }

    #[bench]
    fn multi_operations_3ops_1vector_32_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Medium);
        b.iter(|| {
            vector.execute(|v, buffer| {
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
    fn multi_operations_3ops_1vector_32_reference(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Medium);
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
    fn multi_operations_3ops_2vectors_32_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Medium);
        b.iter(|| {
            vector.execute_with_arg(|v, operand, buffer| {
                let ops = multi_ops2(v, operand);
                let ops = ops.add_ops(|mut v, o| {
                    v.log(10.0);
                    v.mul(&o).unwrap();
                    v.sin();
                    (v, o)
                });
                let (v, operand) = ops.get(buffer).unwrap();
                (v, operand)
            })
        });
    }

    #[bench]
    fn multi_operations_3ops_2vectors_32_reference(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Medium);
        b.iter(|| {
            vector.execute_with_arg(|mut v, o, _| {
                v.log(10.0);
                v.mul(&o).unwrap();
                v.sin();
                (v, o)
            })
        });
    }


    #[bench]
    fn multi_operations_6ops_1vector_32_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Medium);
        b.iter(|| {
            vector.execute(|v, buffer| {
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
    fn multi_operations_6ops_1vector_32_reference(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Medium);
        b.iter(|| {
            vector.execute(|mut x, _| {
                x.square();
                x.scale(6.0);
                x.sin();
                x.log(10.0);
                x.scale(10.0);
                x.sqrt();
                x
            })
        });
    }

    #[bench]
    fn multi_operations_6ops_2vectors_32_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Medium);
        b.iter(|| {
            vector.execute_with_arg(|v, operand, buffer| {
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
                let (v, operand) = ops.get(buffer).unwrap();
                (v, operand)
            })
        });
    }



    #[bench]
    fn multi_operations_6ops_2vectors_32_reference(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Medium);
        b.iter(|| {
            vector.execute_with_arg(|mut v, o, _| {
                v.square();
                v.mul(&o).unwrap();
                v.sin();
                v.log(10.0);
                v.scale(10.0);
                v.sqrt();
                (v, o)
            })
        });
    }
}
