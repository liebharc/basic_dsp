#![feature(test)]
#![feature(box_syntax)]
extern crate basic_dsp;
extern crate num;
extern crate test;

pub mod tools;

#[cfg(test)]
mod complex {
    use basic_dsp::conv_types::*;
    use basic_dsp::*;
    use num::complex::Complex32;
    use test::Bencher;
    use tools::*;

    #[bench]
    fn complex_offset_32s_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _| {
                v.offset(Complex32::new(2.0, -5.0));
                v
            })
        });
    }

    #[bench]
    fn complex_scale_32s_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _| {
                v.scale(Complex32::new(-2.0, 2.0));
                v
            })
        });
    }

    #[bench]
    fn complex_sin_32s_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _| {
                v.sin();
                v
            })
        });
    }

    #[bench]
    fn complex_conj_32s_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _| {
                v.conj();
                v
            })
        });
    }

    #[bench]
    fn complex_abs_to_complex_32s_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|v, _| {
                let r = v.magnitude();
                r.to_complex().unwrap()
            })
        });
    }

    #[bench]
    fn complex_vector_multiplication_32s_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute_with_arg(|mut v, o, _| {
                v.mul(&o).unwrap();
                (v, o)
            })
        });
    }

    #[bench]
    fn convolve_vector_with_signal_32s_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, buffer| {
                let sinc: SincFunction<f32> = SincFunction::new();
                let table = RealTimeLinearTableLookup::<f32>::from_conv_function(
                    &sinc as &RealImpulseResponse<f32>,
                    0.2,
                    5,
                );
                v.convolve(buffer, &table as &RealImpulseResponse<f32>, 0.5, 10);
                v
            })
        });
    }

    #[bench]
    fn convolve_vector_with_vector_32t_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Tiny);
        b.iter(|| {
            vector.execute_with_arg(|mut v, o, buffer| {
                v.convolve_signal(buffer, &o).unwrap();
                (v, o)
            })
        });
    }

    #[bench]
    fn convolve_vector_with_smaller_vector_32s_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, buffer| {
                let operand = vec![0.0; 100].to_complex_time_vec();
                v.convolve_signal(buffer, &operand).unwrap();
                v
            })
        });
    }

    #[bench]
    fn interpolatei_32t_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, buffer| {
                let len = v.len();
                v.interpolatei(buffer, &RaisedCosineFunction::new(0.35), 10)
                    .unwrap();
                v.resize(len).unwrap();
                v
            })
        });
    }

    #[bench]
    fn interpolatef_32t_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, buffer| {
                let len = v.len();
                v.interpolatef(buffer, &RaisedCosineFunction::new(0.35), 10.0, 0.0, 10);
                v.resize(len).unwrap();
                v
            })
        });
    }

    #[bench]
    fn interpolatef_delayed_32t_benchmark(b: &mut Bencher) {
        let mut vector = ComplexTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, buffer| {
                let len = v.len();
                v.interpolatef(buffer, &RaisedCosineFunction::new(0.35), 10.0, 0.5, 10);
                v.resize(len).unwrap();
                v
            })
        });
    }
}
