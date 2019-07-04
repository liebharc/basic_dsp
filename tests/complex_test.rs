extern crate basic_dsp;
extern crate num;
extern crate rand;
pub mod tools;

mod complex_test {
    use crate::tools::*;
    use basic_dsp::*;
    use num::complex::*;

    fn to_complex(a: &Vec<f32>) -> Vec<Complex32> {
        let mut result = vec![Complex32::new(0.0, 0.0); a.len() / 2];
        for i in 0..result.len() {
            result[i] = Complex32::new(a[2 * i], a[2 * i + 1]);
        }

        result
    }

    fn from_complex(a: &Vec<Complex32>) -> Vec<f32> {
        let mut result = vec![0.0; a.len() * 2];
        for i in 0..a.len() {
            result[2 * i] = a[i].re;
            result[2 * i + 1] = a[i].im;
        }

        result
    }

    fn complex_add_scalar(a: &Vec<f32>, value: Complex32) -> Vec<f32> {
        let complex = to_complex(&a);
        let mut result = vec![Complex32::new(0.0, 0.0); complex.len()];
        for i in 0..complex.len() {
            result[i] = complex[i] + value;
        }

        from_complex(&result)
    }

    #[test]
    fn complex_add_scalar_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(2015111410, iteration, range.start, range.end);
            let scalar = create_data_with_len(2015111413, iteration, 2);
            let scalar = Complex32::new(scalar[0], scalar[1]);
            let expected = complex_add_scalar(&a, scalar);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.to_complex_time_vec_par();
            vector.set_delta(delta);
            vector.offset(scalar);
            assert_vector_eq(&expected, vector.data(..));
            assert_eq!(vector.is_complex(), true);
            assert_eq!(vector.delta(), delta);
        });
    }

    fn complex_multiply_scalar(a: &Vec<f32>, value: Complex32) -> Vec<f32> {
        let complex = to_complex(&a);
        let mut result = vec![Complex32::new(0.0, 0.0); complex.len()];
        for i in 0..complex.len() {
            result[i] = complex[i] * value;
        }

        from_complex(&result)
    }

    #[test]
    fn complex_mutiply_scalar_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(2015111410, iteration, range.start, range.end);
            let scalar = create_data_with_len(2015111413, iteration, 2);
            let scalar = Complex32::new(scalar[0], scalar[1]);
            let expected = complex_multiply_scalar(&a, scalar);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.to_complex_time_vec_par();
            vector.set_delta(delta);
            vector.scale(scalar);
            assert_vector_eq(&expected, vector.data(..));
            assert_eq!(vector.is_complex(), true);
            assert_eq!(vector.delta(), delta);
        });
    }

    fn complex_abs(a: &Vec<f32>) -> Vec<f32> {
        let complex = to_complex(&a);
        let mut result = vec![0.0; complex.len()];
        for i in 0..complex.len() {
            result[i] = (complex[i].re * complex[i].re + complex[i].im * complex[i].im).sqrt();
        }

        result
    }

    #[test]
    fn complex_abs_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(2015111410, iteration, range.start, range.end);
            let expected = complex_abs(&a);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.to_complex_time_vec_par();
            vector.set_delta(delta);
            let vector = vector.magnitude();
            assert_vector_eq(&expected, vector.data(..));
            assert_eq!(vector.is_complex(), false);
            assert_eq!(vector.delta(), delta);
        });
    }

    fn complex_abs_sq(a: &Vec<f32>) -> Vec<f32> {
        let complex = to_complex(&a);
        let mut result = vec![0.0; complex.len()];
        for i in 0..complex.len() {
            result[i] = complex[i].re * complex[i].re + complex[i].im * complex[i].im;
        }

        result
    }

    #[test]
    fn complex_abs_sq_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(2015111410, iteration, range.start, range.end);
            let expected = complex_abs_sq(&a);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.to_complex_time_vec_par();
            vector.set_delta(delta);
            let vector = vector.magnitude_squared();
            assert_vector_eq(&expected, vector.data(..));
            assert_eq!(vector.is_complex(), false);
            assert_eq!(vector.delta(), delta);
        });
    }

    fn complex_vector_mul(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        let a = to_complex(a);
        let b = to_complex(b);
        let mut result = vec![Complex32::new(0.0, 0.0); a.len()];
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }

        from_complex(&result)
    }

    #[test]
    fn complex_mul_vector_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511171, iteration, range.start, range.end);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = complex_vector_mul(&a, &b);
            let delta = create_delta(3561159, iteration);
            let mut vector1 = a.to_complex_time_vec_par();
            vector1.set_delta(delta);
            let mut vector2 = b.to_complex_time_vec_par();
            vector2.set_delta(delta);
            vector1.mul(&vector2).unwrap();
            assert_vector_eq(&expected, vector1.data(..));
            assert_eq!(vector1.is_complex(), true);
            assert_eq!(vector1.delta(), delta);
        });
    }

    fn complex_vector_mul_mod(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        let a = to_complex(a);
        let b = to_complex(b);
        let mut result = vec![Complex32::new(0.0, 0.0); a.len()];
        for i in 0..a.len() {
            result[i] = a[i] * b[i % b.len()];
        }

        from_complex(&result)
    }

    #[test]
    fn complex_mul_smaller_vector_vector32() {
        let a = create_data_with_len(201511171, 1, 240);
        let b = create_data_with_len(201511172, 1, 10);
        let expected = complex_vector_mul_mod(&a, &b);
        let delta = create_delta(3561159, 1);
        let mut vector1 = a.to_complex_time_vec_par();
        vector1.set_delta(delta);
        let mut vector2 = b.to_complex_time_vec_par();
        vector2.set_delta(delta);
        vector1.mul_smaller(&vector2).unwrap();
        assert_vector_eq(&expected, vector1.data(..));
        assert_eq!(vector1.is_complex(), true);
        assert_eq!(vector1.delta(), delta);
    }

    fn complex_vector_div_mod(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        let a = to_complex(a);
        let b = to_complex(b);
        let mut result = vec![Complex32::new(0.0, 0.0); a.len()];
        for i in 0..a.len() {
            result[i] = a[i] / b[i % b.len()];
        }

        from_complex(&result)
    }

    #[test]
    fn complex_div_smaller_vector_vector32() {
        let a = create_data_with_len(201511171, 1, 240);
        let b = create_data_with_len(201511172, 1, 10);
        let expected = complex_vector_div_mod(&a, &b);
        let delta = create_delta(3561159, 1);
        let mut vector1 = a.to_complex_time_vec_par();
        vector1.set_delta(delta);
        let mut vector2 = b.to_complex_time_vec_par();
        vector2.set_delta(delta);
        vector1.div_smaller(&vector2).unwrap();
        assert_vector_eq(&expected, vector1.data(..));
        assert_eq!(vector1.is_complex(), true);
        assert_eq!(vector1.delta(), delta);
    }

    #[test]
    fn complex_dot_product32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511171, iteration, range.start, range.end);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = to_complex(&complex_vector_mul(&a, &b))
                .iter()
                .fold(Complex32::new(0.0, 0.0), |a, b| a + b);
            let delta = create_delta(3561159, iteration);
            let mut vector1 = a.to_complex_time_vec_par();
            vector1.set_delta(delta);
            let mut vector2 = b.to_complex_time_vec_par();
            vector2.set_delta(delta);
            let result = vector1.dot_product(&vector2).unwrap();
            assert_in_tolerance(expected.re, result.re, 0.5);
            assert_in_tolerance(expected.im, result.im, 0.5);
        });
    }

    #[test]
    fn complex_dot_product_prec32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511171, iteration, range.start, range.end);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = to_complex(&complex_vector_mul(&a, &b)).iter().fold(
                Complex64::new(0.0, 0.0),
                |a, b| {
                    let a = Complex64::new(a.re as f64, a.im as f64);
                    let b = Complex64::new(b.re as f64, b.im as f64);
                    a + b
                },
            );
            let delta = create_delta(3561159, iteration);
            let mut vector1 = a.to_complex_time_vec_par();
            vector1.set_delta(delta);
            let mut vector2 = b.to_complex_time_vec_par();
            vector2.set_delta(delta);
            let result = vector1.dot_product_prec(&vector2).unwrap();
            assert_in_tolerance(expected.re as f32, result.re, 1e-2);
            assert_in_tolerance(expected.im as f32, result.im, 1e-2);
        });
    }

    fn complex_vector_div(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        let a = to_complex(a);
        let b = to_complex(b);
        let mut result = vec![Complex32::new(0.0, 0.0); a.len()];
        for i in 0..a.len() {
            result[i] = a[i] / b[i];
        }

        from_complex(&result)
    }

    #[test]
    fn complex_div_vector_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511171, iteration, range.start, range.end);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = complex_vector_div(&a, &b);
            let delta = create_delta(3561159, iteration);
            let mut vector1 = a.to_complex_time_vec_par();
            vector1.set_delta(delta);
            let mut vector2 = b.to_complex_time_vec_par();
            vector2.set_delta(delta);
            vector1.div(&vector2).unwrap();
            assert_vector_eq(&expected, vector1.data(..));
            assert_eq!(vector1.is_complex(), true);
            assert_eq!(vector1.delta(), delta);
        });
    }

    #[test]
    fn complex_real_conversions_vector32() {
        parameterized_vector_test(|iteration, range| {
            let real = create_data(201511191, iteration, range.start, range.end);
            let imag = create_data_with_len(201511192, iteration, real.len());
            let realvec = real.clone().to_real_time_vec_par();
            let imagvec = imag.clone().to_real_time_vec_par();
            let delta = create_delta(3561159, iteration);
            let mut complex = vec![0.0; 0].to_complex_time_vec_par();
            complex.set_delta(delta);
            complex.set_real_imag(&realvec, &imagvec).unwrap();
            assert_eq!(complex.len(), real.len() + imag.len());
            assert_eq!(complex.is_complex(), true);
            let mut real_vector = Vec::new().to_real_time_vec_par();
            let mut imag_vector = Vec::new().to_real_time_vec_par();
            assert_eq!(real_vector.is_complex(), false);
            complex.get_real(&mut real_vector);
            assert_eq!(real_vector.len(), real.len());
            complex.get_imag(&mut imag_vector);
            let real_result = complex.to_real();
            assert_vector_eq_with_reason(&real, real_vector.data(..), "Failure in get_real");
            assert_vector_eq_with_reason(&real, real_result.data(..), "Failure in get_imag");
            assert_vector_eq_with_reason(&imag, imag_vector.data(..), "Failure in to_real");
            assert_eq!(real_vector.is_complex(), false);
            assert_eq!(real_vector.delta(), delta);
            assert_eq!(imag_vector.is_complex(), false);
            assert_eq!(imag_vector.delta(), delta);
            assert_eq!(real_result.is_complex(), false);
            assert_eq!(real_result.delta(), delta);
        });
    }

    #[test]
    fn abs_phase_conversions_vector32() {
        parameterized_vector_test(|iteration, range| {
            let abs =
                create_data_even_in_range(201511191, iteration, range.start, range.end, 0.1, 10.0);
            let phase = create_data_in_range_with_len(201511204, iteration, abs.len(), -1.57, 1.57);

            let absvec = abs.clone().to_real_time_vec_par();
            let phasevec = phase.clone().to_real_time_vec_par();
            let delta = create_delta(3561159, iteration);
            let mut complex = vec![0.0; 0].to_complex_time_vec_par();
            complex.set_delta(delta);
            complex.set_mag_phase(&absvec, &phasevec).unwrap();
            assert_eq!(complex.len(), abs.len() + phase.len());
            assert_eq!(complex.is_complex(), true);
            let mut abs_vector = Vec::new().to_real_time_vec_par();
            let mut phase_vector = Vec::new().to_real_time_vec_par();
            assert_eq!(abs_vector.is_complex(), false);
            complex.get_magnitude(&mut abs_vector);
            assert_eq!(abs_vector.len(), abs.len());
            complex.get_phase(&mut phase_vector);
            let phase_result = complex.phase();
            assert_vector_eq_with_reason(&phase, phase_vector.data(..), "Failure in get_phase");
            assert_vector_eq_with_reason(&abs, abs_vector.data(..), "Failure in get_magnitude");
            assert_vector_eq_with_reason(&phase, phase_result.data(..), "Failure in phase()");
            assert_eq!(abs_vector.is_complex(), false);
            assert_eq!(abs_vector.delta(), delta);
            assert_eq!(phase_vector.is_complex(), false);
            assert_eq!(phase_vector.delta(), delta);
            assert_eq!(phase_result.is_complex(), false);
            assert_eq!(phase_result.delta(), delta);
        });
    }

    fn complex_vector_diff(a: &Vec<f32>) -> Vec<f32> {
        let a = to_complex(a);
        let mut result = vec![Complex32::new(0.0, 0.0); a.len()];
        result[0] = a[0];
        for i in 1..a.len() {
            result[i] = a[i] - a[i - 1];
        }

        from_complex(&result)
    }

    #[test]
    fn complex_diff_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_complex_time_vec_par();
            vector.set_delta(delta);
            let expected = complex_vector_diff(&a);
            vector.diff_with_start();
            assert_vector_eq(&expected, vector.data(..));
            assert_eq!(vector.is_complex(), true);
            assert_eq!(vector.delta(), delta);
        });
    }

    fn complex_vector_cum_sum(a: &Vec<f32>) -> Vec<f32> {
        let a = to_complex(a);
        let mut result = vec![Complex32::new(0.0, 0.0); a.len()];
        result[0] = a[0];
        for i in 1..a.len() {
            result[i] = a[i] + result[i - 1];
        }

        from_complex(&result)
    }

    #[test]
    fn complex_cum_sum_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_complex_time_vec_par();
            vector.set_delta(delta);
            let expected = complex_vector_cum_sum(&a);
            vector.cum_sum();
            assert_vector_eq(&expected, vector.data(..));
            assert_eq!(vector.is_complex(), true);
            assert_eq!(vector.delta(), delta);
        });
    }

    fn complex_exponential(vec: &Vec<f32>, a: f32, b: f32, delta: f32) -> Vec<f32> {
        let a = a * delta;
        let b = b * delta;
        let mut exponential = Complex32::from_polar(&1.0, &b);
        let increment = Complex32::from_polar(&1.0, &a);
        let mut result = to_complex(vec);
        for complex in &mut result {
            *complex = (*complex) * exponential;
            exponential = exponential * increment;
        }

        from_complex(&result)
    }

    #[test]
    fn complex_exponential_vector32() {
        parameterized_vector_test(|iteration, _| {
            // With f32 the errors sums up quite a bit for large arrays
            // in `complex_exponential` and therefore the length is limited
            // in this test.
            let a = create_data_with_len(201511210, iteration, 10000);
            let args = create_data_with_len(201511210, iteration, 2);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_complex_time_vec_par();
            vector.set_delta(delta);
            let expected = complex_exponential(&a, args[0], args[1], delta);
            vector.multiply_complex_exponential(args[0], args[1]);
            assert_vector_eq_with_reason_and_tolerance(&expected, vector.data(..), 1e-2, "");
            assert_eq!(vector.is_complex(), true);
            assert_eq!(vector.delta(), delta);
        });
    }

    #[test]
    fn statistics_test32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let c = to_complex(&a);
            let mut vector = a.clone().to_complex_time_vec_par();
            vector.set_delta(delta);
            let sum = c.iter().fold(Complex32::new(0.0, 0.0), |a, b| a + b);
            let sum_sq = c
                .iter()
                .map(|v| v * v)
                .fold(Complex32::new(0.0, 0.0), |a, b| a + b);
            let rms = (sum_sq / a.len() as f32).sqrt();
            let result = vector.statistics();
            assert_complex_in_tolerance(result.sum, sum, 0.5);
            assert_complex_in_tolerance(result.rms, rms, 0.5);
        });
    }

    #[test]
    fn statistics_vs_sum_test32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_complex_time_vec_par();
            vector.set_delta(delta);
            let sum = vector.sum();
            let sum_sq = vector.sum_sq();
            let rms = (sum_sq / a.len() as f32).sqrt();
            let result = vector.statistics();
            assert_complex_in_tolerance(result.sum, sum, 0.5);
            assert_complex_in_tolerance(result.rms, rms, 0.5);
        });
    }

    #[test]
    fn statistics_prec_test32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let c = to_complex(&a);
            let mut vector = a.clone().to_complex_time_vec_par();
            vector.set_delta(delta);
            let sum = c.iter().fold(Complex64::new(0.0, 0.0), |a, b| {
                let a = Complex64::new(a.re as f64, a.im as f64);
                let b = Complex64::new(b.re as f64, b.im as f64);
                a + b
            });
            let sum_sq = c
                .iter()
                .map(|v| v * v)
                .fold(Complex64::new(0.0, 0.0), |a, b| {
                    let a = Complex64::new(a.re as f64, a.im as f64);
                    let b = Complex64::new(b.re as f64, b.im as f64);
                    a + b
                });
            let rms = (sum_sq / a.len() as f64).sqrt();
            let result = vector.statistics_prec();
            assert_complex_in_tolerance(
                Complex32::new(result.sum.re as f32, result.sum.im as f32),
                Complex32::new(sum.re as f32, sum.im as f32),
                1e-2,
            );
            assert_complex_in_tolerance(
                Complex32::new(result.rms.re as f32, result.rms.im as f32),
                Complex32::new(rms.re as f32, rms.im as f32),
                0.3,
            );
        });
    }

    #[test]
    fn statistics_prec_vs_sum_prec_test32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_complex_time_vec_par();
            vector.set_delta(delta);
            let sum = vector.sum_prec();
            let sum_sq = vector.sum_sq_prec();
            let rms = (sum_sq / a.len() as f64).sqrt();
            let result = vector.statistics_prec();
            assert_complex_in_tolerance(
                Complex32::new(result.sum.re as f32, result.sum.im as f32),
                Complex32::new(sum.re as f32, sum.im as f32),
                1e-2,
            );
            assert_complex_in_tolerance(
                Complex32::new(result.rms.re as f32, result.rms.im as f32),
                Complex32::new(rms.re as f32, rms.im as f32),
                0.3,
            );
        });
    }

    #[test]
    fn split_merge_test32() {
        let a = create_data(201511210, 0, 1000, 1000);
        let vector = a.clone().to_complex_time_vec_par();
        let empty: Vec<f32> = Vec::new();
        let mut split = [
            Box::new(empty.clone().to_complex_time_vec_par()),
            Box::new(empty.clone().to_complex_time_vec_par()),
            Box::new(empty.clone().to_complex_time_vec_par()),
            Box::new(empty.clone().to_complex_time_vec_par()),
            Box::new(empty.clone().to_complex_time_vec_par()),
        ];
        {
            let mut dest: Vec<_> = split.iter_mut().map(|x| x.as_mut()).collect();
            vector.split_into(&mut dest[..]).unwrap();
        }
        let mut merge = empty.to_complex_time_vec_par();
        let src: Vec<_> = split.iter().map(|x| x.as_ref()).collect();
        merge.merge(&src[..]).unwrap();
        assert_vector_eq(&a, merge.data(..));
    }

    #[test]
    fn split_test32() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let vector = a.to_complex_time_vec_par();
        let empty: Vec<f32> = Vec::new();
        let mut split = [
            &mut empty.clone().to_complex_time_vec_par(),
            &mut empty.clone().to_complex_time_vec_par(),
        ];
        vector.split_into(&mut split).unwrap();
        assert_vector_eq(&[1.0, 2.0, 5.0, 6.0], split[0].data(..));
        assert_vector_eq(&[3.0, 4.0, 7.0, 8.0], split[1].data(..));
    }

    #[test]
    fn to_real_imag_and_back32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_complex_time_vec_par();
            vector.set_delta(delta);
            let mut real = Vec::new().to_real_time_vec_par();
            let mut imag = Vec::new().to_real_time_vec_par();
            vector.get_real_imag(&mut real, &mut imag);
            let mut vector2 = vec![0.0; 0].to_complex_time_vec_par();
            vector2.set_real_imag(&real, &imag).unwrap();
            assert_vector_eq(&a, vector2.data(..));
        });
    }

    #[test]
    fn to_mag_phase_and_back32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_complex_time_vec_par();
            vector.set_delta(delta);
            let mut mag = Vec::new().to_real_time_vec_par();
            let mut phase = Vec::new().to_real_time_vec_par();
            let mut mag2 = Vec::new().to_real_time_vec_par();
            let mut phase2 = Vec::new().to_real_time_vec_par();
            vector.get_mag_phase(&mut mag, &mut phase);
            vector.get_magnitude(&mut mag2);
            vector.get_phase(&mut phase2);

            assert_vector_eq_with_reason(mag.data(..), mag2.data(..), "Magnitude differs");
            assert_vector_eq_with_reason(phase.data(..), phase2.data(..), "Phase differs");

            let mut vector2 = vec![0.0; 0].to_complex_time_vec_par();
            vector2.set_mag_phase(&mag, &phase).unwrap();
            assert_vector_eq_with_reason_and_tolerance(&a, vector2.data(..), 1e-4, "Merge differs");
        });
    }
}
