extern crate basic_dsp;
extern crate rand;
extern crate num;
pub mod tools;

mod slow_test {
    use basic_dsp::vector_types2::*;
    use basic_dsp::vector_types2::combined_ops::*;
    use num::complex::*;
    use tools::*;

    fn to_complex(a: &Vec<f32>) -> Vec<Complex32>
    {
        let mut result = vec![Complex32::new(0.0, 0.0); a.len() / 2];
        for i in 0..result.len() {
            result[i] = Complex32::new(a[2 * i], a[2 * i + 1]);
        }

        result
    }

    fn from_complex(a: &Vec<Complex32>) -> Vec<f32>
    {
        let mut result = vec![0.0; a.len() * 2];
        for i in 0..a.len() {
            result[2 * i] = a[i].re;
            result[2 * i + 1] = a[i].im;
        }

        result
    }

    fn complex_add_scalar(a: &Vec<f32>, value: Complex32) -> Vec<f32>
    {
        let complex = to_complex(&a);
        let mut result = vec![Complex32::new(0.0, 0.0); complex.len()];
        for i in 0 .. complex.len() {
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
            let mut vector = a.to_complex_time_vec();
            vector.set_delta(delta);
            vector.offset(scalar);
            assert_vector_eq(&expected, &vector[..]);
            assert_eq!(vector.is_complex(), true);
            assert_eq!(vector.delta(), delta);
        });
    }

    fn complex_multiply_scalar(a: &Vec<f32>, value: Complex32) -> Vec<f32>
    {
        let complex = to_complex(&a);
        let mut result = vec![Complex32::new(0.0, 0.0); complex.len()];
        for i in 0 .. complex.len() {
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
            let mut vector = a.to_complex_time_vec();
            vector.set_delta(delta);
            vector.scale(scalar);
            assert_vector_eq(&expected, &vector[..]);
            assert_eq!(vector.is_complex(), true);
            assert_eq!(vector.delta(), delta);
        });
    }

    fn complex_abs(a: &Vec<f32>) -> Vec<f32>
    {
        let complex = to_complex(&a);
        let mut result = vec![0.0; complex.len()];
        for i in 0 .. complex.len() {
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
            let mut vector = a.to_complex_time_vec();
            vector.set_delta(delta);
            let vector = vector.magnitude();
            assert_vector_eq(&expected, &vector[..]);
            assert_eq!(vector.is_complex(), false);
            assert_eq!(vector.delta(), delta);
        });
    }

    fn complex_abs_sq(a: &Vec<f32>) -> Vec<f32>
    {
        let complex = to_complex(&a);
        let mut result = vec![0.0; complex.len()];
        for i in 0 .. complex.len() {
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
            let mut vector = a.to_complex_time_vec();
            vector.set_delta(delta);
            let vector = vector.magnitude_squared();
            assert_vector_eq(&expected, &vector[..]);
            assert_eq!(vector.is_complex(), false);
            assert_eq!(vector.delta(), delta);
        });
    }

    fn complex_vector_mul(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32>
    {
        let a = to_complex(a);
        let b = to_complex(b);
        let mut result = vec![Complex32::new(0.0, 0.0); a.len()];
        for i in 0 .. a.len() {
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
            let mut vector1 = a.to_complex_time_vec();
            vector1.set_delta(delta);
            let mut vector2 = b.to_complex_time_vec();
            vector2.set_delta(delta);
            vector1.mul(&vector2).unwrap();
            assert_vector_eq(&expected, &vector1[..]);
            assert_eq!(vector1.is_complex(), true);
            assert_eq!(vector1.delta(), delta);
        });
    }

    fn complex_vector_mul_mod(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32>
    {
        let a = to_complex(a);
        let b = to_complex(b);
        let mut result = vec![Complex32::new(0.0, 0.0); a.len()];
        for i in 0 .. a.len() {
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
        let mut vector1 = a.to_complex_time_vec();
        vector1.set_delta(delta);
        let mut vector2 = b.to_complex_time_vec();
        vector2.set_delta(delta);
        vector1.mul_smaller(&vector2).unwrap();
        assert_vector_eq(&expected, &vector1[..]);
        assert_eq!(vector1.is_complex(), true);
        assert_eq!(vector1.delta(), delta);
    }

    fn complex_vector_div_mod(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32>
    {
        let a = to_complex(a);
        let b = to_complex(b);
        let mut result = vec![Complex32::new(0.0, 0.0); a.len()];
        for i in 0 .. a.len() {
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
        let mut vector1 = a.to_complex_time_vec();
        vector1.set_delta(delta);
        let mut vector2 = b.to_complex_time_vec();
        vector2.set_delta(delta);
        vector1.div_smaller(&vector2).unwrap();
        assert_vector_eq(&expected, &vector1[..]);
        assert_eq!(vector1.is_complex(), true);
        assert_eq!(vector1.delta(), delta);
    }

    #[test]
    fn complex_dot_product32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511171, iteration, range.start, range.end);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = to_complex(&complex_vector_mul(&a, &b)).iter().fold(Complex32::new(0.0, 0.0), |a, b| a + b);
            let delta = create_delta(3561159, iteration);
            let mut vector1 = a.to_complex_time_vec();
            vector1.set_delta(delta);
            let mut vector2 = b.to_complex_time_vec();
            vector2.set_delta(delta);
            let result = vector1.dot_product(&vector2).unwrap();
            assert_in_tolerance(expected.re, result.re, 0.5);
            assert_in_tolerance(expected.im, result.im, 0.5);
        });
    }

    fn complex_vector_div(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32>
    {
        let a = to_complex(a);
        let b = to_complex(b);
        let mut result = vec![Complex32::new(0.0, 0.0); a.len()];
        for i in 0 .. a.len() {
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
            let mut vector1 = a.to_complex_time_vec();
            vector1.set_delta(delta);
            let mut vector2 = b.to_complex_time_vec();
            vector2.set_delta(delta);
            vector1.div(&vector2).unwrap();
            assert_vector_eq(&expected, &vector1[..]);
            assert_eq!(vector1.is_complex(), true);
            assert_eq!(vector1.delta(), delta);
        });
    }

    #[test]
    fn complex_real_conversions_vector32() {
        parameterized_vector_test(|iteration, range| {
            let real = create_data(201511191, iteration, range.start, range.end);
            let imag = create_data_with_len(201511192, iteration, real.len());
            let realvec = real.clone().to_real_time_vec();
            let imagvec = imag.clone().to_real_time_vec();
            let delta = create_delta(3561159, iteration);
            let mut complex = vec!(0.0; 0).to_complex_time_vec();
            complex.set_delta(delta);
            complex.set_real_imag(&realvec, &imagvec).unwrap();
            assert_eq!(complex.len(), real.len() + imag.len());
            assert_eq!(complex.is_complex(), true);
            let mut real_vector = Vec::new().to_real_time_vec();
            let mut imag_vector = Vec::new().to_real_time_vec();
            assert_eq!(real_vector.is_complex(), false);
            complex.get_real(&mut real_vector);
            assert_eq!(real_vector.len(), real.len());
            complex.get_imag(&mut imag_vector);
            let real_result = complex.to_real();
            assert_vector_eq_with_reason(&real, &real_vector[..], "Failure in get_real");
            assert_vector_eq_with_reason(&real, &real_result[..], "Failure in get_imag");
            assert_vector_eq_with_reason(&imag, &imag_vector[..], "Failure in to_real");
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
            let abs = create_data_even_in_range(201511191, iteration, range.start, range.end, 0.1, 10.0);
            let phase = create_data_in_range_with_len(201511204, iteration, abs.len(), -1.57, 1.57);

            let absvec = abs.clone().to_real_time_vec();
            let phasevec = phase.clone().to_real_time_vec();
            let delta = create_delta(3561159, iteration);
            let mut complex = vec!(0.0; 0).to_complex_time_vec();
            complex.set_delta(delta);
            complex.set_mag_phase(&absvec, &phasevec).unwrap();
            assert_eq!(complex.len(), abs.len() + phase.len());
            assert_eq!(complex.is_complex(), true);
            let mut abs_vector = Vec::new().to_real_time_vec();
            let mut phase_vector = Vec::new().to_real_time_vec();
            assert_eq!(abs_vector.is_complex(), false);
            complex.get_magnitude(&mut abs_vector);
            assert_eq!(abs_vector.len(), abs.len());
            complex.get_phase(&mut phase_vector);
            let phase_result = complex.phase();
            assert_vector_eq_with_reason(&phase, &phase_vector[..], "Failure in get_phase");
            assert_vector_eq_with_reason(&abs, &abs_vector[..], "Failure in get_magnitude");
            assert_vector_eq_with_reason(&phase, &phase_result[..], "Failure in phase()");
            assert_eq!(abs_vector.is_complex(), false);
            assert_eq!(abs_vector.delta(), delta);
            assert_eq!(phase_vector.is_complex(), false);
            assert_eq!(phase_vector.delta(), delta);
            assert_eq!(phase_result.is_complex(), false);
            assert_eq!(phase_result.delta(), delta);
        });
    }

    fn complex_vector_diff(a: &Vec<f32>) -> Vec<f32>
    {
        let a = to_complex(a);
        let mut result = vec![Complex32::new(0.0, 0.0); a.len()];
        result[0] = a[0];
        for i in 1 .. a.len() {
            result[i] = a[i] - a[i - 1];
        }

        from_complex(&result)
    }

    #[test]
    fn complex_diff_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_complex_time_vec();
            vector.set_delta(delta);
            let expected = complex_vector_diff(&a);
            vector.diff_with_start();
            assert_vector_eq(&expected, &vector[..]);
            assert_eq!(vector.is_complex(), true);
            assert_eq!(vector.delta(), delta);
        });
    }

    fn complex_vector_cum_sum(a: &Vec<f32>) -> Vec<f32>
    {
        let a = to_complex(a);
        let mut result = vec![Complex32::new(0.0, 0.0); a.len()];
        result[0] = a[0];
        for i in 1 .. a.len() {
            result[i] = a[i] + result[i - 1];
        }

        from_complex(&result)
    }

    #[test]
    fn complex_cum_sum_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_complex_time_vec();
            vector.set_delta(delta);
            let expected = complex_vector_cum_sum(&a);
            vector.cum_sum();
            assert_vector_eq(&expected, &vector[..]);
            assert_eq!(vector.is_complex(), true);
            assert_eq!(vector.delta(), delta);
        });
    }

    fn complex_exponential(vec: &Vec<f32>, a: f32, b: f32, delta: f32) -> Vec<f32>
    {
        let a = a * delta;
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
            let mut vector = a.clone().to_complex_time_vec();
            vector.set_delta(delta);
            let expected = complex_exponential(&a, args[0], args[1], delta);
            vector.multiply_complex_exponential(args[0], args[1]);
            assert_vector_eq_with_reason_and_tolerance(&expected, &vector[..], 1e-2, "");
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
            let mut vector = a.clone().to_complex_time_vec();
            vector.set_delta(delta);
            let sum = c.iter().fold(Complex32::new(0.0, 0.0), |a,b| a + b);
            let sum_sq = c.iter().map(|v| v * v).fold(Complex32::new(0.0, 0.0), |a,b| a + b);
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
            let mut vector = a.clone().to_complex_time_vec();
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
    fn split_merge_test32() {
        let a = create_data(201511210, 0, 1000, 1000);
        let vector = a.clone().to_complex_time_vec();
        let empty: Vec<f32> = Vec::new();
        let mut split =
            [
                Box::new(empty.clone().to_complex_time_vec()),
                Box::new(empty.clone().clone().to_complex_time_vec()),
                Box::new(empty.clone().to_complex_time_vec()),
                Box::new(empty.clone().to_complex_time_vec()),
                Box::new(empty.clone().to_complex_time_vec())];
        vector.split_into(&mut split).unwrap();
        let mut merge = empty.to_complex_time_vec();
        merge.merge(&split).unwrap();
        assert_vector_eq(&a, &merge[..]);
    }

    #[test]
    fn split_test32() {
        let a = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let vector = a.to_complex_time_vec();
        let empty: Vec<f32> = Vec::new();
        let mut split =
            [
                Box::new(empty.clone().to_complex_time_vec()),
                Box::new(empty.clone().to_complex_time_vec())];
        vector.split_into(&mut split).unwrap();
        assert_vector_eq(&[1.0, 2.0, 5.0, 6.0], &split[0][..]);
        assert_vector_eq(&[3.0, 4.0, 7.0, 8.0], &split[1][..]);
    }

    #[test]
    fn to_real_imag_and_back32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_complex_time_vec();
            vector.set_delta(delta);
            let mut real = Vec::new().to_real_time_vec();
            let mut imag = Vec::new().to_real_time_vec();
            vector.get_real_imag(&mut real, &mut imag);
            let mut vector2 = vec!(0.0; 0).to_complex_time_vec();
            vector2.set_real_imag(&real, &imag).unwrap();
            assert_vector_eq(&a, &vector2[..]);
        });
    }

    #[test]
    fn to_mag_phase_and_back32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_complex_time_vec();
            vector.set_delta(delta);
            let mut mag = Vec::new().to_real_time_vec();
            let mut phase = Vec::new().to_real_time_vec();
            let mut mag2 = Vec::new().to_real_time_vec();
            let mut phase2 = Vec::new().to_real_time_vec();
            vector.get_mag_phase(&mut mag, &mut phase);
            vector.get_magnitude(&mut mag2);
            vector.get_phase(&mut phase2);

            assert_vector_eq_with_reason(&mag[..], &mag2[..], "Magnitude differs");
            assert_vector_eq_with_reason(&phase[..], &phase2[..], "Phase differs");

            let mut vector2 = vec!(0.0; 0).to_complex_time_vec();
            vector2.set_mag_phase(&mag, &phase).unwrap();
            assert_vector_eq_with_reason_and_tolerance(&a, &vector2[..], 1e-4, "Merge differs");
        });
    }

    #[test]
    fn multi_ops_offset_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 2);
            let a = a.to_complex_time_vec();
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.offset(Complex32::new(b[0], b[1]));
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let mut a_expected = a;
            a_expected.offset(Complex32::new(b[0], b[1]));
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_scale_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 2);
            let a = a.to_complex_time_vec();
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.scale(Complex32::new(b[0], b[1]));
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let mut a_expected = a;
            a_expected.scale(Complex32::new(b[0], b[1]));
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_magnitude_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = a.to_complex_time_vec();
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                a.magnitude()
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let a_expected = a.magnitude();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_magnitude_squared_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = a.to_complex_time_vec();
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                a.magnitude_squared()
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let a_expected = a.magnitude_squared();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_conj_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = a.to_complex_time_vec();
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.conj();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let mut a_expected = a;
            a_expected.conj();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_to_real_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = a.to_complex_time_vec();
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                a.to_real()
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let a_expected = a.to_real();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_to_imag_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = a.to_complex_time_vec();
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                a.to_imag()
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let a_expected = a.to_imag();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_phase_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = a.to_complex_time_vec();
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                a.phase()
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let a_expected = a.phase();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_add_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let b = b.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops2(a.clone(), b.clone());
            let ops = ops.add_ops(|mut a, b| {
                a.add(&b).unwrap();
                (a, b)
            });
            let mut buffer = SingleBuffer::new();
            let (a_actual, b_actual) = ops.get(&mut buffer).unwrap();
            a.add(&b).unwrap();
            let a_expected = a;
            let b_expected = b;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
            assert_vector_eq(&b_expected[..], &b_actual[..]);
        });
    }

    #[test]
    fn multi_ops_sub_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let b = b.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops2(a.clone(), b.clone());
            let ops = ops.add_ops(|mut a, b| {
                a.sub(&b).unwrap();
                (a, b)
            });
            let mut buffer = SingleBuffer::new();
            let (a_actual, b_actual) = ops.get(&mut buffer).unwrap();
            a.sub(&b).unwrap();
            let a_expected = a;
            let b_expected = b;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
            assert_vector_eq(&b_expected[..], &b_actual[..]);
        });
    }

    #[test]
    fn multi_ops_mul_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let b = b.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops2(a.clone(), b.clone());
            let ops = ops.add_ops(|mut a, b| {
                a.mul(&b).unwrap();
                (a, b)
            });
            let mut buffer = SingleBuffer::new();
            let (a_actual, b_actual) = ops.get(&mut buffer).unwrap();
            a.mul(&b).unwrap();
            let a_expected = a;
            let b_expected = b;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
            assert_vector_eq(&b_expected[..], &b_actual[..]);
        });
    }

    #[test]
    fn multi_ops_div_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let b = b.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops2(a.clone(), b.clone());
            let ops = ops.add_ops(|mut a, b| {
                a.div(&b).unwrap();
                (a, b)
            });
            let mut buffer = SingleBuffer::new();
            let (a_actual, b_actual) = ops.get(&mut buffer).unwrap();
            a.div(&b).unwrap();
            let a_expected = a;
            let b_expected = b;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
            assert_vector_eq(&b_expected[..], &b_actual[..]);;
        });
    }

    #[test]
    fn multi_ops_sqrt_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511142, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.sqrt();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.sqrt();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_square_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.square();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.square();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_root_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 1);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.root(b[0]);
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.root(b[0]);
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_powf_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 1);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.powf(b[0]);
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.powf(b[0]);
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_ln_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.ln();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.ln();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_exp_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.exp();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.exp();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_log_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 1);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.log(b[0]);
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.log(b[0]);
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_expf_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 1);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.expf(b[0]);
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.expf(b[0]);
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_sin_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.sin();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.sin();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_cos_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.cos();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.cos();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_tan_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.tan();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.tan();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_asin_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.asin();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.asin();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_acos_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.acos();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.acos();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_atan_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.atan();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.atan();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_sinh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.sinh();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.sinh();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_cosh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.cosh();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.cosh();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_tanh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.tanh();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.tanh();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_asinh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.asinh();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.asinh();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_acosh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.acosh();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.acosh();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_atanh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.atanh();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.atanh();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_complex_exponential_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1030;
            let a = create_data_with_len(201511141, iteration, len);
            let args = create_data_with_len(201511141, iteration, 2);
            let a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.multiply_complex_exponential(args[0], args[1]);
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let mut a_expected = a;
            a_expected.multiply_complex_exponential(args[0], args[1]);
            assert_vector_eq_with_reason_and_tolerance(&a_expected[..], &a_actual[..], 1e-2, "");
        });
    }
}
