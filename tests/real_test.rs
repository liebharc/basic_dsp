extern crate basic_dsp;
extern crate num;
extern crate rand;
pub mod tools;

mod real_test {
    use basic_dsp::*;
    use tools::*;

    #[allow(dead_code)]
    fn real_add(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        let mut result = vec![0.0; a.len()];
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }

        result
    }

    fn real_add_scalar(a: &Vec<f32>, value: f32) -> Vec<f32> {
        let mut result = vec![0.0; a.len()];
        for i in 0..a.len() {
            result[i] = a[i] + value;
        }

        result
    }

    #[test]
    fn add_real_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511141, iteration, range.start, range.end);
            let scalar = create_data_with_len(201511142, iteration, 1);
            let expected = real_add_scalar(&a, scalar[0]);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.to_real_time_vec();
            vector.set_delta(delta);
            vector.offset(scalar[0]);
            assert_vector_eq(&expected, &vector[..]);
            assert_eq!(vector.is_complex(), false);
            assert_eq!(vector.delta(), delta);
        });
    }

    fn real_mulitply_scalar(a: &Vec<f32>, value: f32) -> Vec<f32> {
        let mut result = vec![0.0; a.len()];
        for i in 0..a.len() {
            result[i] = a[i] * value;
        }

        result
    }

    #[test]
    fn multiply_real_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511143, iteration, range.start, range.end);
            let scalar = create_data_with_len(201511142, iteration, 1);
            let expected = real_mulitply_scalar(&a, scalar[0]);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.to_real_time_vec();
            vector.set_delta(delta);
            vector.scale(scalar[0]);
            assert_vector_eq(&expected, &vector[..]);
            assert_eq!(vector.is_complex(), false);
            assert_eq!(vector.delta(), delta);
        });
    }

    fn real_abs(a: &Vec<f32>) -> Vec<f32> {
        let mut result = vec![0.0; a.len()];
        for i in 0..a.len() {
            result[i] = a[i].abs();
        }

        result
    }

    #[test]
    fn abs_real_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511146, iteration, range.start, range.end);
            let expected = real_abs(&a);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.to_real_time_vec();
            vector.set_delta(delta);
            vector.abs();
            assert_vector_eq(&expected, &vector[..]);
            assert_eq!(vector.is_complex(), false);
            assert_eq!(vector.delta(), delta);
        });
    }

    fn real_add_vector(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        let mut result = vec![0.0; a.len()];
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }

        result
    }

    #[test]
    fn real_add_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511171, iteration, range.start, range.end);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = real_add_vector(&a, &b);
            let delta = create_delta(3561159, iteration);
            let mut vector1 = a.to_real_time_vec();
            vector1.set_delta(delta);
            let mut vector2 = b.to_real_time_vec();
            vector2.set_delta(delta);
            vector1.add(&vector2).unwrap();
            assert_vector_eq(&expected, &vector1[..]);
            assert_eq!(vector1.is_complex(), false);
            assert_eq!(vector1.delta(), delta);
        });
    }

    fn real_add_vector_mod(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        let mut result = vec![0.0; a.len()];
        for i in 0..a.len() {
            result[i] = a[i] + b[i % b.len()];
        }

        result
    }

    #[test]
    fn real_add_smaller_vector32() {
        let a = create_data_with_len(201511171, 1, 99);
        let b = create_data_with_len(201511172, 1, 9);
        let expected = real_add_vector_mod(&a, &b);
        let delta = create_delta(3561159, 1);
        let mut vector1 = a.to_real_time_vec();
        vector1.set_delta(delta);
        let mut vector2 = b.to_real_time_vec();
        vector2.set_delta(delta);
        vector1.add_smaller(&vector2).unwrap();
        assert_vector_eq(&expected, &vector1[..]);
        assert_eq!(vector1.is_complex(), false);
        assert_eq!(vector1.delta(), delta);
    }

    fn real_sub_vector(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        let mut result = vec![0.0; a.len()];
        for i in 0..a.len() {
            result[i] = a[i] - b[i];
        }

        result
    }

    #[test]
    fn real_sub_vector_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511171, iteration, range.start, range.end);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = real_sub_vector(&a, &b);
            let delta = create_delta(3561159, iteration);
            let mut vector1 = a.to_real_time_vec();
            vector1.set_delta(delta);
            let mut vector2 = b.to_real_time_vec();
            vector2.set_delta(delta);
            vector1.sub(&vector2).unwrap();
            assert_vector_eq(&expected, &vector1[..]);
            assert_eq!(vector1.is_complex(), false);
            assert_eq!(vector1.delta(), delta);
        });
    }

    fn real_sub_vector_mod(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        let mut result = vec![0.0; a.len()];
        for i in 0..a.len() {
            result[i] = a[i] - b[i % b.len()];
        }

        result
    }

    #[test]
    fn real_sub_smaller_vector_vector32() {
        let a = create_data_with_len(201511171, 1, 99);
        let b = create_data_with_len(201511172, 1, 9);
        let expected = real_sub_vector_mod(&a, &b);
        let delta = create_delta(3561159, 1);
        let mut vector1 = a.to_real_time_vec();
        vector1.set_delta(delta);
        let mut vector2 = b.to_real_time_vec();
        vector2.set_delta(delta);
        vector1.sub_smaller(&vector2).unwrap();
        assert_vector_eq(&expected, &vector1[..]);
        assert_eq!(vector1.is_complex(), false);
        assert_eq!(vector1.delta(), delta);
    }

    fn real_vector_mul(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        let mut result = vec![0.0; a.len()];
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }

        result
    }

    #[test]
    fn real_mul_vector_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511171, iteration, range.start, range.end);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = real_vector_mul(&a, &b);
            let delta = create_delta(3561159, iteration);
            let mut vector1 = a.to_real_time_vec();
            vector1.set_delta(delta);
            let mut vector2 = b.to_real_time_vec();
            vector2.set_delta(delta);
            vector1.mul(&vector2).unwrap();
            assert_vector_eq(&expected, &vector1[..]);
            assert_eq!(vector1.is_complex(), false);
            assert_eq!(vector1.delta(), delta);
        });
    }

    fn real_mul_vector_mod(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        let mut result = vec![0.0; a.len()];
        for i in 0..a.len() {
            result[i] = a[i] * b[i % b.len()];
        }

        result
    }

    #[test]
    fn real_mul_smaller_vector_vector32() {
        let a = create_data_with_len(201511171, 1, 99);
        let b = create_data_with_len(201511172, 1, 9);
        let expected = real_mul_vector_mod(&a, &b);
        let delta = create_delta(3561159, 1);
        let mut vector1 = a.to_real_time_vec();
        vector1.set_delta(delta);
        let mut vector2 = b.to_real_time_vec();
        vector2.set_delta(delta);
        vector1.mul_smaller(&vector2).unwrap();
        assert_vector_eq(&expected, &vector1[..]);
        assert_eq!(vector1.is_complex(), false);
        assert_eq!(vector1.delta(), delta);
    }

    fn real_div_vector_mod(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        let mut result = vec![0.0; a.len()];
        for i in 0..a.len() {
            result[i] = a[i] / b[i % b.len()];
        }

        result
    }

    #[test]
    fn real_div_smaller_vector_vector32() {
        let a = create_data_with_len(201511171, 1, 99);
        let b = create_data_with_len(201511172, 1, 9);
        let expected = real_div_vector_mod(&a, &b);
        let delta = create_delta(3561159, 1);
        let mut vector1 = a.to_real_time_vec();
        vector1.set_delta(delta);
        let mut vector2 = b.to_real_time_vec();
        vector2.set_delta(delta);
        vector1.div_smaller(&vector2).unwrap();
        assert_vector_eq(&expected, &vector1[..]);
        assert_eq!(vector1.is_complex(), false);
        assert_eq!(vector1.delta(), delta);
    }

    #[test]
    fn real_dot_product32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511171, iteration, range.start, range.end);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = real_vector_mul(&a, &b).iter().fold(0.0, |a, b| a + b);
            let delta = create_delta(3561159, iteration);
            let mut vector1 = a.to_real_time_vec();
            vector1.set_delta(delta);
            let mut vector2 = b.to_real_time_vec();
            vector2.set_delta(delta);
            let result = vector1.dot_product(&vector2).unwrap();
            assert_in_tolerance(expected, result, 0.5);
        });
    }

    #[test]
    fn real_dot_product_prec32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511171, iteration, range.start, range.end);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = real_vector_mul(&a, &b)
                .iter()
                .map(|v| *v as f64)
                .fold(0.0, |a, b| a + b);
            let delta = create_delta(3561159, iteration);
            let mut vector1 = a.to_real_time_vec();
            vector1.set_delta(delta);
            let mut vector2 = b.to_real_time_vec();
            vector2.set_delta(delta);
            let result = vector1.dot_product_prec(&vector2).unwrap();
            assert_in_tolerance(expected as f32, result, 1e-2);
        });
    }

    fn real_vector_div(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        let mut result = vec![0.0; a.len()];
        for i in 0..a.len() {
            result[i] = a[i] / b[i];
        }

        result
    }

    #[test]
    fn real_div_vector_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511171, iteration, range.start, range.end);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = real_vector_div(&a, &b);
            let delta = create_delta(3561159, iteration);
            let mut vector1 = a.to_real_time_vec();
            vector1.set_delta(delta);
            let mut vector2 = b.to_real_time_vec();
            vector2.set_delta(delta);
            vector1.div(&vector2).unwrap();
            assert_vector_eq(&expected, &vector1[..]);
            assert_eq!(vector1.is_complex(), false);
            assert_eq!(vector1.delta(), delta);
        });
    }

    #[test]
    fn real_square_sqrt_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a =
                create_data_even_in_range(201511210, iteration, range.start, range.end, 0.0, 10.0);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_real_time_vec();
            vector.set_delta(delta);
            vector.square();
            vector.sqrt();
            assert_vector_eq(&a, &vector[..]);
            assert_eq!(vector.is_complex(), false);
            assert_eq!(vector.delta(), delta);
        });
    }

    #[test]
    fn real_expn_logn_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_real_time_vec();
            vector.set_delta(delta);
            vector.exp();
            vector.ln();
            assert_vector_eq(&a, &vector[..]);
            assert_eq!(vector.is_complex(), false);
            assert_eq!(vector.delta(), delta);
        });
    }

    #[test]
    fn real_exp_log_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511210, iteration, range.start, range.end);
            let base = create_data_even_in_range(201511213, iteration, 1, 2, 0.1, 20.0);
            let base = base[0];
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_real_time_vec();
            vector.set_delta(delta);
            vector.expf(base);
            vector.log(base);
            assert_vector_eq(&a, &vector[..]);
            assert_eq!(vector.is_complex(), false);
            assert_eq!(vector.delta(), delta);
        });
    }

    fn real_vector_diff(a: &Vec<f32>) -> Vec<f32> {
        let mut result = vec![0.0; a.len()];
        result[0] = a[0];
        for i in 1..a.len() {
            result[i] = a[i] - a[i - 1];
        }

        result
    }

    #[test]
    fn real_diff_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_real_time_vec();
            let expected = real_vector_diff(&a);
            vector.set_delta(delta);
            vector.diff_with_start();
            assert_vector_eq(&expected, &vector[..]);
            assert_eq!(vector.is_complex(), false);
            assert_eq!(vector.delta(), delta);
        });
    }

    fn real_vector_cum_sum(a: &Vec<f32>) -> Vec<f32> {
        let mut result = vec![0.0; a.len()];
        result[0] = a[0];
        for i in 1..a.len() {
            result[i] = a[i] + result[i - 1];
        }

        result
    }

    #[test]
    fn real_cum_sum_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_real_time_vec();
            let expected = real_vector_cum_sum(&a);
            vector.set_delta(delta);
            vector.cum_sum();
            assert_vector_eq(&expected, &vector[..]);
            assert_eq!(vector.is_complex(), false);
            assert_eq!(vector.delta(), delta);
        });
    }

    #[test]
    fn real_wrap_unwrap_vector32_positive_large() {
        let a = vec![1.0; RANGE_MULTI_CORE.end];
        let linear_seq = real_vector_cum_sum(&a);
        let delta = 0.1;
        let mut vector = linear_seq.clone().to_real_time_vec();
        vector.set_delta(delta);
        vector.wrap(8.0);
        vector.unwrap(8.0);
        assert_vector_eq(&linear_seq, &vector[..]);
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.delta(), delta);
    }

    #[test]
    fn real_wrap_unwrap_vector32_negative_large() {
        let a = vec![-1.0; RANGE_MULTI_CORE.end];
        let linear_seq = real_vector_cum_sum(&a);
        let delta = 0.1;
        let mut vector = linear_seq.clone().to_real_time_vec();
        vector.set_delta(delta);
        vector.wrap(8.0);
        vector.unwrap(8.0);
        assert_vector_eq(&linear_seq, &vector[..]);
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.delta(), delta);
    }

    #[test]
    fn statistics_test32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_real_time_vec();
            vector.set_delta(delta);
            let sum: f32 = a.iter().fold(0.0, |a, b| a + b);
            let sum_sq: f32 = a.iter().map(|v| v * v).fold(0.0, |a, b| a + b);
            let rms = (sum_sq / a.len() as f32).sqrt();
            let result = vector.statistics();
            assert_eq!(result.sum, sum);
            assert_eq!(result.rms, rms);
        });
    }

    #[test]
    fn statistics_vs_sum_test32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_real_time_vec();
            vector.set_delta(delta);
            let sum: f32 = vector.sum();
            let sum_sq: f32 = vector.sum_sq();
            let rms = (sum_sq / a.len() as f32).sqrt();
            let result = vector.statistics();
            assert_in_tolerance(result.sum, sum, 1e-1);
            assert_in_tolerance(result.rms, rms, 1e-1);
        });
    }

    #[test]
    fn statistics_prec_test32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_real_time_vec();
            vector.set_delta(delta);
            let sum: f64 = a.iter().map(|v| *v as f64).fold(0.0, |a, b| a + b);
            let sum_sq: f64 = a
                .iter()
                .map(|v| *v as f64)
                .map(|v| v * v)
                .fold(0.0, |a, b| a + b);
            let rms = (sum_sq / a.len() as f64).sqrt();
            let result = vector.statistics_prec();
            assert_eq!(result.sum as f32, sum as f32);
            assert_eq!(result.rms as f32, rms as f32);
        });
    }

    #[test]
    fn statistics_prec_vs_sum_prec_test32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_real_time_vec();
            vector.set_delta(delta);
            let sum: f64 = vector.sum_prec();
            let sum_sq: f64 = vector.sum_sq_prec();
            let rms = (sum_sq / a.len() as f64).sqrt();
            let result = vector.statistics_prec();
            assert_in_tolerance(result.sum as f32, sum as f32, 1e-2);
            assert_in_tolerance(result.rms as f32, rms as f32, 1e-2);
        });
    }

    #[test]
    fn statistics_prec_test64() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data64(201511210, iteration, range.start, range.end);
            let vector = a.clone().to_real_time_vec();
            let sum: f64 = a.iter().fold(0.0, |a, b| a + b);
            let sum_sq: f64 = a.iter().map(|v| v * v).fold(0.0, |a, b| a + b);
            let rms = (sum_sq / a.len() as f64).sqrt();
            let result = vector.statistics_prec();
            assert_eq!(result.sum as f32, sum as f32);
            assert_eq!(result.rms as f32, rms as f32);
        });
    }

    #[test]
    fn statistics_prec_vs_sum_prec_test64() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data64(201511210, iteration, range.start, range.end);
            let vector = a.clone().to_real_time_vec();
            let sum: f64 = vector.sum_prec();
            let sum_sq: f64 = vector.sum_sq_prec();
            let rms = (sum_sq / a.len() as f64).sqrt();
            let result = vector.statistics_prec();
            assert_in_tolerance(result.sum as f32, sum as f32, 1e-2);
            assert_in_tolerance(result.rms as f32, rms as f32, 1e-2);
        });
    }

    #[test]
    fn split_merge_test32() {
        let a = create_data(201511210, 0, 1000, 1000);
        let vector = a.clone().to_real_time_vec();
        let empty: Vec<f32> = Vec::new();
        let mut split = [
            Box::new(empty.clone().to_real_time_vec()),
            Box::new(empty.clone().to_real_time_vec()),
            Box::new(empty.clone().to_real_time_vec()),
            Box::new(empty.clone().to_real_time_vec()),
            Box::new(empty.clone().to_real_time_vec()),
        ];
        {
            let mut dest: Vec<_> = split.iter_mut().map(|x| x.as_mut()).collect();
            vector.split_into(&mut dest[..]).unwrap();
        }
        let mut merge = empty.to_real_time_vec();
        let src: Vec<_> = split.iter().map(|x| x.as_ref()).collect();
        merge.merge(&src[..]).unwrap();
        assert_vector_eq(&a, &merge[..]);
    }

    #[test]
    fn real_fft_test32() {
        let data = create_data(201511210, 0, 1001, 1001);
        let mut buffer = SingleBuffer::new();
        let time = data.to_real_time_vec();
        let sym_fft: ComplexFreqVec32 = time.clone().plain_sfft(&mut buffer).unwrap();
        let complex_time = time.clone().to_complex_b(&mut buffer);
        let complex_freq: ComplexFreqVec32 = complex_time.plain_fft(&mut buffer);
        let mut real_mirror = sym_fft.clone();
        real_mirror.mirror(&mut buffer);
        assert_vector_eq_with_reason_and_tolerance(
            &complex_freq[..],
            &real_mirror[..],
            1e-3,
            "Different FFT paths must equal",
        );
        let mut real_ifft: RealTimeVec32 = sym_fft.plain_sifft(&mut buffer).unwrap();
        real_ifft.scale(1.0 / 1001.0);
        assert_vector_eq_with_reason_and_tolerance(
            &time[..],
            &real_ifft[..],
            1e-3,
            "Ifft must give back the original result",
        );
    }
}
