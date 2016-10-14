extern crate basic_dsp;
extern crate rand;
extern crate num;
pub mod tools;

mod slow_test {
    use basic_dsp::vector_types2::*;
    use basic_dsp::vector_types2::combined_ops::*;
    use tools::*;

    #[allow(dead_code)]
    fn real_add(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32>
    {
        let mut result = vec![0.0; a.len()];
        for i in 0 .. a.len() {
            result[i] = a[i] + b[i];
        }

        result
    }

    fn real_add_scalar(a: &Vec<f32>, value: f32) -> Vec<f32>
    {
        let mut result = vec![0.0; a.len()];
        for i in 0 .. a.len() {
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
            let mut vector = a.to_real_time_vec();;
            vector.set_delta(delta);
            vector.offset(scalar[0]);
            assert_vector_eq(&expected, &vector[..]);
            assert_eq!(vector.is_complex(), false);
            assert_eq!(vector.delta(), delta);
        });
    }

    #[test]
    fn multi_ops1_vector32() {
        parameterized_vector_test(|iteration, _| {
            let a = create_data_with_len(201511141, iteration, 500008);
            let mut buffer = SingleBuffer::new();
            let mut vector = a.to_gen_dsp_vec(false, DataDomain::Time);
            let ops = multi_ops1(vector.clone());
            let ops = ops.add_ops(|mut a| {
                a.log(10.0);
                a.scale(10.0);
                a
            });
            let result = ops.get(&mut buffer).unwrap();
            vector.log(10.0);
            vector.scale(10.0);
            assert_vector_eq(&vector[..], &result[..]);
        });
    }

    #[test]
    fn multi_ops2_vector32() {
        parameterized_vector_test(|iteration, _| {
            let len = 10;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 2 * len);
            let mut buffer = SingleBuffer::new();
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
            let b = b.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops2(a.clone(), b.clone());
            let ops = ops.add_ops(|mut r, c| {
                r.sin();
                let c = c.magnitude();
                r.add(&c).unwrap();
                (r, c)
            });
            let (a_actual, b_actual) = ops.get(&mut buffer).unwrap();
            let b = b.magnitude();
            a.sin();
            a.add(&b).unwrap();
            assert_vector_eq_with_reason(&b[..], &b_actual[..], "Complex vec");
            assert_vector_eq_with_reason(&a[..], &a_actual[..], "Real vec");
        });
    }

    #[test]
    fn multi_ops1_extend_vector32() {
        parameterized_vector_test(|iteration, _| {
            let len = 10;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 2 * len);
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
            let mut b = b.to_gen_dsp_vec(true, DataDomain::Time);
            let mut buffer = SingleBuffer::new();
            let ops = multi_ops1(b.clone());
            let ops = ops.add_ops(|mut c| {
                c.conj();
                c.to_imag()
            });
            let ops = ops.extend(a.clone());
            let ops = ops.add_ops(|c, mut r| {
                r.mul(&c).unwrap();
                (c, r)
            });
            let (b_actual, a_actual) = ops.get(&mut buffer).unwrap();
            b.conj();
            let b = b.to_imag();
            a.mul(&b).unwrap();
            assert_vector_eq(&a[..], &a_actual[..]);
            assert_vector_eq(&b[..], &b_actual[..]);
        });
    }

    fn real_mulitply_scalar(a: &Vec<f32>, value: f32) -> Vec<f32>
    {
        let mut result = vec![0.0; a.len()];
        for i in 0 .. a.len() {
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

    fn real_abs(a: &Vec<f32>) -> Vec<f32>
    {
        let mut result = vec![0.0; a.len()];
        for i in 0 .. a.len() {
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

    fn real_add_vector(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32>
    {
        let mut result = vec![0.0; a.len()];
        for i in 0 .. a.len() {
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

    fn real_add_vector_mod(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32>
    {
        let mut result = vec![0.0; a.len()];
        for i in 0 .. a.len() {
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

    fn real_sub_vector(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32>
    {
        let mut result = vec![0.0; a.len()];
        for i in 0 .. a.len() {
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

    fn real_sub_vector_mod(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32>
    {
        let mut result = vec![0.0; a.len()];
        for i in 0 .. a.len() {
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

    fn real_vector_mul(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32>
    {
        let mut result = vec![0.0; a.len()];
        for i in 0 .. a.len() {
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

    fn real_mul_vector_mod(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32>
    {
        let mut result = vec![0.0; a.len()];
        for i in 0 .. a.len() {
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

    fn real_div_vector_mod(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32>
    {
        let mut result = vec![0.0; a.len()];
        for i in 0 .. a.len() {
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

    fn real_vector_div(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32>
    {
        let mut result = vec![0.0; a.len()];
        for i in 0 .. a.len() {
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
            let a = create_data_even_in_range(201511210, iteration, range.start, range.end, 0.0, 10.0);
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

    fn real_vector_diff(a: &Vec<f32>) -> Vec<f32>
    {
        let mut result = vec![0.0; a.len()];
        result[0] = a[0];
        for i in 1 .. a.len() {
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

    fn real_vector_cum_sum(a: &Vec<f32>) -> Vec<f32>
    {
        let mut result = vec![0.0; a.len()];
        result[0] = a[0];
        for i in 1 .. a.len() {
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
            let sum: f32 = a.iter().fold(0.0, |a,b| a + b);
            let sum_sq: f32 = a.iter().map(|v| v * v).fold(0.0, |a,b| a + b);
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
    fn split_merge_test32() {
        let a = create_data(201511210, 0, 1000, 1000);
        let vector = a.clone().to_real_time_vec();
        let empty: Vec<f32> = Vec::new();
        let mut split =
            [
                Box::new(empty.clone().to_real_time_vec()),
                Box::new(empty.clone().to_real_time_vec()),
                Box::new(empty.clone().to_real_time_vec()),
                Box::new(empty.clone().to_real_time_vec()),
                Box::new(empty.clone().to_real_time_vec())];
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
        assert_vector_eq_with_reason_and_tolerance(&complex_freq[..], &real_mirror[..], 1e-3, "Different FFT paths must equal");
        let mut real_ifft: RealTimeVec32 = sym_fft.plain_sifft(&mut buffer).unwrap();
        real_ifft.scale(1.0 / 1001.0);
        assert_vector_eq_with_reason_and_tolerance(&time[..], &real_ifft[..], 1e-3, "Ifft must give back the original result");
    }

    #[test]
    fn multi_ops_noop_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, len);
            let a = a.to_gen_dsp_vec(false, DataDomain::Time);
            let b = b.to_gen_dsp_vec(false, DataDomain::Time);
            let mut buffer = SingleBuffer::new();
            let ops = multi_ops2(a.clone(), b.clone());
            let ops = ops.add_ops(|a, b| {
                (a, b)
            });
            let (a_actual, b_actual) = ops.get(&mut buffer).unwrap();
            let b_expected = b;
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
            assert_vector_eq(&b_expected[..], &b_actual[..]);
        });
    }

    #[test]
    fn multi_ops_offset_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1031;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 1);
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.offset(b[0]);
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.offset(b[0]);
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_scale_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 1);
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.scale(b[0]);
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.scale(b[0]);
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_abs_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.abs();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.abs();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_to_complex_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = a.to_gen_dsp_vec(false, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                a.to_complex().unwrap()
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let a_expected = a.to_complex().unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_add_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
            let b = b.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
            let b = b.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
            let b = b.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
            let b = b.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
            let mut a = a.to_gen_dsp_vec(false, DataDomain::Time);
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
}
