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
        let mut split =
            [
                Box::new(Vec::new().to_real_time_vec()),
                Box::new(Vec::new().to_real_time_vec()),
                Box::new(Vec::new().to_real_time_vec()),
                Box::new(Vec::new().to_real_time_vec()),
                Box::new(Vec::new().to_real_time_vec())];
        vector.split_into(&mut split).unwrap();
        let mut merge = Vec::new().to_real_time_vec();
        merge.merge(&split).unwrap();
        assert_vector_eq(&a, &merge[..]);
    }
/*
    #[test]
    fn real_fft_test32() {
        let data = create_data(201511210, 0, 1001, 1001);
        let time = data.to_real_time_vec();
        let sym_fft = time.clone().plain_sfft().unwrap();
        let complex_time = time.clone().to_complex().unwrap();
        let complex_freq = complex_time.plain_fft().unwrap();
        let real_mirror = sym_fft.clone().mirror().unwrap();
        assert_vector_eq_with_reason_and_tolerance(&complex_freq[..], &real_mirror[..], 1e-3, "Different FFT paths must equal");
        let real_ifft = sym_fft.plain_sifft().unwrap()
                                .real_scale(1.0 / 1001.0).unwrap();
        assert_vector_eq_with_reason_and_tolerance(&time[..], &real_ifft[..], 1e-3, "Ifft must give back the original result");
    }

    #[test]
    fn rededicate_test() {
        // The test case is rather long since it basically
        // just checks all the different combinations. I however
        // sometimes prefer to have explict tests even if they
        // are repetetive.
        let len = 8;
        let data = create_data(20160622, 0, len, len);
        // RealTimeVector32
        let original = RealTimeVector32::from_array(&data);
        original.assert_meta_data();
        assert_eq!(original.len(), len);

        let vector: ComplexTimeVector32 = original.clone().rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector: RealFreqVector32 = original.clone().rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector: ComplexFreqVector32 = original.clone().rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector: DataVec32 = original.clone().rededicate();
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.domain(), DataVecDomain::Time);
        assert_eq!(vector.len(), len);

        // ComplexTimeVector32
        let original = ComplexTimeVector32::from_interleaved(&data);
        original.assert_meta_data();
        assert_eq!(original.len(), len);

        let vector: RealTimeVector32 = original.clone().rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector: RealFreqVector32 = original.clone().rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector: ComplexFreqVector32 = original.clone().rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector: DataVec32 = original.clone().rededicate();
        assert_eq!(vector.is_complex(), true);
        assert_eq!(vector.domain(), DataVecDomain::Time);
        assert_eq!(vector.len(), len);

        // RealFreqVector32
        let original = RealFreqVector32::from_array(&data);
        original.assert_meta_data();
        assert_eq!(original.len(), len);

        let vector: RealTimeVector32 = original.clone().rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector: ComplexTimeVector32 = original.clone().rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector: ComplexFreqVector32 = original.clone().rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector: DataVec32 = original.clone().rededicate();
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.domain(), DataVecDomain::Frequency);
        assert_eq!(vector.len(), len);

        // ComplexFreqVector32
        let original = ComplexFreqVector32::from_interleaved(&data);
        original.assert_meta_data();
        assert_eq!(original.len(), len);

        let vector: RealFreqVector32 = original.clone().rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector: ComplexTimeVector32 = original.clone().rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector: ComplexFreqVector32 = original.clone().rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector: DataVec32 = original.clone().rededicate();
        assert_eq!(vector.is_complex(), true);
        assert_eq!(vector.domain(), DataVecDomain::Frequency);
        assert_eq!(vector.len(), len);
    }

    #[test]
    fn rededicate_from_test() {
        let len = 8;
        let data = create_data(20160622, 0, len, len);
        // RealTimeVector32
        let original = RealTimeVector32::from_array(&data);
        original.assert_meta_data();
        assert_eq!(original.len(), len);

        let vector = ComplexTimeVector32::rededicate_from(original.clone());
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector = RealFreqVector32::rededicate_from(original.clone());
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector = ComplexFreqVector32::rededicate_from(original.clone());
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector = DataVec32::rededicate_from(original.clone());
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.domain(), DataVecDomain::Time);
        assert_eq!(vector.len(), len);

        // ComplexTimeVector32
        let original = ComplexTimeVector32::from_interleaved(&data);
        original.assert_meta_data();
        assert_eq!(original.len(), len);

        let vector = RealTimeVector32::rededicate_from(original.clone());
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector = RealFreqVector32::rededicate_from(original.clone());
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector = ComplexFreqVector32::rededicate_from(original.clone());
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector = DataVec32::rededicate_from(original.clone());
        assert_eq!(vector.is_complex(), true);
        assert_eq!(vector.domain(), DataVecDomain::Time);
        assert_eq!(vector.len(), len);

        // RealFreqVector32
        let original = RealFreqVector32::from_array(&data);
        original.assert_meta_data();
        assert_eq!(original.len(), len);

        let vector = RealTimeVector32::rededicate_from(original.clone());
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector = ComplexTimeVector32::rededicate_from(original.clone());
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector = ComplexFreqVector32::rededicate_from(original.clone());
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector = DataVec32::rededicate_from(original.clone());
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.domain(), DataVecDomain::Frequency);
        assert_eq!(vector.len(), len);

        // ComplexFreqVector32
        let original = ComplexFreqVector32::from_interleaved(&data);
        original.assert_meta_data();
        assert_eq!(original.len(), len);

        let vector = RealFreqVector32::rededicate_from(original.clone());
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector = ComplexTimeVector32::rededicate_from(original.clone());
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector = ComplexFreqVector32::rededicate_from(original.clone());
        vector.assert_meta_data();
        assert_eq!(vector.len(), 0);

        let vector = DataVec32::rededicate_from(original.clone());
        assert_eq!(vector.is_complex(), true);
        assert_eq!(vector.domain(), DataVecDomain::Frequency);
        assert_eq!(vector.len(), len);
    }

    #[test]
    fn rededicate_generic_test() {
        let len = 8;
        let data = create_data(20160622, 0, len, len);
        // RealTimeVector32
        let vector: DataVec32 = RealTimeVector32::from_array(&data).rededicate();
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.domain(), DataVecDomain::Time);
        assert_eq!(vector.len(), len);
        let vector: RealTimeVector32 = vector.rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), len);

        // ComplexTimeVector32
        let vector: DataVec32 = ComplexTimeVector32::from_interleaved(&data).rededicate();
        assert_eq!(vector.is_complex(), true);
        assert_eq!(vector.domain(), DataVecDomain::Time);
        assert_eq!(vector.len(), len);
        let vector: ComplexTimeVector32 = vector.rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), len);

        // RealFreqVector32
        let vector: DataVec32 = RealFreqVector32::from_array(&data).rededicate();
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.domain(), DataVecDomain::Frequency);
        assert_eq!(vector.len(), len);
        let vector: RealFreqVector32 = vector.rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), len);

        // ComplexFreqVector32
        let vector: DataVec32 = ComplexFreqVector32::from_interleaved(&data).rededicate();
        assert_eq!(vector.is_complex(), true);
        assert_eq!(vector.domain(), DataVecDomain::Frequency);
        assert_eq!(vector.len(), len);
        let vector: ComplexFreqVector32 = vector.rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), len);
    }

    #[test]
    fn rededicate_from_generic_test() {
        let len = 8;
        let data = create_data(20160622, 0, len, len);
        // RealTimeVector32
        let vector = DataVec32::rededicate_from(RealTimeVector32::from_array(&data));
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.domain(), DataVecDomain::Time);
        assert_eq!(vector.len(), len);
        let vector = RealTimeVector32::rededicate_from(vector);
        vector.assert_meta_data();
        assert_eq!(vector.len(), len);

        // ComplexTimeVector32
        let vector = DataVec32::rededicate_from(ComplexTimeVector32::from_interleaved(&data));
        assert_eq!(vector.is_complex(), true);
        assert_eq!(vector.domain(), DataVecDomain::Time);
        assert_eq!(vector.len(), len);
        let vector = ComplexTimeVector32::rededicate_from(vector);
        vector.assert_meta_data();
        assert_eq!(vector.len(), len);

        // RealFreqVector32
        let vector = DataVec32::rededicate_from(RealFreqVector32::from_array(&data));
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.domain(), DataVecDomain::Frequency);
        assert_eq!(vector.len(), len);
        let vector = RealFreqVector32::rededicate_from(vector);
        vector.assert_meta_data();
        assert_eq!(vector.len(), len);

        // ComplexFreqVector32
        let vector = DataVec32::rededicate_from(ComplexFreqVector32::from_interleaved(&data));
        assert_eq!(vector.is_complex(), true);
        assert_eq!(vector.domain(), DataVecDomain::Frequency);
        assert_eq!(vector.len(), len);
        let vector = ComplexFreqVector32::rededicate_from(vector);
        vector.assert_meta_data();
        assert_eq!(vector.len(), len);
    }

    #[test]
    fn multi_ops_noop_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let b = DataVec32::from_array(false, DataVecDomain::Time, &b);
            let ops = multi_ops2(a.clone(), b.clone());
            let ops = ops.add_ops(|a, b| {
                (a, b)
            });
            let (a_actual, b_actual) = ops.get().unwrap();
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
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.real_offset(b[0]);
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.real_offset(b[0]))
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_scale_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 1);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.real_scale(b[0]);
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.real_scale(b[0]))
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_abs_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.abs();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.abs())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_to_complex_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.to_complex();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.to_complex())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_add_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let b = DataVec32::from_array(false, DataVecDomain::Time, &b);
            let ops = multi_ops2(a.clone(), b.clone());
            let ops = ops.add_ops(|a, b| {
                let a = a.add(&b);
                (a, b)
            });
            let (a_actual, b_actual) = ops.get().unwrap();
            let a_expected = a.add(&b).unwrap();
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
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let b = DataVec32::from_array(false, DataVecDomain::Time, &b);
            let ops = multi_ops2(a.clone(), b.clone());
            let ops = ops.add_ops(|a, b| {
                let a = a.sub(&b);
                (a, b)
            });
            let (a_actual, b_actual) = ops.get().unwrap();
            let a_expected = a.sub(&b).unwrap();
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
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let b = DataVec32::from_array(false, DataVecDomain::Time, &b);
            let ops = multi_ops2(a.clone(), b.clone());
            let ops = ops.add_ops(|a, b| {
                let a = a.mul(&b);
                (a, b)
            });
            let (a_actual, b_actual) = ops.get().unwrap();
            let a_expected = a.mul(&b).unwrap();
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
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let b = DataVec32::from_array(false, DataVecDomain::Time, &b);
            let ops = multi_ops2(a.clone(), b.clone());
            let ops = ops.add_ops(|a, b| {
                let a = a.div(&b);
                (a, b)
            });
            let (a_actual, b_actual) = ops.get().unwrap();
            let a_expected = a.div(&b).unwrap();
            let b_expected = b;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
            assert_vector_eq(&b_expected[..], &b_actual[..]);
        });
    }

    #[test]
    fn multi_ops_sqrt_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.sqrt();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.sqrt())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_square_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.square();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.square())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_root_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 1);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.root(b[0]);
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.root(b[0]))
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_powf_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 1);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.powf(b[0]);
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.powf(b[0]))
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_ln_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.ln();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.ln())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_exp_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.exp();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.exp())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_log_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 1);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.log(b[0]);
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.log(b[0]))
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_expf_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 1);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.expf(b[0]);
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.expf(b[0]))
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_sin_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.sin();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.sin())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_cos_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.cos();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.cos())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_tan_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.tan();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.tan())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_asin_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.asin();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.asin())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_acos_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.acos();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.acos())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_atan_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.atan();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.atan())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_sinh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.sinh();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.sinh())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_cosh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.cosh();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.cosh())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_tanh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.tanh();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.tanh())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_asinh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.asinh();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.asinh())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_acosh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.acosh();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.acosh())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_atanh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = DataVec32::from_array(false, DataVecDomain::Time, &a);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| {
                let a = a.atanh();
                a
            });
            let a_actual = ops.get().unwrap();
            let a_expected =
                Ok(a)
                .and_then(|a|a.atanh())
                .unwrap();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }*/
}
