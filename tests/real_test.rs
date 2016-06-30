extern crate basic_dsp;
extern crate rand;
extern crate num;
pub mod tools;

#[cfg(feature = "slow_test")]
mod slow_test {
    use basic_dsp::{
        DataVector,
        DataVectorDomain,
        DataVector32,
        RealTimeVector32,
        RealFreqVector32,
        ComplexTimeVector32,
        ComplexFreqVector32,
        TimeDomainOperations,
        SymmetricTimeDomainOperations,
        FrequencyDomainOperations,
        SymmetricFrequencyDomainOperations,
        GenericVectorOps,
        RealVectorOps,
        RededicateVector,
        Operation};
    use basic_dsp::combined_ops::multi_ops1;
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
            let vector = RealTimeVector32::from_array_with_delta(&a, delta);
            let result = vector.real_offset(scalar[0]).unwrap();
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), false);
            assert_eq!(result.delta(), delta);
        });
    }
    
    #[test]
    fn multi_ops_vector32() {
        parameterized_vector_test(|iteration, _| {
            let a = create_data_with_len(201511141, iteration, 500008);
            let vector = DataVector32::from_array(false, DataVectorDomain::Time, &a);
            let mut ops = multi_ops1(vector.clone());
            ops.add_enum_op(Operation::Log(0, 10.0));
            ops.add_enum_op(Operation::MultiplyReal(0, 10.0));
            let result = ops.get().unwrap();
            let expected = vector.log(10.0).and_then(|v| v.real_scale(10.0)).unwrap();
            assert_vector_eq(&expected.data(), &result.data());
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
            let vector = RealTimeVector32::from_array_with_delta(&a, delta);
            let result = vector.real_scale(scalar[0]).unwrap();
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), false);
            assert_eq!(result.delta(), delta);
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
            let vector = RealTimeVector32::from_array_with_delta(&a, delta);
            let result = vector.abs().unwrap();
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), false);
            assert_eq!(result.delta(), delta);
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
    fn real_add_vector_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511171, iteration, range.start, range.end);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = real_add_vector(&a, &b);
            let delta = create_delta(3561159, iteration);
            let vector1 = RealTimeVector32::from_array_with_delta(&a, delta);
            let vector2 = RealTimeVector32::from_array_with_delta(&b, delta);
            let result = vector1.add_vector(&vector2).unwrap();
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), false);
            assert_eq!(result.delta(), delta);
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
    fn real_add_smaller_vector_vector32() {
        let a = create_data_with_len(201511171, 1, 99);
        let b = create_data_with_len(201511172, 1, 9);
        let expected = real_add_vector_mod(&a, &b);
        let delta = create_delta(3561159, 1);
        let vector1 = RealTimeVector32::from_array_with_delta(&a, delta);
        let vector2 = RealTimeVector32::from_array_with_delta(&b, delta);
        let result = vector1.add_smaller_vector(&vector2).unwrap();
        assert_vector_eq(&expected, &result.data());
        assert_eq!(result.is_complex(), false);
        assert_eq!(result.delta(), delta);
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
            let vector1 = RealTimeVector32::from_array_with_delta(&a, delta);
            let vector2 = RealTimeVector32::from_array_with_delta(&b, delta);
            let result = vector1.subtract_vector(&vector2).unwrap();
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), false);
            assert_eq!(result.delta(), delta);
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
        let vector1 = RealTimeVector32::from_array_with_delta(&a, delta);
        let vector2 = RealTimeVector32::from_array_with_delta(&b, delta);
        let result = vector1.subtract_smaller_vector(&vector2).unwrap();
        assert_vector_eq(&expected, &result.data());
        assert_eq!(result.is_complex(), false);
        assert_eq!(result.delta(), delta);
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
            let vector1 = RealTimeVector32::from_array_with_delta(&a, delta);
            let vector2 = RealTimeVector32::from_array_with_delta(&b, delta);
            let result = vector1.multiply_vector(&vector2).unwrap();
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), false);
            assert_eq!(result.delta(), delta);
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
        let vector1 = RealTimeVector32::from_array_with_delta(&a, delta);
        let vector2 = RealTimeVector32::from_array_with_delta(&b, delta);
        let result = vector1.multiply_smaller_vector(&vector2).unwrap();
        assert_vector_eq(&expected, &result.data());
        assert_eq!(result.is_complex(), false);
        assert_eq!(result.delta(), delta);
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
        let vector1 = RealTimeVector32::from_array_with_delta(&a, delta);
        let vector2 = RealTimeVector32::from_array_with_delta(&b, delta);
        let result = vector1.divide_smaller_vector(&vector2).unwrap();
        assert_vector_eq(&expected, &result.data());
        assert_eq!(result.is_complex(), false);
        assert_eq!(result.delta(), delta);
    }
    
    #[test]
    fn real_dot_product32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511171, iteration, range.start, range.end);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = real_vector_mul(&a, &b).iter().fold(0.0, |a, b| a + b);
            let delta = create_delta(3561159, iteration);
            let vector1 = RealTimeVector32::from_array_with_delta(&a, delta);
            let vector2 = RealTimeVector32::from_array_with_delta(&b, delta);
            let result = vector1.real_dot_product(&vector2).unwrap();
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
            let vector1 = RealTimeVector32::from_array_with_delta(&a, delta);
            let vector2 = RealTimeVector32::from_array_with_delta(&b, delta);
            let result = vector1.divide_vector(&vector2).unwrap();
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), false);
            assert_eq!(result.delta(), delta);
        });
    }
    
    #[test]
    fn real_square_sqrt_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even_in_range(201511210, iteration, range.start, range.end, 0.0, 10.0);
            let delta = create_delta(3561159, iteration);
            let vector = RealTimeVector32::from_array_with_delta(&a, delta);
            let result = vector.square().unwrap().sqrt().unwrap();
            assert_vector_eq(&a, &result.data());
            assert_eq!(result.is_complex(), false);
            assert_eq!(result.delta(), delta);
        });
    }
    
    #[test]
    fn real_expn_logn_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let vector = RealTimeVector32::from_array_with_delta(&a, delta);
            let result = vector.exp().unwrap().ln().unwrap();
            assert_vector_eq(&a, &result.data());
            assert_eq!(result.is_complex(), false);
            assert_eq!(result.delta(), delta);
        });
    }
    
    #[test]
    fn real_exp_log_vector32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511210, iteration, range.start, range.end);
            let base = create_data_even_in_range(201511213, iteration, 1, 2, 0.1, 20.0);
            let base = base[0];
            let delta = create_delta(3561159, iteration);
            let vector = RealTimeVector32::from_array_with_delta(&a, delta);
            let result = vector.expf(base).unwrap().log(base).unwrap();
            assert_vector_eq(&a, &result.data());
            assert_eq!(result.is_complex(), false);
            assert_eq!(result.delta(), delta);
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
            let vector = RealTimeVector32::from_array_with_delta(&a, delta);
            let expected = real_vector_diff(&a);
            let result = vector.diff_with_start().unwrap();
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), false);
            assert_eq!(result.delta(), delta);
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
            let vector = RealTimeVector32::from_array_with_delta(&a, delta);
            let expected = real_vector_cum_sum(&a);
            let result = vector.cum_sum().unwrap();
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), false);
            assert_eq!(result.delta(), delta);
        });
    }
     
    #[test]
    fn real_wrap_unwrap_vector32_positive_large() {
        let a = vec![1.0; RANGE_MULTI_CORE.end];
        let linear_seq = real_vector_cum_sum(&a);
        let delta = 0.1;
        let vector = RealTimeVector32::from_array_with_delta(&linear_seq, delta);
        // Be careful because of the two meanings of unwrap depending on the type
        let result = vector.wrap(8.0).unwrap().unwrap(8.0).unwrap();
        assert_vector_eq(&linear_seq, &result.data());
        assert_eq!(result.is_complex(), false);
        assert_eq!(result.delta(), delta);
    }
    
    #[test]
    fn real_wrap_unwrap_vector32_negative_large() {
        let a = vec![-1.0; RANGE_MULTI_CORE.end];
        let linear_seq = real_vector_cum_sum(&a);
        let delta = 0.1;
        let vector = RealTimeVector32::from_array_with_delta(&linear_seq, delta);
        // Be careful because of the two meanings of unwrap depending on the type
        let result = vector.wrap(8.0).unwrap().unwrap(8.0).unwrap();
        assert_vector_eq(&linear_seq, &result.data());
        assert_eq!(result.is_complex(), false);
        assert_eq!(result.delta(), delta);
    }
    
    #[test]
    fn statistics_test32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let vector = RealTimeVector32::from_array_with_delta(&a, delta);
            let sum: f32 = a.iter().fold(0.0, |a,b| a + b);
            let sum_sq: f32 = a.iter().map(|v| v * v).fold(0.0, |a,b| a + b);
            let rms = (sum_sq / a.len() as f32).sqrt();
            let result = vector.real_statistics();
            assert_eq!(result.sum, sum);
            assert_eq!(result.rms, rms);
        });
    }
    
    #[test]
    fn split_merge_test32() {
        let a = create_data(201511210, 0, 1000, 1000);
        let vector = RealTimeVector32::from_array(&a);
        let mut split = 
            [
                Box::new(RealTimeVector32::empty()),
                Box::new(RealTimeVector32::empty()),
                Box::new(RealTimeVector32::empty()),
                Box::new(RealTimeVector32::empty()),
                Box::new(RealTimeVector32::empty())];
        vector.split_into(&mut split).unwrap();
        let merge = RealTimeVector32::empty();
        let result = merge.merge(&split).unwrap();
        assert_vector_eq(&a, &result.data());
    }
    
    #[test]
    fn real_fft_test32() {
        let data = create_data(201511210, 0, 1001, 1001);
        let time = RealTimeVector32::from_array(&data);
        time.assert_meta_data();
        let sym_fft = time.clone().plain_sfft().unwrap();
        sym_fft.assert_meta_data();
        let complex_time = time.clone().to_complex().unwrap();
        complex_time.assert_meta_data();
        let complex_freq = complex_time.plain_fft().unwrap();
        complex_freq.assert_meta_data();
        let real_mirror = sym_fft.clone().mirror().unwrap();
        real_mirror.assert_meta_data();
        assert_vector_eq_with_reason_and_tolerance(&complex_freq.data(), &real_mirror.data(), 1e-3, "Different FFT paths must equal");
        let real_ifft = sym_fft.plain_sifft().unwrap()
                                .real_scale(1.0 / 1001.0).unwrap();
        assert_vector_eq_with_reason_and_tolerance(&time.data(), &real_ifft.data(), 1e-3, "Ifft must give back the original result");
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
        
        let vector: DataVector32 = original.clone().rededicate();
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.domain(), DataVectorDomain::Time);
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
        
        let vector: DataVector32 = original.clone().rededicate();
        assert_eq!(vector.is_complex(), true);
        assert_eq!(vector.domain(), DataVectorDomain::Time);
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
        
        let vector: DataVector32 = original.clone().rededicate();
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.domain(), DataVectorDomain::Frequency);
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
        
        let vector: DataVector32 = original.clone().rededicate();
        assert_eq!(vector.is_complex(), true);
        assert_eq!(vector.domain(), DataVectorDomain::Frequency);
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
        
        let vector = DataVector32::rededicate_from(original.clone());
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.domain(), DataVectorDomain::Time);
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
        
        let vector = DataVector32::rededicate_from(original.clone());
        assert_eq!(vector.is_complex(), true);
        assert_eq!(vector.domain(), DataVectorDomain::Time);
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
        
        let vector = DataVector32::rededicate_from(original.clone());
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.domain(), DataVectorDomain::Frequency);
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
        
        let vector = DataVector32::rededicate_from(original.clone());
        assert_eq!(vector.is_complex(), true);
        assert_eq!(vector.domain(), DataVectorDomain::Frequency);
        assert_eq!(vector.len(), len);
    }
    
    #[test]
    fn rededicate_generic_test() {
        let len = 8;
        let data = create_data(20160622, 0, len, len);
        // RealTimeVector32
        let vector: DataVector32 = RealTimeVector32::from_array(&data).rededicate();
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.domain(), DataVectorDomain::Time);
        assert_eq!(vector.len(), len);
        let vector: RealTimeVector32 = vector.rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), len);
        
        // ComplexTimeVector32
        let vector: DataVector32 = ComplexTimeVector32::from_interleaved(&data).rededicate();
        assert_eq!(vector.is_complex(), true);
        assert_eq!(vector.domain(), DataVectorDomain::Time);
        assert_eq!(vector.len(), len);
        let vector: ComplexTimeVector32 = vector.rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), len);
        
        // RealFreqVector32
        let vector: DataVector32 = RealFreqVector32::from_array(&data).rededicate();
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.domain(), DataVectorDomain::Frequency);
        assert_eq!(vector.len(), len);
        let vector: RealFreqVector32 = vector.rededicate();
        vector.assert_meta_data();
        assert_eq!(vector.len(), len);
        
        // ComplexFreqVector32
        let vector: DataVector32 = ComplexFreqVector32::from_interleaved(&data).rededicate();
        assert_eq!(vector.is_complex(), true);
        assert_eq!(vector.domain(), DataVectorDomain::Frequency);
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
        let vector = DataVector32::rededicate_from(RealTimeVector32::from_array(&data));
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.domain(), DataVectorDomain::Time);
        assert_eq!(vector.len(), len);
        let vector = RealTimeVector32::rededicate_from(vector);
        vector.assert_meta_data();
        assert_eq!(vector.len(), len);
        
        // ComplexTimeVector32
        let vector = DataVector32::rededicate_from(ComplexTimeVector32::from_interleaved(&data));
        assert_eq!(vector.is_complex(), true);
        assert_eq!(vector.domain(), DataVectorDomain::Time);
        assert_eq!(vector.len(), len);
        let vector = ComplexTimeVector32::rededicate_from(vector);
        vector.assert_meta_data();
        assert_eq!(vector.len(), len);
        
        // RealFreqVector32
        let vector = DataVector32::rededicate_from(RealFreqVector32::from_array(&data));
        assert_eq!(vector.is_complex(), false);
        assert_eq!(vector.domain(), DataVectorDomain::Frequency);
        assert_eq!(vector.len(), len);
        let vector = RealFreqVector32::rededicate_from(vector);
        vector.assert_meta_data();
        assert_eq!(vector.len(), len);
        
        // ComplexFreqVector32
        let vector = DataVector32::rededicate_from(ComplexFreqVector32::from_interleaved(&data));
        assert_eq!(vector.is_complex(), true);
        assert_eq!(vector.domain(), DataVectorDomain::Frequency);
        assert_eq!(vector.len(), len);
        let vector = ComplexFreqVector32::rededicate_from(vector);
        vector.assert_meta_data();
        assert_eq!(vector.len(), len);
    }
}