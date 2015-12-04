
#[cfg(feature = "slow_test")]
mod slow_test {
    use basic_dsp::{
        DataVector,
        RealTimeVector32,
        GenericVectorOperations,
        RealVectorOperations};
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
            let result = vector.real_offset(scalar[0]);
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), false);
            assert_eq!(result.delta(), delta);
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
            let result = vector.real_scale(scalar[0]);
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
            let result = vector.real_abs();
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
            let result = vector.real_square().real_sqrt();
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
            let result = vector.real_expn().real_logn();
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
            let result = vector.real_exp_base(base).real_log_base(base);
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
        let result = vector.wrap(8.0).unwrap(8.0);
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
        let result = vector.wrap(8.0).unwrap(8.0);
        assert_vector_eq(&linear_seq, &result.data());
        assert_eq!(result.is_complex(), false);
        assert_eq!(result.delta(), delta);
    }
}