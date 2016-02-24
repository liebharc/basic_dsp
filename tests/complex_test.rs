#[cfg(feature = "slow_test")]
mod slow_test {
    use basic_dsp::{
        DataVector,
        RealTimeVector32,
        GenericVectorOperations,
        ComplexVectorOperations,
        ComplexTimeVector32};
    use num::complex::Complex32;
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
            let vector = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let result = vector.complex_offset(scalar).unwrap();
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), true);
            assert_eq!(result.delta(), delta);
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
            let vector = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let result = vector.complex_scale(scalar).unwrap();
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), true);
            assert_eq!(result.delta(), delta);
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
            let vector = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let result = vector.magnitude().unwrap();
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), false);
            assert_eq!(result.delta(), delta);
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
            let vector = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let result = vector.magnitude_squared().unwrap();
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), false);
            assert_eq!(result.delta(), delta);
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
            let vector1 = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let vector2 = ComplexTimeVector32::from_interleaved_with_delta(&b, delta);
            let result = vector1.multiply_vector(&vector2).unwrap();
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), true);
            assert_eq!(result.delta(), delta);
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
        let vector1 = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
        let vector2 = ComplexTimeVector32::from_interleaved_with_delta(&b, delta);
        let result = vector1.multiply_smaller_vector(&vector2).unwrap();
        assert_vector_eq(&expected, &result.data());
        assert_eq!(result.is_complex(), true);
        assert_eq!(result.delta(), delta);
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
        let vector1 = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
        let vector2 = ComplexTimeVector32::from_interleaved_with_delta(&b, delta);
        let result = vector1.divide_smaller_vector(&vector2).unwrap();
        assert_vector_eq(&expected, &result.data());
        assert_eq!(result.is_complex(), true);
        assert_eq!(result.delta(), delta);
    }
    
    #[test]
    fn complex_dot_product32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511171, iteration, range.start, range.end);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = to_complex(&complex_vector_mul(&a, &b)).iter().fold(Complex32::new(0.0, 0.0), |a, b| a + b);
            let delta = create_delta(3561159, iteration);
            let vector1 = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let vector2 =ComplexTimeVector32::from_interleaved_with_delta(&b, delta);
            let result = vector1.complex_dot_product(&vector2).unwrap();
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
            let vector1 = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let vector2 = ComplexTimeVector32::from_interleaved_with_delta(&b, delta);
            let result = vector1.divide_vector(&vector2).unwrap();
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), true);
            assert_eq!(result.delta(), delta);
        });
    }
    
    #[test]
    fn complex_real_conversions_vector32() {
        parameterized_vector_test(|iteration, range| {
            let real = create_data(201511191, iteration, range.start, range.end);
            let imag = create_data_with_len(201511192, iteration, real.len());
            let delta = create_delta(3561159, iteration);
            let complex = ComplexTimeVector32::from_real_imag_with_delta(&real, &imag, delta);
            let mut real_vector = RealTimeVector32::from_array_no_copy(vec![0.0; 0]);
            let mut imag_vector = RealTimeVector32::from_array_no_copy(vec![0.0; 0]);
            complex.get_real(&mut real_vector).unwrap();
            complex.get_imag(&mut imag_vector).unwrap();
            let real_result = complex.to_real().unwrap();
            assert_vector_eq_with_reason(&real, &real_vector.data(), "Failure in get_real");
            assert_vector_eq_with_reason(&real, &real_result.data(), "Failure in get_imag");
            assert_vector_eq_with_reason(&imag, &imag_vector.data(), "Failure in to_real");
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
            let abs = create_data_even_in_range(201511203, iteration, range.start, range.end, 0.1, 10.0);
            let phase = create_data_in_range_with_len(201511204, iteration, abs.len(), -1.57, 1.57);
            let delta = create_delta(3561159, iteration);
            let complex = ComplexTimeVector32::from_mag_phase_with_delta(&abs, &phase, delta);
            let mut abs_vector = RealTimeVector32::from_array_no_copy(vec![0.0; 0]);
            let mut phase_vector = RealTimeVector32::from_array_no_copy(vec![0.0; 0]);
            complex.get_magnitude(&mut abs_vector).unwrap();
            complex.get_phase(&mut phase_vector).unwrap();
            let phase_result = complex.phase().unwrap();
            assert_vector_eq_with_reason(&abs, &abs_vector.data(), "Failure in get_complex_abs");
            assert_vector_eq_with_reason(&phase, &phase_vector.data(), "Failure in get_phase");
            assert_vector_eq_with_reason(&phase, &phase_result.data(), "Failure in phase");
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
            let vector = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let expected = complex_vector_diff(&a);
            let result = vector.diff_with_start().unwrap();
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), true);
            assert_eq!(result.delta(), delta);
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
            let vector = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let expected = complex_vector_cum_sum(&a);
            let result = vector.cum_sum().unwrap();
            assert_vector_eq(&expected, &result.data());
            assert_eq!(result.is_complex(), true);
            assert_eq!(result.delta(), delta);
        });
    }
    
    #[test]
    fn statistics_test32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let c = to_complex(&a);
            let vector = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let sum = c.iter().fold(Complex32::new(0.0, 0.0), |a,b| a + b);
            let sum_sq = c.iter().map(|v| v * v).fold(Complex32::new(0.0, 0.0), |a,b| a + b);
            let rms = (sum_sq / a.len() as f32).sqrt();
            let result = vector.complex_statistics();
            assert_complex_in_tolerance(result.sum, sum, 0.5);
            assert_complex_in_tolerance(result.rms, rms, 0.5);
        });
    }
    
    #[test]
    fn split_merge_test32() {
        let a = create_data(201511210, 0, 1000, 1000);
        let vector = ComplexTimeVector32::from_interleaved(&a);
        let mut split = 
            [
                Box::new(ComplexTimeVector32::empty()),
                Box::new(ComplexTimeVector32::empty()),
                Box::new(ComplexTimeVector32::empty()),
                Box::new(ComplexTimeVector32::empty()),
                Box::new(ComplexTimeVector32::empty())];
        vector.split_into(&mut split).unwrap();
        let merge = ComplexTimeVector32::empty();
        let result = merge.merge(&split).unwrap();
        assert_vector_eq(&a, &result.data());
    }
    
    #[test]
    fn split_test32() {
        let a = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let vector = ComplexTimeVector32::from_interleaved(a);
        let mut split = 
            [
                Box::new(ComplexTimeVector32::empty()),
                Box::new(ComplexTimeVector32::empty())];
        vector.split_into(&mut split).unwrap();
        assert_vector_eq(&[1.0, 2.0, 5.0, 6.0], &split[0].data());
        assert_vector_eq(&[3.0, 4.0, 7.0, 8.0], &split[1].data()); 
    }
    
    #[test]
    fn to_real_imag_and_back32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let vector = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let mut real = RealTimeVector32::empty();
            let mut imag = RealTimeVector32::empty();
            vector.get_real_imag(&mut real, &mut imag).unwrap();
            let vector2 = ComplexTimeVector32::empty();
            let result = vector2.set_real_imag(&real, &imag).unwrap();
            assert_vector_eq(&a, result.data());
        });
    }
    
    #[test]
    fn to_mag_phase_and_back32() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201511210, iteration, range.start, range.end);
            let delta = create_delta(3561159, iteration);
            let vector = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let mut mag = RealTimeVector32::empty();
            let mut phase = RealTimeVector32::empty();
            let mut mag2 = RealTimeVector32::empty();
            let mut phase2 = RealTimeVector32::empty();;
            vector.get_mag_phase(&mut mag, &mut phase).unwrap();
            vector.get_magnitude(&mut mag2).unwrap();
            vector.get_phase(&mut phase2).unwrap();
            
            assert_vector_eq_with_reason(mag.data(), mag2.data(), "Magnitude differs");
            assert_vector_eq_with_reason(phase.data(), phase2.data(), "Phase differs");
            
            let vector2 = ComplexTimeVector32::empty();
            let result = vector2.set_mag_phase(&mag, &phase).unwrap();
            assert_vector_eq_with_reason_and_tolerance(&a, result.data(), 1e-4, "Merge differs");
        });
    }
}