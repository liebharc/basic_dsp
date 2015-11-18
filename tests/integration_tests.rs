extern crate basic_dsp;
extern crate rand;
extern crate num;

#[cfg(feature = "slow_test")]
mod slow_test {
    use rand::*;
    use basic_dsp::{
        DataVector,
        RealTimeVector32,
        GenericVectorOperations,
        RealVectorOperations,
        ComplexVectorOperations,
        ComplexTimeVector32};
    use num::complex::Complex32;
        
    fn assert_vector_eq(left: &[f32], right: &[f32]) {
        let mut errors = Vec::new();
        if left.len() != right.len()
        {
            errors.push(format!("Size difference {} != {}", left.len(), right.len()));
        }
        
        let len = if left.len() < right.len() { left.len() } else { right.len() };
        let mut differences = 0;
        let mut first_difference = false;
        let tolerance = 1e-12;
        for i in 0 .. len {
            if (left[i] - right[i]).abs() > tolerance
            {
                differences += 1;
                if !first_difference
                {
                    errors.push(format!("First difference at index {}, left: {} != right: {}", i, left[i], right[i]));
                    first_difference = true;
                }
            }
        }
        
        if differences > 0
        {
            errors.push(format!("Total number of differences: {}/{}={}%", differences, len, differences*100/len));
        }
        
        if errors.len() > 0
        {
            let all_errors = errors.join("\n");
            let header = "-----------------------".to_owned();
            let full_text = format!("\n{}\n{}\n{}\n", header, all_errors, header);
            panic!(full_text);
        }
    }
    
    fn create_data(seed: usize, iteration: usize, from: usize, to: usize) -> Vec<f32>
    {
        let len_seed: &[_] = &[seed, iteration];
        let mut rng: StdRng = SeedableRng::from_seed(len_seed);
        let len = rng.gen_range(from, to);
        create_data_with_len(seed, iteration, len)
    }
    
    fn create_data_even(seed: usize, iteration: usize, from: usize, to: usize) -> Vec<f32>
    {
        let len_seed: &[_] = &[seed, iteration];
        let mut rng: StdRng = SeedableRng::from_seed(len_seed);
        let len = rng.gen_range(from, to);
        let len = len + len % 2;
        create_data_with_len(seed, iteration, len)
    }
    
    fn create_data_with_len(seed: usize, iteration: usize, len: usize) -> Vec<f32>
    {
        let seed: &[_] = &[seed, iteration];
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        let mut data = vec![0.0; len];
        for i in 0..len {
            data[i] = rng.gen_range(-10.0, 10.0);
        }
        data
    }
    
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
    fn add_real_vector32_small() {
        for iteration in 0 .. 10 {
            let a = create_data(201511141, iteration, 10000, 1000000);
            let scalar = create_data_with_len(201511142, iteration, 1);
            let expected = real_add_scalar(&a, scalar[0]);
            let vector = RealTimeVector32::from_array(&a);
            let result = vector.real_offset(scalar[0]);
            assert_vector_eq(&expected, &result.data());
        }
    }
    
    #[test]
    fn add_real_vector32_large() {
        for iteration in 0 .. 3 {
            let a = create_data(201511142, iteration, 1000001, 2000000);
            let scalar = create_data_with_len(201511143, iteration, 1);
            let expected = real_add_scalar(&a, scalar[0]);
            let vector = RealTimeVector32::from_array(&a);
            let result = vector.real_offset(scalar[0]);
            assert_vector_eq(&expected, &result.data());
        }
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
    fn multiply_real_vector32_small() {
        for iteration in 0 .. 10 {
            let a = create_data(201511143, iteration, 10000, 1000000);
            let scalar = create_data_with_len(201511142, iteration, 1);
            let expected = real_mulitply_scalar(&a, scalar[0]);
            let vector = RealTimeVector32::from_array(&a);
            let result = vector.real_scale(scalar[0]);
            assert_vector_eq(&expected, &result.data());
        }
    }
    
    #[test]
    fn multiply_real_vector32_large() {
        for iteration in 0 .. 3 {
            let a = create_data(201511144, iteration, 1000001, 2000000);
            let scalar = create_data_with_len(201511143, iteration, 1);
            let expected = real_mulitply_scalar(&a, scalar[0]);
            let vector = RealTimeVector32::from_array(&a);
            let result = vector.real_scale(scalar[0]);
            assert_vector_eq(&expected, &result.data());
        }
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
    fn abs_real_vector32_small() {
        for iteration in 0 .. 10 {
            let a = create_data(201511146, iteration, 10000, 1000000);
            let expected = real_abs(&a);
            let vector = RealTimeVector32::from_array(&a);
            let result = vector.real_abs();
            assert_vector_eq(&expected, &result.data());
        }
    }
    
    #[test]
    fn abs_real_vector32_large() {
        for iteration in 0 .. 3 {
            let a = create_data(201511147, iteration, 1000001, 2000000);
            let expected = real_abs(&a);
            let vector = RealTimeVector32::from_array(&a);
            let result = vector.real_abs();
            assert_vector_eq(&expected, &result.data());
        }
    }
    
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
    fn complex_add_scalar_vector32_small() {
        for iteration in 0 .. 10 {
            let a = create_data_even(2015111410, iteration, 10000, 1000000);
            let scalar = create_data_with_len(2015111413, iteration, 2);
            let scalar = Complex32::new(scalar[0], scalar[1]);
            let expected = complex_add_scalar(&a, scalar);
            let vector = ComplexTimeVector32::from_interleaved(&a);
            let result = vector.complex_offset(scalar);
            assert_vector_eq(&expected, &result.data());
        }
    }
    
    #[test]
    fn complex_add_scalar_vector32_large() {
        for iteration in 0 .. 3 {
            let a = create_data_even(2015111411, iteration, 1000001, 2000000);
            let scalar = create_data_with_len(2015111414, iteration, 2);
            let scalar = Complex32::new(scalar[0], scalar[1]);
            let expected = complex_add_scalar(&a, scalar);
            let vector = ComplexTimeVector32::from_interleaved(&a);
            let result = vector.complex_offset(scalar);
            assert_vector_eq(&expected, &result.data());
        }
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
    fn complex_mutiply_scalar_vector32_small() {
        for iteration in 0 .. 10 {
            let a = create_data_even(2015111410, iteration, 10000, 1000000);
            let scalar = create_data_with_len(2015111413, iteration, 2);
            let scalar = Complex32::new(scalar[0], scalar[1]);
            let expected = complex_multiply_scalar(&a, scalar);
            let vector = ComplexTimeVector32::from_interleaved(&a);
            let result = vector.complex_scale(scalar);
            assert_vector_eq(&expected, &result.data());
        }
    }
    
    #[test]
    fn complex_mutiply_scalar_vector32_large() {
        for iteration in 0 .. 3 {
            let a = create_data_even(2015111411, iteration, 1000001, 2000000);
            let scalar = create_data_with_len(2015111414, iteration, 2);
            let scalar = Complex32::new(scalar[0], scalar[1]);
            let expected = complex_multiply_scalar(&a, scalar);
            let vector = ComplexTimeVector32::from_interleaved(&a);
            let result = vector.complex_scale(scalar);
            assert_vector_eq(&expected, &result.data());
        }
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
    fn complex_abs_vector32_small() {
        for iteration in 0 .. 10 {
            let a = create_data_even(2015111410, iteration, 10000, 1000000);
            let expected = complex_abs(&a);
            let vector = ComplexTimeVector32::from_interleaved(&a);
            let result = vector.complex_abs();
            assert_vector_eq(&expected, &result.data());
        }
    }
    
    #[test]
    fn complex_abs_vector32_large() {
        for iteration in 0 .. 3 {
            let a = create_data_even(2015111411, iteration, 1000001, 2000000);
            let expected = complex_abs(&a);
            let vector = ComplexTimeVector32::from_interleaved(&a);
            let result = vector.complex_abs();
            assert_vector_eq(&expected, &result.data());
        }
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
    fn complex_abs_sq_vector32_small() {
        for iteration in 0 .. 10 {
            let a = create_data_even(2015111410, iteration, 10000, 1000000);
            let expected = complex_abs_sq(&a);
            let vector = ComplexTimeVector32::from_interleaved(&a);
            let result = vector.complex_abs_squared();
            assert_vector_eq(&expected, &result.data());
        }
    }
    
    #[test]
    fn complex_abs_sq_vector32_large() {
        for iteration in 0 .. 3 {
            let a = create_data_even(2015111411, iteration, 1000001, 2000000);
            let expected = complex_abs_sq(&a);
            let vector = ComplexTimeVector32::from_interleaved(&a);
            let result = vector.complex_abs_squared();
            assert_vector_eq(&expected, &result.data());
        }
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
    fn real_add_vector_vector32_small() {
        for iteration in 0 .. 10 {
            let a = create_data(201511171, iteration, 10000, 1000000);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = real_add_vector(&a, &b);
            let vector1 = RealTimeVector32::from_array(&a);
            let vector2 = RealTimeVector32::from_array(&b);
            let result = vector1.add_vector(&vector2);
            assert_vector_eq(&expected, &result.data());
        }
    }
    
    #[test]
    fn real_add_vector_vector32_large() {
        for iteration in 0 .. 3 {
            let a = create_data(201511173, iteration, 1000001, 2000000);
            let b = create_data_with_len(201511174, iteration, a.len());
            let expected = real_add_vector(&a, &b);
            let vector1 = RealTimeVector32::from_array(&a);
            let vector2 = RealTimeVector32::from_array(&b);
            let result = vector1.add_vector(&vector2);
            assert_vector_eq(&expected, &result.data());
        }
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
    fn real_sub_vector_vector32_small() {
        for iteration in 0 .. 10 {
            let a = create_data(201511171, iteration, 10000, 1000000);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = real_sub_vector(&a, &b);
            let vector1 = RealTimeVector32::from_array(&a);
            let vector2 = RealTimeVector32::from_array(&b);
            let result = vector1.subtract_vector(&vector2);
            assert_vector_eq(&expected, &result.data());
        }
    }
    
    #[test]
    fn real_sub_vector_vector32_large() {
        for iteration in 0 .. 3 {
            let a = create_data(201511173, iteration, 1000001, 2000000);
            let b = create_data_with_len(201511174, iteration, a.len());
            let expected = real_sub_vector(&a, &b);
            let vector1 = RealTimeVector32::from_array(&a);
            let vector2 = RealTimeVector32::from_array(&b);
            let result = vector1.subtract_vector(&vector2);
            assert_vector_eq(&expected, &result.data());
        }
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
    fn real_mul_vector_vector32_small() {
        for iteration in 0 .. 10 {
            let a = create_data(201511171, iteration, 10000, 1000000);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = real_vector_mul(&a, &b);
            let vector1 = RealTimeVector32::from_array(&a);
            let vector2 = RealTimeVector32::from_array(&b);
            let result = vector1.multiply_vector(&vector2);
            assert_vector_eq(&expected, &result.data());
        }
    }
    
    #[test]
    fn real_mul_vector_vector32_large() {
        for iteration in 0 .. 3 {
            let a = create_data(201511173, iteration, 1000001, 2000000);
            let b = create_data_with_len(201511174, iteration, a.len());
            let expected = real_vector_mul(&a, &b);
            let vector1 = RealTimeVector32::from_array(&a);
            let vector2 = RealTimeVector32::from_array(&b);
            let result = vector1.multiply_vector(&vector2);
            assert_vector_eq(&expected, &result.data());
        }
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
    fn real_div_vector_vector32_small() {
        for iteration in 0 .. 10 {
            let a = create_data(201511171, iteration, 10000, 1000000);
            let b = create_data_with_len(201511172, iteration, a.len());
            let expected = real_vector_div(&a, &b);
            let vector1 = RealTimeVector32::from_array(&a);
            let vector2 = RealTimeVector32::from_array(&b);
            let result = vector1.divide_vector(&vector2);
            assert_vector_eq(&expected, &result.data());
        }
    }
    
    #[test]
    fn real_div_vector_vector32_large() {
        for iteration in 0 .. 3 {
            let a = create_data(201511173, iteration, 1000001, 2000000);
            let b = create_data_with_len(201511174, iteration, a.len());
            let expected = real_vector_div(&a, &b);
            let vector1 = RealTimeVector32::from_array(&a);
            let vector2 = RealTimeVector32::from_array(&b);
            let result = vector1.divide_vector(&vector2);
            assert_vector_eq(&expected, &result.data());
        }
    }
}