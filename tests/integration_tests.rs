extern crate basic_dsp;
extern crate rand;
extern crate num;

#[cfg(feature = "slow_test")]
mod slow_test {
    
    use rand::*;
    use basic_dsp::{
        DataVector,
        RealTimeVector32};
    
    fn create_data(seed: usize, iteration: usize, from: usize, to: usize) -> Vec<f32>
    {
        let seed: &[_] = &[seed, iteration];
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        let len = rng.gen_range(from, to);
        let mut data = vec![0.0; len];
        for i in 0..len {
            data[i] = rng.gen_range(-10.0, 10.0);
        }
        data
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
            assert_eq!(expected, result.data());
            
        }
    }
    
    #[test]
    fn add_real_vector32_large() {
        for iteration in 0 .. 3 {
            let a = create_data(201511141, iteration, 1000001, 2000000);
            let scalar = create_data_with_len(201511143, iteration, 1);
            let expected = real_add_scalar(&a, scalar[0]);
            let vector = RealTimeVector32::from_array(&a);
            let result = vector.real_offset(scalar[0]);
            assert_eq!(expected, result.data());
        }
    }
}