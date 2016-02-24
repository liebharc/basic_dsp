#[cfg(feature = "slow_test")]
mod slow_test {
    use basic_dsp::*;
    use basic_dsp::conv_types::*;
    use tools::*;
          
    #[test]
    fn compare_interpolatef_and_interpolatei() {
        for iteration in 0 .. 3 {
            let a = create_data_even(201511212, iteration, 2002, 4000);
            let delta = create_delta(3561159, iteration);
            let time = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let factor = iteration as u32 + 1;
            let left = time.clone().interpolatef(&fun as &RealImpulseResponse<f32>, factor as f32, 0.0, 10).unwrap();
            let right = time.interpolatei(&fun as &RealFrequencyResponse<f32>, factor).unwrap();
            assert_vector_eq_with_reason_and_tolerance(&left.data(), &right.data(), 0.1, &format!("Results should match independent if done with interpolatei or interpolatef, factor={}", factor));
        }
    }
    
    #[test]
    fn compare_interpolatef_and_interpolatef_optimized() {
        for iteration in 0 .. 3 {
            let offset = 50e-6; // This offset is just enough to trigger the non optimized code path
            let a = create_data_even(201602221, iteration, 2002, 4000);
            let delta = create_delta(201602222, iteration);
            let time = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let factor = iteration as u32 + 2;
            let left = time.clone().interpolatef(&fun as &RealImpulseResponse<f32>, factor as f32, offset, 10).unwrap();
            let right = time.interpolatef(&fun as &RealImpulseResponse<f32>, factor as f32, 0.0, 10).unwrap();
            assert_vector_eq_with_reason_and_tolerance(&left.data(), &right.data(), 1e-2, &format!("Results should match independent if done with optimized or non optimized interpolatef, factor={}", factor));
        }
    }
    
    #[test]
    fn compare_real_and_complex_interpolatef() {
        for iteration in 0 .. 3 {
            let a = create_data_even(2015112121, iteration, 2002, 4000);
            let delta = create_delta(35611592, iteration);
            let real = RealTimeVector32::from_array_with_delta(&a, delta);
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let factor = iteration as f32 + 1.0;
            let left = real.clone().interpolatef(&fun as &RealImpulseResponse<f32>, factor, 0.0, 10).unwrap();
            let right = real.to_complex().unwrap().interpolatef(&fun as &RealImpulseResponse<f32>, factor, 0.0, 10).unwrap();
            let right = right.to_real().unwrap();
            assert_vector_eq_with_reason_and_tolerance(&left.data(), &right.data(), 0.1, &format!("Results should match independent if done in real or complex number space, factor={}", factor));
        }
    }
    
    #[test]
    fn compare_real_and_complex_interpolatei() {
        for iteration in 0 .. 3 {
            let a = create_data_even(2015112123, iteration, 2002, 4000);
            let delta = create_delta(35611594, iteration);
            let real = RealTimeVector32::from_array_with_delta(&a, delta);
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let factor = iteration as u32 + 1;
            let left = real.clone().interpolatei(&fun as &RealFrequencyResponse<f32>, factor).unwrap();
            let right = real.to_complex().unwrap().interpolatei(&fun as &RealFrequencyResponse<f32>, factor).unwrap();
            let right = right.to_real().unwrap();
            assert_vector_eq_with_reason_and_tolerance(&left.data(), &right.data(), 0.1, "Results should match independent if done in real or complex number space");
        }
    }
    
    #[test]
    fn upsample_downsample() {
        for iteration in 0 .. 3 {
            let a = create_data_even(2015112125, iteration, 2002, 4000);
            let delta = create_delta(35611596, iteration);
            let time = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let factor = (iteration as f32 + 4.0) * 0.5;
            let upsample = time.clone().interpolatef(&fun as &RealImpulseResponse<f32>, factor, 0.0, 10).unwrap();
            let downsample = upsample.interpolatef(&fun as &RealImpulseResponse<f32>, 1.0 / factor, 0.0, 10).unwrap();
            assert_vector_eq_with_reason_and_tolerance(&time.data(), &downsample.data(), 0.2, &format!("Downsampling should be the inverse of upsampling, factor={}", factor));
        }
    }
}