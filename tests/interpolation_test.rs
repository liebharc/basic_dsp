extern crate basic_dsp;
extern crate rand;
extern crate num;
pub mod tools;

mod inter_test {
    use basic_dsp::*;
    use basic_dsp::conv_types::*;
    use tools::*;
    use num::complex::*;

    #[test]
    fn compare_interpolatef_and_interpolatei() {
        for iteration in 0..3 {
            let a = create_data_even(201511212, iteration, 2002, 4000);
            let delta = create_delta(3561159, iteration);
            let mut time = a.to_complex_time_vec();
            time.set_delta(delta);
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let factor = iteration as u32 + 1;
            let mut buffer = SingleBuffer::new();
            let mut left = time.clone();
            left.interpolatef(&mut buffer,
                              &fun as &RealImpulseResponse<f32>,
                              factor as f32,
                              0.0,
                              10);
            let mut right = time;
            right.interpolatei(&mut buffer, &fun as &RealFrequencyResponse<f32>, factor).unwrap();
            assert_vector_eq_with_reason_and_tolerance(&left[..],
                                                       &right[..],
                                                       0.1,
                                                       &format!("Results should match \
                                                                 independent if done with \
                                                                 interpolatei or interpolatef, \
                                                                 factor={}",
                                                                factor));
        }
    }

    #[test]
    fn compare_interpolate_and_interpolatei() {
        for iteration in 0..3 {
            let a = create_random_multitones(201511212, iteration, 2002, 4000, 5);
            let delta = create_delta(3561159, iteration);
            let time = a.to_real_time_vec();
            let mut time = time.to_complex().unwrap();
            time.scale(Complex32::new(0.9, -0.1));
            time.set_delta(delta);
            let fun: SincFunction<f32> = SincFunction::new();
            let factor = iteration as u32 + 1;
            let mut buffer = SingleBuffer::new();
            let mut left = time.clone();
            left.interpolate(&mut buffer,
                              Some(&fun as &RealFrequencyResponse<f32>),
                              time.points() * factor as usize,
                              0.0).unwrap();
            let mut right = time;
            right.interpolatei(&mut buffer, &fun as &RealFrequencyResponse<f32>, factor).unwrap();
            assert_vector_eq_with_reason_and_tolerance(&left[..],
                                                       &right[..],
                                                       0.1,
                                                       &format!("Results should match \
                                                                 independent if done with \
                                                                 interpolatei or interpolate, \
                                                                 factor={}",
                                                                factor));
        }
    }

    #[test]
    fn compare_interpolatef_and_interpolatef_optimized() {
        for iteration in 0..3 {
            // This offset is just enough to trigger the non optimized code path
            let offset = 1.0001e-6;
            let a = create_data_even(201602221, iteration, 2002, 4000);
            let delta = create_delta(201602222, iteration);
            let mut time = a.to_complex_time_vec();
            time.set_delta(delta);
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let mut buffer = SingleBuffer::new();
            let factor = iteration as u32 + 2;
            let mut left = time.clone();
            left.interpolatef(&mut buffer,
                              &fun as &RealImpulseResponse<f32>,
                              factor as f32 + offset,
                              0.0,
                              12);
            let mut right = time;
            right.interpolatef(&mut buffer,
                               &fun as &RealImpulseResponse<f32>,
                               factor as f32,
                               0.0,
                               12);
            assert_vector_eq_with_reason_and_tolerance(&left[..],
                                                       &right[..],
                                                       1e-2,
                                                       &format!("Results should match \
                                                                 independent if done with \
                                                                 optimized or non optimized \
                                                                 interpolatef, factor={}",
                                                                factor));
        }
    }

    #[test]
    fn compare_interpolatef_and_interpolate() {
        for iteration in 0..3 {
            let a = create_random_multitones(201511212, iteration, 2002, 4000, 5);
            let delta = create_delta(201602222, iteration);
            let time = a.to_real_time_vec();
            let mut time = time.to_complex().unwrap();
            time.scale(Complex32::new(0.45, -0.3));
            time.set_delta(delta);
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let mut buffer = SingleBuffer::new();
            let factor = iteration as u32 + 2;
            let mut left = time.clone();
            left.interpolatef(&mut buffer,
                              &fun as &RealImpulseResponse<f32>,
                              factor as f32,
                              0.0,
                              12);
            let mut right = time;
            right.interpolate(&mut buffer,
                               Some(&fun as &RealFrequencyResponse<f32>),
                               left.points(),
                               0.0).unwrap();
            assert_vector_eq_with_reason_and_tolerance(&left[..],
                                                       &right[..],
                                                       0.1,
                                                       &format!("Results should match \
                                                                 independent if done with \
                                                                 optimized with interpolate or \
                                                                 interpolatef, factor={}",
                                                                factor));
        }
    }

    #[test]
    fn compare_interpolatef_and_interpolatef_optimized_with_delay() {
        for iteration in 0..3 {
            // This offset is just enough to trigger the non optimized code path
            let offset = 1.0001e-6;
            let a = create_data_even(201602221, iteration, 2002, 4000);
            let delta = create_delta(201602222, iteration);
            let mut time = a.to_complex_time_vec();
            time.set_delta(delta);
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let mut buffer = SingleBuffer::new();
            let factor = iteration as u32 + 2;
            let delay = 1.0 / (iteration as f32 + 2.0);
            let mut left = time.clone();
            left.interpolatef(&mut buffer,
                              &fun as &RealImpulseResponse<f32>,
                              factor as f32 + offset,
                              delay,
                              12);
            let mut right = time;
            right.interpolatef(&mut buffer,
                               &fun as &RealImpulseResponse<f32>,
                               factor as f32,
                               delay,
                               12);
            assert_vector_eq_with_reason_and_tolerance(&left[..],
                                                       &right[..],
                                                       0.1,
                                                       &format!("Results should match \
                                                                 independent if done with \
                                                                 optimized or non optimized \
                                                                 interpolatef, factor={}",
                                                                factor));
        }
    }

    #[test]
    fn compare_real_and_complex_interpolatef() {
        for iteration in 0..3 {
            let a = create_data_even(2015112121, iteration, 2002, 4000);
            let delta = create_delta(35611592, iteration);
            let mut real = a.to_real_time_vec();
            real.set_delta(delta);
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let mut buffer = SingleBuffer::new();
            let factor = iteration as f32 + 1.0;
            let mut left = real.clone();
            left.interpolatef(&mut buffer,
                              &fun as &RealImpulseResponse<f32>,
                              factor,
                              0.0,
                              12);
            let mut right = real.to_complex().unwrap();
            right.interpolatef(&mut buffer,
                               &fun as &RealImpulseResponse<f32>,
                               factor,
                               0.0,
                               12);
            let right = right.to_real();
            assert_vector_eq_with_reason_and_tolerance(&left[..],
                                                       &right[..],
                                                       0.1,
                                                       &format!("Results should match \
                                                                 independent if done in real or \
                                                                 complex number space, factor={}",
                                                                factor));
        }
    }

    #[test]
    fn compare_real_and_complex_interpolatei() {
        for iteration in 0..3 {
            let a = create_data_even(2015112123, iteration, 2002, 4000);
            let delta = create_delta(35611594, iteration);
            let mut real = a.to_real_time_vec();
            real.set_delta(delta);
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let mut buffer = SingleBuffer::new();
            let factor = iteration as u32 + 1;
            let mut left = real.clone();
            left.interpolatei(&mut buffer, &fun as &RealFrequencyResponse<f32>, factor).unwrap();
            let mut right = real.to_complex().unwrap();
            right.interpolatei(&mut buffer, &fun as &RealFrequencyResponse<f32>, factor).unwrap();
            let right = right.to_real();
            assert_vector_eq_with_reason_and_tolerance(&left[..],
                                                       &right[..],
                                                       0.1,
                                                       "Results should match independent if done \
                                                        in real or complex number space");
        }
    }

    #[test]
    fn upsample_downsample() {
        for iteration in 0..3 {
            let a = create_data_even(2015112125, iteration, 2002, 4000);
            let delta = create_delta(35611596, iteration);
            let mut time = a.to_complex_time_vec();
            time.set_delta(delta);
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let mut buffer = SingleBuffer::new();
            let factor = (iteration as f32 + 4.0) * 0.5;
            let mut upsample = time.clone();
            upsample.interpolatef(&mut buffer,
                                  &fun as &RealImpulseResponse<f32>,
                                  factor,
                                  0.0,
                                  13);
            upsample.interpolatef(&mut buffer,
                                  &fun as &RealImpulseResponse<f32>,
                                  1.0 / factor,
                                  0.0,
                                  13);
            assert_vector_eq_with_reason_and_tolerance(&time[..],
                                                       &upsample[..],
                                                       0.2,
                                                       &format!("Downsampling should be the \
                                                                 inverse of upsampling, \
                                                                 factor={}",
                                                                factor));
        }
    }
}
