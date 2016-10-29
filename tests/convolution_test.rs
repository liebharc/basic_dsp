extern crate basic_dsp;
extern crate rand;
extern crate num;
pub mod tools;

mod conv_test {
    use basic_dsp::*;
    use basic_dsp::conv_types::*;
    use basic_dsp::window_functions::*;
    use tools::*;
    use std::os::raw::c_void;

    #[test]
    fn compare_conv_freq_multiplication_for_rc() {
        for iteration in 0..3 {
            let a = create_data_even(201602211, iteration, 1001, 2000);
            let delta = create_delta(201602212, iteration);
            let mut time = a.to_complex_time_vec();
            time.set_delta(delta);
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let mut buffer = SingleBuffer::new();
            let mut freq = time.clone().fft(&mut buffer);
            let points = time.points();
            // Should get us a range [0.0 .. 1.0] and hopefully we are not that unlucky to get 0.0
            let ratio = create_delta(20160229, iteration).abs() / 10.0;
            time.convolve(&mut buffer,
                          &fun as &RealImpulseResponse<f32>,
                          ratio,
                          points);
            freq.multiply_frequency_response(&fun as &RealFrequencyResponse<f32>, 1.0 / ratio);
            let ifreq_res = freq.ifft(&mut buffer);
            assert_vector_eq_with_reason_and_tolerance(&ifreq_res[..],
                                                       &time[..],
                                                       0.2,
                                                       "Results should match independent if done \
                                                        in time or frequency domain");
        }
    }

    #[test]
    fn compare_conv_freq_multiplication_for_sinc() {
        for iteration in 0..3 {
            let a = create_data_even(201602214, iteration, 2001, 4000);
            let delta = create_delta(201602215, iteration);
            let mut time = a.to_complex_time_vec();
            time.set_delta(delta);
            let fun: SincFunction<f32> = SincFunction::new();
            let mut buffer = SingleBuffer::new();
            let mut freq = time.clone().fft(&mut buffer);
            let points = time.points();
            // Should get us a range [0.5 .. 1.0]
            let ratio = create_delta(201602216, iteration).abs() / 20.0 + 0.5;
            time.convolve(&mut buffer,
                          &fun as &RealImpulseResponse<f32>,
                          ratio,
                          points);
            freq.multiply_frequency_response(&fun as &RealFrequencyResponse<f32>, 1.0 / ratio);
            let ifreq_res = freq.ifft(&mut buffer);
            assert_vector_eq_with_reason_and_tolerance(&ifreq_res[..],
                                                       &time[..],
                                                       0.3,
                                                       "Results should match independent if done \
                                                        in time or frequency domain");
        }
    }

    #[test]
    fn compare_optimized_and_non_optimized_conv() {
        for iteration in 0..3 {
            // This offset is small enough to now have a big impact on the results
            // (for the RC function)
            // but the code will use a different non-optimized branch since it won't
            // recognize ratio as an
            // integer
            let offset = 2e-6;
            let a = create_data_even(201602217, iteration, 2002, 4000);
            let b = create_data_even(201602218, iteration, 50, 202);
            let delta = create_delta(201602219, iteration);
            let mut time = a.to_complex_time_vec();
            time.set_delta(delta);
            let mut buffer = SingleBuffer::new();
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let ratio = iteration as f32 + 1.0;
            let mut left = time.clone();
            left.convolve(&mut buffer,
                          &fun as &RealImpulseResponse<f32>,
                          ratio,
                          b.len());
            time.convolve(&mut buffer,
                          &fun as &RealImpulseResponse<f32>,
                          ratio + offset,
                          b.len());
            assert_vector_eq_with_reason_and_tolerance(&left[..],
                                                       &time[..],
                                                       0.1,
                                                       "Results should match independent if done \
                                                        in optimized or non optimized code branch");
        }
    }

    #[test]
    fn compare_vector_conv_freq_multiplication() {
        for iteration in 0..3 {
            let a = create_data_even(201601171, iteration, 502, 1000);
            let b = create_data_with_len(201601172, iteration, a.len());
            let delta = create_delta(201601173, iteration);
            let mut time1 = a.to_complex_time_vec();
            time1.set_delta(delta);
            let mut time2 = b.to_complex_time_vec();
            time2.set_delta(delta);
            let mut buffer = SingleBuffer::new();
            let mut left = time1.clone();
            left.convolve_vector(&mut buffer, &time2).unwrap();
            let mut freq1 = time1.fft(&mut buffer);
            let freq2 = time2.fft(&mut buffer);
            freq1.mul(&freq2).unwrap();
            let right = freq1.ifft(&mut buffer);
            assert_vector_eq_with_reason_and_tolerance(&left[..],
                                                       &conv_swap(&right[..])[0..left.len()],
                                                       0.2,
                                                       "Results should match independent if done \
                                                        in time or frequency domain");
        }
    }

    #[test]
    fn compare_smaller_vector_conv_with_zero_padded_conv() {
        for iteration in 0..3 {
            let a = create_data_even(201601174, iteration, 1002, 2000);
            let b = create_data_even(201601175, iteration, 50, 202);
            let delta = create_delta(201601176, iteration);
            let mut time1 = a.to_complex_time_vec();
            time1.set_delta(delta);
            let mut time2 = b.clone().to_complex_time_vec();
            time2.set_delta(delta);
            let mut buffer = SingleBuffer::new();
            let mut left = time1.clone();
            left.convolve_vector(&mut buffer, &time2).unwrap();
            let mut time2 = conv_zero_pad(&b, time1.len(), true).to_complex_time_vec();
            time2.set_delta(delta);
            time1.convolve_vector(&mut buffer, &time2).unwrap();
            assert_vector_eq_with_reason_and_tolerance(&left[..],
                                                       &time1[..],
                                                       0.2,
                                                       "Results should match independent if done \
                                                        with a smaller vector or with a zero \
                                                        padded vector of the same size");
        }
    }

    #[test]
    fn compare_smaller_vector_conv_with_zero_padded_conv_real() {
        for iteration in 0..3 {
            let a = create_data_even(201601177, iteration, 1002, 2000);
            let b = create_data_even(201601178, iteration, 50, 202);
            let delta = create_delta(201601179, iteration);
            let mut time1 = a.to_real_time_vec();
            time1.set_delta(delta);
            let mut time2 = b.clone().to_real_time_vec();
            time2.set_delta(delta);
            let mut buffer = SingleBuffer::new();
            let mut left = time1.clone();
            left.convolve_vector(&mut buffer, &time2).unwrap();
            let mut time2 = conv_zero_pad(&b, time1.len(), false).to_real_time_vec();
            time2.set_delta(delta);
            time1.convolve_vector(&mut buffer, &time2).unwrap();
            assert_vector_eq_with_reason_and_tolerance(&left[..],
                                                       &time1[..],
                                                       0.2,
                                                       "Results should match independent if done \
                                                        with a smaller vector or with a zero \
                                                        padded vector of the same size");
        }
    }

    fn conv_zero_pad(data: &[f32], len: usize, is_complex: bool) -> Vec<f32> {
        if is_complex {
            let mut result = vec![0.0; len];
            let points = len / 2;
            let data_points = data.len() / 2;
            let diff = points - data_points;
            let left = diff - diff / 2;
            for i in 0..data.len() {
                result[2 * left + i] = data[i];
            }
            result
        } else {
            let mut result = vec![0.0; len];
            let data_len = data.len();
            let diff = len - data_len;
            let left = diff - diff / 2;
            for i in 0..data.len() {
                result[left + i] = data[i];
            }
            result
        }
    }

    /// This kind of swapping is necessary since we defined
    /// our conv to have 0s in the center
    fn conv_swap(data: &[f32]) -> Vec<f32> {
        let len = data.len();
        let mut result = vec![0.0; len];
        let points = len / 2;
        let half = 2 * (points / 2 + 1);
        for i in 0..len {
            if i < half {
                result[i] = data[len - half + i];
            } else if i >= len - half {
                result[i] = data[i - half];
            } else
            // Center
            {
                result[i] = data[i];
            }
        }
        result
    }

    /// Calls to another window with the only
    /// difference that it doesn't allow to make use of symmetry
    fn unsym_triag_window() -> ForeignWindowFunction<f32> {
        ForeignWindowFunction {
            window_data: 0,
            window_function: call_triag,
            is_symmetric: false,
        }
    }

    extern "C" fn call_triag(_arg: *const c_void, i: usize, points: usize) -> f32 {
        let triag: &WindowFunction<f32> = &TriangularWindow;
        triag.window(i, points)
    }

    /// Calls to another window with the only
    /// difference that it doesn't allow to make use of symmetry
    fn unsym_rc_mul() -> ForeignRealConvolutionFunction<f32> {
        ForeignRealConvolutionFunction {
            conv_data: 0,
            conv_function: call_freq_rc,
            is_symmetric: false,
        }
    }

    extern "C" fn call_freq_rc(_arg: *const c_void, x: f32) -> f32 {
        let rc: &RealFrequencyResponse<f32> = &RaisedCosineFunction::new(0.35);
        rc.calc(x)
    }

    #[test]
    fn compare_sym_optimized_window_with_normal_version() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(20160116, iteration, range.start, range.end);
            let delta = create_delta(201601161, iteration);
            let mut time = a.to_complex_time_vec();
            time.set_delta(delta);
            let triag_sym = TriangularWindow;
            let triag_unsym = unsym_triag_window();
            let mut result_sym = time.clone();
            result_sym.apply_window(&triag_sym);
            let mut result_unsym = time;
            result_unsym.apply_window(&triag_unsym);
            assert_vector_eq_with_reason_and_tolerance(&result_sym[..],
                                                       &result_unsym[..],
                                                       1e-2,
                                                       "Results should match with or without \
                                                        symmetry optimization");
        });
    }

    #[test]
    fn compare_sym_optimized_freq_mul_with_normal_version() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(201601162, iteration, range.start, range.end);
            let delta = create_delta(201601163, iteration);
            let mut freq = a.to_complex_freq_vec();
            freq.set_delta(delta);
            let rc_sym: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let rc_unsym = unsym_rc_mul();
            // Should get us a range [0.5 .. 1.0]
            let ratio = create_delta(201601164, iteration).abs() / 20.0 + 0.5;
            let mut result_sym = freq.clone();
            result_sym.multiply_frequency_response(
                &rc_sym as &RealFrequencyResponse<f32>, 1.0 / ratio);
            let mut result_unsym = freq;
            result_unsym.multiply_frequency_response(
                &rc_unsym as &RealFrequencyResponse<f32>, 1.0 / ratio);
            assert_vector_eq_with_reason_and_tolerance(&result_sym[..],
                                                       &result_unsym[..],
                                                       1e-2,
                                                       "Results should match with or without \
                                                        symmetry optimization");
        });
    }
}
