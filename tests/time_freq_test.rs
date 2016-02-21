
#[cfg(feature = "slow_test")]
mod slow_test {
    use basic_dsp::*;
    use basic_dsp::conv_types::*;
    use basic_dsp::interop_facade::facade32::*;
    use num::complex::Complex32;
    use basic_dsp::window_functions::*;
    use tools::*;
    use std::f64::consts::PI;
    use std::os::raw::c_void;
       
    #[test]
    fn complex_plain_fft_plain_ifft_vector32_large() {
        for iteration in 0 .. 3 {
            let a = create_data_even(201511212, iteration, 10001, 20000);
            let points = (a.len() / 2) as f32;
            let delta = create_delta(3561159, iteration);
            let vector = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let freq = vector.plain_fft().unwrap().complex_scale(Complex32::new(1.0 / points, 0.0)).unwrap();
            let result= freq.plain_ifft().unwrap();
            assert_vector_eq_with_reason_and_tolerance(&a, &result.data(), 1e-4, "IFFT must invert FFT");
            assert_eq!(result.is_complex(), true);
        }
    }
    
    #[test]
    fn window_real_vs_complex_vector64() {
        let vector = new_sinusoid_vector();
        let complex = vector.clone().to_complex().unwrap();
        let complex_windowed = complex.apply_window(&HammingWindow::default()).unwrap();
        let real_windowed = vector.apply_window(&HammingWindow::default()).unwrap();
        assert_eq!(real_windowed.data(), complex_windowed.to_real().unwrap().data());
    }
    
    #[test]
    fn fft_vector64() {
        let vector = new_sinusoid_vector();
        let complex = vector.to_complex().unwrap();
        let fft = complex.fft().unwrap().magnitude().unwrap();
        // Expected data has been created with GNU Octave
        let expected: &[f64] = &[0.9292870138334854, 0.9306635099648193, 0.9348162621613968, 0.9418153274362542, 0.9517810621190216, 0.9648895430587848, 0.9813809812325847, 
            1.0015726905449405, 1.0258730936123666, 1.0548108445331859, 1.0890644245480268, 1.1295083134069603, 1.1772879726812928, 1.2339182289598294, 
            1.301437989279902, 1.3826534754026867, 1.4815340275011206, 1.6038793282853527, 1.7585157812279568, 1.9595783851339075, 2.2312382613655144,
            2.6185925930596348, 3.2167138068850805, 4.266740801517487, 6.612395930080317, 16.722094841103452, 23.622177170007486, 6.303697095969605, 
            3.404295797341746, 2.210968575749469, 1.5819040732615888, 1.246194569535693, 1.1367683431144981, 1.2461951854260518, 1.581903667468762, 
            2.210968517938972, 3.40429586037563, 6.303698000270388, 23.622176749343343, 16.722094721382852, 6.612395731182459, 4.266740005002631, 
            3.216713364304185, 2.618592497323997, 2.23123801189946, 1.9595783052844522, 1.7585159098930296, 1.6038802182584422, 1.4815339648659298,
            1.3826531545500815, 1.3014374693633786, 1.2339180461884898, 1.177287968900429, 1.1295077116182717, 1.0890636132326164, 1.0548115826822455, 
            1.0258732601724936, 1.0015721588901556, 0.9813817215431422, 0.9648899510832059, 0.951781283968659, 0.9418152796531379, 0.9348164516683282, 
            0.9306639008658044];
        assert_vector_eq(&expected, fft.data());
    }
    
    #[test]
    fn windowed_fft_vector64() {
        let vector = new_sinusoid_vector();
        let complex = vector.to_complex().unwrap();
        let fft = complex.windowed_fft(&HammingWindow::default()).unwrap().magnitude().unwrap();
        // Expected data has been created with GNU Octave
        let expected: &[f64] = &[0.07411808515197066, 0.07422272322333621, 0.07453841468679659, 0.07506988195440296, 0.07582541343880053, 
            0.07681696328361777, 0.07806061998281554, 0.07957802869766938, 0.08139483126598358, 0.08354572044699357, 0.0860733404576818, 
            0.08902894849492062, 0.09248035198400738, 0.09650916590034163, 0.10121825314935218, 0.10673436981991158, 0.11320884208865986, 
            0.12081079743775351, 0.12968456814874033, 0.13980104112377398, 0.15043320855074566, 0.15812824378762533, 0.14734167412188875,
            0.04249205387030338, 1.3969045117052756, 12.846276122172032, 14.9883680193849, 2.9493349550502477, 0.12854704555683252, 
            0.07502029346769173, 0.08639361740278063, 0.08219267572562121, 0.07964010102931768, 0.0821927994000342, 0.08639332990145131, 
            0.07502038796334394, 0.12854702502866483, 2.9493345269345794, 14.988367552410944, 12.846276615795297, 1.3969044843995686, 
            0.04249183333909037, 0.14734154414942854, 0.15812800570779315, 0.1504332002895389, 0.13980122658303065, 0.12968464820669545, 
            0.12081166709283893, 0.11320883727454012, 0.10673405922210086, 0.10121809170575471, 0.09650909536592596, 0.09248034765427705, 
            0.08902919211450319, 0.08607303669204663, 0.08354618763280168, 0.08139478829605633, 0.07957758721938324, 0.07806104687040545, 
            0.07681675159617712, 0.07582540168807066, 0.0750699488332293, 0.07453849632122858, 0.07422306880326296];
        assert_vector_eq(&expected, fft.data());
    }
    
    #[test]
    fn fft_ifft_vector64() {
        let vector = new_sinusoid_vector();
        let complex = vector.clone().to_complex().unwrap();
        let fft = complex.fft().unwrap();
        let ifft = fft.ifft().unwrap().to_real().unwrap();
        assert_vector_eq(vector.data(), ifft.data());   
    }
    
    #[test]
    fn windowed_fft_windowed_ifft_vector64() {
        let vector = new_sinusoid_vector();
        let complex = vector.clone().to_complex().unwrap();
        let fft = complex.windowed_fft(&HammingWindow::default()).unwrap();
        let ifft = fft.windowed_ifft(&HammingWindow::default()).unwrap().to_real().unwrap();
        assert_vector_eq(vector.data(), ifft.data());   
    }
    
    fn new_sinusoid_vector() -> RealTimeVector64 {
        let n: usize = 64;
        let f = 0.1;
        let phi = 0.25;
        let range: Vec<_> = (0..n).map(|v|(v as f64)*f).collect();
        let vector = RealTimeVector64::from_array(&range);
        let vector = 
            vector.real_scale(2.0 * PI)
            .and_then(|v|v.real_offset(phi))
            .and_then(|v|v.cos())
            .unwrap();
        vector
    }
    
    #[test]
    fn complex_fft_ifft_vector32_large() {
        for iteration in 0 .. 3 {
            let a = create_data_even(201511212, iteration, 10001, 20000);
            let delta = create_delta(3561159, iteration);
            let vector = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let freq = vector.fft().unwrap();
            let result= freq.ifft().unwrap();
            assert_vector_eq_with_reason_and_tolerance(&a, &result.data(), 1e-4, "IFFT must invert FFT");
            assert_eq!(result.is_complex(), true);
        }
    }
    
    #[test]
    fn compare_conv_freq_multiplication_for_rc() {
        for iteration in 0 .. 3 {
            let a = create_data_even(201511212, iteration, 1001, 2000);
            let delta = create_delta(3561159, iteration);
            let time = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let freq = time.clone().fft().unwrap();
            let points = time.points();
            let ratio = create_delta(201601091, iteration).abs() / 10.0; // Should get us a range [0.0 .. 1.0] and hopefully we are not that unlucky to get 0.0
            let time_res = time.convolve(&fun as &RealImpulseResponse<f32>, ratio, points).unwrap();
            let freq_res = freq.multiply_frequency_response(&fun as &RealFrequencyResponse<f32>, 1.0 / ratio).unwrap();
            let ifreq_res = freq_res.ifft().unwrap();
            let left = &ifreq_res.data();
            let right = &time_res.data();
            assert_vector_eq_with_reason_and_tolerance(&left, &right, 0.1, "Results should match independent if done in time or frequency domain");
        }
    }
    
    #[test]
    fn compare_conv_freq_multiplication_for_sinc() {
        for iteration in 0 .. 3 {
            let a = create_data_even(201511212, iteration, 2001, 4000);
            let delta = create_delta(3561159, iteration);
            let time = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let fun: SincFunction<f32> = SincFunction::new();
            let freq = time.clone().fft().unwrap();
            let points = time.points();
            let ratio = create_delta(201601093, iteration).abs() / 20.0 + 0.5; // Should get us a range [0.5 .. 1.0]
            let time_res = time.convolve(&fun as &RealImpulseResponse<f32>, ratio, points).unwrap();
            let freq_res = freq.multiply_frequency_response(&fun as &RealFrequencyResponse<f32>, 1.0 / ratio).unwrap();
            let ifreq_res = freq_res.ifft().unwrap();
            let left = &ifreq_res.data();
            let right = &time_res.data();
            assert_vector_eq_with_reason_and_tolerance(&left, &right, 0.2, "Results should match independent if done in time or frequency domain");
        }
    }
    
    #[test]
    fn compare_optimized_and_non_optimized_conv() {
        for iteration in 0 .. 3 {
            // This offset is small enough to now have a big impact on the results (for the RC function)
            // but the code will use a different non-optimized branch since it won't recognize ratio as an
            // integer
            let offset = 2e-6; 
            let a = create_data_even(201511212, iteration, 2002, 4000);
            let b = create_data_even(201601172, iteration, 50, 202);
            let delta = create_delta(3561159, iteration);
            let time = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let ratio = iteration as f32 + 1.0;
            let left = time.clone().convolve(&fun as &RealImpulseResponse<f32>, ratio, b.len()).unwrap();
            let right = time.convolve(&fun as &RealImpulseResponse<f32>, ratio + offset, b.len()).unwrap();
            assert_vector_eq_with_reason_and_tolerance(&left.data(), &right.data(), 0.1, "Results should match independent if done in optimized or non optimized code branch");
        }
    }
    
    #[test]
    fn compare_vector_conv_freq_multiplication() {
        for iteration in 0 .. 3 {
            let a = create_data_even(201601171, iteration, 502, 1000);
            let b = create_data_with_len(201601172, iteration, a.len());
            let delta = create_delta(201601173, iteration);
            let time1 = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let time2 = ComplexTimeVector32::from_interleaved_with_delta(&b, delta);
            let left = time1.clone().convolve_vector(&time2).unwrap();
            let freq1 = time1.fft().unwrap();
            let freq2 = time2.fft().unwrap();
            let freq_res = freq1.multiply_vector(&freq2).unwrap();
            let right = freq_res.ifft().unwrap();
            assert_vector_eq_with_reason_and_tolerance(
                left.data(), 
                &conv_swap(right.data())[0..left.len()], 
                0.2, 
                "Results should match independent if done in time or frequency domain");
        }
    }
    
    #[test]
    fn compare_smaller_vector_conv_with_zero_padded_conv() {
        for iteration in 0 .. 3 {
            let a = create_data_even(201601171, iteration, 1002, 2000);
            let b = create_data_even(201601172, iteration, 50, 202);
            let delta = create_delta(201601173, iteration);
            let time1 = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let time2 = ComplexTimeVector32::from_interleaved_with_delta(&b, delta);
            let left = time1.clone().convolve_vector(&time2).unwrap();
            let time2 = ComplexTimeVector32::from_interleaved_with_delta(&conv_zero_pad(&b, time1.len(), true), delta);
            let right = time1.convolve_vector(&time2).unwrap();
            assert_vector_eq_with_reason_and_tolerance(
                left.data(), 
                right.data(), 
                0.2, 
                "Results should match independent if done with a smaller vector or with a zero padded vector of the same size");
        }
    }
    
    #[test]
    fn compare_smaller_vector_conv_with_zero_padded_conv_real() {
        for iteration in 0 .. 3 {
            let a = create_data_even(201601171, iteration, 1002, 2000);
            let b = create_data_even(201601172, iteration, 50, 202);
            let delta = create_delta(201601173, iteration);
            let time1 = RealTimeVector32::from_array_with_delta(&a, delta);
            let time2 = RealTimeVector32::from_array_with_delta(&b, delta);
            let left = time1.clone().convolve_vector(&time2).unwrap();
            let time2 = RealTimeVector32::from_array_with_delta(&conv_zero_pad(&b, time1.len(), false), delta);
            let right = time1.convolve_vector(&time2).unwrap();
            assert_vector_eq_with_reason_and_tolerance(
                left.data(), 
                right.data(), 
                0.2, 
                "Results should match independent if done with a smaller vector or with a zero padded vector of the same size");
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
        }
        else {
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
        let half = 2 *(points / 2 + 1);
        for i in 0..len {
            if i < half {
                result[i] = data[len - half + i];
            }
            else if i >= len - half {
                result[i] = data[i - half];
            }
            else /* Center */ { 
                result[i] = data[i]; 
            }
        }
        result
    }
    
    /// Calls to another window with the only
    /// difference that it doesn't allow to make use of symmetry
    fn unsym_triag_window() -> ForeignWindowFunction {
        ForeignWindowFunction {
            window_data: 0,
            window_function: call_triag,
            is_symmetric: false
        }
    }
       
    extern fn call_triag(_arg: *const c_void, i: usize, points: usize) -> f32 {
        let triag: &WindowFunction<f32> = &TriangularWindow;
        triag.window(i, points)
    }
    
    /// Calls to another window with the only
    /// difference that it doesn't allow to make use of symmetry
    fn unsym_rc_mul() -> ForeignRealConvolutionFunction {
        ForeignRealConvolutionFunction {
            conv_data: 0,
            conv_function: call_freq_rc,
            is_symmetric: false
        }
    }
       
    extern fn call_freq_rc(_arg: *const c_void, x: f32) -> f32 {
        let rc: &RealFrequencyResponse<f32> = &RaisedCosineFunction::new(0.35);
        rc.calc(x)
    }
    
    #[test]
    fn compare_sym_optimized_window_with_normal_version() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(20160116, iteration, range.start, range.end);
            let delta = create_delta(201601161, iteration);
            let time = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let triag_sym = TriangularWindow;
            let triag_unsym = unsym_triag_window();
            let result_sym = time.clone().apply_window(&triag_sym).unwrap();
            let result_unsym = time.apply_window(&triag_unsym).unwrap();
            let left = &result_sym.data();
            let right = &result_unsym.data();
            assert_vector_eq_with_reason_and_tolerance(&left, &right, 1e-2, "Results should match with or without symmetry optimization");
        });
    }
    
    #[test]
    fn compare_sym_optimized_freq_mul_with_normal_version() {
        parameterized_vector_test(|iteration, range| {
            let a = create_data_even(20160116, iteration, range.start, range.end);
            let delta = create_delta(201601161, iteration);
            let freq = ComplexFreqVector32::from_interleaved_with_delta(&a, delta);
            let rc_sym: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let rc_unsym = unsym_rc_mul();
            let ratio = create_delta(201601093, iteration).abs() / 20.0 + 0.5; // Should get us a range [0.5 .. 1.0]
            let result_sym = freq.clone().multiply_frequency_response(&rc_sym as &RealFrequencyResponse<f32>, 1.0 / ratio).unwrap();
            let result_unsym = freq.multiply_frequency_response(&rc_unsym as &RealFrequencyResponse<f32>, 1.0 / ratio).unwrap();
            let left = &result_sym.data();
            let right = &result_unsym.data();
            assert_vector_eq_with_reason_and_tolerance(&left, &right, 1e-2, "Results should match with or without symmetry optimization");
        });
    }
    
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
            assert_vector_eq_with_reason_and_tolerance(&left.data(), &right.data(), 0.1, "Results should match independent if done with interpolatei or interpolatef");
        }
    }
    
    #[test]
    fn compare_real_and_complex_interpolatef() {
        for iteration in 0 .. 3 {
            let a = create_data_even(201511212, iteration, 2002, 4000);
            let delta = create_delta(3561159, iteration);
            let real = RealTimeVector32::from_array_with_delta(&a, delta);
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let factor = iteration as f32 + 1.0;
            let left = real.clone().interpolatef(&fun as &RealImpulseResponse<f32>, factor, 0.0, 10).unwrap();
            let right = real.to_complex().unwrap().interpolatef(&fun as &RealImpulseResponse<f32>, factor, 0.0, 10).unwrap();
            let right = right.to_real().unwrap();
            assert_vector_eq_with_reason_and_tolerance(&left.data(), &right.data(), 0.1, "Results should match independent if done in real or complex number space");
        }
    }
    
    #[test]
    fn compare_real_and_complex_interpolatei() {
        for iteration in 0 .. 3 {
            let a = create_data_even(201511212, iteration, 2002, 4000);
            let delta = create_delta(3561159, iteration);
            let real = RealTimeVector32::from_array_with_delta(&a, delta);
            let fun: RaisedCosineFunction<f32> = RaisedCosineFunction::new(0.35);
            let factor = iteration as u32 + 1;
            let left = real.clone().interpolatei(&fun as &RealFrequencyResponse<f32>, factor).unwrap();
            let right = real.to_complex().unwrap().interpolatei(&fun as &RealFrequencyResponse<f32>, factor).unwrap();
            let right = right.to_real().unwrap();
            assert_vector_eq_with_reason_and_tolerance(&left.data(), &right.data(), 0.1, "Results should match independent if done in real or complex number space");
        }
    }
}