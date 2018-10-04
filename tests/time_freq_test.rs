extern crate basic_dsp;
extern crate num;
extern crate rand;
pub mod tools;

mod time_freq_test {
    use basic_dsp::window_functions::*;
    use basic_dsp::*;
    use num::complex::*;
    use std::f64::consts::PI;
    use tools::*;

    #[test]
    fn complex_plain_fft_plain_ifft_vector32_large() {
        for iteration in 0..3 {
            let a = create_data_even(201511212, iteration, 10001, 20000);
            let points = (a.len() / 2) as f32;
            let delta = create_delta(3561159, iteration);
            let mut buffer = SingleBuffer::new();
            let mut vector = a.clone().to_complex_time_vec_par();
            vector.set_delta(delta);
            let mut freq = vector.plain_fft(&mut buffer);
            freq.scale(Complex32::new(1.0 / points, 0.0));
            let result = freq.plain_ifft(&mut buffer);
            assert_vector_eq_with_reason_and_tolerance(
                &a,
                &result[..],
                1e-4,
                "IFFT must invert FFT",
            );
            assert_eq!(result.is_complex(), true);
        }
    }

    #[test]
    fn window_real_vs_complex_vector64() {
        let mut vector = new_sinusoid_vector();
        let mut complex = vector.clone().to_complex().unwrap();
        complex.apply_window(&HammingWindow::default());
        let real = complex.to_real();
        vector.apply_window(&HammingWindow::default());
        assert_eq!(&real[..], &vector[..]);
    }

    #[test]
    fn fft_vector64() {
        let vector = new_sinusoid_vector();
        let mut buffer = SingleBuffer::new();
        let complex = vector.clone().to_complex().unwrap();
        let freq = complex.fft(&mut buffer);
        let result = freq.magnitude();
        // Expected data has been created with GNU Octave
        let expected: &[f64] = &[
            0.9292870138334854,
            0.9306635099648193,
            0.9348162621613968,
            0.9418153274362542,
            0.9517810621190216,
            0.9648895430587848,
            0.9813809812325847,
            1.0015726905449405,
            1.0258730936123666,
            1.0548108445331859,
            1.0890644245480268,
            1.1295083134069603,
            1.1772879726812928,
            1.2339182289598294,
            1.301437989279902,
            1.3826534754026867,
            1.4815340275011206,
            1.6038793282853527,
            1.7585157812279568,
            1.9595783851339075,
            2.2312382613655144,
            2.6185925930596348,
            3.2167138068850805,
            4.266740801517487,
            6.612395930080317,
            16.722094841103452,
            23.622177170007486,
            6.303697095969605,
            3.404295797341746,
            2.210968575749469,
            1.5819040732615888,
            1.246194569535693,
            1.1367683431144981,
            1.2461951854260518,
            1.581903667468762,
            2.210968517938972,
            3.40429586037563,
            6.303698000270388,
            23.622176749343343,
            16.722094721382852,
            6.612395731182459,
            4.266740005002631,
            3.216713364304185,
            2.618592497323997,
            2.23123801189946,
            1.9595783052844522,
            1.7585159098930296,
            1.6038802182584422,
            1.4815339648659298,
            1.3826531545500815,
            1.3014374693633786,
            1.2339180461884898,
            1.177287968900429,
            1.1295077116182717,
            1.0890636132326164,
            1.0548115826822455,
            1.0258732601724936,
            1.0015721588901556,
            0.9813817215431422,
            0.9648899510832059,
            0.951781283968659,
            0.9418152796531379,
            0.9348164516683282,
            0.9306639008658044,
        ];
        assert_vector_eq(&expected, &result[..]);
    }

    #[test]
    fn windowed_fft_vector64() {
        let vector = new_sinusoid_vector();
        let mut buffer = SingleBuffer::new();
        let complex = vector.clone().to_complex().unwrap();
        let freq = complex.windowed_fft(&mut buffer, &HammingWindow::default());
        let result = freq.magnitude();
        // Expected data has been created with GNU Octave
        let expected: &[f64] = &[
            0.07411808515197066,
            0.07422272322333621,
            0.07453841468679659,
            0.07506988195440296,
            0.07582541343880053,
            0.07681696328361777,
            0.07806061998281554,
            0.07957802869766938,
            0.08139483126598358,
            0.08354572044699357,
            0.0860733404576818,
            0.08902894849492062,
            0.09248035198400738,
            0.09650916590034163,
            0.10121825314935218,
            0.10673436981991158,
            0.11320884208865986,
            0.12081079743775351,
            0.12968456814874033,
            0.13980104112377398,
            0.15043320855074566,
            0.15812824378762533,
            0.14734167412188875,
            0.04249205387030338,
            1.3969045117052756,
            12.846276122172032,
            14.9883680193849,
            2.9493349550502477,
            0.12854704555683252,
            0.07502029346769173,
            0.08639361740278063,
            0.08219267572562121,
            0.07964010102931768,
            0.0821927994000342,
            0.08639332990145131,
            0.07502038796334394,
            0.12854702502866483,
            2.9493345269345794,
            14.988367552410944,
            12.846276615795297,
            1.3969044843995686,
            0.04249183333909037,
            0.14734154414942854,
            0.15812800570779315,
            0.1504332002895389,
            0.13980122658303065,
            0.12968464820669545,
            0.12081166709283893,
            0.11320883727454012,
            0.10673405922210086,
            0.10121809170575471,
            0.09650909536592596,
            0.09248034765427705,
            0.08902919211450319,
            0.08607303669204663,
            0.08354618763280168,
            0.08139478829605633,
            0.07957758721938324,
            0.07806104687040545,
            0.07681675159617712,
            0.07582540168807066,
            0.0750699488332293,
            0.07453849632122858,
            0.07422306880326296,
        ];
        assert_vector_eq(&expected, &result[..]);
    }

    #[test]
    fn fft_ifft_vector64() {
        let vector = new_sinusoid_vector();
        let mut buffer = SingleBuffer::new();
        let complex = vector.clone().to_complex().unwrap();
        let fft = complex.fft(&mut buffer);
        let ifft = fft.ifft(&mut buffer).to_real();
        assert_vector_eq(&vector[..], &ifft[..]);
    }

    #[test]
    fn windowed_fft_windowed_ifft_vector64() {
        let vector = new_sinusoid_vector();
        let mut buffer = SingleBuffer::new();
        let complex = vector.clone().to_complex().unwrap();
        let fft = complex.windowed_fft(&mut buffer, &HammingWindow::default());
        let ifft = fft
            .windowed_ifft(&mut buffer, &HammingWindow::default())
            .to_real();
        assert_vector_eq(&vector[..], &ifft[..]);
    }

    fn new_sinusoid_vector() -> RealTimeVec64 {
        let n: usize = 64;
        let f = 0.1;
        let phi = 0.25;
        let range: Vec<_> = (0..n).map(|v| (v as f64) * f).collect();
        let mut vector = range.to_real_time_vec_par();
        vector.scale(2.0 * PI);
        vector.offset(phi);
        vector.cos();
        vector
    }

    #[test]
    fn complex_fft_ifft_vector32_large() {
        for iteration in 0..3 {
            let a = create_data_even(201511212, iteration, 10001, 20000);
            let delta = create_delta(3561159, iteration);
            let mut vector = a.clone().to_complex_time_vec_par();
            vector.set_delta(delta);
            let mut buffer = SingleBuffer::new();
            let freq = vector.fft(&mut buffer);
            let result = freq.ifft(&mut buffer);
            assert_vector_eq_with_reason_and_tolerance(
                &a,
                &result[..],
                1e-4,
                "IFFT must invert FFT",
            );
            assert_eq!(result.is_complex(), true);
        }
    }
}
