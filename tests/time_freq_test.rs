
#[cfg(feature = "slow_test")]
mod slow_test {
    use basic_dsp::{
        DataVector,
        TimeDomainOperations,
        FrequencyDomainOperations,
        ComplexVectorOperations,
        ComplexTimeVector32};
    use num::complex::Complex32;
    use tools::*;
       
    #[test]
    fn complex_fft_ifft_vector32_large() {
        for iteration in 0 .. 3 {
            let a = create_data_even(201511212, iteration, 10001, 20000);
            let points = (a.len() / 2) as f32;
            let delta = create_delta(3561159, iteration);
            let vector = ComplexTimeVector32::from_interleaved_with_delta(&a, delta);
            let freq = vector.plain_fft().complex_scale(Complex32::new(1.0 / points, 0.0));
            let result= freq.plain_ifft();
            assert_vector_eq_with_reason_and_tolerance(&a, &result.data(), 1e-4, "IFFT must invert FFT");
            assert_eq!(result.is_complex(), true);
            assert!((result.delta() - delta) < 1e-4);
        }
    }
}