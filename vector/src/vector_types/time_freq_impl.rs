use super::definitions::{
    DataVector,
    VecResult,
    DataVectorDomain,
    ErrorReason,
    ComplexVectorOps,
    RealVectorOps};
use super::{
    GenericDataVector,
    RealTimeVector,
    ComplexFreqVector,
    ComplexTimeVector,
    round_len};
use rustfft::FFT;
use RealNumber;
use std::ptr;
use window_functions::WindowFunction;
use num::complex::Complex;
use std::ops::{Mul, Div};

/// Defines all operations which are valid on `DataVectors` containing time domain data.
/// # Failures
/// All operations in this trait fail with `VectorMustBeInTimeDomain` if the vector isn't in time domain.
pub trait TimeDomainOperations<T> : DataVector<T> 
    where T : RealNumber {
    type FreqPartner;
    
    /// Performs a Fast Fourier Transformation transforming a time domain vector
    /// into a frequency domain vector. 
    /// 
    /// This version of the FFT neither applies a window nor does it scale the 
    /// vector.
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{ComplexTimeVector32, TimeDomainOperations, DataVector};
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254]);
    /// let result = vector.plain_fft().expect("Ignoring error handling in examples");
    /// let actual = result.data();
    /// let expected = &[0.0, 0.0, 3.0, 0.0, 0.0, 0.0];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!((actual[i] - expected[i]).abs() < 1e-4);
    /// }
    /// ```
    fn plain_fft(self) -> VecResult<Self::FreqPartner>;
    
    /// Performs a Fast Fourier Transformation transforming a time domain vector
    /// into a frequency domain vector. 
    /// # Unstable
    /// FFTs of real vectors are unstable.
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{ComplexTimeVector32, TimeDomainOperations, DataVector};
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254]);
    /// let result = vector.fft().expect("Ignoring error handling in examples");
    /// let actual = result.data();
    /// let expected = &[0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!((actual[i] - expected[i]).abs() < 1e-4);
    /// }
    /// ```
    fn fft(self) -> VecResult<Self::FreqPartner>;
    
    /// Applies a FFT window and performs a Fast Fourier Transformation transforming a time domain vector
    /// into a frequency domain vector. 
    fn windowed_fft(self, window: &WindowFunction<T>) -> VecResult<Self::FreqPartner>;
    
    /// Applies a window to the data vector.
    fn apply_window(self, window: &WindowFunction<T>) -> VecResult<Self>;
    
    /// Removes a window from the data vector.
    fn unapply_window(self, window: &WindowFunction<T>) -> VecResult<Self>;
}

/// Defines all operations which are valid on `DataVectors` containing real time domain data.
/// # Failures
/// All operations in this trait fail with `VectorMustBeInTimeDomain` if the vector isn't in time domain or
/// with `VectorMustHaveAnOddLength` if `self.points()` isn't and odd number.
pub trait SymmetricTimeDomainOperations<T> : DataVector<T> 
    where T : RealNumber {
    type FreqPartner;
    
    /// Performs a Symmetric Fast Fourier Transformation under the assumption that `self`
    /// is symmetric around the center. This assumption
    /// isn't verified and no error is raised if the vector isn't symmetric.
    ///
    /// This version of the IFFT neither applies a window nor does it scale the 
    /// vector.
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `VectorMustBeReal`: if `self` is in complex number space.
    /// 2. `VectorMustBeInTimeDomain`: if `self` is in frequency domain.
    /// 
    /// # Unstable
    /// Symmetric IFFTs are unstable and may only work under certain conditions.
    fn plain_sfft(self) -> VecResult<Self::FreqPartner>;
    
    /// Performs a Symmetric Fast Fourier Transformation under the assumption that `self`
    /// is symmetric around the center. This assumption
    /// isn't verified and no error is raised if the vector isn't symmetric.
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `VectorMustBeReal`: if `self` is in complex number space.
    /// 2. `VectorMustBeInTimeDomain`: if `self` is in frequency domain.
    /// 
    /// # Unstable
    /// Symmetric IFFTs are unstable and may only work under certain conditions.
    fn sfft(self) -> VecResult<Self::FreqPartner>;
    
    /// Performs a Symmetric Fast Fourier Transformation under the assumption that `self`
    /// is symmetric around the center. This assumption
    /// isn't verified and no error is raised if the vector isn't symmetric.
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `VectorMustBeReal`: if `self` is in complex number space.
    /// 2. `VectorMustBeInTimeDomain`: if `self` is in frequency domain.
    /// 
    /// # Unstable
    /// Symmetric IFFTs are unstable and may only work under certain conditions.
    fn windowed_sfft(self, window: &WindowFunction<T>) -> VecResult<Self::FreqPartner>;
}

/// Defines all operations which are valid on `DataVectors` containing frequency domain data.
/// # Failures
/// All operations in this trait fail with `VectorMustBeInFrquencyDomain` or `VectorMustBeComplex` 
/// if the vector isn't in frequency domain and complex number space.
pub trait FrequencyDomainOperations<T> : DataVector<T> 
    where T : RealNumber {
    type ComplexTimePartner;
    
    /// Performs an Inverse Fast Fourier Transformation transforming a frequency domain vector
    /// into a time domain vector.
    /// 
    /// This version of the IFFT neither applies a window nor does it scale the 
    /// vector.
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{ComplexFreqVector32, FrequencyDomainOperations, DataVector};
    /// let vector = ComplexFreqVector32::from_interleaved(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    /// let result = vector.plain_ifft().expect("Ignoring error handling in examples");
    /// let actual = result.data();
    /// let expected = &[1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!((actual[i] - expected[i]).abs() < 1e-4);
    /// }
    /// ```
    fn plain_ifft(self) -> VecResult<Self::ComplexTimePartner>;
        
    /// This function mirrors the spectrum vector to transform a symmetric spectrum
    /// into a full spectrum with the DC element at index 0 (no FFT shift/swap halves).
    ///
    /// The argument indicates whether the resulting real vector should have `2*N` or `2*N-1` points.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{ComplexFreqVector32, FrequencyDomainOperations, DataVector};
    /// let vector = ComplexFreqVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let result = vector.mirror().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, -6.0, 3.0, -4.0], result.data());
    /// ```
    fn mirror(self) -> VecResult<Self>;
    
    /// Performs an Inverse Fast Fourier Transformation transforming a frequency domain vector
    /// into a time domain vector.
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{ComplexFreqVector32, FrequencyDomainOperations, DataVector};
    /// let vector = ComplexFreqVector32::from_interleaved(&[0.0, 0.0, 0.0, 0.0, 3.0, 0.0]);
    /// let result = vector.ifft().expect("Ignoring error handling in examples");
    /// let actual = result.data();
    /// let expected = &[1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!((actual[i] - expected[i]).abs() < 1e-4);
    /// }
    /// ```
    fn ifft(self) -> VecResult<Self::ComplexTimePartner>;
    
    /// Performs an Inverse Fast Fourier Transformation transforming a frequency domain vector
    /// into a time domain vector and removes the FFT window.
    fn windowed_ifft(self, window: &WindowFunction<T>) -> VecResult<Self::ComplexTimePartner>;
    
    /// Swaps vector halves after a Fourier Transformation.
    fn fft_shift(self) -> VecResult<Self>;
    
    /// Swaps vector halves before an Inverse Fourier Transformation.
    fn ifft_shift(self) -> VecResult<Self>;
}

/// Defines all operations which are valid on `DataVectors` containing frequency domain data and
/// the data is assumed to half of complex conjugate symmetric spectrum round 0 Hz where 
/// the 0 Hz element itself is real.
/// # Failures
/// All operations in this trait fail with `VectorMustBeInFrquencyDomain`
/// if the vector isn't in frequency domain or with `VectorMustBeConjSymmetric` if the first element (0Hz)
/// isn't real.
pub trait SymmetricFrequencyDomainOperations<T> : DataVector<T> 
    where T : RealNumber {
    type RealTimePartner;
    /// Performs a Symmetric Inverse Fast Fourier Transformation under the assumption that `self`
    /// contains half of a symmetric spectrum starting from 0 Hz. This assumption
    /// isn't verified and no error is raised if the spectrum isn't symmetric. The reason
    /// for this is that there is no robust verification possible.
    ///
    /// The argument indicates whether the resulting real vector should have `2*N` or `2*N-1` points.
    ///
    /// This version of the IFFT neither applies a window nor does it scale the 
    /// vector.
    fn plain_sifft(self) -> VecResult<Self::RealTimePartner>;
    
    /// Performs a Symmetric Inverse Fast Fourier Transformation under the assumption that `self`
    /// contains half of a symmetric spectrum starting from 0 Hz. This assumption
    /// isn't verified and no error is raised if the spectrum isn't symmetric. The reason
    /// for this is that there is no robust verification possible.
    ///
    /// The argument indicates whether the resulting real vector should have `2*N` or `2*N-1` points.
    fn sifft(self) -> VecResult<Self::RealTimePartner>;
    
    /// Performs a Symmetric Inverse Fast Fourier Transformation (SIFFT) and removes the FFT window. 
    /// The SIFFT is performed under the assumption that `self`
    /// contains half of a symmetric spectrum starting from 0 Hz. This assumption
    /// isn't verified and no error is raised if the spectrum isn't symmetric. The reason
    /// for this is that there is no robust verification possible.
    ///
    /// The argument indicates whether the resulting real vector should have `2*N` or `2*N-1` points.
    fn windowed_sifft(self, window: &WindowFunction<T>) -> VecResult<Self::RealTimePartner>;
}

macro_rules! define_time_domain_forward {
    ($($name:ident, $data_type:ident);*) => {
        $( 
            impl TimeDomainOperations<$data_type> for $name<$data_type> {
                type FreqPartner = ComplexFreqVector<$data_type>;
                fn plain_fft(self) -> VecResult<Self::FreqPartner> {
                    Self::FreqPartner::from_genres(self.to_gen().plain_fft())
                }
                
                fn apply_window(self, window: &WindowFunction<$data_type>) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().apply_window(window))
                }
    
                fn unapply_window(self, window: &WindowFunction<$data_type>) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().unapply_window(window))
                }
                
                fn fft(self) -> VecResult<Self::FreqPartner> {
                    Self::FreqPartner::from_genres(self.to_gen().fft())
                }
                
                fn windowed_fft(self, window: &WindowFunction<$data_type>) -> VecResult<Self::FreqPartner> {
                    Self::FreqPartner::from_genres(self.to_gen().windowed_fft(window))
                }
            }
        )*
    }
}

macro_rules! implement_window_function {
    ($($name:ident, $data_type:ident, $operation: ident);*) => {
        $( 
            fn $name(self, window: &WindowFunction<$data_type>) -> VecResult<Self> {
                    assert_time!(self);
                    if self.is_complex {
                        Ok(self.multiply_window_priv(
                            window.is_symmetric(),
                            |array|Self::array_to_complex_mut(array),
                            window,
                            |f,i,p|Complex::<$data_type>::new(1.0.$operation(f.window(i, p)), 0.0)))
                    } else {
                        Ok(self.multiply_window_priv(
                            window.is_symmetric(),
                            |array|array,
                            window,
                            |f,i,p|1.0.$operation(f.window(i, p))))
                    }
                }
        )*
    }
}

macro_rules! add_time_freq_impl {
    ($($data_type:ident);*)
     =>
     {     
        $(
            impl TimeDomainOperations<$data_type> for GenericDataVector<$data_type> {
                type FreqPartner = GenericDataVector<$data_type>;
                fn plain_fft(mut self) -> VecResult<Self> {
                    assert_time!(self);
                    if self.is_complex {
                        {
                            let points = self.points();
                            let rbw = (points as $data_type) * self.delta;
                            self.delta = rbw;
                            let mut fft = FFT::new(points, false);
                            let signal = &self.data;
                            let spectrum = temp_mut!(self, signal.len());
                            let signal = Self::array_to_complex(signal);
                            let spectrum = Self::array_to_complex_mut(spectrum);
                            fft.process(&signal[0..points], &mut spectrum[0..points]);
                            self.domain = DataVectorDomain::Frequency;
                        }
                        
                        Ok(self.swap_data_temp())
                    } else {
                        self.to_complex().and_then(|v|v.plain_fft())
                    }
                }
                
                implement_window_function!(apply_window, $data_type, mul; unapply_window, $data_type, div);
                
                fn fft(self) -> VecResult<Self::FreqPartner> {
                    self.plain_fft()
                    .and_then(|v|v.fft_shift())
                }
                
                fn windowed_fft(self, window: &WindowFunction<$data_type>) -> VecResult<Self::FreqPartner> {
                    self.apply_window(window)
                    .and_then(|v|v.plain_fft())
                    .and_then(|v|v.fft_shift())
                }
            }
            
            impl SymmetricTimeDomainOperations<$data_type> for GenericDataVector<$data_type> {
                type FreqPartner = GenericDataVector<$data_type>;
                fn plain_sfft(self) -> VecResult<GenericDataVector<$data_type>> {
                    reject_if!(self, self.points() % 2 == 0, ErrorReason::VectorMustHaveAnOddLength);
                    self.to_complex()
                    .and_then(|v|v.plain_fft())
                    .and_then(|v|v.unmirror())
                }
                
                fn sfft(self) -> VecResult<GenericDataVector<$data_type>> {
                    reject_if!(self, self.points() % 2 == 0, ErrorReason::VectorMustHaveAnOddLength);
                    self.to_complex()
                    .and_then(|v|v.plain_fft())
                    .and_then(|v|v.unmirror())
                }
                
                fn windowed_sfft(self, window: &WindowFunction<$data_type>) -> VecResult<GenericDataVector<$data_type>> {
                    reject_if!(self, self.points() % 2 == 0, ErrorReason::VectorMustHaveAnOddLength);
                    self.to_complex()
                    .and_then(|v|v.apply_window(window))
                    .and_then(|v|v.plain_fft())
                    .and_then(|v|v.unmirror())
                }
            }
            
            impl GenericDataVector<$data_type> {
                fn unmirror(mut self) -> VecResult<Self> {
                    let len = self.len();
                    let len = len / 2 + 1;
                    self.valid_len = len;
                    Ok(self)
                }
            }
            
            define_time_domain_forward!(ComplexTimeVector, $data_type; RealTimeVector, $data_type);
            
            impl SymmetricTimeDomainOperations<$data_type> for RealTimeVector<$data_type> {
                type FreqPartner = ComplexFreqVector<$data_type>;
                fn plain_sfft(self) -> VecResult<Self::FreqPartner> {
                    Self::FreqPartner::from_genres(self.to_gen().plain_sfft())
                }
                
                fn sfft(self) -> VecResult<Self::FreqPartner> {
                    Self::FreqPartner::from_genres(self.to_gen().sfft())
                }
                
                fn windowed_sfft(self, window: &WindowFunction<$data_type>) -> VecResult<Self::FreqPartner> {
                    Self::FreqPartner::from_genres(self.to_gen().windowed_sfft(window))
                }
            }
            
            impl FrequencyDomainOperations<$data_type> for GenericDataVector<$data_type> {
                type ComplexTimePartner = GenericDataVector<$data_type>;
                fn plain_ifft(mut self) -> VecResult<GenericDataVector<$data_type>> {
                     {
                        assert_complex!(self);
                        assert_freq!(self);
                        let points = self.points();
                        let mut fft = FFT::new(points, true);
                        let delta = (points as $data_type)  / self.delta;
                        self.delta = delta;
                        let signal = &self.data;
                        let spectrum = temp_mut!(self, signal.len());
                        let signal = Self::array_to_complex(signal);
                        let spectrum = Self::array_to_complex_mut(spectrum);
                        fft.process(&signal[0..points], &mut spectrum[0..points]);
                        self.domain = DataVectorDomain::Time;
                    }
                    Ok(self.swap_data_temp())
                }
                
                fn mirror(mut self) -> VecResult<Self> {
                    assert_freq!(self);
                    assert_complex!(self);
                    {
                        let len = self.len();
                        let data = &self.data;
                        let step = 2;
                        let temp_len = 2 * len - step;
                        let mut temp = temp_mut!(self, temp_len);
                        {
                            let data = &data[step] as *const $data_type;
                            let target = &mut temp[step] as *mut $data_type;
                            unsafe {
                                ptr::copy(data, target, len - step);
                            }
                        }
                        {
                            let data = &data[0] as *const $data_type;
                            let target = &mut temp[0] as *mut $data_type;
                            unsafe {
                                ptr::copy(data, target, step);
                                }
                            }
                            let mut j = step + 1;
                            let mut i = temp_len - 1;
                            while i >= len {
                                temp[i] = -data[j];
                                temp[i - 1] = data[j - 1];
                                i -= 2;
                                j += 2;
                            }
                        
                            self.valid_len = temp_len;
                    }
                    Ok(self.swap_data_temp())
                }
                
                fn ifft(self) -> VecResult<Self::ComplexTimePartner> {
                    
                    let points = self.points();
                    self.real_scale(1.0 / points as $data_type)
                    .and_then(|v| v.ifft_shift())
                    .and_then(|v| v.plain_ifft())
                }
                
                fn windowed_ifft(self, window: &WindowFunction<$data_type>) -> VecResult<Self::ComplexTimePartner> {
                    let points = self.points();
                    self.real_scale(1.0 / points as $data_type)
                    .and_then(|v| v.ifft_shift())
                    .and_then(|v| v.plain_ifft())
                    .and_then(|v|v.unapply_window(window))
                }
                
                fn fft_shift(self) -> VecResult<Self> {
                    self.swap_halves_priv(true)
                }
                
                fn ifft_shift(self) -> VecResult<Self> {
                    self.swap_halves_priv(false)
                }
            }
            
            impl SymmetricFrequencyDomainOperations<$data_type> for GenericDataVector<$data_type> {
                type RealTimePartner = GenericDataVector<$data_type>;
                
                fn plain_sifft(self) -> VecResult<GenericDataVector<$data_type>> {
                    assert_complex!(self);
                    assert_freq!(self);
                    reject_if!(self, self.points() > 0 && self.data[1].abs() > 1e-10, ErrorReason::VectorMustBeConjSymmetric);
                    self.mirror()
                        .and_then(|v|v.plain_ifft())
                        .and_then(|v|v.to_real())
                }
                
                fn sifft(self) -> VecResult<Self::RealTimePartner> {
                    let points = self.points();
                    self.real_scale(1.0 / points as $data_type)
                    .and_then(|v| v.ifft_shift())
                    .and_then(|v| v.plain_sifft())
                }
                fn windowed_sifft(self, window: &WindowFunction<$data_type>) -> VecResult<Self::RealTimePartner> {
                    let points = self.points();
                    self.real_scale(1.0 / points as $data_type)
                    .and_then(|v| v.ifft_shift())
                    .and_then(|v| v.plain_sifft())
                    .and_then(|v|v.unapply_window(window))
                }
            }
            
            impl FrequencyDomainOperations<$data_type> for ComplexFreqVector<$data_type> {
                type ComplexTimePartner = ComplexTimeVector<$data_type>;
                fn plain_ifft(self) -> VecResult<ComplexTimeVector<$data_type>> {
                    Self::ComplexTimePartner::from_genres(self.to_gen().plain_ifft())
                }
                                
                fn mirror(self) -> VecResult<ComplexFreqVector<$data_type>> {
                    Self::from_genres(self.to_gen().mirror())
                }
                
                fn ifft(self) -> VecResult<Self::ComplexTimePartner> {
                    Self::ComplexTimePartner::from_genres(self.to_gen().ifft())
                }
                
                fn windowed_ifft(self, window: &WindowFunction<$data_type>) -> VecResult<Self::ComplexTimePartner> {
                    Self::ComplexTimePartner::from_genres(self.to_gen().windowed_ifft(window))
                }
                
                fn fft_shift(self) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().fft_shift())
                }
                
                fn ifft_shift(self) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().ifft_shift())
                }
            }
            
            impl SymmetricFrequencyDomainOperations<$data_type> for ComplexFreqVector<$data_type> {
                type RealTimePartner = RealTimeVector<$data_type>;
                fn plain_sifft(self) -> VecResult<Self::RealTimePartner> {
                    Self::RealTimePartner::from_genres(self.to_gen().plain_sifft())
                }
                
                fn sifft(self) -> VecResult<Self::RealTimePartner> {
                    Self::RealTimePartner::from_genres(self.to_gen().sifft())
                }
                
                fn windowed_sifft(self, window: &WindowFunction<$data_type>) -> VecResult<Self::RealTimePartner> {
                    Self::RealTimePartner::from_genres(self.to_gen().windowed_sifft(window))
                }
            }
        )*
     }
}
add_time_freq_impl!(f32; f64);

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{
        ComplexTimeVector32,
        ComplexFreqVector32,
        RealTimeVector32,
        ComplexVectorOps,
        DataVector};
    use num::complex::Complex32;
    use window_functions::*;
    use RealNumber;
    use std::fmt::Debug;
    
    fn assert_eq_tol<T>(left: &[T], right: &[T], tol: T) 
        where T: RealNumber + Debug {
        assert_eq!(left.len(), right.len());
        for i in 0..left.len() {
            if (left[i] - right[i]).abs() > tol {
                panic!("assertion failed: {:?} != {:?}", left, right);
            }
        }
    }

    #[test]
    fn plain_fft_test()
    {
        let vector = ComplexTimeVector32::from_interleaved(&[1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254]);
        let result = vector.plain_fft().unwrap();
        let expected = &[0.0, 0.0, 3.0, 0.0, 0.0, 0.0];
        let result = result.data();
        for i in 0..result.len() {
            assert!((result[i] - expected[i]).abs() < 1e-4);
        }
    }
    
    #[test]
    fn plain_ifft_test()
    {
        let vector = ComplexFreqVector32::from_interleaved(&[0.0, 0.0, 3.0, 0.0, 0.0, 0.0]);
        let result = vector.plain_ifft().unwrap();
        let expected = &[3.0, 0.0, -1.5, 2.598076, -1.5, -2.598076];
        let result = result.data();
        for i in 0..result.len() {
            assert!((result[i] - expected[i]).abs() < 1e-4);
        }
    }
    
    #[test]
    fn plain_sfft_test()
    {
        let vector = RealTimeVector32::from_array(&[1.0, -0.5, -0.5]);
        let result = vector.plain_sfft().unwrap();
        let expected = &[0.0, 0.0, 1.5, 0.0];
        let result = result.data();
        for i in 0..result.len() {
            assert!((result[i] - expected[i]).abs() < 1e-4);
        }
    }
    
    #[test]
    fn mirror_test()
    {
        let vector = ComplexFreqVector32::from_interleaved(&[0.0, 0.0, 1.5, 0.0]);
        let result = vector.mirror().unwrap();
        let expected = &[0.0, 0.0, 1.5, 0.0, 1.5, 0.0];
        assert_eq!(result.data(), expected);
    }
    
    #[test]
    fn plain_sifft_test() {
        let vector = ComplexFreqVector32::from_interleaved(&[0.0, 0.0, 1.5, 0.0]);
        let result = vector.plain_sifft().unwrap();
        let expected = &[3.0, -1.5, -1.5];
        assert_eq!(result.is_complex(), false);
        let result = result.data();
        for i in 0..result.len() {
            assert!((result[i] - expected[i]).abs() < 1e-4);
        }
    }
    
    #[test]
    fn sfft_test()
    {
        let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let result = vector.sfft().unwrap();
        let expected = &[28.0, 0.0, -3.5, 7.2678, -3.5000, 2.7912, -3.5000, 0.7989];
        let result = result.data();
        for i in 0..result.len() {
            assert!((result[i] - expected[i]).abs() < 1e-4);
        }
    }
    
    #[test]
    fn ifft_shift_test()
    {
        let mut a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let c = ComplexFreqVector32::from_interleaved(&mut a);
        let r = c.ifft_shift().unwrap();
        assert_eq!(r.data(), &[3.0, 4.0, 5.0, 6.0, 1.0, 2.0]);
    }
    
    #[test]
    fn fft_shift_test()
    {
        let mut a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let c = ComplexFreqVector32::from_interleaved(&mut a);
        let r = c.fft_shift().unwrap();
        assert_eq!(r.data(), &[5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
    }
    
    #[test]
    fn window_test()
    {
        let c = ComplexTimeVector32::from_constant(Complex32::new(1.0, 0.0), 10);
        let triag = TriangularWindow;
        let r = c.apply_window(&triag).unwrap().magnitude().unwrap();
        let expected = [0.1, 0.3, 0.5, 0.7, 0.9, 0.9, 0.7, 0.5, 0.3, 0.1];
        let r = r.data();
        assert_eq_tol(r, &expected, 1e-4);
    }
    
    #[test]
    fn window_odd_test()
    {
        let c = ComplexTimeVector32::from_constant(Complex32::new(1.0, 0.0), 9);
        let triag = TriangularWindow;
        let r = c.apply_window(&triag).unwrap().magnitude().unwrap();
        let expected = [0.111, 0.333, 0.555, 0.777, 1.0, 0.777, 0.555, 0.333, 0.111];
        let r = r.data();
        assert_eq_tol(r, &expected, 1e-2);
    }
    
    #[test]
    fn unapply_window_test()
    {
        let c = RealTimeVector32::from_array(&[0.1, 0.3, 0.5, 0.7, 0.9, 0.9, 0.7, 0.5, 0.3, 0.1]);
        let triag = TriangularWindow;
        let r = c.unapply_window(&triag).unwrap();
        let expected = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let r = r.data();
        assert_eq_tol(r, &expected, 1e-2);
    }
}