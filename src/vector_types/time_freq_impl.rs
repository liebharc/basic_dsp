use super::definitions::{
	DataVector,
    VecResult,
    DataVectorDomain,
    ErrorReason,
    ComplexVectorOperations,
    RealVectorOperations};
use super::{
    GenericDataVector,
    RealTimeVector,
    ComplexFreqVector,
    ComplexTimeVector};
use rustfft::FFT;
use RealNumber;
use std::ptr;

/// Argument for some operations to determine if the result should have an even 
/// or odd number of points.
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum EvenOdd {
	/// Even number of points
	Even,
	/// Odd number of points
    Odd
}

/// Defines all operations which are valid on `DataVectors` containing real data.
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
	///		assert!((actual[i] - expected[i]).abs() < 1e-4);
	/// }
	/// ```
	fn plain_fft(self) -> VecResult<Self::FreqPartner>;
	
	// TODO add fft method which also applies a window
}

/// Defines all operations which are valid on `DataVectors` containing complex data.
pub trait FrequencyDomainOperations<T> : DataVector<T> 
    where T : RealNumber {
    type RealTimePartner;
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
	///		assert!((actual[i] - expected[i]).abs() < 1e-4);
	/// }
	/// ```
	fn plain_ifft(self) -> VecResult<Self::ComplexTimePartner>;
    
    /// Performs a Symmetric Inverse Fast Fourier Transformation under the assumption that `self`
    /// contains half of a symmetric spectrum starting from 0 Hz. This assumption
    /// isn't verified and no error is raised if the spectrum isn't symmetric. The reason
    /// for this is that there is no robust verification possible.
    ///
    /// The argument indicates whether the resulting real vector should have `2*N` or `2*N-1` points.
    ///
    /// This version of the IFFT neither applies a window nor does it scale the 
	/// vector.
    fn plain_sifft(self, even_odd: EvenOdd) -> VecResult<Self::RealTimePartner>;
    
    /// This function mirrors the spectrum vector to transform a symmetric spectrum
    /// into a full spectrum with the DC element at index 0 (no fft shift/swap halves).
	///
    /// The argument indicates whether the resulting real vector should have `2*N` or `2*N-1` points.
    ///
	/// # Example
	///
	/// ```
	/// use basic_dsp::{ComplexFreqVector32, FrequencyDomainOperations, DataVector, EvenOdd};
	/// let vector = ComplexFreqVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
	/// let result = vector.mirror(EvenOdd::Odd).expect("Ignoring error handling in examples");
	/// assert_eq!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, -6.0, 3.0, -4.0], result.data());
	/// ```
    fn mirror(self, even_odd: EvenOdd) -> VecResult<Self>;
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
                            let rbw = (points as $data_type)  / self.delta;
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
                        let mut fft = try! { self.to_complex().and_then(|v|v.plain_fft()) };
                        let len = fft.len();
                        let len = len / 2 + 1;
                        let len = len + len % 2;
                        fft.valid_len = len;
                        Ok(fft)
                    }
                }
            }
            
            impl TimeDomainOperations<$data_type> for ComplexTimeVector<$data_type> {
                type FreqPartner = ComplexFreqVector<$data_type>;
                fn plain_fft(self) -> VecResult<Self::FreqPartner> {
                    Self::FreqPartner::from_genres(self.to_gen().plain_fft())
                }
            }
            
            impl TimeDomainOperations<$data_type> for RealTimeVector<$data_type> {
                type FreqPartner = ComplexFreqVector<$data_type>;
                fn plain_fft(self) -> VecResult<Self::FreqPartner> {
                    Self::FreqPartner::from_genres(self.to_gen().plain_fft())
                }
            }
            
            impl FrequencyDomainOperations<$data_type> for GenericDataVector<$data_type> {
                type ComplexTimePartner = GenericDataVector<$data_type>;
                type RealTimePartner = GenericDataVector<$data_type>;
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
                
                fn plain_sifft(self, even_odd: EvenOdd) -> VecResult<GenericDataVector<$data_type>> {
                    assert_complex!(self);
                    assert_freq!(self);
                    self.mirror(even_odd)
                        .and_then(|v|v.plain_ifft())
                        .and_then(|v|v.to_real())
                }
                
                fn mirror(mut self, even_odd: EvenOdd) -> VecResult<Self> {
                    assert_freq!(self);
                    assert_complex!(self);
                    {
                        let len = self.len();
                        let data = &self.data;
                        let step = 2;
                        let temp_len = if even_odd == EvenOdd::Even { 2 * (len - step) } else { 2 * len - step };
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
            }
            
            impl FrequencyDomainOperations<$data_type> for ComplexFreqVector<$data_type> {
                type ComplexTimePartner = ComplexTimeVector<$data_type>;
                type RealTimePartner = RealTimeVector<$data_type>;
                fn plain_ifft(self) -> VecResult<ComplexTimeVector<$data_type>> {
                    Self::ComplexTimePartner::from_genres(self.to_gen().plain_ifft())
                }
                
                fn plain_sifft(self, even_odd: EvenOdd) -> VecResult<RealTimeVector<$data_type>> {
                    Self::RealTimePartner::from_genres(self.to_gen().plain_sifft(even_odd))
                }
                
                fn mirror(self, even_odd: EvenOdd) -> VecResult<ComplexFreqVector<$data_type>> {
                    Self::from_genres(self.to_gen().mirror(even_odd))
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
        DataVector};

	#[test]
	fn plain_fft_test()
	{
		let vector = ComplexTimeVector32::from_interleaved(&[1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254]);
        let result = vector.plain_fft().unwrap();
        let expected = &[0.0, 0.0, 3.0, 0.0, 0.0, 0.0];
        assert_eq!(result.data(), expected);
	}
    
    #[test]
	fn plain_ifft_test()
	{
		let vector = ComplexFreqVector32::from_interleaved(&[0.0, 0.0, 3.0, 0.0, 0.0, 0.0]);
        let result = vector.plain_ifft().unwrap();
        let expected = &[3.0, 0.0, -1.5, 2.598076, -1.5, -2.598076];
        assert_eq!(result.data(), expected);
	}
    
    #[test]
	fn plain_sfft_test()
	{
		let vector = RealTimeVector32::from_array(&[1.0, -0.5, -0.5]);
        let result = vector.plain_fft().unwrap();
        let expected = &[0.0, 0.0, 1.5, 0.0];
        assert_eq!(result.data(), expected);
	}
    
    #[test]
	fn mirror_test()
	{
		let vector = ComplexFreqVector32::from_interleaved(&[0.0, 0.0, 1.5, 0.0]);
        let result = vector.mirror(EvenOdd::Odd).unwrap();
        let expected = &[0.0, 0.0, 1.5, 0.0, 1.5, 0.0];
        assert_eq!(result.data(), expected);
	}
    
    #[test]
    fn plain_sifft_test() {
        let vector = ComplexFreqVector32::from_interleaved(&[0.0, 0.0, 1.5, 0.0]);
        let result = vector.plain_sifft(EvenOdd::Odd).unwrap();
        let expected = &[3.0, -1.5, -1.5];
        assert_eq!(result.data(), expected);
        assert_eq!(result.is_complex(), false);
    }
    
    #[test]
	fn plain_sfft_test2()
	{
		let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = vector.plain_fft().unwrap();
        let expected = &[21.0, 0.0, -3.0, 5.1961527, -2.9999998, 1.7320508, -3.0, 0.0];
        assert_eq!(result.data(), expected);
	}
    
    #[test]
	fn mirror_test2()
	{
		let vector = ComplexFreqVector32::from_interleaved(&[21.0, 0.0, -3.0, 5.1961527, -5.0, 1.7320508, -3.0, 0.0]);
        let result = vector.mirror(EvenOdd::Even).unwrap();
        let expected = &[21.0, 0.0, -3.0, 5.1961527, -5.0, 1.7320508, -3.0, 0.0, -5.0, -1.7320508, -3.0, -5.1961527];
        assert_eq!(result.data(), expected);
	}
}