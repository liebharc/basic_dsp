use super::definitions::{
	DataVector,
    VecResult,
    DataVectorDomain,
    ErrorReason,
	TimeDomainOperations,
	FrequencyDomainOperations};
use super::{
    GenericDataVector,
    ComplexFreqVector,
    ComplexTimeVector};
use rustfft::FFT;

macro_rules! add_time_freq_impl {
    ($($data_type:ident);*)
	 =>
	 {	 
        $(
            impl TimeDomainOperations<$data_type> for GenericDataVector<$data_type> {
                type FreqPartner = GenericDataVector<$data_type>;
                fn plain_fft(mut self) -> VecResult<Self> {
                    {
                        assert_time!(self);
                        assert_complex!(self);
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
                }
            }
            
            impl TimeDomainOperations<$data_type> for ComplexTimeVector<$data_type> {
                type FreqPartner = ComplexFreqVector<$data_type>;
                fn plain_fft(self) -> VecResult<Self::FreqPartner> {
                    Self::FreqPartner::from_genres(self.to_gen().plain_fft())
                }
            }
            
            impl FrequencyDomainOperations<$data_type> for GenericDataVector<$data_type> {
                type TimePartner = GenericDataVector<$data_type>;
                fn plain_ifft(mut self) -> VecResult<Self::TimePartner> {
                    {
                        assert_freq!(self);
                        assert_complex!(self);
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
            }
            
            impl FrequencyDomainOperations<$data_type> for ComplexFreqVector<$data_type> {
                type TimePartner = ComplexTimeVector<$data_type>;
                fn plain_ifft(self) -> VecResult<Self::TimePartner> {
                    Self::TimePartner::from_genres(self.to_gen().plain_ifft())
                }
            }
        )*
     }
}
add_time_freq_impl!(f32; f64);