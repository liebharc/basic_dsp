use super::super::general::{
	DataVector,
	TimeDomainOperations,
	FrequencyDomainOperations};
use super::{DataVector32, ComplexTimeVector32, ComplexFreqVector32};
use rustfft::FFT;

impl TimeDomainOperations for DataVector32 {
	type FreqPartner = DataVector32;
	fn plain_fft(mut self) -> DataVector32 {
		{
			let points = self.points();
			let rbw = (points as f32)  / self.delta;
			self.delta = rbw;
			let mut fft = FFT::new(points, false);
			let signal = &self.data;
			let spectrum = &mut self.temp;
			let signal = DataVector32::array_to_complex(signal);
			let spectrum = DataVector32::array_to_complex_mut(spectrum);
			fft.process(signal, spectrum);
		}
		self.swap_data_temp()
	}
}

impl TimeDomainOperations for ComplexTimeVector32 {
	type FreqPartner = ComplexFreqVector32;
	fn plain_fft(self) -> ComplexFreqVector32 {
		ComplexFreqVector32::from_gen(self.to_gen().plain_fft())
	}
}

impl FrequencyDomainOperations for DataVector32 {
	type TimePartner = DataVector32;
	fn plain_ifft(mut self) -> DataVector32 {
		{
			let points = self.points();
			let mut fft = FFT::new(points, true);
			let delta = (points as f32)  / self.delta;
			self.delta = delta;
			let signal = &self.data;
			let spectrum = &mut self.temp;
			let signal = DataVector32::array_to_complex(signal);
			let spectrum = DataVector32::array_to_complex_mut(spectrum);
			fft.process(signal, spectrum);
		}
		self.swap_data_temp()
	}
}

impl FrequencyDomainOperations for ComplexFreqVector32 {
	type TimePartner = ComplexTimeVector32;
	fn plain_ifft(self) -> ComplexTimeVector32 {
		ComplexTimeVector32::from_gen(self.to_gen().plain_ifft())
	}
}