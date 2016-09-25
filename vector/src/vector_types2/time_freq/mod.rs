mod freq;
pub use self::freq::*;
mod time;
pub use self::time::*;
mod time_to_freq;
pub use self::time_to_freq::*;
mod freq_to_time;
pub use self::freq_to_time::*;
mod correlation;
pub use self::correlation::*;

use std::mem;
use rustfft::FFT;
use RealNumber;
use super::{
	array_to_complex, array_to_complex_mut,
    Buffer, Vector,
    DspVec, ToSliceMut, NumberSpace, Domain
};

fn fft<S, T, N, D, B>(vec: &mut DspVec<S, T, N, D>, buffer: &mut B, reverse: bool)
	where S: ToSliceMut<T>,
    	  T: RealNumber,
		  N: NumberSpace,
		  D: Domain,
		  B: Buffer<S, T> {
	let len = vec.len();
	let mut temp = buffer.get(len);
	{
		let temp = temp.to_slice_mut();
		let points = vec.points();
		let rbw = (T::from(points).unwrap()) * vec.delta;
		vec.delta = rbw;
		let mut fft = FFT::new(points, reverse);
		let signal = vec.data.to_slice();
		let spectrum = &mut temp[0..len];
		let signal = array_to_complex(&signal[0..len]);
		let spectrum = array_to_complex_mut(spectrum);
		fft.process(signal, spectrum);
	}

	mem::swap(&mut vec.data, &mut temp);
	buffer.free(temp);
}
