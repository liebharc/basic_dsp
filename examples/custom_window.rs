extern crate basic_dsp;
use basic_dsp::conv_types::*;
use basic_dsp::*;

struct Identity;

impl RealImpulseResponse<f64> for Identity {
    fn is_symmetric(&self) -> bool {
        true
    }

    fn calc(&self, x: f64) -> f64 {
        if x == 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

/// This example is just supposed to make the syntax clearer on how to implement one
/// of the traits which are indended to be extended by users as required:
/// `RealImpulseResponse`, `RealFrequencyResponse`, `ComplexImpulseResponse`, `ComplexFrequencyResponse`
/// and `WindowFunction`.
fn main() {
    let number_of_symbols = 100;
    // for this example the data doesn't really matter
    let mut data = vec![0.0; number_of_symbols].to_real_time_vec();
    let mut buffer = SingleBuffer::new();
    data.convolve(&mut buffer, &Identity, 1.0, 12);
}
