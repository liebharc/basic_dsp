extern crate basic_dsp;
use basic_dsp::conv_types::*;
use basic_dsp::*;

use std::f64::consts::PI;
use std::fs::File;
use std::io;
use std::io::prelude::*;

// In this example start with creating some data using a PRBS15 which is a standard way to create
// pseudo random data. Afterwards we interpolate the data to get a waveform which fits in a small
// frequency block and finally we shift this frequency block to a desired target frequency.
// Modern communication technologies like UMTS and LTE use similar approaches in order to
// transmit data, however needless to say that those technologies are far more complicated.
fn main() {
    let number_of_symbols = 10000;
    let mut prbs = Prbs15::new();
    let mut channel1 = vec![0.0; number_of_symbols].to_real_time_vec();
    let mut channel2 = vec![0.0; number_of_symbols].to_real_time_vec();
    let mut complex = vec![0.0; 0].to_complex_time_vec();
    let mut buffer = SingleBuffer::new();

    for i in 0..3 {
        fill_vectors_with_prbs(&mut channel1, &mut channel2, &mut prbs);
        complex
            .set_real_imag(&channel1, &channel2)
            .expect("Channel 1 and channel 2 should always have the same size");
        complex.interpolatef(&mut buffer, &RaisedCosineFunction::new(0.35), 10.0, 0.0, 10);
        let mut file = File::create(format!("baseband_time{}.csv", i))
            .expect("Failed to create baseband time file");
        complex_vector_to_file(&complex, &mut file).expect("Failed to write baseband time file");

        complex.multiply_complex_exponential(0.25 * PI, 0.0);

        let real = complex.to_real();

        let mut file = File::create(format!("modulated_time{}.csv", i))
            .expect("Failed to create modulated time file");
        real_vector_to_file(&real, &mut file).expect("Failed to write modulated time file");

        complex = real.rededicate(); // Reuse memory
    }
}

struct Prbs15 {
    lfsr: u32,
}

impl Prbs15 {
    fn new() -> Self {
        Prbs15 { lfsr: 0x1 }
    }

    fn next(&mut self) -> f64 {
        let bit = (self.lfsr ^ self.lfsr >> 14) & 0x1;
        self.lfsr = (self.lfsr >> 1) | (bit << 14);
        (bit as f64 - 0.5)
    }
}

fn fill_vectors_with_prbs(
    channel1: &mut RealTimeVec64,
    channel2: &mut RealTimeVec64,
    prbs: &mut Prbs15,
) {
    assert!(channel1.points() == channel2.points());
    for i in 0..channel1.points() {
        *channel2.data_mut(i) = prbs.next();
        *channel1.data_mut(i) = prbs.next();
    }
}

fn complex_vector_to_file(vector: &ComplexTimeVec64, f: &mut File) -> io::Result<()> {
    let mut i = 0;
    let vector = vector.data(..);
    while i < vector.len() {
        writeln!(f, "{}, {}", vector[i], vector[i + 1])?;
        i += 2;
    }
    Ok(())
}

fn real_vector_to_file(vector: &RealTimeVec64, f: &mut File) -> io::Result<()> {
    let mut i = 0;
    let vector = vector.data(..);
    while i < vector.len() {
        writeln!(f, "{}", vector[i])?;
        i += 2;
    }
    Ok(())
}
