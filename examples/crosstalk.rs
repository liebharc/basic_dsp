extern crate basic_dsp;
extern crate docopt;
extern crate hound;

use basic_dsp::matrix::*;
use basic_dsp::*;
use docopt::Docopt;
use std::env;
use std::i16;

const USAGE: &'static str = "
This program takes a source wav file, adds crosstalk
and then writes the result to the dest file.

Crosstalk means that you will hear parts of channel 1 in
channel 2 and vice versa.

Usage: crosstalk <source> <dest>
       crosstalk (--help)

Options:
    -h, --help  Display usage.
";

fn main() {
    let argv = env::args();
    let args = Docopt::new(USAGE)
        .and_then(|d| d.argv(argv.into_iter()).parse())
        .unwrap_or_else(|e| e.exit());
    if args.get_bool("-h") || args.get_bool("--help") {
        println!("{}", USAGE);
        std::process::exit(0);
    }

    let source = args.get_str("<source>");
    let dest = args.get_str("<dest>");

    let mut reader = hound::WavReader::open(source).expect("Failed to open input waveform");
    let samples: Vec<f32> = reader
        .samples::<f32>()
        .map(|x| x.expect("Failed to read sample"))
        .collect();
    assert_eq!(reader.spec().channels, 2);
    assert_eq!(reader.spec().sample_rate, 44100);
    assert_eq!(reader.spec().bits_per_sample, 32);

    let mut complex = samples.to_complex_time_vec();
    let mut channel1 = Vec::new().to_real_time_vec();
    let mut channel2 = Vec::new().to_real_time_vec();
    complex.get_real_imag(&mut channel1, &mut channel2);

    let mut mat = [channel1, channel2].to_mat();
    // The attenuation impulse response also adds an echo (expressed
    // by the value at index 0), but since the echo is only 3 samples and the
    // sample rate is 44.1 kHz the echo is < 1ms and no one will be able to hear that,
    // but it might be interesting enough for an example.
    let attenuation = vec![0.2, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0].to_real_time_vec();
    let crosstalk = vec![0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0].to_real_time_vec();

    let imp_resp = [[&attenuation, &crosstalk], [&crosstalk, &attenuation]];

    let mut buffer = SingleBuffer::new();
    mat.convolve_signal(&mut buffer, &imp_resp).unwrap();

    let rows = &mat.rows();
    complex
        .set_real_imag(&rows[0], &rows[1])
        .expect("Should never fail, this Vec can be resized");

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(dest, spec).expect("Failed to open output waveform");
    let mut sample_no = 0;
    for sample in &complex[..] {
        let amplitude = i16::MAX as f32;
        let sample = (sample * amplitude) as i16;
        writer
            .write_sample(sample)
            .expect(&format!("Failed to write sample {}", sample_no));
        sample_no += 1;
    }

    writer.finalize().expect("Failed to close output waveform");

    let len = complex.points();
    println!("Finished processing {} samples", len);
}
