extern crate hound;
extern crate basic_dsp;
extern crate docopt;

use basic_dsp::*;
use basic_dsp::conv_types::*;
use std::i16;
use std::env;
use docopt::Docopt;

const USAGE: &'static str = "
This program takes a source wav file, slows it down by a factor of 1.5 
and then writes the result to the dest file.

Usage: slow_down <source> <dest>
       slow_down (--help) 

Options:
    -h, --help  Display usage.
";

// The approach to slow down a wav file is to interpolate the channels
// so that they contain more samples. If we then keep the sample rate 
// constant then the resulting music track will be longer and the music
// played slower. With this method the pitch will be mostly preserved.
// How good the pitch is preserved is determined by the interpolation
// function. A Sinc function is a very simple model but the result
// sounds good enough.
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
    let samples: Vec<f32> = reader.samples::<f32>().map(|x|x.expect("Failed to read sample")).collect(); 
    assert_eq!(reader.spec().channels, 2);
    assert_eq!(reader.spec().sample_rate, 44100);
    assert_eq!(reader.spec().bits_per_sample, 32);
    
    // If we assume stereo data then the samples are interleaved.
    // For convenience we put them into a complex vector since
    // basic_dsp stores complex data as interleaved too and so
    // we get an easy way to interpolate both channels.
    let mut complex = samples.to_complex_time_vec();
    let mut buffer = SingleBuffer::new();
    let function = SincFunction::new();
    complex.interpolatef(&mut buffer, &function, 1.5, 0.0, 10);
    
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
        writer.write_sample(sample).expect(&format!("Failed to write sample {}", sample_no));
        sample_no += 1;
    }
    
    writer.finalize().expect("Failed to close output waveform");
    
    let len = complex.points();
    println!("Finished processing {} samples", len);
}