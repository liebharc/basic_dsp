#![feature(box_syntax)] 
extern crate simd;
extern crate num_cpus;
extern crate simple_parallel;
extern crate num;
mod vector_types;
pub use vector_types::
	{
		DataVector,
		DataVector32, 
		RealTimeVector32,
		ComplexTimeVector32, 
		RealFreqVector32,
		ComplexFreqVector32,
		DataVector64, 
		RealTimeVector64,
		ComplexTimeVector64, 
		RealFreqVector64,
		ComplexFreqVector64,
		DataBuffer
	};

