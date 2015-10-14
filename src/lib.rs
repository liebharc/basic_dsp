#![feature(box_syntax)] 
extern crate simd;
extern crate num_cpus;
extern crate simple_parallel;
extern crate num;
mod vector_types;
mod multicore_support;
pub use multicore_support::DataBuffer;
pub use vector_types::
	{
		DataVectorDomain,
		DataVector,
		DataVector32, 
		RealTimeVector32,
		ComplexTimeVector32, 
		RealFreqVector32,
		ComplexFreqVector32,
		/*DataVector64, 
		RealTimeVector64,
		ComplexTimeVector64, 
		RealFreqVector64,
		ComplexFreqVector64*/
	};

