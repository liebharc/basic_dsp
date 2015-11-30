#[cfg(test)]
mod bench {
	use test::Bencher;
	use basic_dsp::{
		DataVector,
		RealVectorOperations,
		ComplexVectorOperations,
        TimeDomainOperations,
        FrequencyDomainOperations,
		DataVector32, 
		RealTimeVector32, 
		ComplexTimeVector32, 
		Operation32};
	use num::complex::Complex32;
	use std::boxed::Box;
    use tools::{VectorBox, DEFAULT_DATA_SIZE};
	
	#[bench]
	fn plain_fft_32_benchmark(b: &mut Bencher)
	{
		let mut vector = VectorBox::<DataVector32>::with_size(true, 10000);
		b.iter(|| {
			vector.execute(|v|  { v.plain_fft() } )
		});
	}
	
	#[bench]
	fn plain_ifft_32_benchmark(b: &mut Bencher)
	{
		let mut vector = VectorBox::<DataVector32>::with_size(true, 10000);
		b.iter(|| {
			vector.execute(|v|  { v.plain_ifft() } )
		});
	}
}