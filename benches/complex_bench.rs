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
	fn complex_offset_32_benchmark(b: &mut Bencher)
	{
		let mut vector = VectorBox::<ComplexTimeVector32>::new();
		b.iter(|| {
			vector.execute(|v|  { v.complex_offset(Complex32::new(2.0, -5.0)) } )
		});
	}
	
	#[bench]
	fn complex_scale_32_benchmark(b: &mut Bencher)
	{
		let mut vector = VectorBox::<ComplexTimeVector32>::new();
		b.iter(|| {
			vector.execute(|v|  { v.complex_scale(Complex32::new(-2.0, 2.0)) } )
		});
	}
}