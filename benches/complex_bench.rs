#[cfg(test)]
mod bench {
	use test::Bencher;
	use basic_dsp::{
		ComplexVectorOperations,
		ComplexTimeVector32};
	use num::complex::Complex32;
    use tools::{VectorBox, Size};
    
    #[bench]
	fn complex_offset_32s_benchmark(b: &mut Bencher)
	{
		let mut vector = VectorBox::<ComplexTimeVector32>::new(Size::Small);
		b.iter(|| {
			vector.execute_res(|v|  { v.complex_offset(Complex32::new(2.0, -5.0)) } )
		});
	}
	
	#[bench]
	fn complex_scale_32s_benchmark(b: &mut Bencher)
	{
		let mut vector = VectorBox::<ComplexTimeVector32>::new(Size::Small);
		b.iter(|| {
			vector.execute_res(|v|  { v.complex_scale(Complex32::new(-2.0, 2.0)) } )
		});
	}
    
    #[bench]
	fn complex_conj_32s_benchmark(b: &mut Bencher)
	{
		let mut vector = VectorBox::<ComplexTimeVector32>::new(Size::Small);
		b.iter(|| {
			vector.execute_res(|v|  { v.complex_conj() } )
		});
	}
}