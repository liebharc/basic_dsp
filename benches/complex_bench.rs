#[cfg(test)]
mod bench {
	use test::Bencher;
	use basic_dsp::{
        DataVector,
        GenericVectorOperations,
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
    
    #[bench]
	fn complex_vector_multiplication_32s_benchmark(b: &mut Bencher)
	{
		let mut vector = VectorBox::<ComplexTimeVector32>::new(Size::Small);
		b.iter(|| {
			vector.execute_res(|v| {
                let len = v.len(); 
                let operand = ComplexTimeVector32::complex_from_constant(0.0, len);
                v.multiply_vector(&operand) 
            } )
		});
	}
}