#[cfg(test)]
mod bench {
	use test::Bencher;
	use basic_dsp::{
        TimeDomainOperations,
        FrequencyDomainOperations,
		DataVector32};
    use tools::VectorBox;
    use basic_dsp::window_functions::TriangularWindow;
	
	#[bench]
	fn plain_fft_ifft_32_benchmark(b: &mut Bencher)
	{
		let mut vector = VectorBox::<DataVector32>::with_size(true, 10000);
		b.iter(|| {
			vector.execute_res(|v|  
            { 
                v.plain_fft()
                .and_then(|v|v.plain_ifft()) 
            } )
		});
	}
    
    #[bench]
	fn window_32_benchmark(b: &mut Bencher)
	{
		let mut vector = VectorBox::<DataVector32>::with_size(true, 10000);
		b.iter(|| {
			vector.execute_res(|v|  
            { 
                let triag = TriangularWindow;
                v.apply_window(&triag) 
            } )
		});
	}
}