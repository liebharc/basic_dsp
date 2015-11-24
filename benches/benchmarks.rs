#![feature(test)]
#![feature(box_syntax)] 
extern crate test;
extern crate basic_dsp;
extern crate num;

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
	
	pub struct VectorBox<T>
	{
		vector: *mut T
	}
	
	impl VectorBox<DataVector32>
	{
		fn with_size(size: usize) -> VectorBox<DataVector32>
		{
			let data = vec![0.0; size];
			let vector = DataVector32::from_interleaved_no_copy(data);
			VectorBox
			{
				vector: Box::into_raw(Box::new(vector))
			}
		}
		
		fn new() -> VectorBox<DataVector32>
		{
			let data = vec![0.0; DEFAULT_DATA_SIZE];
			let vector = DataVector32::from_interleaved_no_copy(data);
			VectorBox
			{
				vector: Box::into_raw(Box::new(vector))
			}
		}
	}
	
	impl VectorBox<RealTimeVector32>
	{
		fn new() -> VectorBox<RealTimeVector32>
		{
			let data = vec![0.0; DEFAULT_DATA_SIZE];
			let vector = RealTimeVector32::from_array_no_copy(data);
			VectorBox
			{
				vector: Box::into_raw(Box::new(vector))
			}
		}
	}
	
	impl VectorBox<ComplexTimeVector32>
	{
		fn new() -> VectorBox<ComplexTimeVector32>
		{
			let data = vec![0.0; DEFAULT_DATA_SIZE];
			let vector = ComplexTimeVector32::from_interleaved_no_copy(data);
			VectorBox
			{
				vector: Box::into_raw(Box::new(vector))
			}
		}
	}
	
	#[allow(dead_code)]
	impl<T> VectorBox<T>
	{
		fn execute<F>(&mut self, function: F) -> bool
			where F: Fn(T) -> T + 'static + Sync
		{
			unsafe {
				let vector = Box::from_raw(self.vector);
				let result = function(*vector);
				self.vector = Box::into_raw(Box::new(result));
			}
			
			true
		}
	}
	
	impl<T> Drop for VectorBox<T>
	{
		fn drop(&mut self) {
			unsafe {
				let _ = Box::from_raw(self.vector); // make sure that the vector is deleted
			}
		}
	}
	
	pub fn add_offset_reference(array: &mut [f32], offset: f32) 
	{
		let mut i = 0;
		while i < array.len()
		{
			array[i] = array[i] + offset;
			i += 1;
		}
	}

	const DEFAULT_DATA_SIZE: usize = 10000;
	
	#[bench]
	fn add_real_one_scalar_32_benchmark(b: &mut Bencher)
	{
		let mut data: Box<[f32]> = box [0.0; DEFAULT_DATA_SIZE];
		b.iter(move|| {
			add_offset_reference(&mut data, 1.0);
			return data[0];
			});
	}
	
	#[bench]
	fn vector_creation_32_benchmark(b: &mut Bencher)
	{
		b.iter(|| {
			let data = vec![0.0; DEFAULT_DATA_SIZE];
			let result = DataVector32::from_interleaved_no_copy(data);
			return result.delta();;
			});
	}
	
	#[bench]
	fn add_real_32_benchmark(b: &mut Bencher)
	{
		let mut vector = VectorBox::<RealTimeVector32>::new();
		b.iter(|| {
			vector.execute(|v|  { v.real_offset(2.0) } )
		});
	}
    
    #[bench]
	fn add_complex_32_benchmark(b: &mut Bencher)
	{
		let mut vector = VectorBox::<ComplexTimeVector32>::new();
		b.iter(|| {
			vector.execute(|v|  { v.complex_offset(Complex32::new(2.0, -5.0)) } )
		});
	}
	
	#[bench]
	fn multi_operations_vector_32_benchmark(b: &mut Bencher)
	{
		let mut vector = VectorBox::<DataVector32>::new();
		b.iter(|| {
			vector.execute(|v|  
				{
					  v.perform_operations(
						&[Operation32::AddReal(1.0),
						Operation32::AddComplex(Complex32::new(1.0, 1.0)),
						Operation32::MultiplyComplex(Complex32::new(-1.0, 1.0))])
				})
		});
	}
	
	#[bench]
	fn scale_complex_vector_32_benchmark(b: &mut Bencher)
	{
		let mut vector = VectorBox::<ComplexTimeVector32>::new();
		b.iter(|| {
			vector.execute(|v|  { v.complex_scale(Complex32::new(-2.0, 2.0)) } )
		});
	}
	
	#[bench]
	fn abs_real_vector_32_benchmark(b: &mut Bencher)
	{
		let mut vector = VectorBox::<RealTimeVector32>::new();
		b.iter(|| {
			vector.execute(|v|  { v.real_abs() } )
		});
	}
	
	#[bench]
	fn fft_complex_vector_32_benchmark(b: &mut Bencher)
	{
		let mut vector = VectorBox::<DataVector32>::with_size(10000);
		b.iter(|| {
			vector.execute(|v|  { v.plain_fft() } )
		});
	}
	
	#[bench]
	fn ifft_complex_vector_32_benchmark(b: &mut Bencher)
	{
		let mut vector = VectorBox::<DataVector32>::with_size(10000);
		b.iter(|| {
			vector.execute(|v|  { v.plain_ifft() } )
		});
	}
	
	/*
	#[bench]
	fn add_real_vector_64_benchmark(b: &mut Bencher)
	{
		let mut data: Box<[f64]> = box [0.0; DEFAULT_DATA_SIZE];
		let mut result = RealTimeVector64::from_array(&mut data);
		let mut buffer = DataBuffer::new("test");
		b.iter(|| {
			result.inplace_real_offset(2.0, &mut buffer);
			return result.data(&mut buffer)[0];;
			});
	}
	
	#[bench]
	fn scale_complex_vector_64_benchmark(b: &mut Bencher)
	{
		let mut data: Box<[f64]> = box [0.0; DEFAULT_DATA_SIZE];
		let mut result = ComplexTimeVector64::from_interleaved(&mut data);
		let mut buffer = DataBuffer::new("test");
		b.iter(|| {
			result.inplace_complex_scale(Complex64::new(-2.0, 2.0), &mut buffer);
			return result.data(&mut buffer)[0];;
			});
	}
	
	#[bench]
	fn abs_real_vector_64_benchmark(b: &mut Bencher)
	{
		let mut data: Box<[f64]> = box [0.0; DEFAULT_DATA_SIZE];
		let mut result = RealTimeVector64::from_array(&mut data);
		let mut buffer = DataBuffer::new("test");
		b.iter(|| {
			result.inplace_real_abs(&mut buffer);
			return result.data(&mut buffer)[0];;
			});
	}
	
	
	#[bench]
	fn abs_complex_vector_64_benchmark(b: &mut Bencher)
	{
		let mut data: Box<[f64]> = box [0.0; DEFAULT_DATA_SIZE];
		let mut result = ComplexTimeVector64::from_interleaved(&mut data);
		let mut buffer = DataBuffer::new("test");
		b.iter(|| {
			result.inplace_complex_abs(&mut buffer);
			return result.data(&mut buffer)[0];;
			});
	}*/
}