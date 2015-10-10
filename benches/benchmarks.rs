#![feature(test)]
extern crate test;
use test::Bencher;

extern crate basic_dsp;
use basic_dsp::vector_types::*;

pub fn add_one_scalar(data: &mut DataVector) 
{
	let data_length = data.len();
	let mut array = &mut data.data;
	let mut i = 0;
    while i < data_length
	{ 
		array[i] = array[i] + 1.0;
		i += 1;
	}
}

#[allow(dead_code)]
const DEFAULT_DATA_SIZE: usize = 100000;

#[bench]
fn add_real_vector_benchmark(b: &mut Bencher)
{
	let mut data = [0.0; DEFAULT_DATA_SIZE];
	let mut result = DataVector::new(&mut data);
	let mut buffer = DataBuffer::new("test", PerformanceHint::None);
	b.iter(|| {
		result.inplace_real_offset(1.0, &mut buffer);
		return result.data[0];
		});
}

#[bench]
fn add_real_vector_multicore_benchmark(b: &mut Bencher)
{
	let mut data = [0.0; DEFAULT_DATA_SIZE];
	let mut result = DataVector::new(&mut data);
	let mut buffer = DataBuffer::new("test", PerformanceHint::AggressiveParallize);
	b.iter(|| {
		result.inplace_real_offset(1.0, &mut buffer);
		return result.data[0];
		});
}


#[bench]
fn add_complex_vector_benchmark(b: &mut Bencher)
{
	let mut data = [0.0; DEFAULT_DATA_SIZE];
	let mut result = DataVector::new(&mut data);
	let mut buffer = DataBuffer::new("test", PerformanceHint::None);
	b.iter(move|| {
		let complex = Complex::new(1.0, -1.0);
		result.inplace_complex_offset(complex, &mut buffer);
		return result.data[0];
		});
}

#[bench]
fn scale_real_one_vector_benchmark(b: &mut Bencher)
{
	let mut data = [0.0; DEFAULT_DATA_SIZE];
	let mut result = DataVector::new(&mut data);
	let mut buffer = DataBuffer::new("test", PerformanceHint::None);
	b.iter(move|| {
		result.inplace_real_scale(2.0, &mut buffer);
		return result.data[0];
		});
}

#[bench]
fn scale_complex_vector_benchmark(b: &mut Bencher)
{
	let mut data = [0.0; DEFAULT_DATA_SIZE];
	let mut result = DataVector::new(&mut data);
	let mut buffer = DataBuffer::new("test", PerformanceHint::None);
	b.iter(move|| {
		let complex = Complex::new(2.0, -2.0);
		result.inplace_complex_scale(complex, &mut buffer);
		return result.data[0];
		});
}

#[bench]
fn add_real_one_scalar_benchmark(b: &mut Bencher)
{
	let mut data = [0.0; DEFAULT_DATA_SIZE];
	let mut result = DataVector::new(&mut data);
	b.iter(move|| {
		add_one_scalar(&mut result);
		return result.data[0];
		});
}