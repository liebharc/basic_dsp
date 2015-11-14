#![feature(test)]
#![feature(box_syntax)] 
extern crate test;
use test::Bencher;

extern crate basic_dsp;
use basic_dsp::{
	DataVector,
	DataVector32, 
	RealTimeVector32, 
	ComplexTimeVector32, 
	Operation32};

extern crate num;
use num::complex::Complex32;

pub fn add_offset_reference(array: &mut [f32], offset: f32) 
{
	let mut i = 0;
	while i < array.len()
	{ 
		array[i] = array[i] + offset;
		i += 1;
	}
}

#[allow(dead_code)]
const DEFAULT_DATA_SIZE: usize = 10000000;

#[bench]
fn add_real_one_scalar_32_benchmark(b: &mut Bencher)
{
		b.iter(move|| {
		let mut data: Box<[f32]> = box [0.0; DEFAULT_DATA_SIZE];
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
fn add_real_vector_32_benchmark(b: &mut Bencher)
{
	b.iter(|| {
		let data = vec![0.0; DEFAULT_DATA_SIZE];
		let result = RealTimeVector32::from_array_no_copy(data);
		let result = result.inplace_real_offset(2.0);
		return result.delta();;
		});
}

#[bench]
fn multi_operations_vector_32_benchmark(b: &mut Bencher)
{
	b.iter(|| {
		let data = vec![0.0; DEFAULT_DATA_SIZE];
		let result = ComplexTimeVector32::from_interleaved_no_copy(data);
		let result = result.perform_operations(
			&[Operation32::AddReal(1.0),
			Operation32::AddComplex(Complex32::new(1.0, 1.0)),
			Operation32::MultiplyComplex(Complex32::new(-1.0, 1.0))]);
		return result.delta();
		});
}

#[bench]
fn scale_complex_vector_32_benchmark(b: &mut Bencher)
{
	b.iter(|| {
		let data = vec![0.0; DEFAULT_DATA_SIZE];
		let result = ComplexTimeVector32::from_interleaved_no_copy(data);
		let result = result.inplace_complex_scale(Complex32::new(-2.0, 2.0));
		return result.delta();
		});
}

#[bench]
fn abs_real_vector_32_benchmark(b: &mut Bencher)
{
	b.iter(|| {
		let data = vec![0.0; DEFAULT_DATA_SIZE];
		let result = RealTimeVector32::from_array_no_copy(data);
		let result = result.inplace_real_abs();
		return result.delta();
		});
}

#[bench]
fn abs_complex_vector_32_benchmark(b: &mut Bencher)
{
	b.iter(|| {
		let data = vec![0.0; DEFAULT_DATA_SIZE];
		let result = ComplexTimeVector32::from_interleaved_no_copy(data);
		let result = result.inplace_complex_abs();
		return result.delta();
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