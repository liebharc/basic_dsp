#![feature(test)]
#![feature(box_syntax)] 
extern crate test;
use test::Bencher;

extern crate basic_dsp;
use basic_dsp::vector_types::*;

pub fn add_offset_reference(data: &mut DataVector, offset: f32) 
{
	let data_length = data.len();
	let mut array = &mut data.data;
	let mut i = 0;
    while i < data_length
	{ 
		array[i] = array[i] + offset;
		i += 1;
	}
}

#[allow(dead_code)]
const DEFAULT_DATA_SIZE: usize = 10000000;

#[bench]
fn add_real_one_scalar_benchmark(b: &mut Bencher)
{
	let mut data: Box<[f32]> = box [0.0; DEFAULT_DATA_SIZE];
	let mut result = DataVector::new(&mut data);
	b.iter(move|| {
		add_offset_reference(&mut result, 1.0);
		return result.data()[0];
		});
}

#[bench]
fn add_real_vector_benchmark(b: &mut Bencher)
{
	let mut data: Box<[f32]> = box [0.0; DEFAULT_DATA_SIZE];
	let mut result = DataVector::new(&mut data);
	let mut buffer = DataBuffer::new("test");
	b.iter(|| {
		result.inplace_real_offset(2.0, &mut buffer);
		return result.data()[0];
		});
}

#[bench]
fn scale_complex_vector_benchmark(b: &mut Bencher)
{
	let mut data: Box<[f32]> = box [0.0; DEFAULT_DATA_SIZE];
	let mut result = DataVector::new(&mut data);
	let mut buffer = DataBuffer::new("test");
	b.iter(|| {
		result.inplace_complex_scale(Complex::new(-2.0, 2.0), &mut buffer);
		return result.data()[0];
		});
}