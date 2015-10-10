#![feature(test)]
#![feature(core_simd)]
#![feature(step_by)]

mod vector_types;
use vector_types::*;
extern crate test;
use test::Bencher;
use std::simd::f32x4;

pub fn add_one(data: &mut DataVector) 
{
	let increment_vector = f32x4(1.0, 1.0, 1.0, 1.0); 
	let data_length = data.len();
	let scalar_length = data_length % 4;
	let vectorization_length = data_length - scalar_length;
	let mut array = &mut data.data;
	for i in (0..vectorization_length).step_by(4)
	{ 
		let vector = f32x4(array[i], array[i+1], array[i+2], array[i+3]);
		let incremted = vector + increment_vector;
		array[i] = incremted.0;
		array[i + 1] = incremted.1;
		array[i + 2] = incremted.2;
		array[i + 3] = incremted.3;
	}
	
	for i in vectorization_length..data_length
	{
		array[i] = array[i] + 1.0;
	}
}

pub fn add_one_scalar(data: &mut DataVector) 
{
	let data_length = data.len();
	let mut array = &mut data.data;
	for i in 0..data_length
	{ 
		array[i] = array[i] + 1.0;
	}
}

#[test]
fn add_one_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = DataVector::new(&mut data);
	add_one(&mut result);
	assert_eq!(result.data, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn add_one_test_odd_number_of_elements()
{
	let mut data = [1.0, 2.0, 3.0];
	let mut result = DataVector::new(&mut data);
	add_one(&mut result);
	assert_eq!(result.data, [2.0, 3.0, 4.0]);
}

#[bench]
fn add_one_benchmark(b: &mut Bencher)
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
	let mut result = DataVector::new(&mut data);
	b.iter(|| {
		add_one(&mut result);
		return result.data[0];
		});
}

#[bench]
fn add_one_scalar_benchmark(b: &mut Bencher)
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
	let mut result = DataVector::new(&mut data);
	b.iter(|| {
		add_one_scalar(&mut result);
		return result.data[0];
		});
}