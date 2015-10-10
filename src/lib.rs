#![feature(test)]
mod vector_types;
use vector_types::*;
extern crate test;
#[allow(unused_imports)]
use test::Bencher;
extern crate simd;
use simd::f32x4;

pub fn add_one(data: &mut DataVector) 
{
	let increment_vector = f32x4::splat(1.0); 
	let data_length = data.len();
	let scalar_length = data_length % 4;
	let vectorization_length = data_length - scalar_length;
	let mut array = &mut data.data;
	let mut i = 0;
    while i < vectorization_length
	{ 
		let vector = f32x4::load(array, i);
		let incremented = vector + increment_vector;
		incremented.store(array, i);
		i += 4;
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
	let mut i = 0;
    while i < data_length
	{ 
		array[i] = array[i] + 1.0;
		i += 1;
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
	let mut data = [0.0; 20000];
	let mut result = DataVector::new(&mut data);
	b.iter(|| {
		add_one(&mut result);
		return result.data[0];
		});
}

#[bench]
fn add_one_scalar_benchmark(b: &mut Bencher)
{
	let mut data = [0.0; 20000];
	let mut result = DataVector::new(&mut data);
	b.iter(|| {
		add_one_scalar(&mut result);
		return result.data[0];
		});
}