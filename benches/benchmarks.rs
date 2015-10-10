#![feature(test)]
extern crate test;
use test::Bencher;

extern crate basic_dsp;
use basic_dsp::*;
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

#[bench]
fn add_one_vector_benchmark(b: &mut Bencher)
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