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

#[bench]
fn add_real_one_vector_benchmark(b: &mut Bencher)
{
	let mut data = [0.0; 10000];
	let mut result = DataVector::new(&mut data);
	b.iter(|| {
		result.inplace_real_offset(1.0);
		return result.data[0];
		});
}

#[bench]
fn add_complex_vector_benchmark(b: &mut Bencher)
{
	let mut data = [0.0; 10000];
	let mut result = DataVector::new(&mut data);
	b.iter(|| {
		let complex = Complex::new(1.0, -1.0);
		result.inplace_complex_offset(complex);
		return result.data[0];
		});
}

#[bench]
fn add_real_one_scalar_benchmark(b: &mut Bencher)
{
	let mut data = [0.0; 10000];
	let mut result = DataVector::new(&mut data);
	b.iter(|| {
		add_one_scalar(&mut result);
		return result.data[0];
		});
}