use super::definitions::{
	DataVector,
    VecResult,
    VoidResult,
	DataVectorDomain,
	GenericVectorOperations,
	RealVectorOperations,
	ComplexVectorOperations};
use num::complex::Complex64;
use std::ops::{Index, IndexMut, Range, RangeTo, RangeFrom, RangeFull};
use std::mem;
use super::super::multicore_support::MultiCoreSettings;

define_vector_struct!(struct DataVector64, f64);
define_real_basic_struct_members!(impl DataVector64, DataVectorDomain::Time);
define_complex_basic_struct_members!(impl DataVector64, DataVectorDomain::Frequency);

define_vector_struct!(struct RealTimeVector64, f64);
define_real_basic_struct_members!(impl RealTimeVector64, DataVectorDomain::Time);
define_generic_operations_forward!(from: RealTimeVector64, to: DataVector64);
define_real_operations_forward!(from: RealTimeVector64, to: DataVector64, complex_partner: ComplexTimeVector64);

define_vector_struct!(struct RealFreqVector64, f64);
define_real_basic_struct_members!(impl RealFreqVector64, DataVectorDomain::Frequency);
define_generic_operations_forward!(from: RealFreqVector64, to: DataVector64);
define_real_operations_forward!(from: RealFreqVector64, to: DataVector64, complex_partner: ComplexFreqVector64);

define_vector_struct!(struct ComplexTimeVector64, f64);
define_complex_basic_struct_members!(impl ComplexTimeVector64, DataVectorDomain::Time);
define_generic_operations_forward!(from: ComplexTimeVector64, to: DataVector64);
define_complex_operations_forward!(from: ComplexTimeVector64, to: DataVector64, complex: Complex64, real_partner: RealTimeVector64);

define_vector_struct!(struct ComplexFreqVector64, f64);
define_complex_basic_struct_members!(impl ComplexFreqVector64, DataVectorDomain::Frequency);
define_generic_operations_forward!(from: ComplexFreqVector64, to: DataVector64);
define_complex_operations_forward!(from: ComplexFreqVector64, to: DataVector64, complex: Complex64, real_partner: RealTimeVector64);

#[inline]
#[allow(unused_variables)]
impl GenericVectorOperations for DataVector64
{
	fn add_vector(self, summand: &Self) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}

	fn subtract_vector(self, subtrahend: &Self) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
	
	fn multiply_vector(self, factor: &Self) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
	
	fn divide_vector(self, divisor: &Self) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
	
	fn zero_pad(self, points: usize) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
	
	fn zero_interleave(self) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
	
	fn diff(self) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
	
	fn diff_with_start(self) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
	
	fn cum_sum(self) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
    
    fn sqrt(self) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
	
	fn square(self) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
	
	fn root(self, degree: Self::E) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
	
	fn power(self, exponent: Self::E) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
	
	fn logn(self) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
	
	fn expn(self) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}

	fn log_base(self, base: Self::E) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
	
	fn exp_base(self, base: Self::E) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
    
    fn sin(self) -> VecResult<Self>
    {
        panic!("Unimplemented");
    }
    
    fn cos(self) -> VecResult<Self>
    {
        panic!("Unimplemented");
    }
    
    fn swap_halves(self) -> VecResult<Self>
    {
        panic!("Unimplemented");
    }
}

#[inline]
#[allow(unused_variables)]
impl RealVectorOperations for DataVector64
{
	type ComplexPartner = DataVector64;
	
	fn real_offset(self, offset: f64) -> VecResult<Self>
	{
		panic!("Unimplemented");
		// self.inplace_offset(&[offset, offset], buffer);
	}
	
	fn real_scale(self, factor: f64) -> VecResult<Self>
	{
		panic!("Unimplemented");
		/*let data_length = self.len();
		let mut array = &mut self.data;
		Chunk::execute_partial_with_arguments(&mut array, data_length, 1, buffer, DataVector64::inplace_real_scale_par, factor);*/
	}
	
	fn real_abs(self) -> VecResult<Self>
	{
		panic!("Unimplemented");
		/*let mut array = &mut self.data;
		let length = array.len();
		Chunk::execute_partial(&mut array, length, 1, buffer, DataVector32::inplace_abs_real_par);*/
	}
	
	fn to_complex(self) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
	
	fn wrap(self, divisor: Self::E) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
	
	fn unwrap(self, divisor: Self::E) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
}

#[inline]
#[allow(unused_variables)]
impl ComplexVectorOperations for DataVector64
{
	type RealPartner = DataVector64;
	type Complex = Complex64;
	
	fn complex_offset(self, offset: Complex64) -> VecResult<Self>
	{
		panic!("Unimplemented");
		// self.inplace_offset(&[offset.re, offset.im], buffer);
	}
	
	fn complex_scale(self, factor: Complex64) -> VecResult<Self>
	{
		panic!("Unimplemented");
		/*
		let data_length = self.len();
		let mut array = &mut self.data;
		Chunk::execute_partial_with_arguments(&mut array, data_length, 1, buffer, DataVector64::inplace_complex_scale_par, factor);*/
	}
	
	fn complex_abs(self) -> VecResult<Self>
	{
		panic!("Unimplemented");
		/*
		let data_length = self.len();
		let mut array = &mut self.data;
		Chunk::execute_partial(&mut array, data_length, 1, buffer, DataVector64::inplace_complex_abs_par);*/
	}
	
	fn get_complex_abs(&self, destination: &mut DataVector64) -> VoidResult
	{
		panic!("Unimplemented");
	}
	
	fn complex_abs_squared(self) -> VecResult<Self>
	{
		panic!("Unimplemented");
		/*
		let data_length = self.len();
		let mut array = &mut self.data;
		Chunk::execute_partial(&mut array, data_length, 1, buffer, DataVector64::inplace_complex_abs_squared_par);*/
	}
	
	fn complex_conj(self) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
	
	fn to_real(self) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}

	fn to_imag(self) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}	
			
	fn get_real(&self, destination: &mut Self) -> VoidResult
	{
		panic!("Unimplemented");
	}
	
	fn get_imag(&self, destination: &mut Self) -> VoidResult
	{
		panic!("Unimplemented");
	}
	
	fn phase(self) -> VecResult<Self>
	{
		panic!("Unimplemented");
	}
	
	fn get_phase(&self, destination: &mut Self) -> VoidResult
	{
		panic!("Unimplemented");
	}
}

#[inline]
#[allow(unused_variables)]
impl DataVector64
{
	
	
	
	/*
	fn inplace_offset(&mut self, offset: &[f64; 2], buffer: &mut DataBuffer) 
	{
		let data_length = self.len();
		let mut array = &mut self.data;
		Chunk::execute_partial_with_arguments(&mut array, data_length, 1, buffer, DataVector64::inplace_offset_parallel, offset);
	}
		
	fn inplace_offset_parallel(array: &mut [f64], increment_vector: &[f64;2])
	{
		let mut i = 0;
		while i < array.len()
		{ 
			array[i] = array[i] + increment_vector[i % 2];
			i += 1;
		}
	}*/

	
	/*
	fn inplace_real_scale_par(array: &mut [f64], factor: f64)
	{
		let mut i = 0;
		while i < array.len()
		{ 
			array[i] = array[i] * factor;
			i += 1;
		}
	}*/
	
	
	/*
	fn inplace_complex_scale_par(array: &mut [f64], factor: Complex64)
	{
		let mut i = 0;
		while i < array.len()
		{
			let real = array[i];
			let imag = array[i + 1];
			array[i] = real * factor.re - imag * factor.im;
			array[i + 1] = real * factor.im + imag * factor.re;
			i += 2;
		}
	}*/
		
	
	/*
	fn inplace_complex_abs_par(array: &mut [f64])
	{
		let mut i = 0;
		while i < array.len()
		{ 
			let real = array[i];
			let imag = array[i + 1];
			array[i / 2] = (real * real + imag * imag).sqrt();
			i += 2;
		}
	}*/
	
	
	
	/*
	fn inplace_complex_abs_squared_par(array: &mut [f64])
	{
		let mut i = 0;
		while i < array.len()
		{ 
			let real = array[i];
			let imag = array[i + 1];
			array[i / 2] = real * real + imag * imag;
			i += 2;
		}
	}*/
}
/*
#[test]
fn add_real_one_64_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = DataVector64::from_array(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_real_offset(1.0, &mut buffer);
	let expected = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
	assert_eq!(result.data, expected);
	assert_eq!(result.delta, 1.0);
}

#[test]
fn multiply_real_two_64_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = DataVector64::from_array(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_real_scale(2.0, &mut buffer);
	let expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
	assert_eq!(result.data, expected);
	assert_eq!(result.delta, 1.0);
}

#[test]
fn multiply_complex_64_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = DataVector64::from_interleaved(&mut data);
	let mut buffer = DataBuffer::new("test");
	result.inplace_complex_scale(Complex64::new(2.0, -3.0), &mut buffer);
	let expected = [8.0, 1.0, 18.0, -1.0, 28.0, -3.0, 38.0, -5.0];
	assert_eq!(result.data, expected);
	assert_eq!(result.delta, 1.0);
}*/