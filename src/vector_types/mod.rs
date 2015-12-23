macro_rules! reject_if {
    ($self_: ident, $condition: expr, $message: expr) => {
        if $condition {
            return Err(($message, $self_));
        }
    }
}

macro_rules! assert_meta_data {
    ($self_: ident, $other: ident) => {
         {        
            let delta_ratio = $self_.delta / $other.delta;
            if $self_.is_complex != $other.is_complex ||
                $self_.domain != $other.domain ||
                delta_ratio > 1.1 || delta_ratio < 0.9 {
                return Err((ErrorReason::VectorMetaDataMustAgree, $self_));
            }
         }
    }
}

macro_rules! assert_real {
    ($self_: ident) => {
        if $self_.is_complex {
            return Err((ErrorReason::VectorMustBeReal, $self_));
        }
    }
}

macro_rules! assert_complex {
    ($self_: ident) => {
        if !$self_.is_complex {
            return Err((ErrorReason::VectorMustBeComplex, $self_));
        }
    }
}

macro_rules! assert_time {
    ($self_: ident) => {
        if $self_.domain != DataVectorDomain::Time {
            return Err((ErrorReason::VectorMustBeInTimeDomain, $self_));
        }
    }
}

macro_rules! assert_freq {
    ($self_: ident) => {
        if $self_.domain != DataVectorDomain::Frequency {
            return Err((ErrorReason::VectorMustBeInFrquencyDomain, $self_));
        }
    }
}

macro_rules! temp_mut {
    ($self_: ident, $len: expr) => {
        if $self_.temp.len() < $len {
            $self_.temp = vec![0.0; $len];
            &mut $self_.temp
        }
        else {
            &mut $self_.temp
        }
    }
}

#[macro_use]
mod struct_macros;
#[macro_use]
mod real_forward;
#[macro_use]
mod complex_forward;
#[macro_use]
mod general_forward;
#[macro_use]
mod basic_functions;
pub mod definitions;
pub mod general_impl;
pub mod real_impl;
pub mod complex_impl;
pub mod time_freq_impl;
mod stats_impl;

pub use vector_types::definitions::{
		DataVectorDomain,
		DataVector,
        VecResult,
        VoidResult,
        ScalarResult,
        ErrorReason,
		GenericVectorOperations,
		RealVectorOperations,
		ComplexVectorOperations,
		TimeDomainOperations,
		FrequencyDomainOperations,		
        Statistics,
        RededicateVector
	};
use num::complex::Complex;
use RealNumber;
use multicore_support::{Chunk, Complexity, MultiCoreSettings};
use std::mem;
use simd_extensions::{Simd, Reg32, Reg64};
use std::ops::{Index, IndexMut, Range, RangeTo, RangeFrom, RangeFull};
    
define_vector_struct!(struct GenericDataVector);
add_basic_private_impl!(f32, Reg32; f64, Reg64);

define_vector_struct!(struct RealTimeVector);
define_real_basic_struct_members!(impl RealTimeVector, DataVectorDomain::Time);
define_generic_operations_forward!(from: RealTimeVector, to: GenericDataVector, f32, f64);
define_real_operations_forward!(from: RealTimeVector, to: GenericDataVector, complex_partner: ComplexTimeVector, f32, f64);

define_vector_struct!(struct RealFreqVector);
define_real_basic_struct_members!(impl RealFreqVector, DataVectorDomain::Frequency);
define_generic_operations_forward!(from: RealFreqVector, to: GenericDataVector, f32, f64);
define_real_operations_forward!(from: RealFreqVector, to: GenericDataVector, complex_partner: ComplexFreqVector, f32, f64);

define_vector_struct!(struct ComplexTimeVector);
define_complex_basic_struct_members!(impl ComplexTimeVector, DataVectorDomain::Time);
define_generic_operations_forward!(from: ComplexTimeVector, to: GenericDataVector, f32, f64);
define_complex_operations_forward!(from: ComplexTimeVector, to: GenericDataVector, complex: Complex, real_partner: RealTimeVector, f32, f64);

define_vector_struct!(struct ComplexFreqVector);
define_complex_basic_struct_members!(impl ComplexFreqVector, DataVectorDomain::Frequency);
define_generic_operations_forward!(from: ComplexFreqVector, to: GenericDataVector, f32, f64);
define_complex_operations_forward!(from: ComplexFreqVector, to: GenericDataVector, complex: Complex, real_partner: RealTimeVector, f32, f64);

define_vector_struct_type_alias!(struct DataVector32, based_on: GenericDataVector, f32);
define_vector_struct_type_alias!(struct RealTimeVector32, based_on: RealTimeVector, f32);
define_vector_struct_type_alias!(struct RealFreqVector32, based_on: RealFreqVector, f32);
define_vector_struct_type_alias!(struct ComplexTimeVector32, based_on: ComplexTimeVector, f32);
define_vector_struct_type_alias!(struct ComplexFreqVector32, based_on: ComplexFreqVector, f32);
define_vector_struct_type_alias!(struct DataVector64, based_on: GenericDataVector, f64);
define_vector_struct_type_alias!(struct RealTimeVector64, based_on: RealTimeVector, f64);
define_vector_struct_type_alias!(struct RealFreqVector64, based_on: RealFreqVector, f64);
define_vector_struct_type_alias!(struct ComplexTimeVector64, based_on: ComplexTimeVector, f64);
define_vector_struct_type_alias!(struct ComplexFreqVector64, based_on: ComplexFreqVector, f64);

/// An alternative way to define operations on a vector.
/// Warning: Highly unstable and not even fully implemented right now.
///
/// In future this enum will likely be deleted or hidden and be replaced with a builder
/// pattern. The advantage of this is that with the builder we have the means to define at 
/// compile time what kind of vector will result from the given set of operations.
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum Operation<T>
{
	AddReal(T),
	AddComplex(Complex<T>),
	//AddVector(&'a DataVector32<'a>),
	MultiplyReal(T),
	MultiplyComplex(Complex<T>),
	//MultiplyVector(&'a DataVector32<'a>),
	AbsReal,
	AbsComplex,
	Sqrt
}

#[inline]
impl<T> GenericDataVector<T> 
    where T: RealNumber
{  
    fn reallocate(&mut self, length: usize)
	{
		if length > self.allocated_len()
		{
			let data = &mut self.data;
			data.resize(length, T::zero());
            if self.multicore_settings.early_temp_allocation {
                let temp = &mut self.temp;
                temp.resize(length, T::zero());
            }
		}
		
		self.valid_len = length;
	}
	
	fn swap_data_temp(mut self) -> Self
	{
		let temp = self.temp;
		self.temp = self.data;
		self.data = temp;
		self
	}
	
	fn array_to_complex(array: &[T]) -> &[Complex<T>] {
		unsafe { 
			let len = array.len();
			let trans: &[Complex<T>] = mem::transmute(array);
			&trans[0 .. len / 2]
		}
	}
	
	fn array_to_complex_mut(array: &mut [T]) -> &mut [Complex<T>] {
		unsafe { 
			let len = array.len();
			let trans: &mut [Complex<T>] = mem::transmute(array);
			&mut trans[0 .. len / 2]			
		}
	}
}

impl GenericDataVector<f32> {
    /// Perform a set of operations on the given vector. 
	/// Warning: Highly unstable and not even fully implemented right now.
	///
	/// With this approach we change how we operate on vectors. If you perform
	/// `M` operations on a vector with the length `N` you iterate wit hall other methods like this:
	///
	/// ```
	/// // pseudocode:
	/// // for m in M:
	/// //  for n in N:
	/// //    execute m on n
	/// ```
	///
	/// with this method the pattern is changed slighly:
	///
	/// ```
	/// // pseudocode:
	/// // for n in N:
	/// //  for m in M:
	/// //    execute m on n
	/// ```
	///
	/// Both variants have the same complexity however the second one is benificial since we
	/// have increased locality this way. This should help us by making better use of registers and 
	/// CPU buffers. This might also help since for large data we might have the chance in future to 
	/// move the data to a GPU, run all operations and get the result back. In this case the GPU is fast
	/// for many operations but the roundtrips on the bus should be minimized to keep the speed advantage.
	pub fn perform_operations(mut self, operations: &[Operation<f32>])
		-> Self
	{
        if operations.len() == 0
		{
			return DataVector32 { data: self.data, .. self };
		}
		
		let data_length = self.len();
		let scalar_length = data_length % Reg32::len();
		let vectorization_length = data_length - scalar_length;
		if scalar_length > 0
		{
			panic!("perform_operations requires right now that the array length is dividable by 4")
		}
		
		{
			let mut array = &mut self.data;
			Chunk::execute_partial(
                Complexity::Large, &self.multicore_settings,
                &mut array, vectorization_length, Reg32::len(), 
                operations, 
                DataVector32::perform_operations_par);
		}
		DataVector32 { data: self.data, .. self }
	}
    
	fn perform_operations_par(array: &mut [f32], operations: &[Operation<f32>])
	{
		let mut i = 0;
		while i < array.len()
		{ 
			let mut vector = Reg32::load(array, i);
			let mut j = 0;
			while j < operations.len()
			{
				let operation = &operations[j];
				match *operation
				{
					Operation::AddReal(value) =>
					{
						vector = vector.add_real(value);
					}
					Operation::AddComplex(value) =>
					{
						vector = vector.add_complex(value);
					}
					/*Operation32::AddVector(value) =>
					{
						// TODO
					}*/
					Operation::MultiplyReal(value) =>
					{
						vector = vector.scale_real(value);
					}
					Operation::MultiplyComplex(value) =>
					{
						vector = vector.scale_complex(value);
					}
					/*Operation32::MultiplyVector(value) =>
					{
						// TODO
					}*/
					Operation::AbsReal =>
					{
						vector.store(array, i);
						{
							let mut content = &mut array[i .. i + Reg32::len()];
							let mut k = 0;
							while k < Reg32::len()
							{
								content[k] = content[k].abs();
								k = k + 1;
							}
						}
						vector = Reg32::load(array, i);
					}
					Operation::AbsComplex =>
					{
						vector = vector.complex_abs();
					}
					Operation::Sqrt =>
					{
						vector.store(array, i);
						{
							let mut content = &mut array[i .. i + Reg32::len()];
							let mut k = 0;
							while k < Reg32::len()
							{
								content[k] = content[k].sqrt();
								k = k + 1;
							}
						}
						vector = Reg32::load(array, i);
					}
				}
				j += 1;
			}
		
			vector.store(array, i);	
			i += Reg32::len();
		}
	}
}
   
#[cfg(test)]
mod tests {
	use super::*;
	use num::complex::Complex32;

	#[test]
	fn construct_real_time_vector_32_test()
	{
		let array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
		let vector = RealTimeVector32::from_array(&array);
		assert_eq!(vector.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
		assert_eq!(vector.delta(), 1.0);
		assert_eq!(vector.domain(), DataVectorDomain::Time);
	}
	
	#[test]
	fn add_real_one_32_test()
	{
		let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
		let vector = RealTimeVector32::from_array(&mut data);
		let result = vector.real_offset(1.0).unwrap();
		let expected = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
		assert_eq!(result.data, expected);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn add_real_two_32_test()
	{
		// Test also that vector calls are possible
		let data = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0);
		let result = RealTimeVector32::from_array(&data);
		let result = result.real_offset(2.0).unwrap();
		assert_eq!(result.data, [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn add_complex_32_test()
	{
		let data = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
		let result = ComplexTimeVector32::from_interleaved(&data);
		let result = result.complex_offset(Complex32::new(1.0, -1.0)).unwrap();
		assert_eq!(result.data, [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0]);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn multiply_real_two_32_test()
	{
		let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
		let result = RealTimeVector32::from_array(&data);
		let result = result.real_scale(2.0).unwrap();
		let expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
		assert_eq!(result.data, expected);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn multiply_complex_32_test()
	{
		let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
		let result = ComplexTimeVector32::from_interleaved(&data);
		let result = result.complex_scale(Complex32::new(2.0, -3.0)).unwrap();
		let expected = [8.0, 1.0, 18.0, -1.0, 28.0, -3.0, 38.0, -5.0];
		assert_eq!(result.data, expected);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn abs_real_32_test()
	{
		let data = [-1.0, 2.0, -3.0, 4.0, -5.0, -6.0, 7.0, -8.0];
		let result = RealTimeVector32::from_array(&data);
		let result = result.real_abs().unwrap();
		let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
		assert_eq!(result.data, expected);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn abs_complex_32_test()
	{
		let data = [3.0, 4.0, -3.0, 4.0, 3.0, -4.0, -3.0, -4.0];
		let result = ComplexTimeVector32::from_interleaved(&data);
		let result = result.complex_abs().unwrap();
		let expected = [5.0, 5.0, 5.0, 5.0];
		assert_eq!(result.data(), expected);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn abs_complex_squared_32_test()
	{
		let data = [-1.0, 2.0, -3.0, 4.0, -5.0, -6.0, 7.0, -8.0, 9.0, 10.0];
		let result = ComplexTimeVector32::from_interleaved(&data);
		let result = result.complex_abs_squared().unwrap();
		let expected = [5.0, 25.0, 61.0, 113.0, 181.0];
		assert_eq!(result.data(), expected);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn indexer_test()
	{
		let data = [1.0, 2.0, 3.0, 4.0];
		let mut result = ComplexTimeVector32::from_interleaved(&data);
		assert_eq!(result[0], 1.0);
		result[0] = 5.0;
		assert_eq!(result[0], 5.0);
		let expected = [5.0, 2.0, 3.0, 4.0];
		assert_eq!(result.data(), expected);
	}
	
	#[test]
	fn add_vector_test()
	{
		let data1 = [1.0, 2.0, 3.0, 4.0];
		let vector1 = ComplexTimeVector32::from_interleaved(&data1);
		let data2 = [5.0, 7.0, 9.0, 11.0];
		let vector2 = ComplexTimeVector32::from_interleaved(&data2);
		let result = vector1.add_vector(&vector2).unwrap();
		let expected = [6.0, 9.0, 12.0, 15.0];
		assert_eq!(result.data(), expected);
	}
	
	#[test]
	fn multiply_complex_vector_32_test()
	{
		let a = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
		let b = ComplexTimeVector32::from_interleaved(&[2.0, -3.0, -2.0, 3.0, 2.0, -3.0, -2.0, 3.0]);
		let result = a.multiply_vector(&b).unwrap();
		let expected = [8.0, 1.0, -18.0, 1.0, 28.0, -3.0, -38.0, 5.0];
		assert_eq!(result.data, expected);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn divide_complex_vector_32_test()
	{
		let a = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
		let b = ComplexTimeVector32::from_interleaved(&[-1.0, 0.0, 0.0, 1.0, 2.0, -3.0]);
		let result = a.divide_vector(&b).unwrap();
		let expected = [-1.0, -2.0, 4.0, -3.0, -8.0/13.0, 27.0/13.0];
		assert_eq!(result.data, expected);
		assert_eq!(result.delta, 1.0);
	}
	
	#[test]
	fn array_to_complex_test()
	{
		let a = [1.0; 10];
		let c = DataVector32::array_to_complex(&a);
		let expected = [Complex32::new(1.0, 1.0); 5];
		assert_eq!(&expected, c);
	}
	
	#[test]
	fn array_to_complex_mut_test()
	{
		let mut a = [1.0; 10];
		let c = DataVector32::array_to_complex_mut(&mut a);
		let expected = [Complex32::new(1.0, 1.0); 5];
		assert_eq!(&expected, c);
	}
    
    #[test]
	fn swap_halves_real_even_test()
	{
		let mut a = [1.0, 2.0, 3.0, 4.0];
		let c = RealTimeVector32::from_array(&mut a);
        let r = c.swap_halves().unwrap();
		assert_eq!(r.data(), &[3.0, 4.0, 1.0, 2.0]);
	}
    
    #[test]
	fn swap_halves_real_odd_test()
	{
		let mut a = [1.0, 2.0, 3.0, 4.0, 5.0];
		let c = RealTimeVector32::from_array(&mut a);
        let r = c.swap_halves().unwrap();
		assert_eq!(r.data(), &[4.0, 5.0, 3.0, 1.0, 2.0]);
	}
    
    #[test]
	fn swap_halves_complex_even_test()
	{
		let mut a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
		let c = ComplexTimeVector32::from_interleaved(&mut a);
        let r = c.swap_halves().unwrap();
		assert_eq!(r.data(), &[5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]);
	}
    
    #[test]
	fn swap_halves_complex_odd_test()
	{
		let mut a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
		let c = ComplexTimeVector32::from_interleaved(&mut a);
        let r = c.swap_halves().unwrap();
		assert_eq!(r.data(), &[7.0, 8.0, 9.0, 10.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
	}
}