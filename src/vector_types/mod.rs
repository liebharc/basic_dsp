use multicore_support::Chunk;
use databuffer::DataBuffer;
use simd::f32x4;
use simd::x86::sse3::Sse3F32x4;
use simd_extensions::SimdExtensions32;
use num::complex::{Complex32,Complex64};
use std::mem;
use num::traits::Float;

macro_rules! define_vector_struct {
    (struct $name:ident,$data_type:ident) => {
		pub struct $name<'a>
		{
			data: &'a mut [$data_type],
			delta: $data_type,
			domain: DataVectorDomain,
			is_complex: bool
		}
		
		#[inline]
		impl<'a> DataVector for $name<'a>
		{
			type E = $data_type;
			
			fn len(&self) -> usize 
			{
				self.data.len()
			}
			
			fn data(&mut self, buffer: &mut DataBuffer) -> &[$data_type]
			{
				// self.perfom_pending_operations(buffer);
				self.data
			}
			
			fn delta(&self) -> $data_type
			{
				self.delta
			}
			
			fn domain(&self) -> DataVectorDomain
			{
				self.domain
			}
			
			fn is_complex(&self) -> bool
			{
				self.is_complex
			}
		}
    }
}

macro_rules! define_real_basic_struct_members {
    (impl $name:ident, DataVectorDomain::$domain:ident)
	 =>
	 {
		#[inline]
		impl<'a> $name<'a>
		{
			pub fn from_array<'b>(data: &'b mut [<$name as DataVector>::E]) -> $name<'b>
			{
				$name 
				{ 
				  data: data, 
				  delta: 1.0,
				  domain: DataVectorDomain::$domain,
				  is_complex: false
				}
			}
			
			pub fn from_array_with_delta<'b>(data: &'b mut [<$name as DataVector>::E], delta: <$name as DataVector>::E) -> $name<'b>
			{
				$name 
				{ 
				  data: data, 
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: false
				}
			}
		}
	 }
}

macro_rules! define_generic_operations_forward {
    (from: $name:ident, to: $gen_type:ident)
	 =>
	 {
		#[inline]
		impl<'a> $name<'a>
		{
			fn perfom_pending_operations(&mut self, buffer: &mut DataBuffer) -> $name
			{
				$name::from_gen(self.to_gen().perfom_pending_operations(buffer))
			}
		}
	}
}


macro_rules! define_real_operations_forward {
    (from: $name:ident, to: $gen_type:ident)
	 =>
	 {
		#[inline]
		impl<'a> $name<'a>
		{
			pub fn inplace_real_offset(&mut self, offset: <$name as DataVector>::E, buffer: &mut DataBuffer) 
			{
				self.to_gen().inplace_real_offset(offset, buffer);
			}
			
			pub fn inplace_real_scale(&mut self, factor: <$name as DataVector>::E, buffer: &mut DataBuffer) 
			{
				self.to_gen().inplace_real_scale(factor, buffer);
			}
					
			pub fn inplace_real_abs(&mut self, buffer: &mut DataBuffer)
			{
				self.to_gen().inplace_real_abs(buffer);
			}
			
			fn to_gen(&mut self) -> &mut $gen_type
			{
				unsafe { mem::transmute(self) }
			}
			
			fn from_gen(other: $gen_type) -> $name
			{
				unsafe { mem::transmute(other) }
			}
		}
	 }
}

macro_rules! define_complex_basic_struct_members {
    (impl $name:ident, DataVectorDomain::$domain:ident)
	 =>
	 {
		#[inline]
		impl<'a> $name<'a>
		{
			pub fn from_interleaved<'b>(data: &'b mut [<$name as DataVector>::E]) -> $name<'b>
			{
				$name 
				{ 
				  data: data, 
				  delta: 1.0,
				  domain: DataVectorDomain::$domain,
				  is_complex: true
				}
			}
			
			pub fn from_interleaved_with_delta<'b>(data: &'b mut [<$name as DataVector>::E], delta: <$name as DataVector>::E) -> $name<'b>
			{
				$name 
				{ 
				  data: data, 
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: true
				}
			}
		} 
	 }
}

macro_rules! define_complex_operations_forward {
    (from: $name:ident, to: $gen_type:ident, complex: $complex_type:ident)
	 =>
	 { 
		#[inline]
		impl<'a> $name<'a>
		{
			pub fn inplace_complex_offset(&mut self, offset: $complex_type, buffer: &mut DataBuffer) 
			{
				self.to_gen().inplace_complex_offset(offset, buffer);
			}		
			
			// We are keeping this since scaling with a real number should be faster
			pub fn inplace_real_scale(&mut self, factor: <$name as DataVector>::E, buffer: &mut DataBuffer) 
			{
				self.to_gen().inplace_real_scale(factor, buffer);
			}
				
			pub fn inplace_complex_scale(&mut self, factor: $complex_type, buffer: &mut DataBuffer) 
			{
				self.to_gen().inplace_complex_scale(factor, buffer);
			}
			
			pub fn inplace_complex_abs(&mut self, buffer: &mut DataBuffer)
			{
				self.to_gen().inplace_complex_abs(buffer);
			}
			
			pub fn inplace_complex_abs_squared(&mut self, buffer: &mut DataBuffer)
			{
				self.to_gen().inplace_complex_abs_squared(buffer);
			}
			
			fn to_gen(&mut self) -> &mut $gen_type
			{
				unsafe { mem::transmute(self) }
			}
			
			fn from_gen(other: $gen_type) -> $name
			{
				unsafe { mem::transmute(other) }
			}
		} 
	 }
}

pub mod general;
pub mod vector32;
pub mod vector64;

pub use vector_types::general::
	{
		DataVectorDomain,
		DataVector,
	};
pub use vector_types::vector32::
	{
		DataVector32, 
		RealTimeVector32,
		ComplexTimeVector32, 
		RealFreqVector32,
		ComplexFreqVector32,
	};
pub use vector_types::vector64::
	{	
		DataVector64, 
		RealTimeVector64,
		ComplexTimeVector64, 
		RealFreqVector64,
		ComplexFreqVector64
	};