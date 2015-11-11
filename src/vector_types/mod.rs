macro_rules! define_vector_struct {
    (struct $name:ident,$data_type:ident) => {
		pub struct $name<'a>
		{
			data: &'a mut [$data_type],
			delta: $data_type,
			domain: DataVectorDomain,
			is_complex: bool,
			points: usize
			// We could need here (or in one of the traits/impl):
			// - A view for complex data types with transmute
			// - A temporary array to store data in
		}
		
		#[inline]
		#[allow(unused_variables)]
		impl<'a> DataVector for $name<'a>
		{
			type E = $data_type;
			
			fn len(&self) -> usize 
			{
				self.data.len()
			}
			
			fn data(&mut self) -> &[$data_type]
			{
				let valid_length =
				 if self.is_complex
				 {
					 self.points * 2
				 }
				 else
				 {
					 self.points
				 };
				 
				&self.data[0 .. valid_length]
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
				let data_length = data.len();
				$name 
				{ 
				  data: data, 
				  delta: 1.0,
				  domain: DataVectorDomain::$domain,
				  is_complex: false,
				  points: data_length
				}
			}
			
			pub fn from_array_with_delta<'b>(data: &'b mut [<$name as DataVector>::E], delta: <$name as DataVector>::E) -> $name<'b>
			{
				let data_length = data.len();
				$name 
				{ 
				  data: data, 
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: false,
				  points: data_length
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
			pub fn perform_operations(&mut self, operations: &[Operation32]) -> $name
			{
				$name::from_gen(self.to_gen().perform_operations(operations))
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
			pub fn inplace_real_offset(&mut self, offset: <$name as DataVector>::E) 
			{
				self.to_gen().inplace_real_offset(offset);
			}
			
			pub fn inplace_real_scale(&mut self, factor: <$name as DataVector>::E) 
			{
				self.to_gen().inplace_real_scale(factor);
			}
					
			pub fn inplace_real_abs(&mut self)
			{
				self.to_gen().inplace_real_abs();
			}
			
			#[allow(dead_code)]
			fn to_gen(&mut self) -> &mut $gen_type
			{
				unsafe { mem::transmute(self) }
			}
			
			#[allow(dead_code)]
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
				let data_length = data.len();
				$name 
				{ 
				  data: data, 
				  delta: 1.0,
				  domain: DataVectorDomain::$domain,
				  is_complex: true,
				  points: data_length / 2
				}
			}
			
			pub fn from_interleaved_with_delta<'b>(data: &'b mut [<$name as DataVector>::E], delta: <$name as DataVector>::E) -> $name<'b>
			{
				let data_length = data.len();
				$name 
				{ 
				  data: data, 
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: true,
				  points: data_length / 2
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
			pub fn inplace_complex_offset(&mut self, offset: $complex_type) 
			{
				self.to_gen().inplace_complex_offset(offset);
			}		
			
			// We are keeping this since scaling with a real number should be faster
			pub fn inplace_real_scale(&mut self, factor: <$name as DataVector>::E) 
			{
				self.to_gen().inplace_real_scale(factor);
			}
				
			pub fn inplace_complex_scale(&mut self, factor: $complex_type) 
			{
				self.to_gen().inplace_complex_scale(factor);
			}
			
			pub fn inplace_complex_abs(&mut self)
			{
				self.to_gen().inplace_complex_abs();
			}
			
			pub fn inplace_complex_abs_squared(&mut self)
			{
				self.to_gen().inplace_complex_abs_squared();
			}
			
			#[allow(dead_code)]
			fn to_gen(&mut self) -> &mut $gen_type
			{
				unsafe { mem::transmute(self) }
			}
			
			#[allow(dead_code)]
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
		Operation32
	};
pub use vector_types::vector64::
	{	
		DataVector64, 
		RealTimeVector64,
		ComplexTimeVector64, 
		RealFreqVector64,
		ComplexFreqVector64
	};