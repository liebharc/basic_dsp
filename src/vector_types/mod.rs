//! A module for very basic digital signal processing (DSP) operations on data vectors.

macro_rules! define_vector_struct {
    (struct $name:ident,$data_type:ident) => {
		/// A 1xN (one times N elements) or Nx1 data vector as used for most digital signal processing (DSP) operations.
		/// All data vector operations consume the vector they operate on and return a new vector. A consumed vector
		/// must not be accessed again.
		///
		/// Vectors come in different flavors:
		///
		/// 1. Time or Frequency domain
		/// 2. Real or Complex numbers
		/// 3. 32bit or 64bit floating point numbers
		///
		/// The first two flavors define meta information about the vector and provide compile time information what
		/// operations are available with the given vector and how this will transform the vector. This makes sure that
		/// some invalid operations are already discovered at compile time. In case that this isn't desired or the information
		/// about the vector isn't known at compile time there are the generic [`DataVector32`](struct.DataVector32.html) and [`DataVector64`](struct.DataVector64.html) vectors
		/// available.
		///
		/// 32bit and 64bit flavors trade performance and memory consumption against accuracy. 32bit vectors are roughly
		/// two times faster than 64bit vectors for most operations. But remember that you should benchmark first
		/// before you give away accuracy for performance unless however you are sure that 32bit accuracy is certainly good
		/// enough.
		pub struct $name
		{
			data: Vec<$data_type>,
			temp: Vec<$data_type>,
			delta: $data_type,
			domain: DataVectorDomain,
			is_complex: bool,
			points: usize
			// We could need here (or in one of the traits/impl):
			// - A view for complex data types with transmute
		}
		
		#[inline]
		#[allow(unused_variables)]
		impl DataVector for $name
		{
			type E = $data_type;
			
			
			fn len(&self) -> usize 
			{
				self.data.len()
			}
			
			fn data(&self) -> &[$data_type]
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
			
			fn points(&self) -> usize
			{
				if self.is_complex
				{
					self.len() / 2
				}
				else
				{
					self.len()
				}
			}
		}
		
		impl Index<usize> for $name
		{
			type Output = $data_type;
		
			fn index(&self, index: usize) -> &$data_type
			{
				&self.data[index]
			}
		}
		
		impl IndexMut<usize> for $name
		{
			fn index_mut(&mut self, index: usize) -> &mut $data_type
			{
				&mut self.data[index]
			}
		}
		
		impl Index<Range<usize>> for $name
		{
			type Output = [$data_type];
		
			fn index(&self, index: Range<usize>) -> &[$data_type]
			{
				&self.data[index]
			}
		}
		
		impl IndexMut<Range<usize>> for $name
		{
			fn index_mut(&mut self, index: Range<usize>) -> &mut [$data_type]
			{
				&mut self.data[index]
			}
		}
		
		impl Index<RangeFrom<usize>> for $name
		{
			type Output = [$data_type];
		
			fn index(&self, index: RangeFrom<usize>) -> &[$data_type]
			{
				&self.data[index]
			}
		}
		
		impl IndexMut<RangeFrom<usize>> for $name
		{
			fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [$data_type]
			{
				&mut self.data[index]
			}
		}
		
		impl Index<RangeTo<usize>> for $name
		{
			type Output = [$data_type];
		
			fn index(&self, index: RangeTo<usize>) -> &[$data_type]
			{
				&self.data[index]
			}
		}
		
		impl IndexMut<RangeTo<usize>> for $name
		{
			fn index_mut(&mut self, index: RangeTo<usize>) -> &mut [$data_type]
			{
				&mut self.data[index]
			}
		}
		
		impl Index<RangeFull> for $name
		{
			type Output = [$data_type];
		
			fn index(&self, index: RangeFull) -> &[$data_type]
			{
				&self.data[index]
			}
		}
		
		impl IndexMut<RangeFull> for $name
		{
			fn index_mut(&mut self, index: RangeFull) -> &mut [$data_type]
			{
				&mut self.data[index]
			}
		}
    }
}

macro_rules! define_generic_operations_forward {
	(from: $name:ident, to: $gen_type:ident)
	 =>
	 {
	 	#[inline]
		impl GenericVectorOperations for $name
		{
			fn add_vector(self, summand: &Self) -> Self
			{
				$name::from_gen(self.to_gen().add_vector(&summand.to_gen_borrow()))
			}
	
			fn subtract_vector(self, subtrahend: &Self) -> Self
			{
				$name::from_gen(self.to_gen().subtract_vector(&subtrahend.to_gen_borrow()))
			}
			
			fn multiply_vector(self, factor: &Self) -> Self
			{
				$name::from_gen(self.to_gen().multiply_vector(&factor.to_gen_borrow()))
			}
			
			fn divide_vector(self, divisor: &Self) -> Self
			{
				$name::from_gen(self.to_gen().divide_vector(&divisor.to_gen_borrow()))
			}
		}
	}	
}

macro_rules! define_real_basic_struct_members {
    (impl $name:ident, DataVectorDomain::$domain:ident)
	 =>
	 {
		#[inline]
		impl $name
		{
			/// Creates a real `DataVector` by consuming a `Vec`. 
			///
			/// This operation is more memory efficient than the other options to create a vector,
			/// however if used outside of Rust then it holds the risk that the user will access 
			/// the data parameter after the vector has been created causing all types of issues.  
			pub fn from_array_no_copy(data: Vec<<$name as DataVector>::E>) -> $name
			{
				let data_length = data.len();
				$name 
				{ 
				  data: data, 
				  temp: vec![0.0; data_length],
				  delta: 1.0,
				  domain: DataVectorDomain::$domain,
				  is_complex: false,
				  points: data_length
				}
			}
		
			/// Creates a real `DataVector` from an array or sequence. `delta` is defaulted to `1`.
			pub fn from_array(data: &[<$name as DataVector>::E]) -> $name
			{
				let data_length = data.len();
				$name 
				{ 
				  data: data.to_vec(), 
				  temp: vec![0.0; data_length],
				  delta: 1.0,
				  domain: DataVectorDomain::$domain,
				  is_complex: false,
				  points: data_length
				}
			}
			
			/// Creates a real `DataVector` from an array or sequence and sets `delta` to the given value.
			pub fn from_array_with_delta(data: &[<$name as DataVector>::E], delta: <$name as DataVector>::E) -> $name
			{
				let data_length = data.len();
				$name 
				{ 
				  data: data.to_vec(), 
				  temp: vec![0.0; data_length],
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: false,
				  points: data_length
				}
			}
		}
	 }
}

macro_rules! define_real_operations_forward {
    (from: $name:ident, to: $gen_type:ident)
	 =>
	 {
	 	#[inline]
		impl RealVectorOperations for $name
		{
			fn real_offset(self, offset: <$name as DataVector>::E) -> $name
			{
				$name::from_gen(self.to_gen().real_offset(offset))
			}
			
			fn real_scale(self, factor: <$name as DataVector>::E) -> $name
			{
				$name::from_gen(self.to_gen().real_scale(factor))
			}
					
			fn real_abs(self) -> $name
			{
				$name::from_gen(self.to_gen().real_abs()) 
			}
			
			fn real_sqrt(self) -> $name
			{
				$name::from_gen(self.to_gen().real_sqrt()) 
			}
		}
	 
		#[inline]
		impl $name
		{		
			fn to_gen(self) -> $gen_type
			{
				$gen_type 
				{ 
				  data: self.data,
				  temp: self.temp,
				  delta: self.delta,
				  domain: self.domain,
				  is_complex: self.is_complex,
				  points: self.points
				}
			}
			
			fn to_gen_borrow(&self) -> &$gen_type
			{
				unsafe { mem::transmute(self) }
			}
			
			fn from_gen(other: $gen_type) -> $name
			{
				$name 
				{ 
				  data: other.data,
				  temp: other.temp,
				  delta: other.delta,
				  domain: other.domain,
				  is_complex: other.is_complex,
				  points: other.points
				}
			}
		}
	 }
}

macro_rules! define_complex_basic_struct_members {
    (impl $name:ident, DataVectorDomain::$domain:ident)
	 =>
	 {
		#[inline]
		impl $name
		{
			/// Creates a complex `DataVector` by consuming a `Vec`. Data is in interleaved format: `i0, q0, i1, q1, ...`. 
			///
			/// This operation is more memory efficient than the other options to create a vector,
			/// however if used outside of Rust then it holds the risk that the user will access 
			/// the data parameter after the vector has been created causing all types of issues.  
			pub fn from_interleaved_no_copy(data: Vec<<$name as DataVector>::E>) -> $name
			{
				let data_length = data.len();
				$name 
				{ 
				  data: data,
				  temp: vec![0.0; data_length],
				  delta: 1.0,
				  domain: DataVectorDomain::$domain,
				  is_complex: true,
				  points: data_length / 2
				}
			}
			
			/// Creates a complex `DataVector` by consuming a `Vec`. Data is in interleaved format: `i0, q0, i1, q1, ...`. `delta` is defaulted to `1`.
			pub fn from_interleaved(data: &[<$name as DataVector>::E]) -> $name
			{
				let data_length = data.len();
				$name 
				{ 
				  data: data.to_vec(), 
				  temp: vec![0.0; data_length],
				  delta: 1.0,
				  domain: DataVectorDomain::$domain,
				  is_complex: true,
				  points: data_length / 2
				}
			}
			
			/// Creates a complex `DataVector` by consuming a `Vec`. Data is in interleaved format: `i0, q0, i1, q1, ...`. `delta` is set to the given value.
			pub fn from_interleaved_with_delta(data: &[<$name as DataVector>::E], delta: <$name as DataVector>::E) -> $name
			{
				let data_length = data.len();
				$name 
				{ 
				  data: data.to_vec(), 
				  temp: vec![0.0; data_length],
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
    (from: $name:ident, to: $gen_type:ident, complex: $complex_type:ident, real_partner: $real_partner:ident)
	 =>
	 { 
	 	#[inline]
		impl ComplexVectorOperations for $name
		{
			type RealPartner = $real_partner;
			type Complex = $complex_type;
			
			fn complex_offset(self, offset: $complex_type) -> $name
			{
				$name::from_gen(self.to_gen().complex_offset(offset))
			}
				
			fn complex_scale(self, factor: $complex_type) -> $name
			{
				$name::from_gen(self.to_gen().complex_scale(factor))
			}
			
			fn complex_abs(self) -> $real_partner
			{
				$real_partner::from_gen(self.to_gen().complex_abs())
			}
			
			fn complex_abs_squared(self) -> $real_partner
			{
				$real_partner::from_gen(self.to_gen().complex_abs_squared())
			}
			
			fn complex_conj(self) -> $name
			{
				$name::from_gen(self.to_gen().complex_conj())
			}
		}
	 	
		#[inline]
		impl $name
		{
			fn to_gen(self) -> $gen_type
			{
				$gen_type 
				{ 
				  data: self.data,
				  temp: self.temp,
				  delta: self.delta,
				  domain: self.domain,
				  is_complex: self.is_complex,
				  points: self.points
				}
			}
			
			fn to_gen_borrow(&self) -> &$gen_type
			{
				unsafe { mem::transmute(self) }
			}
			
			fn from_gen(other: $gen_type) -> $name
			{
				$name 
				{ 
				  data: other.data,
				  temp: other.temp,
				  delta: other.delta,
				  domain: other.domain,
				  is_complex: other.is_complex,
				  points: other.points
				}
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
		GenericVectorOperations,
		RealVectorOperations,
		ComplexVectorOperations,
		TimeDomainOperations,
		FrequencyDomainOperations,		
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