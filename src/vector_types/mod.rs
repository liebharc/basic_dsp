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
		/// you will never trigger an invalid operation during runtime. In case that this isn't desired or the information
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

macro_rules! define_generic_operations_forward {
    (from: $name:ident, to: $gen_type:ident)
	 =>
	 {
		#[inline]
		impl $name
		{
			pub fn perform_operations(self, operations: &[Operation32]) -> $name
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
		impl $name
		{
			pub fn real_offset(self, offset: <$name as DataVector>::E) -> $name
			{
				$name::from_gen(self.to_gen().real_offset(offset))
			}
			
			pub fn real_scale(self, factor: <$name as DataVector>::E) -> $name
			{
				$name::from_gen(self.to_gen().real_scale(factor))
			}
					
			pub fn real_abs(self) -> $name
			{
				$name::from_gen(self.to_gen().real_abs()) 
			}
			
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
		impl $name
		{
			pub fn complex_offset(self, offset: $complex_type) -> $name
			{
				$name::from_gen(self.to_gen().complex_offset(offset))
			}		
			
			// We are keeping this since scaling with a real number should be faster
			pub fn real_scale(self, factor: <$name as DataVector>::E) -> $name
			{
				$name::from_gen(self.to_gen().real_scale(factor))
			}
				
			pub fn complex_scale(self, factor: $complex_type) -> $name
			{
				$name::from_gen(self.to_gen().complex_scale(factor))
			}
			
			pub fn complex_abs(self) -> $real_partner
			{
				$real_partner::from_gen(self.to_gen().complex_abs())
			}
			
			pub fn complex_abs_squared(self) -> $real_partner
			{
				$real_partner::from_gen(self.to_gen().complex_abs_squared())
			}
			
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