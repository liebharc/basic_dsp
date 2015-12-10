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
        #[derive(Debug)]        
		pub struct $name
		{
			data: Vec<$data_type>,
			temp: Vec<$data_type>,
			delta: $data_type,
			domain: DataVectorDomain,
			is_complex: bool,
			valid_len: usize,
            multicore_settings: MultiCoreSettings 
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
				self.valid_len
			}
			
			fn allocated_len(&self) -> usize
			{
				self.data.len()
			}
			
			fn data(&self) -> &[$data_type]
			{
				let valid_length = self.len();
				 
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
				self.valid_len / if self.is_complex { 2 } else { 1 }
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
			fn add_vector(self, summand: &Self) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().add_vector(&summand.to_gen_borrow()))
			}
	
			fn subtract_vector(self, subtrahend: &Self) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().subtract_vector(&subtrahend.to_gen_borrow()))
			}
			
			fn multiply_vector(self, factor: &Self) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().multiply_vector(&factor.to_gen_borrow()))
			}
			
			fn divide_vector(self, divisor: &Self) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().divide_vector(&divisor.to_gen_borrow()))
			}
			
			fn zero_pad(self, points: usize) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().zero_pad(points))
			}
			
			fn zero_interleave(self) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().zero_interleave())
			}
			
			fn diff(self) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().diff())
			}
			
			fn diff_with_start(self) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().diff_with_start())
			}
			
			fn cum_sum(self) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().cum_sum())
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
				  valid_len: data_length,
                  multicore_settings: MultiCoreSettings::new()
				}
			}
		
			/// Creates a real `DataVector` from an array or sequence. `delta` is defaulted to `1`.
			pub fn from_array(data: &[<$name as DataVector>::E]) -> $name
			{
				$name::from_array_with_delta(data, 1.0)
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
				  valid_len: data_length,
                  multicore_settings: MultiCoreSettings::new()
				}
			}
            
            /// Creates a real and empty `DataVector` and sets `delta` to 1.0 value.
            pub fn real_empty() -> $name
            {
                $name 
				{ 
				  data: vec![0.0; 0], 
				  temp: vec![0.0; 0],
				  delta: 1.0,
				  domain: DataVectorDomain::$domain,
				  is_complex: false,
				  valid_len: 0,
                  multicore_settings: MultiCoreSettings::new()
				}
            }
            
            /// Creates a real and empty `DataVector` and sets `delta` to the given value.
            pub fn real_empty_with_delta(delta: <$name as DataVector>::E) -> $name
            {
                $name 
				{ 
				  data: vec![0.0; 0], 
				  temp: vec![0.0; 0],
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: false,
				  valid_len: 0,
                  multicore_settings: MultiCoreSettings::new()
				}
            }
            
            /// Creates a real `DataVector` with `length` elements all set to the value of `constant`. `delta` is defaulted to `1`.
			pub fn real_from_constant(constant: <$name as DataVector>::E, length: usize) -> $name
			{
				$name::real_from_constant_with_delta(constant, length, 1.0)
			}
			
			/// Creates a real `DataVector` with `length` elements all set to the value of `constant` and sets `delta` to the given value.
			pub fn real_from_constant_with_delta(constant: <$name as DataVector>::E, length: usize, delta: <$name as DataVector>::E) -> $name
			{
				$name 
				{ 
				  data: vec![constant; length],
				  temp: vec![0.0; length],
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: false,
				  valid_len: length,
                  multicore_settings: MultiCoreSettings::new()
				}
			}
		}
	 }
}

macro_rules! define_real_operations_forward {
    (from: $name:ident, to: $gen_type:ident, complex_partner: $complex_partner:ident)
	 =>
	 {	 
	 	#[inline]
		impl RealVectorOperations for $name
		{
			type ComplexPartner = $complex_partner; 
			
			fn real_offset(self, offset: <$name as DataVector>::E) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().real_offset(offset))
			}
			
			fn real_scale(self, factor: <$name as DataVector>::E) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().real_scale(factor))
			}
					
			fn real_abs(self) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().real_abs()) 
			}
			
			fn real_sqrt(self) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().real_sqrt()) 
			}
			
			fn real_square(self) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().real_square()) 
			}
			
			fn real_root(self, degree: Self::E) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().real_root(degree)) 
			}
			
			fn real_power(self, exponent: Self::E) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().real_power(exponent)) 
			}
			
			fn real_logn(self) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().real_logn()) 
			}
			
			fn real_expn(self) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().real_expn()) 
			}
		
			fn real_log_base(self, base: Self::E) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().real_log_base(base)) 
			}
            
            fn real_sin(self) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().real_sin()) 
			}
            
            fn real_cos(self) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().real_cos()) 
			}
			
			fn real_exp_base(self, base: Self::E) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().real_exp_base(base)) 
			}
			
			fn to_complex(self) -> VecResult<$complex_partner>
			{
				$complex_partner::from_genres(self.to_gen().to_complex()) 
			}
			
			fn wrap(self, divisor: Self::E) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().wrap(divisor))
			}
			
			fn unwrap(self, divisor: Self::E) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().unwrap(divisor))
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
				  valid_len: self.valid_len,
                  multicore_settings: MultiCoreSettings::new()
				}
			}
			
			fn to_gen_borrow(&self) -> &$gen_type
			{
				unsafe { mem::transmute(self) }
			}
			
			#[allow(dead_code)]
			fn to_gen_mut_borrow(&mut self) -> &mut $gen_type
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
				  valid_len: other.valid_len,
                  multicore_settings: MultiCoreSettings::new()
				}
			}
            
            fn from_genres(other: VecResult<$gen_type>) -> VecResult<$name>
			{
				match other {
                    Ok(v) => Ok($name::from_gen(v)),
                    Err((r, v)) => Err((r, $name::from_gen(v)))
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
				  valid_len: data_length,
                  multicore_settings: MultiCoreSettings::new()
				}
			}
			
			/// Creates a complex `DataVector` from an array or sequence. Data is in interleaved format: `i0, q0, i1, q1, ...`. `delta` is defaulted to `1`.
			pub fn from_interleaved(data: &[<$name as DataVector>::E]) -> $name
			{
				$name::from_interleaved_with_delta(data, 1.0)
			}
			
			/// Creates a complex `DataVector` from an array or sequence. Data is in interleaved format: `i0, q0, i1, q1, ...`. `delta` is set to the given value.
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
				  valid_len: data_length,
                  multicore_settings: MultiCoreSettings::new()
				}
			}
            
            /// Creates a complex and empty `DataVector` and sets `delta` to 1.0 value.
            pub fn complex_empty() -> $name
            {
                $name 
				{ 
				  data: vec![0.0; 0], 
				  temp: vec![0.0; 0],
				  delta: 1.0,
				  domain: DataVectorDomain::$domain,
				  is_complex: true,
				  valid_len: 0,
                  multicore_settings: MultiCoreSettings::new()
				}
            }
            
            /// Creates a complex and empty `DataVector` and sets `delta` to the given value.
            pub fn complex_empty_with_delta(delta: <$name as DataVector>::E) -> $name
            {
                $name 
				{ 
				  data: vec![0.0; 0], 
				  temp: vec![0.0; 0],
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: true,
				  valid_len: 0,
                  multicore_settings: MultiCoreSettings::new()
				}
            }
            
            /// Creates a complex `DataVector` with `length` elements all set to the value of `constant`. `delta` is defaulted to `1`.
			pub fn complex_from_constant(constant: <$name as DataVector>::E, length: usize) -> $name
			{
				$name::complex_from_constant_with_delta(constant, length, 1.0)
			}
			
			/// Creates a complex `DataVector` with `length` elements all set to the value of `constant` and sets `delta` to the given value.
			pub fn complex_from_constant_with_delta(constant: <$name as DataVector>::E, length: usize, delta: <$name as DataVector>::E) -> $name
			{
				$name 
				{ 
				  data: vec![constant; length],
				  temp: vec![0.0; length],
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: true,
				  valid_len: length,
                  multicore_settings: MultiCoreSettings::new()
				}
			}
			
			/// Creates a complex  `DataVector` from an array with real and an array imaginary data. `delta` is set to 1.
			///
			/// Arrays must have the same length.
			pub fn from_real_imag(real: &[<$name as DataVector>::E], imag: &[<$name as DataVector>::E])
				-> $name
			{
				$name::from_real_imag_with_delta(real, imag, 1.0)
			}
			
			/// Creates a complex  `DataVector` from an array with real and an array imaginary data. `delta` is set to the given value 1.
			///
			/// Arrays must have the same length.
			pub fn from_real_imag_with_delta(real: &[<$name as DataVector>::E], imag: &[<$name as DataVector>::E], delta: <$name as DataVector>::E)
				-> $name
			{
				if real.len() != imag.len()
				{
					panic!("Input lengths differ: real has {} elements and imag has {} elements", real.len(), imag.len());
				}
				
				let mut data = Vec::with_capacity(real.len() + imag.len());
				for i in 0 .. real.len() {
					data.push(real[i]);
					data.push(imag[i]);
				}
				
				let data_length = data.len();
				
				$name 
				{ 
				  data: data, 
				  temp: vec![0.0; data_length],
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: true,
				  valid_len: data_length,
                  multicore_settings: MultiCoreSettings::new()
				}
			}
			
			/// Creates a complex  `DataVector` from an array with magnitude and an array with phase data. `delta` is set to 1.
			///
			/// Arrays must have the same length. Phase must be in [rad].
			pub fn from_mag_phase(magnitude: &[<$name as DataVector>::E], phase: &[<$name as DataVector>::E])
				-> $name
			{
				$name::from_mag_phase_with_delta(magnitude, phase, 1.0)
			}
			
			/// Creates a complex  `DataVector` from an array with magnitude and an array with phase data. `delta` is set to the given value 1.
			///
			/// Arrays must have the same length. Phase must be in [rad].
			pub fn from_mag_phase_with_delta(magnitude: &[<$name as DataVector>::E], phase: &[<$name as DataVector>::E], delta: <$name as DataVector>::E)
				-> $name
			{
				if magnitude.len() != phase.len()
				{
					panic!("Input lengths differ: magnitude has {} elements and phase has {} elements", magnitude.len(), phase.len());
				}
				
				let mut data = Vec::with_capacity(magnitude.len() + phase.len());
				for i in 0 .. magnitude.len() {
					let complex = <$name as ComplexVectorOperations>::Complex::from_polar(&magnitude[i], &phase[i]);
					data.push(complex.re);
					data.push(complex.im);
				}
				
				let data_length = data.len();
				
				$name 
				{ 
				  data: data, 
				  temp: vec![0.0; data_length],
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: true,
				  valid_len: data_length,
                  multicore_settings: MultiCoreSettings::new()
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
			
			fn complex_offset(self, offset: $complex_type) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().complex_offset(offset))
			}
				
			fn complex_scale(self, factor: $complex_type) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().complex_scale(factor))
			}
			
			fn complex_abs(self) -> VecResult<$real_partner>
			{
				$real_partner::from_genres(self.to_gen().complex_abs())
			}
			
			fn get_complex_abs(&self, destination: &mut Self::RealPartner) -> VoidResult
			{
				self.to_gen_borrow().get_complex_abs(destination.to_gen_mut_borrow())
			}
			
			fn complex_abs_squared(self) -> VecResult<$real_partner>
			{
				$real_partner::from_genres(self.to_gen().complex_abs_squared())
			}
			
			fn complex_conj(self) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().complex_conj())
			}
			
			fn to_real(self) -> VecResult<Self::RealPartner>
			{
				$real_partner::from_genres(self.to_gen().to_real())
			}
	
			fn to_imag(self) -> VecResult<Self::RealPartner>
			{
				$real_partner::from_genres(self.to_gen().to_imag())
			}	
					
			fn get_real(&self, destination: &mut Self::RealPartner) -> VoidResult
			{
				self.to_gen_borrow().get_real(destination.to_gen_mut_borrow())
			}
			
			fn get_imag(&self, destination: &mut Self::RealPartner) -> VoidResult
			{
				self.to_gen_borrow().get_imag(destination.to_gen_mut_borrow())
			}
			
			fn phase(self) -> VecResult<Self::RealPartner>
			{
				$real_partner::from_genres(self.to_gen().phase())
			}
			
			fn get_phase(&self, destination: &mut Self::RealPartner) -> VoidResult
			{
				self.to_gen_borrow().get_phase(destination.to_gen_mut_borrow())
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
				  valid_len: self.valid_len,
                  multicore_settings: MultiCoreSettings::new()
				}
			}
			
			fn to_gen_borrow(&self) -> &$gen_type
			{
				unsafe { mem::transmute(self) }
			}
			
			#[allow(dead_code)]
			fn to_gen_mut_borrow(&mut self) -> &mut $gen_type
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
				  valid_len: other.valid_len,
                  multicore_settings: MultiCoreSettings::new()
				}
			}
            
            fn from_genres(other: VecResult<$gen_type>) -> VecResult<$name>
			{
				match other {
                    Ok(v) => Ok($name::from_gen(v)),
                    Err((r, v)) => Err((r, $name::from_gen(v)))
                }
			}
		} 
	 }
}

macro_rules! reject_if {
    ($self_: ident, $condition: expr, $message: expr) => {
        if $condition
        {
            return Err(($message, $self_));
        }
    }
}

pub mod definitions;
pub mod vector32;
pub mod vector64;

pub use vector_types::definitions::
	{
		DataVectorDomain,
		DataVector,
        VecResult,
        VoidResult,
        ErrorReason,
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