//! A module for very basic digital signal processing (DSP) operations on data vectors.

macro_rules! define_vector_struct {
    (struct $name:ident) => {
        #[derive(Debug)]        
        pub struct $name<T>
            where T: RealNumber
        {
            data: Vec<T>,
            temp: Vec<T>,
            delta: T,
            domain: DataVectorDomain,
            is_complex: bool,
            valid_len: usize,
            multicore_settings: MultiCoreSettings 
            // We could need here (or in one of the traits/impl):
            // - A view for complex data types with transmute
        }
        
        #[inline]
		#[allow(unused_variables)]
		impl<T> DataVector<T> for $name<T>
            where T: RealNumber
		{
			fn len(&self) -> usize 
			{
				self.valid_len
			}
			
			fn allocated_len(&self) -> usize
			{
				self.data.len()
			}
			
			fn data(&self) -> &[T]
			{
				let valid_length = self.len();
				 
				&self.data[0 .. valid_length]
			}
			
			fn delta(&self) -> T
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
		
		impl<T> Index<usize> for $name<T>
            where T: RealNumber
		{
			type Output = T;
		
			fn index(&self, index: usize) -> &T
			{
				&self.data[index]
			}
		}
		
		impl<T> IndexMut<usize> for $name<T>
            where T: RealNumber
		{
			fn index_mut(&mut self, index: usize) -> &mut T
			{
				&mut self.data[index]
			}
		}
		
		impl<T> Index<Range<usize>> for $name<T>
            where T: RealNumber
		{
			type Output = [T];
		
			fn index(&self, index: Range<usize>) -> &[T]
			{
				&self.data[index]
			}
		}
		
		impl<T> IndexMut<Range<usize>> for $name<T>
            where T: RealNumber
		{
			fn index_mut(&mut self, index: Range<usize>) -> &mut [T]
			{
				&mut self.data[index]
			}
		}
		
		impl<T> Index<RangeFrom<usize>> for $name<T>
            where T: RealNumber
		{
			type Output = [T];
		
			fn index(&self, index: RangeFrom<usize>) -> &[T]
			{
				&self.data[index]
			}
		}
		
		impl<T> IndexMut<RangeFrom<usize>> for $name<T>
            where T: RealNumber
		{
			fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [T]
			{
				&mut self.data[index]
			}
		}
		
		impl<T> Index<RangeTo<usize>> for $name<T>
            where T: RealNumber
		{
			type Output = [T];
		
			fn index(&self, index: RangeTo<usize>) -> &[T]
			{
				&self.data[index]
			}
		}
		
		impl<T> IndexMut<RangeTo<usize>> for $name<T>
            where T: RealNumber
		{
			fn index_mut(&mut self, index: RangeTo<usize>) -> &mut [T]
			{
				&mut self.data[index]
			}
		}
		
		impl<T> Index<RangeFull> for $name<T>
            where T: RealNumber
		{
			type Output = [T];
		
			fn index(&self, index: RangeFull) -> &[T]
			{
				&self.data[index]
			}
		}
		
		impl<T> IndexMut<RangeFull> for $name<T>
            where T: RealNumber
		{
			fn index_mut(&mut self, index: RangeFull) -> &mut [T]
			{
				&mut self.data[index]
			}
		}
        
        impl<T> Clone for $name<T> 
            where T: RealNumber {
            fn clone(&self) -> Self {
                let data_length = self.data.len(); 
                $name 
				{ 
				  data: self.data.clone(), 
				  temp: vec![T::zero(); data_length],
				  delta: self.delta,
				  domain: self.domain,
				  is_complex: self.is_complex,
				  valid_len: self.valid_len,
                  multicore_settings: self.multicore_settings.clone()
				}
            }

            fn clone_from(&mut self, source: &Self) {
                 self.data = source.data.clone();
                 self.temp.resize(self.data.len(), T::zero());
                 self.domain = source.domain;
                 self.is_complex = source.is_complex;
                 self.valid_len = source.valid_len;
                 self.multicore_settings = source.multicore_settings.clone();
            }
        }
    }
}

macro_rules! define_vector_struct_type_alias {
    (struct $name:ident,based_on: $base:ident, $data_type:ident) => {
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
        pub type $name = $base<$data_type>;
    }
}

macro_rules! define_generic_operations_forward {
	(from: $name:ident, to: $gen_type:ident, $($data_type:ident),*)
	 =>
	 {
         $(
            #[inline]
            impl GenericVectorOperations<$data_type> for $name<$data_type>
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
                
                fn sqrt(self) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().sqrt()) 
                }
                
                fn square(self) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().square()) 
                }
                
                fn root(self, degree: $data_type) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().root(degree)) 
                }
                
                fn power(self, exponent: $data_type) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().power(exponent)) 
                }
                
                fn logn(self) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().logn()) 
                }
                
                fn expn(self) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().expn()) 
                }
            
                fn log_base(self, base: $data_type) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().log_base(base)) 
                }
                
                fn sin(self) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().sin()) 
                }
                
                fn cos(self) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().cos()) 
                }
                
                fn swap_halves(self) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().swap_halves()) 
                }
                
                fn exp_base(self, base: $data_type) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().exp_base(base)) 
                }
            }
       )*
	}	
}

macro_rules! define_real_basic_struct_members {
    (impl $name:ident, DataVectorDomain::$domain:ident)
	 =>
	 {
		#[inline]
		impl<T> $name<T> 
            where T: RealNumber
		{
			/// Creates a real `DataVector` by consuming a `Vec`. 
			///
			/// This operation is more memory efficient than the other options to create a vector,
			/// however if used outside of Rust then it holds the risk that the user will access 
			/// the data parameter after the vector has been created causing all types of issues.  
			pub fn from_array_no_copy(data: Vec<T>) -> Self
			{
				let data_length = data.len();
				$name 
				{ 
				  data: data, 
				  temp: vec![T::zero(); data_length],
				  delta: T::one(),
				  domain: DataVectorDomain::$domain,
				  is_complex: false,
				  valid_len: data_length,
                  multicore_settings: MultiCoreSettings::new()
				}
			}
		
			/// Creates a real `DataVector` from an array or sequence. `delta` is defaulted to `1`.
			pub fn from_array(data: &[T]) -> Self
			{
				$name::from_array_with_delta(data, T::one())
			}
			
			/// Creates a real `DataVector` from an array or sequence and sets `delta` to the given value.
			pub fn from_array_with_delta(data: &[T], delta: T) -> Self
			{
				let data_length = data.len();
				$name 
				{ 
				  data: data.to_vec(), 
				  temp: vec![T::zero(); data_length],
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: false,
				  valid_len: data_length,
                  multicore_settings: MultiCoreSettings::new()
				}
			}
            
            /// Creates a real and empty `DataVector` and sets `delta` to 1.0 value.
            pub fn real_empty() -> Self
            {
                $name 
				{ 
				  data: vec![T::zero(); 0], 
				  temp: vec![T::zero(); 0],
				  delta: T::one(),
				  domain: DataVectorDomain::$domain,
				  is_complex: false,
				  valid_len: 0,
                  multicore_settings: MultiCoreSettings::new()
				}
            }
            
            /// Creates a real and empty `DataVector` and sets `delta` to the given value.
            pub fn real_empty_with_delta(delta: T) -> Self
            {
                $name 
				{ 
				  data: vec![T::zero(); 0], 
				  temp: vec![T::zero(); 0],
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: false,
				  valid_len: 0,
                  multicore_settings: MultiCoreSettings::new()
				}
            }
            
            /// Creates a real `DataVector` with `length` elements all set to the value of `constant`. `delta` is defaulted to `1`.
			pub fn real_from_constant(constant: T, length: usize) -> Self
			{
				$name::real_from_constant_with_delta(constant, length, T::one())
			}
			
			/// Creates a real `DataVector` with `length` elements all set to the value of `constant` and sets `delta` to the given value.
			pub fn real_from_constant_with_delta(constant: T, length: usize, delta: T) -> Self
			{
				$name 
				{ 
				  data: vec![constant; length],
				  temp: vec![T::zero(); length],
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
    (from: $name:ident, to: $gen_type:ident, complex_partner: $complex_partner:ident, $($data_type:ident),*)
	 =>
	 {	 
        $(
            #[inline]
            impl RealVectorOperations<$data_type> for $name<$data_type>
            {
                type ComplexPartner = $complex_partner<$data_type>; 
                
                fn real_offset(self, offset: $data_type) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().real_offset(offset))
                }
                
                fn real_scale(self, factor: $data_type) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().real_scale(factor))
                }
                        
                fn real_abs(self) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().real_abs()) 
                }
                            
                fn to_complex(self) -> VecResult<Self::ComplexPartner>
                {
                    $complex_partner::from_genres(self.to_gen().to_complex()) 
                }
                
                fn wrap(self, divisor: $data_type) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().wrap(divisor))
                }
                
                fn unwrap(self, divisor: $data_type) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().unwrap(divisor))
                }
            }
        )*
        
        #[inline]
        impl<T> $name<T>
            where T: RealNumber
        {		
            fn to_gen(self) -> $gen_type<T>
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
            
            fn to_gen_borrow(&self) -> &$gen_type<T>
            {
                unsafe { mem::transmute(self) }
            }
            
            #[allow(dead_code)]
            fn to_gen_mut_borrow(&mut self) -> &mut $gen_type<T>
            {
                unsafe { mem::transmute(self) }
            }
            
            fn from_gen(other: $gen_type<T>) -> Self
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
            
            fn from_genres(other: VecResult<$gen_type<T>>) -> VecResult<$name<T>>
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
		impl<T> $name<T>
            where T: RealNumber
		{
			/// Creates a complex `DataVector` by consuming a `Vec`. Data is in interleaved format: `i0, q0, i1, q1, ...`. 
			///
			/// This operation is more memory efficient than the other options to create a vector,
			/// however if used outside of Rust then it holds the risk that the user will access 
			/// the data parameter after the vector has been created causing all types of issues.  
			pub fn from_interleaved_no_copy(data: Vec<T>) -> Self
			{
				let data_length = data.len();
				$name 
				{ 
				  data: data,
				  temp: vec![T::zero(); data_length],
				  delta: T::one(),
				  domain: DataVectorDomain::$domain,
				  is_complex: true,
				  valid_len: data_length,
                  multicore_settings: MultiCoreSettings::new()
				}
			}
			
			/// Creates a complex `DataVector` from an array or sequence. Data is in interleaved format: `i0, q0, i1, q1, ...`. `delta` is defaulted to `1`.
			pub fn from_interleaved(data: &[T]) -> Self
			{
				$name::from_interleaved_with_delta(data, T::one())
			}
			
			/// Creates a complex `DataVector` from an array or sequence. Data is in interleaved format: `i0, q0, i1, q1, ...`. `delta` is set to the given value.
			pub fn from_interleaved_with_delta(data: &[T], delta: T) -> Self
			{
				let data_length = data.len();
				$name 
				{ 
				  data: data.to_vec(), 
				  temp: vec![T::zero(); data_length],
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: true,
				  valid_len: data_length,
                  multicore_settings: MultiCoreSettings::new()
				}
			}
            
            /// Creates a complex and empty `DataVector` and sets `delta` to 1.0 value.
            pub fn complex_empty() -> Self
            {
                $name
				{ 
				  data: vec![T::zero(); 0], 
				  temp: vec![T::zero(); 0],
				  delta: T::one(),
				  domain: DataVectorDomain::$domain,
				  is_complex: true,
				  valid_len: 0,
                  multicore_settings: MultiCoreSettings::new()
				}
            }
            
            /// Creates a complex and empty `DataVector` and sets `delta` to the given value.
            pub fn complex_empty_with_delta(delta: T) -> Self
            {
                $name 
				{ 
				  data: vec![T::zero(); 0], 
				  temp: vec![T::zero(); 0],
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: true,
				  valid_len: 0,
                  multicore_settings: MultiCoreSettings::new()
				}
            }
            
            /// Creates a complex `DataVector` with `length` elements all set to the value of `constant`. `delta` is defaulted to `1`.
			pub fn complex_from_constant(constant: T, length: usize) -> Self
			{
				$name::complex_from_constant_with_delta(constant, length, T::one())
			}
			
			/// Creates a complex `DataVector` with `length` elements all set to the value of `constant` and sets `delta` to the given value.
			pub fn complex_from_constant_with_delta(constant: T, length: usize, delta: T) -> Self
			{
				$name 
				{ 
				  data: vec![constant; length],
				  temp: vec![T::zero(); length],
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
			pub fn from_real_imag(real: &[T], imag: &[T])
				-> Self
			{
				$name::from_real_imag_with_delta(real, imag, T::one())
			}
			
			/// Creates a complex  `DataVector` from an array with real and an array imaginary data. `delta` is set to the given value 1.
			///
			/// Arrays must have the same length.
			pub fn from_real_imag_with_delta(real: &[T], imag: &[T], delta: T)
				-> Self
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
				  temp: vec![T::zero(); data_length],
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
			pub fn from_mag_phase(magnitude: &[T], phase: &[T])
				-> Self
			{
				$name::from_mag_phase_with_delta(magnitude, phase, T::one())
			}
			
			/// Creates a complex  `DataVector` from an array with magnitude and an array with phase data. `delta` is set to the given value 1.
			///
			/// Arrays must have the same length. Phase must be in [rad].
			pub fn from_mag_phase_with_delta(magnitude: &[T], phase: &[T], delta: T)
				-> Self
			{
				if magnitude.len() != phase.len()
				{
					panic!("Input lengths differ: magnitude has {} elements and phase has {} elements", magnitude.len(), phase.len());
				}
				
				let mut data = Vec::with_capacity(magnitude.len() + phase.len());
				for i in 0 .. magnitude.len() {
					let complex = Complex::from_polar(&magnitude[i], &phase[i]);
					data.push(complex.re);
					data.push(complex.im);
				}
				
				let data_length = data.len();
				
				$name 
				{ 
				  data: data, 
				  temp: vec![T::zero(); data_length],
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
    (from: $name:ident, to: $gen_type:ident, complex: $complex_type:ident, real_partner: $real_partner:ident, $($data_type:ident),*)
	 =>
	 { 
        $(
	 	#[inline]
		impl ComplexVectorOperations<$data_type> for $name<$data_type>
		{
			type RealPartner = $real_partner<$data_type>;
			
			fn complex_offset(self, offset: Complex<$data_type>) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().complex_offset(offset))
			}
				
			fn complex_scale(self, factor: Complex<$data_type>) -> VecResult<Self>
			{
				$name::from_genres(self.to_gen().complex_scale(factor))
			}
			
			fn complex_abs(self) -> VecResult<Self::RealPartner>
			{
				$real_partner::from_genres(self.to_gen().complex_abs())
			}
			
			fn get_complex_abs(&self, destination: &mut Self::RealPartner) -> VoidResult
			{
				self.to_gen_borrow().get_complex_abs(destination.to_gen_mut_borrow())
			}
			
			fn complex_abs_squared(self) -> VecResult<Self::RealPartner>
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
        )*
        
        #[inline]
        impl<T> $name<T>
            where T: RealNumber
        {
            fn to_gen(self) -> $gen_type<T>
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
            
            fn to_gen_borrow(&self) -> &$gen_type<T>
            {
                unsafe { mem::transmute(self) }
            }
            
            #[allow(dead_code)]
            fn to_gen_mut_borrow(&mut self) -> &mut $gen_type<T>
            {
                unsafe { mem::transmute(self) }
            }
            
            fn from_gen(other: $gen_type<T>) -> Self
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
            
            fn from_genres(other: VecResult<$gen_type<T>>) -> VecResult<Self>
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
        Operation,
		DataVector32, 
		RealTimeVector32,
		ComplexTimeVector32, 
		RealFreqVector32,
		ComplexFreqVector32,
        DataVector64, 
		RealTimeVector64,
		ComplexTimeVector64, 
		RealFreqVector64,
		ComplexFreqVector64
	};