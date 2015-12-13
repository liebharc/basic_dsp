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
                let temp_length = 
                    if self.multicore_settings.early_temp_allocation {
                        data_length
                    } else {
                        0
                    };
                $name 
				{ 
				  data: self.data.clone(), 
				  temp: vec![T::zero(); temp_length],
				  delta: self.delta,
				  domain: self.domain,
				  is_complex: self.is_complex,
				  valid_len: self.valid_len,
                  multicore_settings: self.multicore_settings.clone()
				}
            }

            fn clone_from(&mut self, source: &Self) {
                let temp_length = 
                    if source.multicore_settings.early_temp_allocation {
                        self.data.len()
                    } else {
                        0
                    };
                 self.data = source.data.clone();
                 self.temp.resize(temp_length, T::zero());
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
		/// about the vector isn't known at compile time there are the generic [`DataVector32`](type.DataVector32.html) and [`DataVector64`](type.DataVector64.html) vectors
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
    
                fn tan(self) -> VecResult<Self> {
                    $name::from_genres(self.to_gen().tan()) 
                }
                
                fn asin(self) -> VecResult<Self> {
                    $name::from_genres(self.to_gen().asin()) 
                }
               
                fn acos(self) -> VecResult<Self> {
                    $name::from_genres(self.to_gen().acos()) 
                }
                
                fn atan(self) -> VecResult<Self> {
                    $name::from_genres(self.to_gen().atan()) 
                }
                
                fn sinh(self) -> VecResult<Self> {
                    $name::from_genres(self.to_gen().sinh()) 
                }
                
                fn cosh(self) -> VecResult<Self> {
                    $name::from_genres(self.to_gen().cosh()) 
                }
                
                fn tanh(self) -> VecResult<Self> {
                    $name::from_genres(self.to_gen().tanh()) 
                }
                
                fn asinh(self) -> VecResult<Self> {
                    $name::from_genres(self.to_gen().asinh()) 
                }
                
                fn acosh(self) -> VecResult<Self> {
                    $name::from_genres(self.to_gen().acosh()) 
                }
                
                fn atanh(self) -> VecResult<Self> {
                    $name::from_genres(self.to_gen().atanh()) 
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
                let temp_length = 
                    if MultiCoreSettings::early_temp_allocation_default() {
                        data_length
                    } else {
                        0
                    };
				$name 
				{ 
				  data: data, 
				  temp: vec![T::zero(); temp_length],
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
                let temp_length = 
                    if MultiCoreSettings::early_temp_allocation_default() {
                        data_length
                    } else {
                        0
                    };
				$name 
				{ 
				  data: data.to_vec(), 
				  temp: vec![T::zero(); temp_length],
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
                let temp_length = 
                    if MultiCoreSettings::early_temp_allocation_default() {
                        length
                    } else {
                        0
                    };
				$name 
				{ 
				  data: vec![constant; length],
				  temp: vec![T::zero(); temp_length],
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
                
                fn real_dot_product(&self, factor: &Self) -> ScalarResult<$data_type>
                {
                    self.to_gen_borrow().real_dot_product(&factor.to_gen_borrow())
                }
                
                fn real_statistics(&self) -> Statistics<$data_type> {
                    self.to_gen_borrow().real_statistics()
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
                let temp_length = 
                    if MultiCoreSettings::early_temp_allocation_default() {
                        data_length
                    } else {
                        0
                    };
				$name 
				{ 
				  data: data,
				  temp: vec![T::zero(); temp_length],
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
                let temp_length = 
                    if MultiCoreSettings::early_temp_allocation_default() {
                        data_length
                    } else {
                        0
                    };
				$name 
				{ 
				  data: data.to_vec(), 
				  temp: vec![T::zero(); temp_length],
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
                let temp_length = 
                    if MultiCoreSettings::early_temp_allocation_default() {
                        length
                    } else {
                        0
                    };
				$name 
				{ 
				  data: vec![constant; length],
				  temp: vec![T::zero(); temp_length],
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
                let temp_length = 
                    if MultiCoreSettings::early_temp_allocation_default() {
                        data_length
                    } else {
                        0
                    };
				
				$name 
				{ 
				  data: data, 
				  temp: vec![T::zero(); temp_length],
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
                let temp_length = 
                    if MultiCoreSettings::early_temp_allocation_default() {
                        data_length
                    } else {
                        0
                    };
				
				$name 
				{ 
				  data: data, 
				  temp: vec![T::zero(); temp_length],
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
                
                fn complex_dot_product(&self, factor: &Self) -> ScalarResult<Complex<$data_type>>
                {
                    self.to_gen_borrow().complex_dot_product(&factor.to_gen_borrow())
                }
                
                fn complex_statistics(&self) -> Statistics<Complex<$data_type>> {
                    self.to_gen_borrow().complex_statistics()
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

macro_rules! temp_mut {
    ($self_: ident, $len: expr) => {
        if $self_.temp.len() < $len {
            $self_.temp.resize($len, 0.0);
            &mut $self_.temp
        }
        else {
            &mut $self_.temp
        }
    }
}

pub mod definitions;
pub mod general_impl;
pub mod real_impl;
pub mod complex_impl;
pub mod time_freq_impl;

pub use vector_types::definitions::
	{
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
        Statistics
	};
use num::complex::Complex;
use RealNumber;
use multicore_support::{Chunk, Complexity, MultiCoreSettings};
use std::mem;
use simd_extensions::{Simd, Reg32};
use std::ops::{Index, IndexMut, Range, RangeTo, RangeFrom, RangeFull};
    
define_vector_struct!(struct GenericDataVector);

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
#[allow(dead_code)]
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
		let scalar_length = data_length % 4;
		let vectorization_length = data_length - scalar_length;
		if scalar_length > 0
		{
			panic!("perform_operations requires right now that the array length is dividable by 4")
		}
		
		{
			let mut array = &mut self.data;
			Chunk::execute_partial_with_arguments(
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
		let data = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
		let result = RealTimeVector32::from_array(&data);
		let result = result.real_offset(2.0).unwrap();
		assert_eq!(result.data, [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
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