//! A module for very basic digital signal processing (DSP) operations on data vectors.

macro_rules! define_vector_struct {
    (struct $name:ident) => {
        #[derive(Debug)]
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
        
        impl<T> RededicateVector<T> for $name<T>
            where T: RealNumber {
            fn rededicate_as_complex_time_vector(self, delta: T) -> ComplexTimeVector<T> {
                ComplexTimeVector {
                    delta: delta,
                    is_complex: true,
                    valid_len: 0,
                    domain: DataVectorDomain::Time,
                    data: self.data,
                    temp: self.temp,
                    multicore_settings: self.multicore_settings
                }
            }
            
            fn rededicate_as_complex_freq_vector(self, delta: T) -> ComplexFreqVector<T> {
                ComplexFreqVector {
                    delta: delta,
                    is_complex: true,
                    valid_len: 0,
                    domain: DataVectorDomain::Frequency,
                    data: self.data,
                    temp: self.temp,
                    multicore_settings: self.multicore_settings
                }
            }
            
            fn rededicate_as_real_time_vector(self, delta: T) -> RealTimeVector<T> {
                RealTimeVector {
                    delta: delta,
                    is_complex: false,
                    valid_len: 0,
                    domain: DataVectorDomain::Time,
                    data: self.data,
                    temp: self.temp,
                    multicore_settings: self.multicore_settings
                }
            }
            
            fn rededicate_as_real_freq_vector(self, delta: T) -> RealFreqVector<T> {
                RealFreqVector {
                    delta: delta,
                    is_complex: true,
                    valid_len: 0,
                    domain: DataVectorDomain::Frequency,
                    data: self.data,
                    temp: self.temp,
                    multicore_settings: self.multicore_settings
                }
            }
            
            fn rededicate_as_generic_vector(self, is_complex: bool, domain: DataVectorDomain, delta: T) -> GenericDataVector<T> {
                GenericDataVector {
                    delta: delta,
                    is_complex: is_complex,
                    valid_len: 0,
                    domain: domain,
                    data: self.data,
                    temp: self.temp,
                    multicore_settings: self.multicore_settings
                }
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
        /// Specialization of a vector for a certain data type.
        pub type $name = $base<$data_type>;
    }
}

macro_rules! define_real_basic_struct_members {
    (impl $name:ident, DataVectorDomain::$domain:ident)
	 =>
	 {
		impl<T> $name<T> 
            where T: RealNumber
		{
			/// Same as `from_array_no_copy` but also allows to set multicore options.
            pub fn from_array_no_copy_with_options(data: Vec<T>, options: MultiCoreSettings) -> Self {
				let data_length = data.len();
                let temp_length = 
                    if options.early_temp_allocation {
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
                  multicore_settings: options
				}
			}
		
			/// Same as `from_array` but also allows to set multicore options.
            pub fn from_array_with_options(data: &[T], options: MultiCoreSettings) -> Self {
				$name::from_array_with_delta_and_options(data, T::one(), options)
			}
			
			/// Same as `from_array_with_delta` but also allows to set multicore options.
            pub fn from_array_with_delta_and_options(data: &[T], delta: T, options: MultiCoreSettings) -> Self {
				let data_length = data.len();
                let temp_length = 
                    if options.early_temp_allocation {
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
                  multicore_settings: options
				}
			}
            
            /// Same as `empty` but also allows to set multicore options.
            pub fn empty_with_options(options: MultiCoreSettings) -> Self {
                $name 
				{ 
				  data: vec![T::zero(); 0], 
				  temp: vec![T::zero(); 0],
				  delta: T::one(),
				  domain: DataVectorDomain::$domain,
				  is_complex: false,
				  valid_len: 0,
                  multicore_settings: options
				}
            }
            
            /// Same as `empty_with_delta` but also allows to set multicore options.
            pub fn empty_with_delta_and_options(delta: T, options: MultiCoreSettings) -> Self {
                $name 
				{ 
				  data: vec![T::zero(); 0], 
				  temp: vec![T::zero(); 0],
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: false,
				  valid_len: 0,
                  multicore_settings: options
				}
            }
            
            /// Same as `from_constant` but also allows to set multicore options.
            pub fn from_constant_with_options(constant: T, length: usize, options: MultiCoreSettings) -> Self {
				$name::from_constant_with_delta_and_options(constant, length, T::one(), options)
			}
			
			/// Same as `from_constant_with_delta` but also allows to set multicore options.
            pub fn from_constant_with_delta_and_options(constant: T, length: usize, delta: T, options: MultiCoreSettings) -> Self {
                let temp_length = 
                    if options.early_temp_allocation {
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
                  multicore_settings: options
				}
			}
            
            /// Creates a real `DataVector` by consuming a `Vec`. 
			///
			/// This operation is more memory efficient than the other options to create a vector,
			/// however if used outside of Rust then it holds the risk that the user will access 
			/// the data parameter after the vector has been created causing all types of issues.  
			pub fn from_array_no_copy(data: Vec<T>) -> Self
			{
				Self::from_array_no_copy_with_options(data, MultiCoreSettings::default())
			}
		
			/// Creates a real `DataVector` from an array or sequence. `delta` is defaulted to `1`.
			pub fn from_array(data: &[T]) -> Self
			{
				Self::from_array_with_delta_and_options(data, T::one(), MultiCoreSettings::default())
			}
			
			/// Creates a real `DataVector` from an array or sequence and sets `delta` to the given value.
			pub fn from_array_with_delta(data: &[T], delta: T) -> Self
			{
				Self::from_array_with_delta_and_options(data, delta, MultiCoreSettings::default())
			}
            
            /// Creates a real and empty `DataVector` and sets `delta` to 1.0 value.
            pub fn empty() -> Self
            {
                Self::empty_with_options(MultiCoreSettings::default())
            }
            
            /// Creates a real and empty `DataVector` and sets `delta` to the given value.
            pub fn empty_with_delta(delta: T) -> Self
            {
                Self::empty_with_delta_and_options(delta, MultiCoreSettings::default())
            }
            
            /// Creates a real `DataVector` with `length` elements all set to the value of `constant`. `delta` is defaulted to `1`.
			pub fn from_constant(constant: T, length: usize) -> Self
			{
				Self::from_constant_with_options(constant, length, MultiCoreSettings::default())
			}
			
			/// Creates a real `DataVector` with `length` elements all set to the value of `constant` and sets `delta` to the given value.
			pub fn from_constant_with_delta(constant: T, length: usize, delta: T) -> Self
			{
                Self::from_constant_with_delta_and_options(constant, length, delta, MultiCoreSettings::default())
			}
		}
	 }
}

macro_rules! define_complex_basic_struct_members {
    (impl $name:ident, DataVectorDomain::$domain:ident)
	 =>
	 {
		impl<T> $name<T>
            where T: RealNumber
		{
			/// Same as `from_interleaved_no_copy` but also allows to set multicore options.
            pub fn from_interleaved_no_copy_with_options(data: Vec<T>, options: MultiCoreSettings) -> Self {
				let data_length = data.len();
                let temp_length = 
                    if options.early_temp_allocation {
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
                  multicore_settings: options
				}
			}
			
			/// Same as `from_interleaved` but also allows to set multicore options.
            pub fn from_interleaved_with_options(data: &[T], options: MultiCoreSettings) -> Self {
				$name::from_interleaved_with_delta_and_options(data, T::one(), options)
			}
			
			/// Same as `from_interleaved_with_delta` but also allows to set multicore options.
            pub fn from_interleaved_with_delta_and_options(data: &[T], delta: T, options: MultiCoreSettings) -> Self {
				let data_length = data.len();
                let temp_length = 
                    if options.early_temp_allocation {
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
                  multicore_settings: options
				}
			}
            
            /// Same as `complex_empty` but also allows to set multicore options.
            pub fn empty_with_options(options: MultiCoreSettings) -> Self {
                $name
				{ 
				  data: vec![T::zero(); 0], 
				  temp: vec![T::zero(); 0],
				  delta: T::one(),
				  domain: DataVectorDomain::$domain,
				  is_complex: true,
				  valid_len: 0,
                  multicore_settings: options
				}
            }
            
            /// Same as `complex_empty_with_delta` but also allows to set multicore options.
            pub fn empty_with_delta_and_options(delta: T, options: MultiCoreSettings) -> Self {
                $name 
				{ 
				  data: vec![T::zero(); 0], 
				  temp: vec![T::zero(); 0],
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: true,
				  valid_len: 0,
                  multicore_settings: options
				}
            }
            
            /// Same as `complex_from_constant` but also allows to set multicore options.
            pub fn from_constant_with_options(constant: Complex<T>, length: usize, options: MultiCoreSettings) -> Self {
				$name::from_constant_with_delta_and_options(constant, length, T::one(), options)
			}
			
			/// Same as `complex_from_constant_with_delta` but also allows to set multicore options.
            pub fn from_constant_with_delta_and_options(constant: Complex<T>, length: usize, delta: T, options: MultiCoreSettings) -> Self {
                let temp_length = 
                    if options.early_temp_allocation {
                        2 * length
                    } else {
                        0
                    };
                let data = unsafe {
                    let complex = vec![constant; length];
                    let mut trans: Vec<T> = mem::transmute(complex);
                    trans.set_len(2 * length);
                    trans
                };
				$name 
				{ 
				  data: data,
				  temp: vec![T::zero(); temp_length],
				  delta: delta,
				  domain: DataVectorDomain::$domain,
				  is_complex: true,
				  valid_len: 2 * length,
                  multicore_settings: options
				}
			}
			
			/// Same as `from_real_imag` but also allows to set multicore options.
            pub fn from_real_imag_with_options(real: &[T], imag: &[T], options: MultiCoreSettings) -> Self {
				$name::from_real_imag_with_delta_and_options(real, imag, T::one(), options)
			}
			
			/// Same as `from_real_imag_with_delta` but also allows to set multicore options.
            pub fn from_real_imag_with_delta_and_options(real: &[T], imag: &[T], delta: T, options: MultiCoreSettings) -> Self {
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
                    if options.early_temp_allocation {
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
                  multicore_settings: options
				}
			}
			
			/// Same as `from_mag_phase` but also allows to set multicore options.
            pub fn from_mag_phase_with_options(magnitude: &[T], phase: &[T], options: MultiCoreSettings) -> Self {
				$name::from_mag_phase_with_delta_and_options(magnitude, phase, T::one(), options)
			}
			
			/// Same as `from_mag_phase_with_delta` but also allows to set multicore options.
            pub fn from_mag_phase_with_delta_and_options(magnitude: &[T], phase: &[T], delta: T, options: MultiCoreSettings) -> Self {
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
                    if options.early_temp_allocation {
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
                  multicore_settings: options
				}
			}
            
            /// Creates a complex `DataVector` by consuming a `Vec`. Data is in interleaved format: `i0, q0, i1, q1, ...`. 
			///
			/// This operation is more memory efficient than the other options to create a vector,
			/// however if used outside of Rust then it holds the risk that the user will access 
			/// the data parameter after the vector has been created causing all types of issues.  
			pub fn from_interleaved_no_copy(data: Vec<T>) -> Self
			{
				Self::from_interleaved_no_copy_with_options(data, MultiCoreSettings::default())
			}
			
			/// Creates a complex `DataVector` from an array or sequence. Data is in interleaved format: `i0, q0, i1, q1, ...`. `delta` is defaulted to `1`.
			pub fn from_interleaved(data: &[T]) -> Self
			{
				Self::from_interleaved_with_options(data, MultiCoreSettings::default())
			}
			
			/// Creates a complex `DataVector` from an array or sequence. Data is in interleaved format: `i0, q0, i1, q1, ...`. `delta` is set to the given value.
			pub fn from_interleaved_with_delta(data: &[T], delta: T) -> Self
			{
				Self::from_interleaved_with_delta_and_options(data, delta, MultiCoreSettings::default())
			}
            
            /// Creates a complex and empty `DataVector` and sets `delta` to 1.0 value.
            pub fn empty() -> Self
            {
                Self::empty_with_options(MultiCoreSettings::default())
            }
            
            /// Creates a complex and empty `DataVector` and sets `delta` to the given value.
            pub fn empty_with_delta(delta: T) -> Self
            {
                Self::empty_with_delta_and_options(delta, MultiCoreSettings::default())
            }
            
            /// Creates a complex `DataVector` with `length` elements all set to the value of `constant`. `delta` is defaulted to `1`.
			pub fn from_constant(constant: Complex<T>, length: usize) -> Self
			{
				Self::from_constant_with_options(constant, length, MultiCoreSettings::default())
			}
			
			/// Creates a complex `DataVector` with `length` elements all set to the value of `constant` and sets `delta` to the given value.
			pub fn from_constant_with_delta(constant: Complex<T>, length: usize, delta: T) -> Self
			{
                Self::from_constant_with_delta_and_options(constant, length, delta, MultiCoreSettings::default())
			}
			
			/// Creates a complex  `DataVector` from an array with real and an array imaginary data. `delta` is set to 1.
			///
			/// Arrays must have the same length.
			pub fn from_real_imag(real: &[T], imag: &[T])
				-> Self
			{
				Self::from_real_imag_with_options(real, imag, MultiCoreSettings::default())
			}
			
			/// Creates a complex  `DataVector` from an array with real and an array imaginary data. `delta` is set to the given value.
			///
			/// Arrays must have the same length.
			pub fn from_real_imag_with_delta(real: &[T], imag: &[T], delta: T)
				-> Self
			{
				Self::from_real_imag_with_delta_and_options(real, imag, delta, MultiCoreSettings::default())
			}
			
			/// Creates a complex  `DataVector` from an array with magnitude and an array with phase data. `delta` is set to 1.
			///
			/// Arrays must have the same length. Phase must be in [rad].
			pub fn from_mag_phase(magnitude: &[T], phase: &[T])
				-> Self
			{
				Self::from_mag_phase_with_options(magnitude, phase, MultiCoreSettings::default())
			}
			
			/// Creates a complex  `DataVector` from an array with magnitude and an array with phase data. `delta` is set to the given value.
			///
			/// Arrays must have the same length. Phase must be in [rad].
			pub fn from_mag_phase_with_delta(magnitude: &[T], phase: &[T], delta: T)
				-> Self
			{
				Self::from_mag_phase_with_delta_and_options(magnitude, phase, delta, MultiCoreSettings::default())
			}
            
            /// Creates a complex `DataVector` from an array of complex numbers. `delta` is set to 1.
            pub fn from_complex(complex: &[Complex<T>]) -> Self {
                Self::from_complex_with_delta(complex, T::one())
            }
            
            /// Creates a complex `DataVector` from an array of complex numbers. `delta` is set to the given value.
            pub fn from_complex_with_delta(complex: &[Complex<T>], delta: T) -> Self {
                Self::from_complex_with_delta_and_options(complex, delta, MultiCoreSettings::default())
            }
            /// Creates a complex `DataVector` from an array of complex numbers.
            
            pub fn from_complex_with_delta_and_options(complex: &[Complex<T>], delta: T, options: MultiCoreSettings) -> Self {
                use std::slice;
                let data = unsafe {
                    let len = complex.len();
                    let trans: &[T] = mem::transmute(complex);
                    slice::from_raw_parts(&trans[0] as *const T, len * 2)
                };
                Self::from_interleaved_with_delta_and_options(data, delta, options)
            }
		} 
	 }
}