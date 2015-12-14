use multicore_support::{Chunk, Complexity};
use super::definitions::{
	DataVector,
    DataVectorDomain,
	GenericVectorOperations,
    VecResult,
    VoidResult,
    ErrorReason};
use super::GenericDataVector;
use num::complex::Complex;
use num::traits::Float;
use complex_extensions::ComplexExtensions;
use simd_extensions::{Simd, Reg32, Reg64};
use multicore_support::MultiCoreSettings;

macro_rules! impl_real_complex_dispatch {
    ($function_name: ident, $real_op: ident, $complex_op: ident)
    => {
        fn $function_name(self) -> VecResult<Self>
        {
            if self.is_complex() {
                Self::$complex_op(self)  
            }
            else {
                Self::$real_op(self)
            }
        }
    }
}

macro_rules! impl_real_complex_arg_dispatch {
    ($function_name: ident, $arg_type: ident, $arg: ident, $real_op: ident, $complex_op: ident)
    => {
        fn $function_name(self, $arg: $arg_type) -> VecResult<Self>
        {
            if self.is_complex() {
                Self::$complex_op(self, $arg)  
            }
            else {
                Self::$real_op(self, $arg)
            }
        }
    }
}

macro_rules! impl_function_call_real_complex {
    ($data_type: ident; $real_name: ident, $real_op: ident; $complex_name: ident, $complex_op: ident) => {
        fn $real_name(mut self) -> VecResult<Self>
        {
            {
                let mut array = &mut self.data;
                let length = array.len();
                Chunk::execute_partial(
                    Complexity::Medium, &self.multicore_settings,
                    &mut array, length, 1, 
                    |array| {
                        let mut i = 0;
                        while i < array.len()
                        {
                            array[i] = array[i].$real_op();
                            i += 1;
                        }
                });
            }
            Ok(self)
        }
        
        fn $complex_name(mut self) -> VecResult<Self>
        {
            {
                let mut array = &mut self.data;
                let length = array.len();
                Chunk::execute_partial(
                    Complexity::Medium, &self.multicore_settings,
                    &mut array, length, 2, 
                    |array| {
                        let mut i = 0;
                        while i < array.len()
                        {
                            let complex = Complex::<$data_type>::new(array[i], array[i + 1]);
                            let result = complex.$complex_op();
                            array[i] = result.re;
                            array[i + 1] = result.im;
                            i += 2;
                        }
                });
            }
            Ok(self)
        }
    }
}

macro_rules! impl_function_call_real_arg_complex {
    ($data_type: ident; $real_name: ident, $real_op: ident; $complex_name: ident, $complex_op: ident) => {
        fn $real_name(mut self, value: $data_type) -> VecResult<Self>
        {
            {
                let mut array = &mut self.data;
                let length = array.len();
                Chunk::execute_partial_with_arguments(
                    Complexity::Medium, &self.multicore_settings,
                    &mut array, length, 1, value, 
                    |array, value| {
                        let mut i = 0;
                        while i < array.len()
                        {
                            array[i] = array[i].$real_op(value);
                            i += 1;
                        }
                });
            }
            Ok(self)
        }
        
        fn $complex_name(mut self, value: $data_type) -> VecResult<Self>
        {
            {
                let mut array = &mut self.data;
                let length = array.len();
                Chunk::execute_partial_with_arguments(
                    Complexity::Medium, &self.multicore_settings,
                    &mut array, length, 2, value, 
                    |array, value| {
                        let mut i = 0;
                        while i < array.len()
                        {
                            let complex = Complex::<$data_type>::new(array[i], array[i + 1]);
                            let result = complex.$complex_op(value);
                            array[i] = result.re;
                            array[i + 1] = result.im;
                            i += 2;
                        }
                });
            }
            Ok(self)
        }
    }
}

macro_rules! add_general_impl {
    ($($data_type:ident, $reg:ident);*)
	 =>
	 {	 
        $(
            impl GenericDataVector<$data_type> {
                /// Same as `new` but also allows to set multicore options.
                pub fn new_with_options(is_complex: bool, domain: DataVectorDomain, init_value: $data_type, length: usize, delta: $data_type, options: MultiCoreSettings) -> Self {
                    GenericDataVector
                     {
                        data: vec![init_value; length],
                        temp: vec![0.0; length],
                        delta: delta,
                        domain: domain,
                        is_complex: is_complex,
                        valid_len: length,
                        multicore_settings: options
                    }
                }
                
                /// Same as `from_array` but also allows to set multicore options.
                pub fn from_array_with_options(is_complex: bool, domain: DataVectorDomain, data: &[$data_type], options: MultiCoreSettings) -> Self {
                    let length = data.len();
                    GenericDataVector
                    { 
                        data: data.to_vec(),
                        temp: vec![0.0; length],
                        delta: 1.0,
                        domain: domain,
                        is_complex: is_complex,
                        valid_len: length,
                        multicore_settings: options
                    }
                }
                
                /// Same as `from_array_no_copy` but also allows to set multicore options.
                pub fn from_array_no_copy_with_options(is_complex: bool, domain: DataVectorDomain, data: Vec<$data_type>, options: MultiCoreSettings) -> Self {
                    let length = data.len();
                    GenericDataVector
                    { 
                        data: data,
                        temp: vec![0.0; length],
                        delta: 1.0,
                        domain: domain,
                        is_complex: is_complex,
                        valid_len: length,
                        multicore_settings: options
                    }
                }
                
                /// Same as `from_array_with_delta` but also allows to set multicore options.
                pub fn from_array_with_delta_and_options(is_complex: bool, domain: DataVectorDomain, data: &[$data_type], delta: $data_type, options: MultiCoreSettings) -> Self {
                    let length = data.len();
                    GenericDataVector
                    { 
                        data: data.to_vec(),
                        temp: vec![0.0; length],
                        delta: delta,
                        domain: domain,
                        is_complex: is_complex,
                        valid_len: length,
                        multicore_settings: options
                    }
                }
                
                /// Same as `from_array_no_copy_with_delta` but also allows to set multicore options.
                pub fn from_array_no_copy_with_delta_and_options(is_complex: bool, domain: DataVectorDomain, data: Vec<$data_type>, delta: $data_type, options: MultiCoreSettings) -> Self {
                    let length = data.len();
                    GenericDataVector
                    { 
                        data: data,
                        temp: vec![0.0; length],
                        delta: delta,
                        domain: domain,
                        is_complex: is_complex,
                        valid_len: length,
                        multicore_settings: options
                    }
                }
                
                /// Creates a new generic data vector from the given arguments.
                pub fn new(is_complex: bool, domain: DataVectorDomain, init_value: $data_type, length: usize, delta: $data_type) -> Self {
                    Self::new_with_options(is_complex, domain, init_value, length, delta, MultiCoreSettings::default())
                }
                
                /// Creates a new generic data vector from the given arguments.
                pub fn from_array(is_complex: bool, domain: DataVectorDomain, data: &[$data_type]) -> Self {
                    Self::from_array_with_options(is_complex, domain, data, MultiCoreSettings::default())
                }
                
                /// Creates a new generic data vector from the given arguments.
                pub fn from_array_no_copy(is_complex: bool, domain: DataVectorDomain, data: Vec<$data_type>) -> Self {
                    Self::from_array_no_copy_with_options(is_complex, domain, data, MultiCoreSettings::default())
                }
                
                /// Creates a new generic data vector from the given arguments.
                pub fn from_array_with_delta(is_complex: bool, domain: DataVectorDomain, data: &[$data_type], delta: $data_type) -> Self {
                    Self::from_array_with_delta_and_options(is_complex, domain, data, delta, MultiCoreSettings::default())
                }
                
                /// Creates a new generic data vector from the given arguments.
                pub fn from_array_no_copy_with_delta(is_complex: bool, domain: DataVectorDomain, data: Vec<$data_type>, delta: $data_type) -> Self {
                    Self::from_array_no_copy_with_delta_and_options(is_complex, domain, data, delta, MultiCoreSettings::default())
                }
            }
            
            #[inline]
            impl GenericVectorOperations<$data_type> for GenericDataVector<$data_type> {
                fn add_vector(mut self, summand: &Self) -> VecResult<Self>
                {
                    {
                        let len = self.len();
                        reject_if!(self, len != summand.len(), ErrorReason::VectorsMustHaveTheSameSize);
                        assert_meta_data!(self, summand);
                        
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        let mut array = &mut self.data;
                        let other = &summand.data;
                        Chunk::execute_original_to_target(
                            Complexity::Small, &self.multicore_settings,
                            &other, vectorization_length, $reg::len(), 
                            &mut array, vectorization_length, $reg::len(), 
                            |original, range, target| {
                                let mut i = 0;
                                let mut j = range.start;
                                while i < target.len()
                                { 
                                    let vector1 = $reg::load(original, j);
                                    let vector2 = $reg::load(target, i);
                                    let result = vector1 + vector2;
                                    result.store(target, i);
                                    i += $reg::len();
                                    j += $reg::len();
                                }
                        });
                        let mut i = vectorization_length;
                        while i < data_length
                        {
                            array[i] = array[i] + other[i];
                            i += 1;
                        }
                    }
                    
                    Ok(self)
                }
                
                fn subtract_vector(mut self, subtrahend: &Self) -> VecResult<Self>
                {
                    {
                        let len = self.len();
                        reject_if!(self, len != subtrahend.len(), ErrorReason::VectorsMustHaveTheSameSize);
                        assert_meta_data!(self, subtrahend);
                            
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        let mut array = &mut self.data;
                        let other = &subtrahend.data;
                        Chunk::execute_original_to_target(
                            Complexity::Small, &self.multicore_settings,
                            &other, vectorization_length, $reg::len(), 
                            &mut array, vectorization_length, $reg::len(), 
                            |original, range, target| {
                                let mut i = 0;
                                let mut j = range.start;
                                while i < target.len()
                                { 
                                    let vector1 = $reg::load(original, j);
                                    let vector2 = $reg::load(target, i);
                                    let result = vector2 - vector1;
                                    result.store(target, i);
                                    i += $reg::len();
                                    j += $reg::len();
                                }
                        });
                        let mut i = vectorization_length;
                        while i < data_length
                        {
                            array[i] = array[i] - other[i];
                            i += 1;
                        }
                    }
                    
                    Ok(self)
                }
                
                fn multiply_vector(self, factor: &Self) -> VecResult<Self>
                {
                    let len = self.len();
                    reject_if!(self, len != factor.len(), ErrorReason::VectorsMustHaveTheSameSize);
                    assert_meta_data!(self, factor);
                    
                    if self.is_complex
                    {
                        Ok(self.multiply_vector_complex(factor))
                    }
                    else
                    {
                        Ok(self.multiply_vector_real(factor))
                    }
                }
                
                fn divide_vector(self, divisor: &Self) -> VecResult<Self>
                {
                    let len = self.len();
                    reject_if!(self, len != divisor.len(), ErrorReason::VectorsMustHaveTheSameSize);
                    assert_meta_data!(self, divisor);
                    
                    if self.is_complex
                    {
                        Ok(self.divide_vector_complex(divisor))
                    }
                    else
                    {
                        Ok(self.divide_vector_real(divisor))
                    }
                }
                
                fn zero_pad(mut self, points: usize) -> VecResult<Self>
                {
                    {
                        let len_before = self.len();
                        let len = if self.is_complex { 2 * points } else { points };
                        self.reallocate(len);
                        let array = &mut self.data;
                        for i in len_before..len
                        {
                            array[i] = 0.0;
                        }
                    }
                    
                    Ok(self)
                }
                
                impl_real_complex_dispatch!(zero_interleave, zero_interleave_real, zero_interleave_complex);
                
                fn diff(mut self) -> VecResult<Self>
                {
                    {
                        let data_length = self.len();
                        let org = &self.data;
                        let mut target = temp_mut!(self, data_length);
                        if self.is_complex {
                            self.valid_len -= 2;
                            Chunk::execute_original_to_target(
                                Complexity::Small, &self.multicore_settings,
                                &org, data_length, 2, 
                                &mut target, data_length, 2, 
                                |original, range, target| {
                                    let mut i = 0;
                                    let mut j = range.start;
                                    let mut len = target.len();
                                    if range.end == original.len() - 1
                                    {
                                        len -= 2;
                                    }
                                    
                                    while i < len
                                    { 
                                        target[i] = original[j + 2] - original[i];
                                        i += 1;
                                        j += 1;
                                    }
                            });
                        }
                        else {
                            self.valid_len -= 1;
                            Chunk::execute_original_to_target(
                                Complexity::Small, &self.multicore_settings,
                                &org, data_length, 1, 
                                &mut target, data_length, 1, 
                                |original, range, target| {
                                    let mut i = 0;
                                    let mut j = range.start;
                                    let mut len = target.len();
                                    if range.end >= original.len() - 1
                                    {
                                        len -= 1;
                                    }
                                        
                                    while i < len
                                    { 
                                        target[i] = original[j + 1] - original[j];
                                        i += 1;
                                        j += 1;
                                    }
                            });
                        }
                    }
                    
                    Ok(self.swap_data_temp())
                }
                
                fn diff_with_start(mut self) -> VecResult<Self>
                {
                    {
                        let data_length = self.len();
                        let mut target = temp_mut!(self, data_length);
                        let org = &self.data;
                        if self.is_complex {
                            Chunk::execute_original_to_target(
                                Complexity::Small, &self.multicore_settings,
                                &org, data_length, 2, 
                                &mut target, data_length, 2, 
                                |original, range, target| {
                                    let mut i = 0;
                                    let mut j = range.start;
                                    if j == 0 {
                                        i = 2;
                                        j = 2;
                                        target[0] = original[0];
                                        target[1] = original[1];
                                    }
                                    
                                    while i < target.len()
                                    { 
                                        target[i] = original[j] - original[j - 2];
                                        i += 1;
                                        j += 1;
                                    }
                            });
                        }
                        else {
                            Chunk::execute_original_to_target(
                                Complexity::Small, &self.multicore_settings,
                                &org, data_length, 1, 
                                &mut target, data_length, 1, 
                                |original, range, target| {
                                    let mut i = 0;
                                    let mut j = range.start;
                                    if j == 0 {
                                        i = 1;
                                        j = 1;
                                        target[0] = original[0];
                                    }
                                    
                                    while i < target.len()
                                    { 
                                        target[i] = original[j] - original[j - 1];
                                        i += 1;
                                        j += 1;
                                    }
                            });
                        }
                    }
                    
                    Ok(self.swap_data_temp())
                }
                
                fn cum_sum(mut self) -> VecResult<Self>
                {
                    {
                        let data_length = self.len();
                        let mut data = &mut self.data;
                        let mut i = 0;
                        let mut j = 1;
                        if self.is_complex {
                            j = 2;
                        }
                        
                        while j < data_length {
                            data[j] = data[j] + data[i];
                            i += 1;
                            j += 1;
                        }
                    }
                    Ok(self)
                }
                
                impl_real_complex_dispatch!(sqrt, real_sqrt, complex_sqrt);
                impl_real_complex_dispatch!(square, real_square, complex_square);
                impl_real_complex_arg_dispatch!(root, $data_type, degree, real_root, complex_root);
                impl_real_complex_arg_dispatch!(power, $data_type, exponent, real_power, complex_power);
                impl_real_complex_dispatch!(logn, real_logn, complex_logn);
                impl_real_complex_dispatch!(expn, real_expn, complex_expn);
                impl_real_complex_arg_dispatch!(log_base, $data_type, base, real_log_base, complex_log_base);
                impl_real_complex_arg_dispatch!(exp_base, $data_type, base, real_exp_base, complex_exp_base);
                impl_real_complex_dispatch!(sin, real_sin, complex_sin);
                impl_real_complex_dispatch!(cos, real_cos, complex_cos);
                impl_real_complex_dispatch!(tan, real_tan, complex_tan);
                impl_real_complex_dispatch!(asin, real_asin, complex_asin);
                impl_real_complex_dispatch!(acos, real_acos, complex_acos);
                impl_real_complex_dispatch!(atan, real_atan, complex_atan);
                impl_real_complex_dispatch!(sinh, real_sinh, complex_sinh);
                impl_real_complex_dispatch!(cosh, real_cosh, complex_cosh);
                impl_real_complex_dispatch!(tanh, real_tanh, complex_tanh);
                impl_real_complex_dispatch!(asinh, real_asinh, complex_asinh);
                impl_real_complex_dispatch!(acosh, real_acosh, complex_acosh);
                impl_real_complex_dispatch!(atanh, real_atanh, complex_atanh);
                
                fn swap_halves(mut self) -> VecResult<Self>
                {
                   {
                        use std::ptr;
                        let data_length = self.len();
                        let mut len = self.points();
                        let mut start = len;
                        let is_odd = len % 2 == 1;
                        if !self.is_complex {
                            if is_odd {
                                // Copy middle element
                                let data = &self.data;;
                                let target = temp_mut!(self, data_length);
                                target[len / 2] = data[len / 2];
                                start = start + 1;
                            }
                            
                            len = len / 2;
                            start = start / 2;
                        }
                        else {
                            if is_odd {
                                let data = &self.data;;
                                let target = temp_mut!(self, data_length);
                                // Copy middle element
                                target[len - 1] = data[len - 1];
                                target[len] = data[len];
                                start = start + 1;
                                len = len - 1;
                            }
                        }
                             
                        let mut temp = temp_mut!(self, data_length);   
                        // First half
                        let data = &self.data[start] as *const $data_type;
                        let target = &mut temp[0] as *mut $data_type;
                        unsafe {
                            ptr::copy(data, target, len);
                        }
                        
                        // Second half
                        let data = &self.data[0] as *const $data_type;
                        let target = &mut temp[start] as *mut $data_type;
                        unsafe {
                            ptr::copy(data, target, len);
                        }
                    }
                    
                    Ok(self.swap_data_temp())
                }
                
                fn override_data(mut self, data: &[$data_type]) -> VecResult<Self> {
                    {
                        use std::ptr;
                        self.reallocate(data.len());
                        let target = &mut self.data[0] as *mut $data_type;
                        let source = &data[0] as *const $data_type;
                        unsafe {
                            ptr::copy(source, target, data.len());
                        }
                    }
                    
                    Ok(self)
                }
                
                fn split_into(&self, targets: &mut [Box<Self>]) -> VoidResult {
                    let num_targets = targets.len();
                    let data_length = self.len();
                    if num_targets == 0 || data_length % num_targets != 0 {
                        return Err(ErrorReason::InvalidArgumentLength);
                    }
                    
                    for i in 0..num_targets {
                        targets[i].reallocate(data_length / num_targets);
                    }
                    
                    let data = &self.data;
                    if self.is_complex {
                        for i in 0..(data_length / 2) {
                            let target = &mut targets[i % num_targets];
                            let pos = i / num_targets;
                            target[2 * pos] = data[2 * i];
                            target[2 * pos + 1] = data[2 * i + 1];
                        }
                    } else {
                        for i in 0..data_length {
                            let target = &mut targets[i % num_targets];
                            let pos = i / num_targets;
                            target[pos] = data[i];
                        }
                    }
                    
                    Ok(())
                }
                
                fn merge(mut self, sources: &[Box<Self>]) -> VecResult<Self> {
                    {
                        let num_sources = sources.len();
                        reject_if!(self, num_sources == 0, ErrorReason::InvalidArgumentLength);
                        for i in 1..num_sources {
                            reject_if!(self, sources[0].len() != sources[i].len(), ErrorReason::InvalidArgumentLength);
                        }
                        
                        self.reallocate(sources[0].len() * num_sources);
                        
                        let data_length = self.len();
                        let data = &mut self.data;
                        if self.is_complex {
                            for i in 0..(data_length / 2) {
                                let source = &sources[i % num_sources];
                                let pos = i / num_sources;
                                data[2 * i] = source[2 * pos];
                                data[2 * i + 1] = source[2 * pos + 1];
                            }
                        } else {
                           for i in 0..data_length {
                                let source = &sources[i % num_sources];
                                let pos = i / num_sources;
                                data[i] = source[pos];
                            } 
                        }
                    }
                    
                    Ok(self)
                }
            }
            
            impl GenericDataVector<$data_type> {                
                fn multiply_vector_complex(mut self, factor: &Self) -> Self
                {
                    {
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        let mut array = &mut self.data;
                        let other = &factor.data;
                        Chunk::execute_original_to_target(
                            Complexity::Small, &self.multicore_settings,
                            &other, vectorization_length, $reg::len(), 
                            &mut array, vectorization_length, $reg::len(), 
                            |original, range, target| {
                                let mut i = 0;
                                let mut j = range.start;
                                while i < target.len()
                                { 
                                    let vector1 = $reg::load(original, j);
                                    let vector2 = $reg::load(target, i);
                                    let result = vector2.mul_complex(vector1);
                                    result.store(target, i);
                                    i += 4;
                                    j += 4;
                                }
                        });
                        let mut i = vectorization_length;
                        while i < data_length
                        {
                            let complex1 = Complex::<$data_type>::new(array[i], array[i + 1]);
                            let complex2 = Complex::<$data_type>::new(other[i], other[i + 1]);
                            let result = complex1 * complex2;
                            array[i] = result.re;
                            array[i + 1] = result.im;
                            i += 2;
                        }
                    }
                    self
                }
                
                fn multiply_vector_real(mut self, factor: &Self) -> Self
                {
                    {
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        let mut array = &mut self.data;
                        let other = &factor.data;
                        Chunk::execute_original_to_target(
                            Complexity::Small, &self.multicore_settings,
                            &other, vectorization_length, $reg::len(), 
                            &mut array, vectorization_length, $reg::len(), 
                            |original, range, target| {
                                let mut i = 0;
                                let mut j = range.start;
                                while i < target.len()
                                { 
                                    let vector1 = $reg::load(original, j);
                                    let vector2 = $reg::load(target, i);
                                    let result = vector2 * vector1;
                                    result.store(target, i);
                                    i += $reg::len();
                                    j += $reg::len();
                                }        
                        });
                        let mut i = vectorization_length;
                        while i < data_length
                        {
                            array[i] = array[i] * other[i];
                            i += 1;
                        }
                    }
                    self
                }
                
                fn divide_vector_complex(mut self, divisor: &Self) -> Self
                {
                    {
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        let mut array = &mut self.data;
                        let other = &divisor.data;
                        Chunk::execute_original_to_target(
                            Complexity::Small, &self.multicore_settings,
                            &other, vectorization_length, $reg::len(), 
                            &mut array, vectorization_length, $reg::len(),  
                            |original, range, target| {
                                let mut i = 0;
                                let mut j = range.start;
                                while i < target.len()
                                { 
                                    let vector1 = $reg::load(original, j);
                                    let vector2 = $reg::load(target, i);
                                    let result = vector2.div_complex(vector1);
                                    result.store(target, i);
                                    i += 4;
                                    j += 4;
                                }
                        });
                        let mut i = vectorization_length;
                        while i < data_length
                        {
                            let complex1 = Complex::<$data_type>::new(array[i], array[i + 1]);
                            let complex2 = Complex::<$data_type>::new(other[i], other[i + 1]);
                            let result = complex1 / complex2;
                            array[i] = result.re;
                            array[i + 1] = result.im;
                            i += 2;
                        }
                    }
                    self
                }
                
                fn divide_vector_real(mut self, divisor: &Self) -> Self
                {
                    {
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        let mut array = &mut self.data;
                        let other = &divisor.data;
                        Chunk::execute_original_to_target(
                            Complexity::Small, &self.multicore_settings,
                            &other, vectorization_length, $reg::len(), 
                            &mut array, vectorization_length, $reg::len(), 
                            |original, range, target| {
                                let mut i = 0;
                                let mut j = range.start;
                                while i < target.len()
                                { 
                                    let vector1 = $reg::load(original, j);
                                    let vector2 = $reg::load(target, i);
                                    let result = vector2 / vector1;
                                    result.store(target, i);
                                    i += $reg::len();
                                    j += $reg::len();
                                }
                        });
                        let mut i = vectorization_length;
                        while i < data_length
                        {
                            array[i] = array[i] / other[i];
                            i += 1;
                        }
                    }
                    self
                }
                
                fn zero_interleave_complex(mut self) -> VecResult<Self>
                {
                    {
                        let new_len = 2 * self.len();
                        self.reallocate(new_len);
                        let data_length = new_len;
                        let mut target = temp_mut!(self, data_length);
                        let source = &self.data;
                        Chunk::execute_original_to_target(
                            Complexity::Small, &self.multicore_settings,
                            &source, data_length, $reg::len(), 
                            &mut target, data_length, $reg::len(), 
                            |original, range, target| {
                                let mut i = 0;
                                let mut j = range.start;
                                while i < target.len() / 2 {
                                    if i % 2 == 0
                                    {
                                        target[2 * i] = original[j];
                                        target[2 * i + 1] = original[j + 1];
                                        j += 2;
                                    }
                                    else
                                    {
                                        target[2 * i] = 0.0;
                                        target[2 * i + 1] = 0.0;
                                    }
                                    
                                    i += 1;
                                }
                        });
                    }
                    Ok(self.swap_data_temp())
                }
                
                fn zero_interleave_real(mut self) -> VecResult<Self>
                {
                    {
                        let new_len = 2 * self.len();
                        self.reallocate(new_len);
                        let data_length = new_len;
                        let mut target = temp_mut!(self, data_length);
                        let source = &self.data;
                        Chunk::execute_original_to_target(
                            Complexity::Small, &self.multicore_settings,
                            &source, data_length, 4, 
                            &mut target, data_length, 2, 
                            |original, range, target| {
                                let mut i = 0;
                                let mut j = range.start;
                                while i < target.len() {
                                    if i % 2 == 0
                                    {
                                        target[i] = original[j];
                                        j += 1;
                                    }
                                    else
                                    {
                                        target[i] = 0.0;
                                    }
                                    
                                    i += 1;
                                }
                        });
                    }
                    Ok(self.swap_data_temp())
                }
                
                fn real_sqrt(mut self) -> VecResult<Self>
                {
                    {
                        let data_length = self.len();
                        let mut array = &mut self.data;
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        Chunk::execute_partial(
                            Complexity::Small, &self.multicore_settings,
                            &mut array, vectorization_length, $reg::len(), 
                            |array| {
                                let mut i = 0;
                                while i < array.len()
                                {
                                    let vector = $reg::load(array, i);
                                    let result = vector.sqrt();
                                    result.store(array, i);
                                    i += $reg::len();
                                }
                        });
                        for i in vectorization_length..data_length
                        {
                            array[i] = array[i].sqrt();
                        }
                    }
                    Ok(self)
                }
                
                fn complex_sqrt(mut self) -> VecResult<Self>
                {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial(
                            Complexity::Medium, &self.multicore_settings,
                            &mut array, length, 2, 
                            |array| {
                                let mut i = 0;
                                while i < array.len()
                                {
                                    let complex = Complex::<$data_type>::new(array[i], array[i + 1]);
                                    let result = complex.sqrt();
                                    array[i] = result.re;
                                    array[i + 1] = result.im;
                                    i += 2;
                                }
                        });
                    }
                    Ok(self)
                }
                
                fn real_square(mut self) -> VecResult<Self>
                {
                    {
                        let data_length = self.len();
                        let mut array = &mut self.data;
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        Chunk::execute_partial(
                            Complexity::Small, &self.multicore_settings,
                            &mut array, vectorization_length, $reg::len(), 
                            |array| {
                                let mut i = 0;
                                while i < array.len()
                                {
                                    let a = $reg::load(array, i);
                                    let result = a * a;
                                    result.store(array, i);
                                    i += $reg::len();
                                }
                        });
                        
                        for i in vectorization_length..data_length
                        {
                            array[i] = array[i] * array[i];
                        }
                    }
                    Ok(self)
                }
                
                fn complex_square(mut self) -> VecResult<Self>
                {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial(
                            Complexity::Medium, &self.multicore_settings,
                            &mut array, length, 2, 
                            |array| {
                                let mut i = 0;
                                while i < array.len()
                                {
                                    let complex = Complex::<$data_type>::new(array[i], array[i + 1]);
                                    let result = complex * complex;
                                    array[i] = result.re;
                                    array[i + 1] = result.im;
                                    i += 2;
                                }
                        });
                    }
                    Ok(self)
                }
                
                fn real_root(mut self, degree: $data_type) -> VecResult<Self>
                {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial_with_arguments(
                            Complexity::Medium, &self.multicore_settings,
                            &mut array, length, 1, degree, 
                            |array, base| {
                                let base = 1.0 / base;
                                let mut i = 0;
                                while i < array.len()
                                {
                                    array[i] = array[i].powf(base);
                                    i += 1;
                                }
                        });
                    }
                    Ok(self)
                }
                
                fn complex_root(mut self, base: $data_type) -> VecResult<Self>
                {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial_with_arguments(
                            Complexity::Medium, &self.multicore_settings, 
                            &mut array, length, 2, base, 
                            |array, base| {
                                let base = 1.0 / base;
                                let mut i = 0;
                                while i < array.len()
                                {
                                    let complex = Complex::<$data_type>::new(array[i], array[i + 1]);
                                    let result = complex.powf(base);
                                    array[i] = result.re;
                                    array[i + 1] = result.im;
                                    i += 2;
                                }
                        });
                    }
                    Ok(self)
                }
                
                impl_function_call_real_arg_complex!($data_type; real_power, powf; complex_power, powf);
                impl_function_call_real_complex!($data_type; real_logn, ln; complex_logn, ln);
                impl_function_call_real_complex!($data_type; real_expn, exp; complex_expn, expn);
                impl_function_call_real_arg_complex!($data_type; real_log_base, log; complex_log_base, log_base);
                
                fn real_exp_base(mut self, base: $data_type) -> VecResult<Self>
                {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial_with_arguments(
                            Complexity::Medium, &self.multicore_settings,
                            &mut array, length, 1, base, 
                            |array, base| {
                                let mut i = 0;
                                while i < array.len()
                                {
                                    array[i] = base.powf(array[i]);
                                    i += 1;
                                }
                        });
                    }
                    Ok(self)
                }
                
                fn complex_exp_base(mut self, base: $data_type) -> VecResult<Self>
                {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial_with_arguments(
                            Complexity::Medium, &self.multicore_settings,
                            &mut array, length, 2, base, 
                            |array, base| {
                                let mut i = 0;
                                while i < array.len()
                                {
                                    let complex = Complex::<$data_type>::new(array[i], array[i + 1]);
                                    let result = complex.exp_base(base);
                                    array[i] = result.re;
                                    array[i + 1] = result.im;
                                    i += 2;
                                }
                        });
                    }
                    Ok(self)
                }
                               
                impl_function_call_real_complex!($data_type; real_sin, sin; complex_sin, sin);
                impl_function_call_real_complex!($data_type; real_cos, cos; complex_cos, cos);
                impl_function_call_real_complex!($data_type; real_tan, tan; complex_tan, tan);
                impl_function_call_real_complex!($data_type; real_asin, asin; complex_asin, asin);
                impl_function_call_real_complex!($data_type; real_acos, acos; complex_acos, acos);
                impl_function_call_real_complex!($data_type; real_atan, atan; complex_atan, atan);
                impl_function_call_real_complex!($data_type; real_sinh, sinh; complex_sinh, sinh);
                impl_function_call_real_complex!($data_type; real_cosh, cosh; complex_cosh, cosh);
                impl_function_call_real_complex!($data_type; real_tanh, tanh; complex_tanh, tanh);
                impl_function_call_real_complex!($data_type; real_asinh, asinh; complex_asinh, asinh);
                impl_function_call_real_complex!($data_type; real_acosh, acosh; complex_acosh, acosh);
                impl_function_call_real_complex!($data_type; real_atanh, atanh; complex_atanh, atanh);
            }
        )*
     }
}

add_general_impl!(f32, Reg32; f64, Reg64);