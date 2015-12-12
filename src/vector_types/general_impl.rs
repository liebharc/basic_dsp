use multicore_support::{Chunk, Complexity};
use super::definitions::{
	DataVector,
    DataVectorDomain,
	GenericVectorOperations,
    VecResult,
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

macro_rules! impl_trig_function_real_complex {
    ($data_type: ident; $real_name: ident, $real_op: ident; $complex_name: ident, $complex_op: ident) => {
        fn $real_name(mut self) -> VecResult<Self>
                {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial(Complexity::Medium, &mut array, length, 1, |array| {
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
                        Chunk::execute_partial(Complexity::Medium, &mut array, length, 2, |array| {
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

macro_rules! add_general_impl {
    ($($data_type:ident, $reg:ident);*)
	 =>
	 {	 
        $(
            impl GenericDataVector<$data_type> {
                /// Creates a new generic data vector from the given arguments.
                pub fn new(is_complex: bool, domain: DataVectorDomain, init_value: $data_type, length: usize, delta: $data_type) -> Self {
                    GenericDataVector
                    { 
                        data: vec![init_value; length],
                        temp: vec![0.0; length],
                        delta: delta,
                        domain: domain,
                        is_complex: is_complex,
                        valid_len: length,
                        multicore_settings: MultiCoreSettings::new()
                    }
                }
                
                /// Creates a new generic data vector from the given arguments.
                pub fn from_array(is_complex: bool, domain: DataVectorDomain, data: &[$data_type]) -> Self {
                    let length = data.len();
                    GenericDataVector
                    { 
                        data: data.to_vec(),
                        temp: vec![0.0; length],
                        delta: 1.0,
                        domain: domain,
                        is_complex: is_complex,
                        valid_len: length,
                        multicore_settings: MultiCoreSettings::new()
                    }
                }
                
                /// Creates a new generic data vector from the given arguments.
                pub fn from_array_no_copy(is_complex: bool, domain: DataVectorDomain, data: Vec<$data_type>) -> Self {
                    let length = data.len();
                    GenericDataVector
                    { 
                        data: data,
                        temp: vec![0.0; length],
                        delta: 1.0,
                        domain: domain,
                        is_complex: is_complex,
                        valid_len: length,
                        multicore_settings: MultiCoreSettings::new()
                    }
                }
                
                /// Creates a new generic data vector from the given arguments.
                pub fn from_array_with_delta(is_complex: bool, domain: DataVectorDomain, data: &[$data_type], delta: $data_type) -> Self {
                    let length = data.len();
                    GenericDataVector
                    { 
                        data: data.to_vec(),
                        temp: vec![0.0; length],
                        delta: delta,
                        domain: domain,
                        is_complex: is_complex,
                        valid_len: length,
                        multicore_settings: MultiCoreSettings::new()
                    }
                }
                
                /// Creates a new generic data vector from the given arguments.
                pub fn from_array_no_copy_with_delta(is_complex: bool, domain: DataVectorDomain, data: Vec<$data_type>, delta: $data_type) -> Self {
                    let length = data.len();
                    GenericDataVector
                    { 
                        data: data,
                        temp: vec![0.0; length],
                        delta: delta,
                        domain: domain,
                        is_complex: is_complex,
                        valid_len: length,
                        multicore_settings: MultiCoreSettings::new()
                    }
                }
            }
            
            #[inline]
            impl GenericVectorOperations<$data_type> for GenericDataVector<$data_type> {
                fn add_vector(mut self, summand: &Self) -> VecResult<Self>
                {
                    {
                        let len = self.len();
                        reject_if!(self, len != summand.len(), ErrorReason::VectorsMustHaveTheSameSize);
                        
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        let mut array = &mut self.data;
                        let other = &summand.data;
                        Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, $reg::len(), &mut array, vectorization_length, $reg::len(),  |original, range, target| {
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
                            
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        let mut array = &mut self.data;
                        let other = &subtrahend.data;
                        Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, $reg::len(), &mut array, vectorization_length, $reg::len(), |original, range, target| {
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
                        let mut target = &mut self.temp;
                        let org = &self.data;
                        if self.is_complex {
                            self.valid_len -= 2;
                            Chunk::execute_original_to_target(Complexity::Small, &org, data_length, 2, &mut target, data_length, 2, |original, range, target| {
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
                            Chunk::execute_original_to_target(Complexity::Small, &org, data_length, 1, &mut target, data_length, 1, |original, range, target| {
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
                        let mut target = &mut self.temp;
                        let org = &self.data;
                        if self.is_complex {
                            Chunk::execute_original_to_target(Complexity::Small, &org, data_length, 2, &mut target, data_length, 2, |original, range, target| {
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
                            Chunk::execute_original_to_target(Complexity::Small, &org, data_length, 1, &mut target, data_length, 1, |original, range, target| {
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
                
                fn swap_halves(mut self) -> VecResult<Self>
                {
                   {
                        use std::ptr;
                        let mut len = self.points();
                        let mut start = len;
                        let is_odd = len % 2 == 1;
                        if !self.is_complex {
                            if is_odd {
                                // Copy middle element
                                let data = &self.data;;
                                let target = &mut self.temp;
                                target[len / 2] = data[len / 2];
                                start = start + 1;
                            }
                            
                            len = len / 2;
                            start = start / 2;
                        }
                        else {
                            if is_odd {
                                let data = &self.data;;
                                let target = &mut self.temp;
                                // Copy middle element
                                target[len - 1] = data[len - 1];
                                target[len] = data[len];
                                start = start + 1;
                                len = len - 1;
                            }
                        }
                                
                        // First half
                        let data = &self.data[start] as *const $data_type;
                        let target = &mut self.temp[0] as *mut $data_type;
                        unsafe {
                            ptr::copy(data, target, len);
                        }
                        
                        // Second half
                        let data = &self.data[0] as *const $data_type;
                        let target = &mut self.temp[start] as *mut $data_type;
                        unsafe {
                            ptr::copy(data, target, len);
                        }
                    }
                    
                    Ok(self.swap_data_temp())
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
                        Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, $reg::len(), &mut array, vectorization_length, $reg::len(), |original, range, target| {
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
                        Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, $reg::len(), &mut array, vectorization_length, $reg::len(), |original, range, target| {
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
                        Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, $reg::len(), &mut array, vectorization_length, $reg::len(),  |original, range, target| {
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
                        Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, $reg::len(), &mut array, vectorization_length, $reg::len(), |original, range, target| {
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
                        let mut target = &mut self.temp;
                        let source = &self.data;
                        Chunk::execute_original_to_target(Complexity::Small, &source, data_length, $reg::len(), &mut target, data_length, $reg::len(), |original, range, target| {
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
                        let mut target = &mut self.temp;
                        let source = &self.data;
                        Chunk::execute_original_to_target(Complexity::Small, &source, data_length, 4, &mut target, data_length, 2,  |original, range, target| {
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
                        Chunk::execute_partial(Complexity::Small, &mut array, vectorization_length, $reg::len(), |array| {
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
                        Chunk::execute_partial(Complexity::Medium, &mut array, length, 2, |array| {
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
                        Chunk::execute_partial(Complexity::Small, &mut array, vectorization_length, $reg::len(), |array| {
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
                        Chunk::execute_partial(Complexity::Medium, &mut array, length, 2, |array| {
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
                        Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 1, degree, |array, base| {
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
                        Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 2, base, |array, base| {
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
                
                fn real_power(mut self, exponent: $data_type) -> VecResult<Self>
                {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 1, exponent, |array, base| {
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
                
                fn complex_power(mut self, base: $data_type) -> VecResult<Self>
                {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 2, base, |array, base| {
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
                
                fn real_logn(mut self) -> VecResult<Self>
                {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial(Complexity::Medium, &mut array, length, 1, |array| {
                            let mut i = 0;
                            while i < array.len()
                            {
                                array[i] = array[i].ln();
                                i += 1;
                            }
                        });
                    }
                    Ok(self)
                }
                
                fn complex_logn(mut self) -> VecResult<Self>
                {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial(Complexity::Medium, &mut array, length, 2, |array| {
                            let mut i = 0;
                            while i < array.len()
                            {
                                let complex = Complex::<$data_type>::new(array[i], array[i + 1]);
                                let result = complex.ln();
                                array[i] = result.re;
                                array[i + 1] = result.im;
                                i += 2;
                            }
                        });
                    }
                    Ok(self)
                }
                
                fn real_expn(mut self) -> VecResult<Self>
                {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial(Complexity::Medium, &mut array, length, 1, |array| {
                            let mut i = 0;
                            while i < array.len()
                            {
                                array[i] = array[i].exp();
                                i += 1;
                            }
                        });
                    }
                    Ok(self)
                }
                
                fn complex_expn(mut self) -> VecResult<Self>
                {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial(Complexity::Medium, &mut array, length, 2, |array| {
                            let mut i = 0;
                            while i < array.len()
                            {
                                let complex = Complex::<$data_type>::new(array[i], array[i + 1]);
                                let result = complex.expn();
                                array[i] = result.re;
                                array[i + 1] = result.im;
                                i += 2;
                            }
                        });
                    }
                    Ok(self)
                }
            
                fn real_log_base(mut self, base: $data_type) -> VecResult<Self>
                {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 1, base, |array, base| {
                            let mut i = 0;
                            while i < array.len()
                            {
                                array[i] = array[i].log(base);
                                i += 1;
                            }
                        });
                    }
                    Ok(self)
                }
                
                fn complex_log_base(mut self, base: $data_type) -> VecResult<Self>
                {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 2, base, |array, base| {
                            let mut i = 0;
                            while i < array.len()
                            {
                                let complex = Complex::<$data_type>::new(array[i], array[i + 1]);
                                let result = complex.log_base(base);
                                array[i] = result.re;
                                array[i + 1] = result.im;
                                i += 2;
                            }
                        });
                    }
                    Ok(self)
                }
                
                fn real_exp_base(mut self, base: $data_type) -> VecResult<Self>
                {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 1, base, |array, base| {
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
                        Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 2, base, |array, base| {
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
                               
                impl_trig_function_real_complex!($data_type; real_sin, sin; complex_sin, sin);
                impl_trig_function_real_complex!($data_type; real_cos, cos; complex_cos, cos);
            }
        )*
     }
}

add_general_impl!(f32, Reg32; f64, Reg64);