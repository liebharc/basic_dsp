use multicore_support::{Chunk, Complexity};
use super::definitions::{
    DataVector,
    DataVectorDomain,
    GenericVectorOps,
    VecResult,
    VoidResult,
    ErrorReason,
    PaddingOption};
use super::{
    GenericDataVector,
    round_len};
use num::complex::Complex;
use simd_extensions::*;
use multicore_support::MultiCoreSettings;
use std::ops::{Add, Sub, Mul, Div};
use std::ptr;

macro_rules! impl_real_complex_dispatch {
    (fn $function_name: ident, $real_op: ident, $complex_op: ident)
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
    (fn $function_name: ident, $arg_type: ident, $arg: ident, $real_op: ident, $complex_op: ident)
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
    ($data_type: ident; fn $real_name: ident, $real_op: ident; fn $complex_name: ident, $complex_op: ident) => {
        fn $real_name(self) -> VecResult<Self>
        {
            self.pure_real_operation(|v, _arg| v.$real_op(), (), Complexity::Small)
        }

        fn $complex_name(self) -> VecResult<Self>
        {
            self.pure_complex_operation(|v, _arg| v.$complex_op(), (), Complexity::Small)
        }
    }
}

macro_rules! impl_function_call_real_arg_complex {
    ($data_type: ident; fn $real_name: ident, $real_op: ident; fn $complex_name: ident, $complex_op: ident) => {
        fn $real_name(self, value: $data_type) -> VecResult<Self>
        {
            self.pure_real_operation(|v, arg| v.$real_op(arg), value, Complexity::Medium)
        }

        fn $complex_name(self, value: $data_type) -> VecResult<Self>
        {
            self.pure_complex_operation(|v, arg| v.$complex_op(arg), value, Complexity::Medium)
        }
    }
}

macro_rules! impl_binary_vector_operation {
    ($data_type: ident, $reg: ident, fn $method: ident, $arg_name: ident, $simd_op: ident, $scal_op: ident) => {
        fn $method(mut self, $arg_name: &Self) -> VecResult<Self>
        {
            {
                let len = self.len();
                reject_if!(self, len != $arg_name.len(), ErrorReason::VectorsMustHaveTheSameSize);
                assert_meta_data!(self, $arg_name);

                let data_length = self.len();
                let (scalar_left, scalar_right, vectorization_length) = $reg::calc_data_alignment_reqs(&self.data[0..data_length]);
                let mut array = &mut self.data;
                let other = &$arg_name.data;
                Chunk::from_src_to_dest(
                    Complexity::Small, &self.multicore_settings,
                    &other[0..vectorization_length], $reg::len(),
                    &mut array[scalar_left..vectorization_length], $reg::len(), (),
                    |original, range, target, _arg| {
                        let mut i = 0;
                        let mut j = range.start;
                        while i < target.len()
                        {
                            let vector1 = $reg::load_unchecked(original, j);
                            let vector2 = $reg::load_unchecked(target, i);
                            let result = vector2.$simd_op(vector1);
                            result.store_unchecked(target, i);
                            i += $reg::len();
                            j += $reg::len();
                        }
                });

                for i in 0..scalar_left {
                    array[i] = array[i].$scal_op(other[i]);
                }

                for i in scalar_right..data_length {
                    array[i] = array[i].$scal_op(other[i]);
                }
            }

            Ok(self)
        }
    }
}

macro_rules! impl_binary_complex_vector_operation {
    ($data_type: ident, $reg: ident, fn $method: ident, $arg_name: ident, $simd_op: ident, $scal_op: ident) => {
        fn $method(mut self, $arg_name: &Self) -> VecResult<Self>
        {
            {
                let len = self.len();
                reject_if!(self, len != $arg_name.len(), ErrorReason::VectorsMustHaveTheSameSize);
                assert_meta_data!(self, $arg_name);

                let data_length = self.len();
                let (scalar_left, scalar_right, vectorization_length) = $reg::calc_data_alignment_reqs(&self.data[0..data_length]);
                let mut array = &mut self.data;
                let other = &$arg_name.data;
                Chunk::from_src_to_dest(
                    Complexity::Small, &self.multicore_settings,
                    &other[scalar_left..vectorization_length], $reg::len(),
                    &mut array[scalar_left..vectorization_length], $reg::len(), (),
                    |original, range, target, _arg| {
                        let mut i = 0;
                        let mut j = range.start;
                        while i < target.len()
                        {
                            let vector1 = $reg::load_unchecked(original, j);
                            let vector2 = $reg::load_unchecked(target, i);
                            let result = vector2.$simd_op(vector1);
                            result.store_unchecked(target, i);
                            i += $reg::len();
                            j += $reg::len();
                        }
                });

                let mut i = 0;
                while i < scalar_left {
                    let complex1 = Complex::<$data_type>::new(array[i], array[i + 1]);
                    let complex2 = Complex::<$data_type>::new(other[i], other[i + 1]);
                    let result = complex1.$scal_op(complex2);
                    array[i] = result.re;
                    array[i + 1] = result.im;
                    i += 2;
                }

                let mut i = scalar_right;
                while i < data_length {
                    let complex1 = Complex::<$data_type>::new(array[i], array[i + 1]);
                    let complex2 = Complex::<$data_type>::new(other[i], other[i + 1]);
                    let result = complex1.$scal_op(complex2);
                    array[i] = result.re;
                    array[i + 1] = result.im;
                    i += 2;
                }
            }

            Ok(self)
        }
    }
}

macro_rules! impl_binary_smaller_vector_operation {
    ($data_type: ident, $reg: ident, fn $method: ident, $arg_name: ident, $simd_op: ident, $scal_op: ident) => {
        fn $method(mut self, $arg_name: &Self) -> VecResult<Self>
        {
            {
                let len = self.len();
                reject_if!(self, len % $arg_name.len() != 0, ErrorReason::InvalidArgumentLength);
                assert_meta_data!(self, $arg_name);

                let data_length = self.len();
                let (scalar_left, scalar_right, vectorization_length) = $reg::calc_data_alignment_reqs(&self.data[0..data_length]);
                let mut array = &mut self.data;
                let other = &$arg_name.data;
                Chunk::from_src_to_dest(
                    Complexity::Small, &self.multicore_settings,
                    &other, $reg::len(),
                    &mut array[scalar_left..vectorization_length], $reg::len(), (),
                    |original, range, target, _arg| {
                        // This parallelization likely doesn't make sense for the use
                        // case which we have in mind with this implementation
                        // so we likely have to revisit this code piece in future
                        let mut i = 0;
                        let mut j = range.start;
                        while i < target.len()
                        {
                            let vector1 =
                                if j + $reg::len() < original.len() {
                                    $reg::load_unchecked(original, j)
                                } else {
                                    $reg::load_wrap_unchecked(original, j)
                                };
                            let vector2 = $reg::load_unchecked(target, i);
                            let result = vector2.$simd_op(vector1);
                            result.store_unchecked(target, i);
                            i += $reg::len();
                            j = (j + $reg::len()) % original.len();
                        }
                });

                for i in 0..scalar_left {
                    array[i] = array[i].$scal_op(other[i % $arg_name.len()]);
                }

                for i in scalar_right..data_length {
                    array[i] = array[i].$scal_op(other[i % $arg_name.len()]);
                }
            }

            Ok(self)
        }
    }
}

macro_rules! impl_binary_smaller_complex_vector_operation {
    ($data_type: ident, $reg: ident, fn $method: ident, $arg_name: ident, $simd_op: ident, $scal_op: ident) => {
        fn $method(mut self, $arg_name: &Self) -> VecResult<Self>
        {
            {
                let len = self.len();
                reject_if!(self, len % $arg_name.len() != 0, ErrorReason::InvalidArgumentLength);
                assert_meta_data!(self, $arg_name);

                let data_length = self.len();
                let (scalar_left, scalar_right, vectorization_length) = $reg::calc_data_alignment_reqs(&self.data[0..data_length]);
                let mut array = &mut self.data;
                let other = &$arg_name.data[0..$arg_name.len()];
                Chunk::from_src_to_dest(
                    Complexity::Small, &self.multicore_settings,
                    &other, $reg::len(),
                    &mut array[scalar_left..vectorization_length], $reg::len(), (),
                    |original, range, target, _arg| {
                        // This parallelization likely doesn't make sense for the use
                        // case which we have in mind with this implementation
                        // so we likely have to revisit this code piece in future
                        let mut i = 0;
                        let mut j = range.start;
                        while i < target.len()
                        {
                            let vector1 =
                                if j + $reg::len() < original.len() {
                                    $reg::load_unchecked(original, j)
                                } else {
                                    $reg::load_wrap_unchecked(original, j)
                                };
                            let vector2 = $reg::load_unchecked(target, i);
                            let result = vector2.$simd_op(vector1);
                            result.store_unchecked(target, i);
                            i += $reg::len();
                            j = (j + $reg::len()) % original.len();
                        }
                });

                let mut i = 0;
                while i < scalar_left {
                    let complex1 = Complex::<$data_type>::new(array[i], array[i + 1]);
                    let complex2 = Complex::<$data_type>::new(other[i % other.len()], other[(i + 1) % other.len()]);
                    let result = complex1.$scal_op(complex2);
                    array[i] = result.re;
                    array[i + 1] = result.im;
                    i += 2;
                }

                let mut i = scalar_right;
                while i < data_length {
                    let complex1 = Complex::<$data_type>::new(array[i], array[i + 1]);
                    let complex2 = Complex::<$data_type>::new(other[i % other.len()], other[(i + 1) % other.len()]);
                    let result = complex1.$scal_op(complex2);
                    array[i] = result.re;
                    array[i + 1] = result.im;
                    i += 2;
                }
            }

            Ok(self)
        }
    }
}

macro_rules! vector_diff_par {
    ($self_: ident, $keep_start: ident, $org: ident, $data_length: ident, $target: ident, $step: expr) => {
        if !$keep_start {
            $self_.valid_len -= $step;
        }
        Chunk::from_src_to_dest(
            Complexity::Small, &$self_.multicore_settings,
            &$org[0..$data_length], $step,
            &mut $target[0..$data_length], $step, (),
            |original, range, target, _arg| {
                let mut i = 0;
                let mut j = range.start;
                if $keep_start && j == 0 {
                    i = $step;
                    j = $step;
                    for k in 0..$step {
                        target[k] = original[k];
                    }
                }

                let len =
                    if !$keep_start && range.end >= original.len() - 1 {
                        target.len() - $step
                    } else {
                        target.len()
                    };

                while i < len {
                    target[i] = if $keep_start { original[j] - original[j - $step] } else { original[j + $step] - original[i] };
                    i += 1;
                    j += 1;
                }
        });
    }
}

macro_rules! vector_diff {
    ($self_: ident, $keep_start: ident) => {
        {
            {
                let data_length = $self_.len();
                let mut target = temp_mut!($self_, data_length);
                let org = &$self_.data;
                if $self_.is_complex {
                    vector_diff_par!($self_, $keep_start, org, data_length, target, 2);
                }
                else {
                    vector_diff_par!($self_, $keep_start, org, data_length, target, 1);
                }
            }
            Ok($self_.swap_data_temp())
        }
    }
}

macro_rules! zero_interleave {
    ($self_: ident, $data_type: ident, $step: ident, $tuple: expr) => {
        {
            if $step <= 1 {
                return Ok($self_);
            }

            {
                let step = $step as usize;
                let old_len = $self_.len();
                let new_len = step * old_len;
                $self_.valid_len = new_len;
                let mut target = temp_mut!($self_, new_len);
                let source = &$self_.data;
                Chunk::from_src_to_dest(
                    Complexity::Small, &$self_.multicore_settings,
                    &source[0..old_len], $tuple,
                    &mut target[0..new_len], $tuple * step, (),
                    move|original, range, target, _arg| {
                         // Zero target
                        let ptr = &mut target[0] as *mut $data_type;
                        unsafe {
                            ptr::write_bytes(ptr, 0, new_len);
                        }
                        let skip = step * $tuple;
                        let mut i = 0;
                        let mut j = range.start;
                        while i < target.len() {
                            let original_ptr = unsafe { original.get_unchecked(j) };
                            let target_ptr = unsafe { target.get_unchecked_mut(i) };
                            unsafe {
                                ptr::copy(original_ptr, target_ptr, $tuple);
                            }

                            j += $tuple;
                            i += skip;
                        }
                });
            }
            Ok($self_.swap_data_temp())
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
                        temp: if options.early_temp_allocation { vec![0.0; length] } else { Vec::new() },
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
                        temp: if options.early_temp_allocation { vec![0.0; length] } else { Vec::new() },
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
                        temp: if options.early_temp_allocation { vec![0.0; length] } else { Vec::new() },
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
                        temp: if options.early_temp_allocation { vec![0.0; length] } else { Vec::new() },
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
                        temp: if options.early_temp_allocation { vec![0.0; length] } else { Vec::new() },
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

            impl GenericVectorOps<$data_type> for GenericDataVector<$data_type> {
                impl_binary_vector_operation!($data_type, $reg, fn add_vector, summand, add, add);
                impl_binary_smaller_vector_operation!($data_type, $reg, fn add_smaller_vector, summand, add, add);
                impl_binary_vector_operation!($data_type, $reg, fn subtract_vector, summand, sub, sub);
                impl_binary_smaller_vector_operation!($data_type, $reg, fn subtract_smaller_vector, summand, sub, sub);

                fn multiply_vector(self, factor: &Self) -> VecResult<Self>
                {
                    let len = self.len();
                    reject_if!(self, len != factor.len(), ErrorReason::VectorsMustHaveTheSameSize);
                    assert_meta_data!(self, factor);

                    if self.is_complex
                    {
                        self.multiply_vector_complex(factor)
                    }
                    else
                    {
                        self.multiply_vector_real(factor)
                    }
                }

                fn multiply_smaller_vector(self, factor: &Self) -> VecResult<Self>
                {
                    let len = self.len();
                    reject_if!(self, len % factor.len() != 0, ErrorReason::InvalidArgumentLength);
                    assert_meta_data!(self, factor);

                    if self.is_complex
                    {
                        self.multiply_smaller_vector_complex(factor)
                    }
                    else
                    {
                        self.multiply_smaller_vector_real(factor)
                    }
                }

                fn divide_vector(self, divisor: &Self) -> VecResult<Self>
                {
                    let len = self.len();
                    reject_if!(self, len != divisor.len(), ErrorReason::VectorsMustHaveTheSameSize);
                    assert_meta_data!(self, divisor);

                    if self.is_complex
                    {
                        self.divide_vector_complex(divisor)
                    }
                    else
                    {
                        self.divide_vector_real(divisor)
                    }
                }

                fn divide_smaller_vector(self, divisor: &Self) -> VecResult<Self>
                {
                    let len = self.len();
                    reject_if!(self, len % divisor.len() != 0, ErrorReason::InvalidArgumentLength);
                    assert_meta_data!(self, divisor);

                    if self.is_complex
                    {
                        self.divide_smaller_vector_complex(divisor)
                    }
                    else
                    {
                        self.divide_smaller_vector_real(divisor)
                    }
                }

                fn zero_pad(mut self, points: usize, option: PaddingOption) -> VecResult<Self>
                {
                    {
                        let len_before = self.len();
                        let allocated_len = self.allocated_len();
                        let is_complex = self.is_complex;
                        let len = if is_complex { 2 * points } else { points };
                        if len < len_before {
                            return Ok(self);
                        }
                        if len > self.allocated_len() {
                            self.data.resize(len, 0.0);
                        }
                        self.valid_len = len;
                        let array = &mut self.data;
                        match option {
                            PaddingOption::End => {
                                // Zero target
                                let ptr = &mut array[len_before] as *mut $data_type;
                                unsafe {
                                    ptr::write_bytes(ptr, 0, allocated_len - len_before);
                                }
                            }
                            PaddingOption::Surround => {
                                let diff = (len - len_before) / if is_complex { 2 } else { 1 };
                                let mut right = diff / 2;
                                let mut left = diff - right;
                                if is_complex {
                                    right *= 2;
                                    left *= 2;
                                }

                                unsafe {
                                    let src = &array[0] as *const $data_type;
                                    let dest = &mut array[left] as *mut $data_type;
                                    ptr::copy(src, dest, len_before);
                                    let dest = &mut array[len - right] as *mut $data_type;
                                    ptr::write_bytes(dest, 0, right);
                                    let dest = &mut array[0] as *mut $data_type;
                                    ptr::write_bytes(dest, 0, left);
                                }
                            }
                            PaddingOption::Center => {
                                let mut diff = (len - len_before) / if is_complex { 2 } else { 1 };
                                let mut right = diff / 2;
                                let mut left = diff - right;
                                if is_complex {
                                    right *= 2;
                                    left *= 2;
                                    diff *= 2;
                                }

                                unsafe {
                                    let src = &array[left] as *const $data_type;
                                    let dest = &mut array[len-right] as *mut $data_type;
                                    ptr::copy(src, dest, right);
                                    let dest = &mut array[left] as *mut $data_type;
                                    ptr::write_bytes(dest, 0, len - diff);
                                }
                            }
                        }
                    }

                    Ok(self)
                }

                fn reverse(mut self) -> VecResult<Self> {
                    {
                        let len = self.len();
                        let is_complex = self.is_complex;
                        let src = &self.data[0..len];
                        let dest = temp_mut!(self, len);
                        if is_complex {
                            let src = Self::array_to_complex(&src[0..len]);
                            let dest = Self::array_to_complex_mut(&mut dest[0..len]);
                            for (s, d) in src.iter().rev().zip(dest) {
                                *d = *s;
                            }
                        } else {
                            for (s, d) in src.iter().rev().zip(dest) {
                                *d = *s;
                            }
                        }
                    }

                    Ok(self.swap_data_temp())
                }

                fn zero_interleave(self, factor: u32) -> VecResult<Self> {
                    if self.is_complex {
                        self.zero_interleave_complex(factor)
                    } else {
                        self.zero_interleave_real(factor)
                    }
                }

                fn diff(mut self) -> VecResult<Self>
                {
                    vector_diff!(self, false)
                }

                fn diff_with_start(mut self) -> VecResult<Self>
                {
                    vector_diff!(self, true)
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

                impl_real_complex_dispatch!(fn sqrt, real_sqrt, complex_sqrt);
                impl_real_complex_dispatch!(fn square, real_square, complex_square);
                impl_real_complex_arg_dispatch!(fn root, $data_type, degree, real_root, complex_root);
                impl_real_complex_arg_dispatch!(fn powf, $data_type, exponent, real_powf, complex_powf);
                impl_real_complex_dispatch!(fn ln, real_ln, complex_ln);
                impl_real_complex_dispatch!(fn exp, real_exp, complex_exp);
                impl_real_complex_arg_dispatch!(fn log, $data_type, base, real_log, complex_log);
                impl_real_complex_arg_dispatch!(fn expf, $data_type, base, real_expf, complex_expf);
                impl_real_complex_dispatch!(fn sin, real_sin, complex_sin);
                impl_real_complex_dispatch!(fn cos, real_cos, complex_cos);
                impl_real_complex_dispatch!(fn tan, real_tan, complex_tan);
                impl_real_complex_dispatch!(fn asin, real_asin, complex_asin);
                impl_real_complex_dispatch!(fn acos, real_acos, complex_acos);
                impl_real_complex_dispatch!(fn atan, real_atan, complex_atan);
                impl_real_complex_dispatch!(fn sinh, real_sinh, complex_sinh);
                impl_real_complex_dispatch!(fn cosh, real_cosh, complex_cosh);
                impl_real_complex_dispatch!(fn tanh, real_tanh, complex_tanh);
                impl_real_complex_dispatch!(fn asinh, real_asinh, complex_asinh);
                impl_real_complex_dispatch!(fn acosh, real_acosh, complex_acosh);
                impl_real_complex_dispatch!(fn atanh, real_atanh, complex_atanh);

                fn swap_halves(self) -> VecResult<Self>
                {
                   self.swap_halves_priv(true)
                }

                fn override_data(mut self, data: &[$data_type]) -> VecResult<Self> {
                    {
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
                impl_binary_complex_vector_operation!($data_type, $reg, fn multiply_vector_complex, factor, mul_complex, mul);
                impl_binary_smaller_complex_vector_operation!($data_type, $reg, fn multiply_smaller_vector_complex, factor, mul_complex, mul);
                impl_binary_vector_operation!($data_type, $reg, fn multiply_vector_real, factor, mul, mul);
                impl_binary_smaller_vector_operation!($data_type, $reg, fn multiply_smaller_vector_real, factor, mul, mul);
                impl_binary_complex_vector_operation!($data_type, $reg, fn divide_vector_complex, divisor, div_complex, div);
                impl_binary_smaller_complex_vector_operation!($data_type, $reg, fn divide_smaller_vector_complex, divisor, div_complex, div);
                impl_binary_vector_operation!($data_type, $reg, fn divide_vector_real, divisor, div, div);
                impl_binary_smaller_vector_operation!($data_type, $reg, fn divide_smaller_vector_real, divisor, div, div);

                fn zero_interleave_complex(mut self, factor: u32) -> VecResult<Self>
                {
                    zero_interleave!(self, $data_type, factor, 2)
                }

                fn zero_interleave_real(mut self, factor: u32) -> VecResult<Self>
                {
                    zero_interleave!(self, $data_type, factor, 1)
                }

                fn real_sqrt(self) -> VecResult<Self>
                {
                    self.simd_real_operation(|x,_arg|x.sqrt(), |x,_arg|x.sqrt(), (), Complexity::Small)
                }

                fn complex_sqrt(self) -> VecResult<Self>
                {
                    self.pure_complex_operation(|x,_arg|x.sqrt(), (), Complexity::Small)
                }

                fn real_square(self) -> VecResult<Self>
                {
                    self.simd_real_operation(|x,_arg|x * x, |x,_arg|x * x, (), Complexity::Small)
                }

                fn complex_square(self) -> VecResult<Self>
                {
                    self.pure_complex_operation(|x,_arg|x * x, (), Complexity::Small)
                }

                fn real_root(self, degree: $data_type) -> VecResult<Self>
                {
                    self.pure_real_operation(|x,y|x.powf(1.0 / y), degree, Complexity::Medium)
                }

                fn complex_root(self, base: $data_type) -> VecResult<Self>
                {
                    self.pure_complex_operation(|x,y|x.powf(1.0 / y), base, Complexity::Medium)
                }

                impl_function_call_real_arg_complex!($data_type; fn real_powf, powf; fn complex_powf, powf);
                impl_function_call_real_complex!($data_type; fn real_ln, ln; fn complex_ln, ln);
                impl_function_call_real_complex!($data_type; fn real_exp, exp; fn complex_exp, exp);
                impl_function_call_real_arg_complex!($data_type; fn real_log, log; fn complex_log, log);

                fn real_expf(self, base: $data_type) -> VecResult<Self>
                {
                    self.pure_real_operation(|x,y|y.powf(x), base, Complexity::Medium)
                }

                fn complex_expf(self, base: $data_type) -> VecResult<Self>
                {
                    self.pure_complex_operation(|x,y|x.expf(y), base, Complexity::Medium)
                }

                impl_function_call_real_complex!($data_type; fn real_sin, sin; fn complex_sin, sin);
                impl_function_call_real_complex!($data_type; fn real_cos, cos; fn complex_cos, cos);
                impl_function_call_real_complex!($data_type; fn real_tan, tan; fn complex_tan, tan);
                impl_function_call_real_complex!($data_type; fn real_asin, asin; fn complex_asin, asin);
                impl_function_call_real_complex!($data_type; fn real_acos, acos; fn complex_acos, acos);
                impl_function_call_real_complex!($data_type; fn real_atan, atan; fn complex_atan, atan);
                impl_function_call_real_complex!($data_type; fn real_sinh, sinh; fn complex_sinh, sinh);
                impl_function_call_real_complex!($data_type; fn real_cosh, cosh; fn complex_cosh, cosh);
                impl_function_call_real_complex!($data_type; fn real_tanh, tanh; fn complex_tanh, tanh);
                impl_function_call_real_complex!($data_type; fn real_asinh, asinh; fn complex_asinh, asinh);
                impl_function_call_real_complex!($data_type; fn real_acosh, acosh; fn complex_acosh, acosh);
                impl_function_call_real_complex!($data_type; fn real_atanh, atanh; fn complex_atanh, atanh);
            }
        )*
     }
}

add_general_impl!(f32, Reg32; f64, Reg64);
