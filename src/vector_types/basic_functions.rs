macro_rules! add_basic_private_impl {
    ($($data_type:ident, $reg:ident);*)
    => {
        $(
            impl GenericDataVector<$data_type> {
                #[inline]
                fn pure_real_operation<A, F>(mut self, op: F, argument: A, complexity: Complexity) -> VecResult<Self> 
                    where A: Sync + Copy + Send,
                        F: Fn($data_type, A) -> $data_type + 'static + Sync {
                    {
                        let mut array = &mut self.data;
                        Chunk::execute_partial(
                            complexity, &self.multicore_settings,
                            &mut array, 1, argument,
                            move|array, argument| {
                                for num in array {
                                    *num = op(*num, argument);
                                }
                        });
                    }
                    Ok(self)
                }
                
                #[inline]
                fn simd_real_operation<A, F, G>(mut self, simd_op: F, scalar_op: G, argument: A, complexity: Complexity) -> VecResult<Self> 
                    where A: Sync + Copy + Send,
                            F: Fn($reg, A) -> $reg + 'static + Sync,
                            G: Fn($data_type, A) -> $data_type + 'static + Sync {
                    {
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        let mut array = &mut self.data;
                        Chunk::execute_partial(
                            complexity, &self.multicore_settings,
                            &mut array[0..vectorization_length], $reg::len(), argument, 
                            move |array, argument| {
                                let array = $reg::array_to_regs_mut(array);
                                for reg in array {
                                    *reg = simd_op(*reg, argument);
                                }
                        });
                        for num in &mut array[vectorization_length..data_length]
                        {
                            *num = scalar_op(*num, argument);
                        }
                    }
                    Ok(self)
                }
                
                #[inline]
                fn pure_complex_operation<A, F>(mut self, op: F, argument: A, complexity: Complexity) -> VecResult<Self> 
                    where A: Sync + Copy + Send,
                        F: Fn(Complex<$data_type>, A) -> Complex<$data_type> + 'static + Sync {
                    {
                        let mut array = &mut self.data;
                        Chunk::execute_partial(
                            complexity, &self.multicore_settings,
                            array, 2, argument,
                            move|array, argument| {
                                let array = Self::array_to_complex_mut(array);
                                for num in array {
                                    *num = op(*num, argument);
                                }
                        });
                    }
                    Ok(self)
                }
                
                #[inline]
                fn pure_complex_to_real_operation<A, F>(mut self, op: F, argument: A, complexity: Complexity) -> VecResult<Self> 
                    where A: Sync + Copy + Send,
                        F: Fn(Complex<$data_type>, A) -> $data_type + 'static + Sync {
                    {
                        let data_length = self.len();       
                        let mut array = &mut self.data;
                        let mut temp = temp_mut!(self, data_length / 2);
                        Chunk::from_src_to_dest(
                            complexity, &self.multicore_settings,
                            &mut array[0..data_length], 2, 
                            &mut temp[0..data_length / 2], 1, argument,
                            move |array, range, target, argument| {
                                let array = Self::array_to_complex(&array[range.start..range.end]);
                                for pair in array.iter().zip(target) {
                                    let (src, dest) = pair;
                                    *dest = op(*src, argument);
                                }
                        });
                        self.is_complex = false;
                        self.valid_len = data_length / 2;
                    }
                    Ok(self.swap_data_temp())
                }
                
                #[inline]
                fn simd_complex_operation<A, F, G>(mut self, simd_op: F, scalar_op: G, argument: A, complexity: Complexity) -> VecResult<Self> 
                    where A: Sync + Copy + Send,
                            F: Fn($reg, A) -> $reg + 'static + Sync,
                            G: Fn(Complex<$data_type>, A) -> Complex<$data_type> + 'static + Sync {
                    {
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;           
                        let mut array = &mut self.data;
                        Chunk::execute_partial(
                            complexity, &self.multicore_settings,
                            &mut array[0..vectorization_length], $reg::len(), argument, 
                            move |array, argument| {
                                let array = $reg::array_to_regs_mut(array);
                                for reg in array {
                                    *reg = simd_op(*reg, argument);
                                }
                        });
                        let array = Self::array_to_complex_mut(&mut array[vectorization_length..data_length]);
                        for num in array {
                            *num = scalar_op(*num, argument);
                        }
                    }
                    Ok(self)
                }
                
                #[inline]
                fn simd_complex_to_real_operation<A, F, G>(mut self, simd_op: F, scalar_op: G, argument: A, complexity: Complexity) -> VecResult<Self> 
                    where A: Sync + Copy + Send,
                          F: Fn($reg, A) -> $reg + 'static + Sync,
                          G: Fn(Complex<$data_type>, A) -> $data_type + 'static + Sync {
                    {
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;           
                        let mut array = &mut self.data;
                        let mut temp = temp_mut!(self, data_length);
                        Chunk::from_src_to_dest(
                            complexity, &self.multicore_settings,
                            &mut array[0..vectorization_length], $reg::len(), 
                            &mut temp[0..vectorization_length], $reg::len() / 2, argument,
                            move |array, range, target, argument| {
                                let array = $reg::array_to_regs(&array[range.start..range.end]);
                                let mut j = 0;
                                for reg in array {
                                    let result = simd_op(*reg, argument);
                                    result.store_half(target, j);
                                    j += $reg::len() / 2;
                                }
                        });
                        let array = Self::array_to_complex(&array[vectorization_length..data_length]);
                        for pair in array.iter().zip(&mut temp[vectorization_length/2..data_length/2]) {
                            let (src, dest) = pair;
                            *dest = scalar_op(*src, argument);
                        }
                        self.is_complex = false;
                        self.valid_len = data_length / 2;
                    }
                    Ok(self.swap_data_temp())
                }
                
                #[inline]
                fn swap_halves_priv(mut self, forward: bool) -> VecResult<Self>
                {
                   {
                        let data_length = self.len();
                        let points = self.points();
                        let complex = self.is_complex;
                        let elems_per_point = if complex { 2 } else { 1 };  
                        let mut temp = temp_mut!(self, data_length);  
                         
                        // First half
                        let len = 
                            if forward {
                                points / 2 * elems_per_point 
                            }
                            else {
                                data_length - points / 2 * elems_per_point 
                            };
                        let start = data_length - len; 
                        let data = &self.data[start] as *const $data_type;
                        let target = &mut temp[0] as *mut $data_type;
                        unsafe {
                            ptr::copy(data, target, len);
                        }
                        
                        // Second half
                        let data = &self.data[0] as *const $data_type;
                        let start = len; 
                        let len = data_length - len;
                        let target = &mut temp[start] as *mut $data_type;
                        unsafe {
                            ptr::copy(data, target, len);
                        }
                    }
                    
                    Ok(self.swap_data_temp())
                }
                
                fn multiply_function_priv<T,CMut,FA, F>(
                    mut self, 
                    is_symmetric: bool,
                    ratio: $data_type,
                    convert_mut: CMut,
                    function_arg: FA, 
                    fun: F) -> Self
                        where 
                            CMut: Fn(&mut [$data_type]) -> &mut [T],
                            FA: Copy + Sync + Send,
                            F: Fn(FA, $data_type)->T + 'static + Sync,
                            T: Zero + Mul<Output=T> + Copy + Display + Debug + Send + Sync + From<$data_type>
                {
                    if !is_symmetric {
                        {
                            let len = self.len();
                            let points = self.points();
                            let converted = convert_mut(&mut self.data[0..len]);
                            Chunk::execute_with_range(
                                Complexity::Medium, &self.multicore_settings,
                                converted, 1, function_arg,
                                move |array, range, arg| {
                                    let scale = T::from(ratio);
                                    let offset = if points % 2 != 0 { 1 } else { 0 };
                                    let max = (points - offset) as $data_type / 2.0; 
                                    let mut j = -((points - offset) as $data_type) / 2.0 + (range.start as $data_type);
                                    for num in array {
                                        *num = (*num) * scale * fun(arg, j / max * ratio);
                                        j += 1.0;
                                    }
                                });
                        }
                        self
                    } else {
                        {
                            let len = self.len();
                            let points = self.points();
                            let converted = convert_mut(&mut self.data[0..len]);
                            Chunk::execute_sym_pairs_with_range(
                                Complexity::Medium, &self.multicore_settings,
                                converted, 1, function_arg,
                                move |array1, range1, array2, range2, arg| {
                                    assert!(array1.len() >= array2.len());
                                    assert!(range1.end <= range2.start);let scale = T::from(ratio);
                                    let len1 = array1.len();
                                    let len2 = array2.len();
                                    let offset = if points % 2 != 0 { 1 } else { 0 };
                                    let max = (points - offset) as $data_type / 2.0; 
                                    let mut j1 = -((points - offset) as $data_type) / 2.0 + range1.start as $data_type;
                                    let mut j2 = ((points - offset) as $data_type) / 2.0 - (range2.end - 1) as $data_type; 
                                    let mut i1 = 0;
                                    let mut i2 = 0;
                                    {
                                        let mut iter1 = array1.iter_mut();
                                        let mut iter2 = array2.iter_mut().rev();
                                        while j1 < j2 {
                                            let num = iter1.next().unwrap();
                                            (*num) = (*num) * scale * fun(arg, j1 / max * ratio);
                                            j1 += 1.0;
                                            i1 += 1;
                                        }
                                        while j2 < j1 {
                                            let num = iter2.next().unwrap();
                                            (*num) = (*num) * scale * fun(arg, j2 / max * ratio);
                                            j2 += 1.0;
                                            i2 += 1;
                                        }
                                        // At this point we can be sure that `j1 == j2`
                                        for (num1, num2) in iter1.zip(iter2) {
                                            let arg = scale * fun(arg, j1 / max * ratio);
                                            *num1 = (*num1) * arg;
                                            *num2 = (*num2) * arg;
                                            j1 += 1.0;
                                        }
                                        j2 = j1;
                                    }
                                    
                                    // Now we have to deal with differences in length
                                    // `common_length` is the number of iterations we spent
                                    // in the previous loop.
                                    let pos1 = len1 - i1;
                                    let pos2 = len2 - i2;
                                    let common_length = if pos1 < pos2 { pos1 } else { pos2 };
                                    for num in &mut array1[i1 + common_length..len1] {
                                        (*num) = (*num) * scale * fun(arg, j1 / max * ratio);
                                        j1 += 1.0;
                                    }
                                    for num in &mut array2[0..len2-common_length-i2] {
                                        (*num) = (*num) * scale * fun(arg, j2 / max * ratio);
                                        j2 += 1.0;
                                    }
                                });
                        }
                        self
                    }
                }
                
                fn multiply_window_priv<T,CMut,FA, F>(
                    mut self, 
                    is_symmetric: bool,
                    convert_mut: CMut,
                    function_arg: FA, 
                    fun: F) -> Self
                        where 
                            CMut: Fn(&mut [$data_type]) -> &mut [T],
                            FA: Copy + Sync + Send,
                            F: Fn(FA, usize, usize)->T + 'static + Sync,
                            T: Zero + Mul<Output=T> + Copy + Display + Debug + Send + Sync + From<$data_type>
                {
                    if !is_symmetric {
                        {
                            let len = self.len();
                            let points = self.points();
                            let converted = convert_mut(&mut self.data[0..len]);
                            Chunk::execute_with_range(
                                Complexity::Medium, &self.multicore_settings,
                                converted, 1, function_arg,
                                move |array, range, arg| {
                                    let mut j = range.start;
                                    for num in array {
                                        *num = (*num) * fun(arg, j, points);
                                        j += 1;
                                    }
                                });
                        }
                        self
                    } else {
                        {
                            let len = self.len();
                            let points = self.points();
                            let converted = convert_mut(&mut self.data[0..len]);
                            Chunk::execute_sym_pairs_with_range(
                                Complexity::Medium, &self.multicore_settings,
                                converted, 1, function_arg,
                                move |array1, range, array2, _, arg| {
                                    assert!(array1.len() >= array2.len());
                                    let mut j = range.start;
                                    let len1 = array1.len();
                                    let len_diff = len1 - array2.len();
                                    {
                                        let iter1 = array1.iter_mut();
                                        let iter2 = array2.iter_mut().rev();
                                        for (num1, num2) in iter1.zip(iter2) {
                                            let arg = fun(arg, j, points);
                                            *num1 = (*num1) * arg;
                                            *num2 = (*num2) * arg;
                                            j += 1;
                                        }
                                    }
                                    for num1 in &mut array1[len1-len_diff..len1] {
                                        let arg = fun(arg, j, points);
                                        *num1 = (*num1) * arg;
                                        j += 1;
                                    }
                                });
                        }
                        self
                    }
                }
                
                /// Reallocates the data inside a vector, but 
                /// not temp. The data will not be preserved by this operation
                fn reallocate(&mut self, len: usize)
                {
                    if len > self.allocated_len()
                    {
                        let mut new_data = Vec::with_capacity(round_len(len));
                        unsafe { new_data.set_len(len) };
                        self.data = new_data;
                    }
                    
                    self.valid_len = len;
                }
                
                /// Creates shifted and reversed copies of the given data vector. 
                /// This function is especially designed for convolutions.
                fn create_shifted_copies(vector: &GenericDataVector<$data_type>) -> Vec<Vec<$data_type>>{
                    let step = if vector.is_complex { 2 } else { 1 };
                    let number_of_shifts = $reg::len() / step;
                    let mut shifted_copies = Vec::with_capacity(number_of_shifts);
                    let mut i = 0;
                    while i < number_of_shifts {
                        let mut data = vector.data.iter().rev();
                        
                        // In general (number_of_shifts - i) indicates which prepared vector we need to use
                        // if we later calculate end % number_of_shifts. Some examples:
                        // number_of_shifts: 4, end: 13 -> mod: 1. The code will round end to the next SIMD register
                        // which ends at 16. In order to get back to 13 we therefore have to ignore 3 numbers.
                        // Ignoring is done by shifting and inserting zeros. So in this example the correct shift is 3
                        // which equals number_of_shifts(4) - mod(1).
                        // Now mod: 0 is a special case. This is because if we round up to the next SIMD register then
                        // we still don't need to add any offset and so for the case 0, 0 is the right shift.
                        let shift = match i {
                            0 => 0,
                            x => (number_of_shifts - x) * step
                        };
                        let min_len = vector.len() + shift;
                        let len =  (min_len + $reg::len() - 1) / $reg::len() * $reg::len();
                        let mut copy = Vec::with_capacity(len);
                        
                        let mut j = len;
                        while j > 0 {
                            j -= step;
                            if j < shift || j >= min_len {
                                copy.push(0.0);
                                if step > 1 {
                                    copy.push(0.0);
                                }
                            } else {
                                if step > 1 {
                                    let im = *data.next().unwrap();
                                    let re = *data.next().unwrap();
                                    copy.push(re);
                                    copy.push(im);
                                }
                                else {
                                    copy.push(*data.next().unwrap());
                                }
                            }
                        }
                        
                        assert_eq!(copy.len(), len);
                        shifted_copies.push(copy);
                        i += 1;
                    }
                    shifted_copies
                }
            }
        )*
    }
}