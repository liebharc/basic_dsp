macro_rules! add_basic_private_impl {
    ($($data_type:ident, $reg:ident);*)
    => {
        $(
            impl GenericDataVector<$data_type> {
                #[inline]
                fn pure_real_operation<A, F>(mut self, op: F, argument: A, complexity: Complexity) -> VecResult<Self> 
                    where A: Sync + Copy,
                        F: Fn($data_type, A) -> $data_type + 'static + Sync {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial(
                            complexity, &self.multicore_settings,
                            &mut array, length, 1, argument,
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
                    where A: Sync + Copy,
                            F: Fn($reg, A) -> $reg + 'static + Sync,
                            G: Fn($data_type, A) -> $data_type + 'static + Sync {
                    {
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;           
                        let mut array = &mut self.data;
                        Chunk::execute_partial(
                            complexity, &self.multicore_settings,
                            &mut array, vectorization_length, $reg::len(), argument, 
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
                    where A: Sync + Copy,
                        F: Fn(Complex<$data_type>, A) -> Complex<$data_type> + 'static + Sync {
                    {
                        let length = self.len();
                        let mut array = &mut self.data;
                        Chunk::execute_partial(
                            complexity, &self.multicore_settings,
                            &mut array, length, 2, argument,
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
                    where A: Sync + Copy,
                        F: Fn(Complex<$data_type>, A) -> $data_type + 'static + Sync {
                    {
                        let data_length = self.len();       
                        let mut array = &mut self.data;
                        let mut temp = temp_mut!(self, data_length);
                        Chunk::from_src_to_dest(
                            complexity, &self.multicore_settings,
                            &mut array, data_length, 2, 
                            &mut temp, data_length / 2, 1, argument,
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
                    where A: Sync + Copy,
                            F: Fn($reg, A) -> $reg + 'static + Sync,
                            G: Fn(Complex<$data_type>, A) -> Complex<$data_type> + 'static + Sync {
                    {
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;           
                        let mut array = &mut self.data;
                        Chunk::execute_partial(
                            complexity, &self.multicore_settings,
                            &mut array, vectorization_length, $reg::len(), argument, 
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
                    where A: Sync + Copy,
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
                            &mut array, vectorization_length, $reg::len(), 
                            &mut temp, vectorization_length / 2, $reg::len() / 2, argument,
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
                            FA: Copy + Sync,
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
                                converted, points, 1, function_arg,
                                move |array, range, arg| {
                                    let scale = T::from(ratio);
                                    let offset = if points % 2 != 0 { 1 } else { 0 };
                                    let max = (points - offset) as $data_type / 2.0; 
                                    let mut j = -((points - offset + range.start) as $data_type) / 2.0;
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
                                converted, points, 1, function_arg,
                                move |array1, range, array2, _, arg| {
                                    let len1 = array1.len();
                                    let len2 = array2.len();
                                    let scale = T::from(ratio);
                                    let (offset, skip) = if points % 2 != 0 { (1, 0) } else { (0, 1) };
                                    let max = (points - offset) as $data_type / 2.0; 
                                    let mut j = -((points - offset + range.start) as $data_type) / 2.0 + skip as $data_type;
                                    let mut i = 0;
                                    {
                                    
                                        let iter1 = array1.iter_mut().skip(skip);
                                        let iter2 = array2.iter_mut().rev();
                                        for (num1, num2) in iter1.zip(iter2) {
                                            let arg = scale * fun(arg, j / max * ratio);
                                            *num1 = (*num1) * arg;
                                            *num2 = (*num2) * arg;
                                            j += 1.0;
                                            i += 1;
                                        }
                                    }
                                    // Now we have to look into all the edge cases
                                    // All for loops are expected to a max run for one iteration
                                    for num2 in &mut array2[0..len2-i] {
                                        let arg = scale * fun(arg, j / max * ratio);
                                        (*num2) = (*num2) * arg;
                                        j += 1.0;
                                    }
                                    if skip != 0 {
                                        j = -((points - offset + range.start) as $data_type) / 2.0 as $data_type;
                                        for num1 in &mut array1[0..len1-i] {
                                            let arg = scale * fun(arg, j / max * ratio);
                                            *num1 = (*num1) * arg;
                                            j += 1.0;
                                        }
                                    } else {
                                        for num1 in &mut array1[i .. len1] {
                                            {
                                                let arg = scale * fun(arg, j / max * ratio);
                                                *num1 = (*num1) * arg;
                                                j += 1.0;
                                            }
                                        }
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
                            FA: Copy + Sync,
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
                                converted, points, 1, function_arg,
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
                                converted, points, 1, function_arg,
                                move |array1, range, array2, _, arg| {
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
            }
        )*
    }
}