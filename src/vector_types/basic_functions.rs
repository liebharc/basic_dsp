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
                    ratio: $data_type,
                    convert_mut: CMut,
                    function_arg: FA, 
                    fun: F) -> Self
                        where 
                            CMut: Fn(&mut [$data_type]) -> &mut [T],
                            FA: Copy + Sync,
                            F: Fn(FA, $data_type)->T + 'static + Sync,
                            T: Zero + Mul<Output=T> + Copy + Display + Send + Sync + From<$data_type>
                {
                    {
                        let len = self.len();
                        let points = self.points();
                        let complex = convert_mut(&mut self.data[0..len]);
                        Chunk::execute_with_range(
                            Complexity::Medium, &self.multicore_settings,
                            complex, points, 1, function_arg,
                            move |array, range, arg| {
                                let scale = T::from(ratio);
                                let offset = if points % 2 != 0 { 1 } else { 0 };
                                let max = (points - offset) as $data_type / 2.0; 
                                let mut j = -((points - offset + range.start) as $data_type) / 2.0;
                                for num in array {
                                    (*num) = (*num) * scale * fun(arg, j / max * ratio);
                                    j += 1.0;
                                }
                            });
                    }
                    self
                }
            }
        )*
    }
}