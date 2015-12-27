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
                        let mut array = &mut self.data;
                        let length = array.len();
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
            }
        )*
    }
}