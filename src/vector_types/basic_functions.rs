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
                        Chunk::execute_partial_with_arguments(
                            complexity, &self.multicore_settings,
                            &mut array, length, 1, argument,
                            move|array, argument| {
                                let mut i = 0;
                                while i < array.len()
                                {
                                    array[i] = op(array[i], argument);
                                    i += 1;
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
                        Chunk::execute_partial_with_arguments(
                            complexity, &self.multicore_settings,
                            &mut array, vectorization_length, $reg::len(), argument, 
                            move |array, argument| {
                                let mut i = 0;
                                while i < array.len()
                                { 
                                    let vector = $reg::load(array, i);
                                    let scaled = simd_op(vector, argument);
                                    scaled.store(array, i);
                                    i += $reg::len();
                                }
                        });
                        for i in vectorization_length..data_length
                        {
                            array[i] = scalar_op(array[i], argument);
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
                        Chunk::execute_partial_with_arguments(
                            complexity, &self.multicore_settings,
                            &mut array, length, 2, argument,
                            move|array, argument| {
                                let mut i = 0;
                                while i < array.len()
                                {
                                    let complex = Complex::<$data_type>::new(array[i], array[i + 1]);
                                    let result = op(complex, argument);
                                    array[i] = result.re;
                                    array[i + 1] = result.im;
                                    i += 2;
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
                        Chunk::execute_original_to_target_with_arguments(
                            complexity, &self.multicore_settings,
                            &mut array, data_length, 2, 
                            &mut temp, data_length / 2, 1, argument,
                            move |array, range, target, argument| {
                                let mut i = range.start;
                                let mut j = 0;
                                while j < target.len()
                                { 
                                    let vector = Complex::<$data_type>::new(array[i], array[i + 1]);
                                    let result = op(vector, argument);
                                    target[j] = result;
                                    i += 2;
                                    j += 1;
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
                        Chunk::execute_partial_with_arguments(
                            complexity, &self.multicore_settings,
                            &mut array, vectorization_length, $reg::len(), argument, 
                            move |array, argument| {
                                let mut i = 0;
                                while i < array.len()
                                { 
                                    let vector = $reg::load(array, i);
                                    let scaled = simd_op(vector, argument);
                                    scaled.store(array, i);
                                    i += $reg::len();
                                }
                        });
                        let mut i = vectorization_length;
                        while i < data_length
                        {
                            let complex = Complex::<$data_type>::new(array[i], array[i + 1]);
                            let result = scalar_op(complex, argument);
                            array[i] = result.re;
                            array[i + 1] = result.im;
                            i += 2;
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
                        Chunk::execute_original_to_target_with_arguments(
                            complexity, &self.multicore_settings,
                            &mut array, vectorization_length, $reg::len(), 
                            &mut temp, vectorization_length / 2, $reg::len() / 2, argument,
                            move |array, range, target, argument| {
                                let mut i = range.start;
                                let mut j = 0;
                                while j < target.len()
                                { 
                                    let vector = $reg::load(array, i);
                                    let result = simd_op(vector, argument);
                                    result.store_half(target, j);
                                    i += $reg::len();
                                    j += $reg::len() / 2;
                                }
                        });
                        let mut i = vectorization_length;
                        while i < data_length
                        {
                            let complex = Complex::<$data_type>::new(array[i], array[i + 1]);
                            let result = scalar_op(complex, argument);
                            temp[i / 2] = result;
                            i += 2;
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