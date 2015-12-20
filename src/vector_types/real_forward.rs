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
                
                fn real_statistics_splitted(&self, len: usize) -> Vec<Statistics<$data_type>> {
                    self.to_gen_borrow().real_statistics_splitted(len)
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
                multicore_settings: MultiCoreSettings::default()
                }
            }
            
            fn to_gen_borrow(&self) -> &$gen_type<T>
            {
                unsafe { mem::transmute(self) }
            }
            
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
                multicore_settings: MultiCoreSettings::default()
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