macro_rules! define_real_operations_forward {
    (from: $name:ident, to: $gen_type:ident, complex_partner: $complex_partner:ident, $($data_type:ident),*)
     =>
     {     
        $(
            impl RealVectorOperations<$data_type> for $name<$data_type>
            {
                type ComplexPartner = $complex_partner<$data_type>; 
                
                fn real_offset(self, offset: $data_type) -> VecResult<Self>
                {
                    Self::from_genres(self.to_gen().real_offset(offset))
                }
                
                fn real_scale(self, factor: $data_type) -> VecResult<Self>
                {
                    Self::from_genres(self.to_gen().real_scale(factor))
                }
                        
                fn abs(self) -> VecResult<Self>
                {
                    Self::from_genres(self.to_gen().abs()) 
                }
                            
                fn to_complex(self) -> VecResult<Self::ComplexPartner>
                {
                    Self::ComplexPartner::from_genres(self.to_gen().to_complex()) 
                }
                
                fn wrap(self, divisor: $data_type) -> VecResult<Self>
                {
                    Self::from_genres(self.to_gen().wrap(divisor))
                }
                
                fn unwrap(self, divisor: $data_type) -> VecResult<Self>
                {
                    Self::from_genres(self.to_gen().unwrap(divisor))
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
            
            impl $name<$data_type>
            {        
                fn to_gen(self) -> $gen_type<$data_type>
                {
                    unsafe { mem::transmute(self) }
                }
                
                fn to_gen_borrow(&self) -> &$gen_type<$data_type>
                {
                    unsafe { mem::transmute(self) }
                }
                
                #[allow(dead_code)]
                fn to_gen_mut_borrow(&mut self) -> &mut $gen_type<$data_type>
                {
                    unsafe { mem::transmute(self) }
                }
                
                fn from_gen(other: $gen_type<$data_type>) -> Self
                {
                    unsafe { mem::transmute(other) }
                }
                
                fn from_genres(other: VecResult<$gen_type<$data_type>>) -> VecResult<$name<$data_type>>
                {
                    match other {
                        Ok(v) => Ok($name::<$data_type>::from_gen(v)),
                        Err((r, v)) => Err((r, $name::<$data_type>::from_gen(v)))
                    }
                }
            }
            
            impl Scale<$data_type> for $name<$data_type> {
                fn scale(self, offset: $data_type) -> VecResult<Self> {
                    self.real_scale(offset)
                }
            }
            
            impl Offset<$data_type> for $name<$data_type> {
                fn offset(self, offset: $data_type) -> VecResult<Self> {
                    self.real_offset(offset)
                }
            }
            
            impl DotProduct<$data_type> for $name<$data_type> {
                fn dot_product(&self, factor: &Self) -> ScalarResult<$data_type> {
                    self.real_dot_product(factor)
                }
            }
            
            impl StatisticsOperations<$data_type> for $name<$data_type> {
                fn statistics(&self) -> Statistics<$data_type> {
                    self.real_statistics()
                }
                
                fn statistics_splitted(&self, len: usize) -> Vec<Statistics<$data_type>> {
                    self.real_statistics_splitted(len)
                }
            }
        )*
     }
}