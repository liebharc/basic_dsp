macro_rules! define_real_operations_forward {
    (from: $name:ident, to: $gen_type:ident, complex_partner: $complex_partner:ident, $($data_type:ident),*)
     =>
     {     
        $(
            impl RealVectorOps<$data_type> for $name<$data_type>
            {
                type ComplexPartner = $complex_partner<$data_type>; 
                
                fn real_offset(self, offset: $data_type) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().real_offset(offset))
                }
                
                fn real_scale(self, factor: $data_type) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().real_scale(factor))
                }
                        
                fn abs(self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().abs()) 
                }
                            
                fn to_complex(self) -> TransRes<Self::ComplexPartner>
                {
                    Self::ComplexPartner::from_genres(self.to_gen().to_complex()) 
                }
                
                fn wrap(self, divisor: $data_type) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().wrap(divisor))
                }
                
                fn unwrap(self, divisor: $data_type) -> TransRes<Self>
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
                
                fn real_sum(&self) -> $data_type {
                    self.to_gen_borrow().real_sum()
                }
                
                fn real_sum_sq(&self) -> $data_type {
                    self.to_gen_borrow().real_sum_sq()
                }
                
                fn map_inplace_real<A, F>(self, argument: A, f: F) -> TransRes<Self>
                    where A: Sync + Copy + Send,
                          F: Fn($data_type, usize, A) -> $data_type + 'static + Sync {
                    Self::from_genres(self.to_gen().map_inplace_real(argument, f))
                }
                
                fn map_aggregate_real<A, FMap, FAggr, R>(
                    &self, 
                    argument: A, 
                    map: FMap,
                    aggregate: FAggr) -> ScalarResult<R>
                        where A: Sync + Copy + Send,
                              FMap: Fn($data_type, usize, A) -> R + 'static + Sync,
                              FAggr: Fn(R, R) -> R + 'static + Sync + Send,
                              R: Send {
                    self.to_gen_borrow().map_aggregate_real(argument, map, aggregate) 
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
                
                fn from_genres(other: TransRes<$gen_type<$data_type>>) -> TransRes<$name<$data_type>>
                {
                    match other {
                        Ok(v) => Ok($name::<$data_type>::from_gen(v)),
                        Err((r, v)) => Err((r, $name::<$data_type>::from_gen(v)))
                    }
                }
            }
            
            impl ScaleOps<$data_type> for $name<$data_type> {
                fn scale(self, offset: $data_type) -> TransRes<Self> {
                    self.real_scale(offset)
                }
            }
            
            impl OffsetOps<$data_type> for $name<$data_type> {
                fn offset(self, offset: $data_type) -> TransRes<Self> {
                    self.real_offset(offset)
                }
            }
            
            impl DotProductOps<$data_type> for $name<$data_type> {
                fn dot_product(&self, factor: &Self) -> ScalarResult<$data_type> {
                    self.real_dot_product(factor)
                }
            }
            
            impl StatisticsOps<$data_type> for $name<$data_type> {
                fn statistics(&self) -> Statistics<$data_type> {
                    self.real_statistics()
                }
                
                fn statistics_splitted(&self, len: usize) -> Vec<Statistics<$data_type>> {
                    self.real_statistics_splitted(len)
                }
                
                fn sum(&self) -> $data_type {
                    self.real_sum()
                }
                
                fn sum_sq(&self) -> $data_type {
                    self.real_sum_sq()
                }
            }
            
            impl VectorIter<$data_type> for $name<$data_type> {
                fn map_inplace<A, F>(self, argument: A, map: F) -> TransRes<Self>
                    where A: Sync + Copy + Send,
                          F: Fn($data_type, usize, A) -> $data_type + 'static + Sync {
                    self.map_inplace_real(argument, map)
                }
                
                fn map_aggregate<A, FMap, FAggr, R>(
                    &self, 
                    argument: A, 
                    map: FMap,
                    aggregate: FAggr) -> ScalarResult<R>
                where A: Sync + Copy + Send,
                      FMap: Fn($data_type, usize, A) -> R + 'static + Sync,
                      FAggr: Fn(R, R) -> R + 'static + Sync + Send,
                      R: Send {
                    self.map_aggregate_real(argument, map, aggregate)  
                }
            }
        )*
     }
}