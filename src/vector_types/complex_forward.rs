macro_rules! define_complex_operations_forward {
    (from: $name:ident, to: $gen_type:ident, complex: $complex_type:ident, real_partner: $real_partner:ident, $($data_type:ident),*)
	 =>
	 { 
        $(
            impl ComplexVectorOperations<$data_type> for $name<$data_type>
            {
                type RealPartner = $real_partner<$data_type>;
                
                fn complex_offset(self, offset: Complex<$data_type>) -> VecResult<Self>
                {
                    Self::from_genres(self.to_gen().complex_offset(offset))
                }
                    
                fn complex_scale(self, factor: Complex<$data_type>) -> VecResult<Self>
                {
                    Self::from_genres(self.to_gen().complex_scale(factor))
                }
                
                fn multiply_complex_exponential(self, a: $data_type, b: $data_type) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().multiply_complex_exponential(a, b))
                }
                
                fn magnitude(self) -> VecResult<Self::RealPartner>
                {
                    Self::RealPartner::from_genres(self.to_gen().magnitude())
                }
                
                fn get_magnitude(&self, destination: &mut Self::RealPartner) -> VoidResult
                {
                    self.to_gen_borrow().get_magnitude(destination.to_gen_mut_borrow())
                }
                
                fn magnitude_squared(self) -> VecResult<Self::RealPartner>
                {
                    Self::RealPartner::from_genres(self.to_gen().magnitude_squared())
                }
                
                fn complex_conj(self) -> VecResult<Self>
                {
                    Self::from_genres(self.to_gen().complex_conj())
                }
                
                fn to_real(self) -> VecResult<Self::RealPartner>
                {
                    Self::RealPartner::from_genres(self.to_gen().to_real())
                }
        
                fn to_imag(self) -> VecResult<Self::RealPartner>
                {
                    Self::RealPartner::from_genres(self.to_gen().to_imag())
                }	
                        
                fn get_real(&self, destination: &mut Self::RealPartner) -> VoidResult
                {
                    self.to_gen_borrow().get_real(destination.to_gen_mut_borrow())
                }
                
                fn get_imag(&self, destination: &mut Self::RealPartner) -> VoidResult
                {
                    self.to_gen_borrow().get_imag(destination.to_gen_mut_borrow())
                }
                
                fn phase(self) -> VecResult<Self::RealPartner>
                {
                    Self::RealPartner::from_genres(self.to_gen().phase())
                }
                
                fn get_phase(&self, destination: &mut Self::RealPartner) -> VoidResult
                {
                    self.to_gen_borrow().get_phase(destination.to_gen_mut_borrow())
                }
                
                fn complex_dot_product(&self, factor: &Self) -> ScalarResult<Complex<$data_type>>
                {
                    self.to_gen_borrow().complex_dot_product(&factor.to_gen_borrow())
                }
                
                fn complex_statistics(&self) -> Statistics<Complex<$data_type>> {
                    self.to_gen_borrow().complex_statistics()
                }
                
                fn complex_statistics_splitted(&self, len: usize) -> Vec<Statistics<Complex<$data_type>>> {
                    self.to_gen_borrow().complex_statistics_splitted(len)
                }
                
                fn get_real_imag(&self, real: &mut Self::RealPartner, imag: &mut Self::RealPartner) -> VoidResult {
                    self.to_gen_borrow().get_real_imag(real.to_gen_mut_borrow(), imag.to_gen_mut_borrow())
                }
                
                fn get_mag_phase(&self, mag: &mut Self::RealPartner, phase: &mut Self::RealPartner) -> VoidResult {
                    self.to_gen_borrow().get_mag_phase(mag.to_gen_mut_borrow(), phase.to_gen_mut_borrow())
                }
                
                fn set_real_imag(self, real: &Self::RealPartner, imag: &Self::RealPartner) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().set_real_imag(real.to_gen_borrow(), imag.to_gen_borrow()))
                }
                
                fn set_mag_phase(self, mag: &Self::RealPartner, phase: &Self::RealPartner) -> VecResult<Self> {
                    Self::from_genres(self.to_gen().set_mag_phase(mag.to_gen_borrow(), phase.to_gen_borrow()))
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
                
                fn from_gen(other: $gen_type<$data_type>) -> Self
                {
                    unsafe { mem::transmute(other) }
                }
                
                fn from_genres(other: VecResult<$gen_type<$data_type>>) -> VecResult<Self>
                {
                    match other {
                        Ok(v) => Ok($name::<$data_type>::from_gen(v)),
                        Err((r, v)) => Err((r, $name::<$data_type>::from_gen(v)))
                    }
                }
            }
            
            impl Scale<Complex<$data_type>> for $name<$data_type> {
                fn scale(self, offset: Complex<$data_type>) -> VecResult<Self> {
                    self.complex_scale(offset)
                }
            }
            
            impl Offset<Complex<$data_type>> for $name<$data_type> {
                fn offset(self, offset: Complex<$data_type>) -> VecResult<Self> {
                    self.complex_offset(offset)
                }
            }
        )*
	 }
}