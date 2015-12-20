macro_rules! define_complex_operations_forward {
    (from: $name:ident, to: $gen_type:ident, complex: $complex_type:ident, real_partner: $real_partner:ident, $($data_type:ident),*)
	 =>
	 { 
        $(
            #[inline]
            impl ComplexVectorOperations<$data_type> for $name<$data_type>
            {
                type RealPartner = $real_partner<$data_type>;
                
                fn complex_offset(self, offset: Complex<$data_type>) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().complex_offset(offset))
                }
                    
                fn complex_scale(self, factor: Complex<$data_type>) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().complex_scale(factor))
                }
                
                fn complex_abs(self) -> VecResult<Self::RealPartner>
                {
                    $real_partner::from_genres(self.to_gen().complex_abs())
                }
                
                fn get_complex_abs(&self, destination: &mut Self::RealPartner) -> VoidResult
                {
                    self.to_gen_borrow().get_complex_abs(destination.to_gen_mut_borrow())
                }
                
                fn complex_abs_squared(self) -> VecResult<Self::RealPartner>
                {
                    $real_partner::from_genres(self.to_gen().complex_abs_squared())
                }
                
                fn complex_conj(self) -> VecResult<Self>
                {
                    $name::from_genres(self.to_gen().complex_conj())
                }
                
                fn to_real(self) -> VecResult<Self::RealPartner>
                {
                    $real_partner::from_genres(self.to_gen().to_real())
                }
        
                fn to_imag(self) -> VecResult<Self::RealPartner>
                {
                    $real_partner::from_genres(self.to_gen().to_imag())
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
                    $real_partner::from_genres(self.to_gen().phase())
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
            
            fn from_genres(other: VecResult<$gen_type<T>>) -> VecResult<Self>
            {
                match other {
                    Ok(v) => Ok($name::from_gen(v)),
                    Err((r, v)) => Err((r, $name::from_gen(v)))
                }
            }
        } 
	 }
}