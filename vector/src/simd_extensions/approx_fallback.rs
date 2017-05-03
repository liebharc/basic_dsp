use super::{SimdApproximations, SimdGeneric, Reg32, Reg64};

macro_rules! simd_approx_impl {
    ($data_type:ident,
	 $regf:ident)
    =>
    {
		impl SimdApproximations<$data_type> for $regf {
		    fn ln_approx(self) -> Self {
				self.iter_over_vector(|x: $data_type| x.ln())
			}

		    fn exp_approx(self) -> Self {
				self.iter_over_vector(|x: $data_type| x.exp())
			}

		    fn sin_approx(self) -> Self {
				self.iter_over_vector(|x: $data_type| x.sin())
			}

		    fn cos_approx(self) -> Self {
				self.iter_over_vector(|x: $data_type| x.cos())
			}

		    fn sin_cos_approx(self, is_sin: bool) -> Self {
				if is_sin {
					self.sin_approx()
				}
				else {
					self.cos_approx()
				}
			}
		}
	}
}

simd_approx_impl!(f32, Reg32);
simd_approx_impl!(f64, Reg64);
