//! Fallback implementation in case no explicit `sse` or `avx` support is available.
//! In this case we use the standard versions of all functions which and no approximations.
//! We do that since this way the API doesn't change (so much) depending on the selected features.
//! However at the same time finding an alternative approximation implementation without
//! explicit SIMD support would be too much effort and therefore we use the standard functions.
//! This way we can also be sure that we achieve the promised unertainty :). 
//! The Rust compiler does a good job to remove the otherhead of this implementation.
//! So in benchmarks this implementation has the same speed as the the standard functions.

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
