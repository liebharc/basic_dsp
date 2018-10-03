use super::*;
use basic_dsp_vector::numbers::*;
use basic_dsp_vector::*;

macro_rules! add_mat_impl {
    ($($matrix:ident);*) => {
        $(
            impl<V: Vector<T> + ScaleOps<T>, S: ToSlice<T>, T: RealNumber>
					ScaleOps<T> for $matrix<V, S, T> {
				fn scale(&mut self, factor: T) {
					for v in self.rows_mut() {
						v.scale(factor);
					}
				}
			}

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber> OffsetOps<T>
                    for $matrix<V, S, T>
                    where V: OffsetOps<T> {
				fn offset(&mut self, offset: T) {
					for v in self.rows_mut() {
						v.offset(offset);
					}
				}
			}

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber> ScaleOps<Complex<T>>
                    for $matrix<V, S, T>
                    where V: ScaleOps<Complex<T>> {
				fn scale(&mut self, factor: Complex<T>) {
					for v in self.rows_mut() {
						v.scale(factor);
					}
				}
			}

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber> OffsetOps<Complex<T>>
                    for $matrix<V, S, T>
                    where V: OffsetOps<Complex<T>> {
				fn offset(&mut self, offset: Complex<T>) {
					for v in self.rows_mut() {
						v.offset(offset);
					}
				}
			}

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber, N: NumberSpace, D: Domain> ElementaryOps<$matrix<V, S, T>, T, N, D>
                    for $matrix<V, S, T>
                    where V: ElementaryOps<V, T, N, D> + GetMetaData<T, N, D> {
				fn add(&mut self, summand: &Self) -> VoidResult {
					for (v, o) in self.rows_mut().iter_mut().zip(summand.rows()) {
						try!(v.add(o));
					}

					Ok(())
				}

				fn sub(&mut self, summand: &Self) -> VoidResult {
					for (v, o) in self.rows_mut().iter_mut().zip(summand.rows()) {
						try!(v.sub(o));
					}

					Ok(())
				}

				fn div(&mut self, summand: &Self) -> VoidResult {
					for (v, o) in self.rows_mut().iter_mut().zip(summand.rows()) {
						try!(v.div(o));
					}

					Ok(())
				}

				fn mul(&mut self, summand: &Self) -> VoidResult {
					for (v, o) in self.rows_mut().iter_mut().zip(summand.rows()) {
						try!(v.mul(o));
					}

					Ok(())
				}
			}

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber, N: NumberSpace, D: Domain> ElementaryOps<V, T, N, D>
                    for $matrix<V, S, T>
                    where V: ElementaryOps<V, T, N, D> + GetMetaData<T, N, D> {
				fn add(&mut self, summand: &V) -> VoidResult {
                    for v in self.rows_mut() {
						try!(v.add(summand));
					}

					Ok(())
				}

				fn sub(&mut self, summand: &V) -> VoidResult {
                    for v in self.rows_mut() {
						try!(v.sub(summand));
					}

					Ok(())
				}

				fn div(&mut self, summand: &V) -> VoidResult {
                    for v in self.rows_mut() {
						try!(v.div(summand));
					}

					Ok(())
				}

				fn mul(&mut self, summand: &V) -> VoidResult {
                    for v in self.rows_mut() {
						try!(v.mul(summand));
					}

					Ok(())
				}
			}

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber, N: NumberSpace, D: Domain>
                    ElementaryWrapAroundOps<$matrix<V, S, T>, T, N, D>
                    for $matrix<V, S, T>
                    where V: ElementaryWrapAroundOps<V, T, N, D> + GetMetaData<T, N, D> {
				fn add_smaller(&mut self, summand: &Self) -> VoidResult {
					for (v, o) in self.rows_mut().iter_mut().zip(summand.rows()) {
						try!(v.add_smaller(o));
					}

					Ok(())
				}

				fn sub_smaller(&mut self, summand: &Self) -> VoidResult {
					for (v, o) in self.rows_mut().iter_mut().zip(summand.rows()) {
						try!(v.sub_smaller(o));
					}

					Ok(())
				}

				fn div_smaller(&mut self, summand: &Self) -> VoidResult {
					for (v, o) in self.rows_mut().iter_mut().zip(summand.rows()) {
						try!(v.div_smaller(o));
					}

					Ok(())
				}

				fn mul_smaller(&mut self, summand: &Self) -> VoidResult {
					for (v, o) in self.rows_mut().iter_mut().zip(summand.rows()) {
						try!(v.mul_smaller(o));
					}

					Ok(())
				}
			}

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber, N: NumberSpace, D: Domain> ElementaryWrapAroundOps<V, T, N, D>
                    for $matrix<V, S, T>
                    where V: ElementaryWrapAroundOps<V, T, N, D> + GetMetaData<T, N, D> {
				fn add_smaller(&mut self, summand: &V) -> VoidResult {
                    for v in self.rows_mut() {
						try!(v.add_smaller(summand));
					}

					Ok(())
				}

				fn sub_smaller(&mut self, summand: &V) -> VoidResult {
                    for v in self.rows_mut() {
						try!(v.sub_smaller(summand));
					}

					Ok(())
				}

				fn div_smaller(&mut self, summand: &V) -> VoidResult {
                    for v in self.rows_mut() {
						try!(v.div_smaller(summand));
					}

					Ok(())
				}

				fn mul_smaller(&mut self, summand: &V) -> VoidResult {
                    for v in self.rows_mut() {
						try!(v.mul_smaller(summand));
					}

					Ok(())
				}
			}

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber> ReorganizeDataOps<T>
                    for $matrix<V, S, T>
                    where V: ReorganizeDataOps<T> {
				fn reverse(&mut self)  {
					for v in self.rows_mut() {
						v.reverse()
					}
				}

				fn swap_halves(&mut self)  {
					for v in self.rows_mut() {
						v.swap_halves()
					}
				}
			}

			impl<S: ToSlice<T>, V: Vector<T> + DiffSumOps, T: RealNumber>
					DiffSumOps for $matrix<V, S, T> {
				fn diff(&mut self)  {
					for v in self.rows_mut() {
						v.diff()
					}
				}

				fn diff_with_start(&mut self)  {
					for v in self.rows_mut() {
						v.diff()
					}
				}

				fn cum_sum(&mut self)  {
					for v in self.rows_mut() {
						v.diff()
					}
				}
			}

			impl<S: ToSlice<T>, V: Vector<T> + TrigOps, T: RealNumber>
					TrigOps for $matrix<V, S, T> {
				fn sin(&mut self)  {
					for v in self.rows_mut() {
						v.sin()
					}
				}

				fn cos(&mut self)  {
					for v in self.rows_mut() {
						v.cos()
					}
				}

				fn tan(&mut self)  {
					for v in self.rows_mut() {
						v.tan()
					}
				}

				fn asin(&mut self)  {
					for v in self.rows_mut() {
						v.asin()
					}
				}

				fn acos(&mut self)  {
					for v in self.rows_mut() {
						v.acos()
					}
				}

				fn atan(&mut self)  {
					for v in self.rows_mut() {
						v.atan()
					}
				}

				fn sinh(&mut self)  {
					for v in self.rows_mut() {
						v.sinh()
					}
				}

				fn cosh(&mut self)  {
					for v in self.rows_mut() {
						v.cosh()
					}
				}

				fn tanh(&mut self)  {
					for v in self.rows_mut() {
						v.tanh()
					}
				}

				fn asinh(&mut self)  {
					for v in self.rows_mut() {
						v.asinh()
					}
				}

				fn acosh(&mut self)  {
					for v in self.rows_mut() {
						v.acosh()
					}
				}

				fn atanh(&mut self)  {
					for v in self.rows_mut() {
						v.atanh()
					}
				}
			}

			impl<S: ToSlice<T>, V: Vector<T> + PowerOps<T>, T: RealNumber>
					PowerOps<T> for $matrix<V, S, T> {
				fn sqrt(&mut self)  {
					for v in self.rows_mut() {
						v.sqrt()
					}
				}

				fn square(&mut self)  {
					for v in self.rows_mut() {
						v.square()
					}
				}

				fn root(&mut self, degree: T)  {
					for v in self.rows_mut() {
						v.root(degree)
					}
				}

				fn powf(&mut self, exponent: T)  {
					for v in self.rows_mut() {
						v.powf(exponent)
					}
				}

				fn ln(&mut self)  {
					for v in self.rows_mut() {
						v.ln()
					}
				}

				fn exp(&mut self)  {
					for v in self.rows_mut() {
						v.exp()
					}
				}

				fn log(&mut self, base: T)  {
					for v in self.rows_mut() {
						v.log(base)
					}
				}

				fn expf(&mut self, base: T)  {
					for v in self.rows_mut() {
						v.expf(base)
					}
				}
			}

			impl<S: ToSlice<T>, V: Vector<T> + MapInplaceOps<T>, T: RealNumber>
					MapInplaceOps<T> for $matrix<V, S, T> {
				fn map_inplace<'a, A, F>(&mut self, argument: A, map: &F)
					where A: Sync + Copy + Send,
					  	  F: Fn(T, usize, A) -> T + 'a + Sync {
					for v in self.rows_mut() {
						v.map_inplace(argument, &map)
					}
				}
			}

			impl<S: ToSlice<T>, V: Vector<T> + MapInplaceOps<Complex<T>>, T: RealNumber>
					MapInplaceOps<Complex<T>> for $matrix<V, S, T> {
				fn map_inplace<'a, A, F>(&mut self, argument: A, map: &F)
					where A: Sync + Copy + Send,
					  	  F: Fn(Complex<T>, usize, A) -> Complex<T> + 'a + Sync {
					for v in self.rows_mut() {
						v.map_inplace(argument, &map)
					}
				}
			}
		)*
	}
}

add_mat_impl!(MatrixMxN; Matrix2xN; Matrix3xN; Matrix4xN);
