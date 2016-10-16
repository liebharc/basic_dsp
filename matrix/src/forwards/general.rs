use basic_dsp_vector::*;
use super::super::*;
use num::complex::Complex;

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

			impl<V: Vector<T> + OffsetOps<T>, S: ToSlice<T>, T: RealNumber> OffsetOps<T> for $matrix<V, S, T> {
				fn offset(&mut self, offset: T) {
					for v in self.rows_mut() {
						v.offset(offset);
					}
				}
			}

			impl<V: Vector<T> + ScaleOps<Complex<T>>, S: ToSlice<T>, T: RealNumber> ScaleOps<Complex<T>> for $matrix<V, S, T> {
				fn scale(&mut self, factor: Complex<T>) {
					for v in self.rows_mut() {
						v.scale(factor);
					}
				}
			}

			impl<V: Vector<T> + OffsetOps<Complex<T>>, S: ToSlice<T>, T: RealNumber> OffsetOps<Complex<T>> for $matrix<V, S, T> {
				fn offset(&mut self, offset: Complex<T>) {
					for v in self.rows_mut() {
						v.offset(offset);
					}
				}
			}

			impl<V: Vector<T> + ElementaryOps, S: ToSlice<T>, T: RealNumber> ElementaryOps for $matrix<V, S, T> {
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

			impl<V: Vector<T> + ElementaryWrapAroundOps, S: ToSlice<T>, T: RealNumber> ElementaryWrapAroundOps for $matrix<V, S, T> {
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

			impl<V: Vector<T> + ReorganizeDataOps<T>, S: ToSlice<T>, T: RealNumber> ReorganizeDataOps<T> for $matrix<V, S, T> {
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

			impl<S: ToSliceMut<T>, V: Vector<T> + ReorganizeDataOpsBuffered<S, T>, T: RealNumber>
					ReorganizeDataOpsBuffered<S, T> for $matrix<V, S, T> {
				fn swap_halves_b<B: Buffer<S, T>>(&mut self, buffer: &mut B)  {
					for v in self.rows_mut() {
						v.swap_halves_b(buffer)
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
				fn map_inplace<'a, A, F>(&mut self, argument: A, map: F)
					where A: Sync + Copy + Send,
					  	  F: Fn(T, usize, A) -> T + 'a + Sync {
					for v in self.rows_mut() {
						v.map_inplace(argument, |v, i, a|map(v, i, a))
					}
				}
			}

			impl<S: ToSlice<T>, V: Vector<T> + MapInplaceOps<Complex<T>>, T: RealNumber>
					MapInplaceOps<Complex<T>> for $matrix<V, S, T> {
				fn map_inplace<'a, A, F>(&mut self, argument: A, map: F)
					where A: Sync + Copy + Send,
					  	  F: Fn(Complex<T>, usize, A) -> Complex<T> + 'a + Sync {
					for v in self.rows_mut() {
						v.map_inplace(argument, |v, i, a|map(v, i, a))
					}
				}
			}
/*
			impl<S: ToSlice<T>, V: Vector<T> + MapAggregateOps<T>, T: RealNumber>
					MapAggregateOps<T> for $matrix<V, S, T> {
				fn map_aggregate<'a, A, F>(
						&self,
			            argument: A,
			            map: FMap,
			            aggregate: FAggr) -> ScalarResult<R>
					where A: Sync + Copy + Send,
						  FMap: Fn(T, usize, A) -> R + 'a + Sync,
						  FAggr: Fn(R, R) -> R + 'a + Sync + Send,
						  R: Send {
					for v in self.rows_mut() {
						v.map_inplace(argument, |v, i, a|map(v, i, a))
					}
				}
			}*/

			// TODO DotProductOps, MapAggregateOps, Statistics, Add/sub/mul/div vector to matu
      c


		)*
	}
}

add_mat_impl!(MatrixMxN; Matrix2xN; Matrix3xN; Matrix4xN);
