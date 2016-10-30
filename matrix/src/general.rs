use basic_dsp_vector::*;
use super::*;
use IntoFixedLength;
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

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber> ElementaryOps<$matrix<V, S, T>>
                    for $matrix<V, S, T>
                    where V: ElementaryOps<V> {
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

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber> ElementaryOps<V>
                    for $matrix<V, S, T>
                    where V: ElementaryOps<V> {
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

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber>
                    ElementaryWrapAroundOps<$matrix<V, S, T>>
                    for $matrix<V, S, T>
                    where V: ElementaryWrapAroundOps<V> {
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

			impl<V: Vector<T>, S: ToSlice<T>, T: RealNumber> ElementaryWrapAroundOps<V>
                    for $matrix<V, S, T>
                    where V: ElementaryWrapAroundOps<V> {
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
						v.map_inplace(argument, &map)
					}
				}
			}

			impl<S: ToSlice<T>, V: Vector<T> + MapInplaceOps<Complex<T>>, T: RealNumber>
					MapInplaceOps<Complex<T>> for $matrix<V, S, T> {
				fn map_inplace<'a, A, F>(&mut self, argument: A, map: F)
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

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> DotProductOps<T, MatrixMxN<V, S, T>>
   for MatrixMxN<V, S, T>
    where V: DotProductOps<T, V, Output=ScalarResult<T>> {
    type Output = ScalarResult<Vec<T>>;

    fn dot_product(&self, factor: &MatrixMxN<V, S, T>) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for (v, o) in self.rows().iter().zip(factor.rows()) {
            let res = try!(v.dot_product(o));
            result.push(res);
        }

        Ok(result)
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> DotProductOps<T, Matrix2xN<V, S, T>>
   for Matrix2xN<V, S, T>
    where V: DotProductOps<T, V, Output=ScalarResult<T>> {
    type Output = ScalarResult<Vec<T>>;

    fn dot_product(&self, factor: &Matrix2xN<V, S, T>) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for (v, o) in self.rows().iter().zip(factor.rows()) {
            let res = try!(v.dot_product(o));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> DotProductOps<T, Matrix3xN<V, S, T>>
   for Matrix3xN<V, S, T>
    where V: DotProductOps<T, V, Output=ScalarResult<T>> {
    type Output = ScalarResult<Vec<T>>;

    fn dot_product(&self, factor: &Matrix3xN<V, S, T>) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for (v, o) in self.rows().iter().zip(factor.rows()) {
            let res = try!(v.dot_product(o));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> DotProductOps<T, Matrix4xN<V, S, T>>
   for Matrix4xN<V, S, T>
    where V: DotProductOps<T, V, Output=ScalarResult<T>> {
    type Output = ScalarResult<Vec<T>>;

    fn dot_product(&self, factor: &Matrix4xN<V, S, T>) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for (v, o) in self.rows().iter().zip(factor.rows()) {
            let res = try!(v.dot_product(o));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> DotProductOps<T, V> for MatrixMxN<V, S, T>
    where V: DotProductOps<T, V, Output = ScalarResult<T>>
{
    type Output = ScalarResult<Vec<T>>;

    fn dot_product(&self, factor: &V) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = try!(v.dot_product(factor));
            result.push(res);
        }

        Ok(result)
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> DotProductOps<T, V> for Matrix2xN<V, S, T>
    where V: DotProductOps<T, V, Output = ScalarResult<T>>
{
    type Output = ScalarResult<[T; 2]>;

    fn dot_product(&self, factor: &V) -> ScalarResult<[T; 2]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = try!(v.dot_product(factor));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> DotProductOps<T, V> for Matrix3xN<V, S, T>
    where V: DotProductOps<T, V, Output = ScalarResult<T>>
{
    type Output = ScalarResult<[T; 3]>;

    fn dot_product(&self, factor: &V) -> ScalarResult<[T; 3]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = try!(v.dot_product(factor));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> DotProductOps<T, V> for Matrix4xN<V, S, T>
    where V: DotProductOps<T, V, Output = ScalarResult<T>>
{
    type Output = ScalarResult<[T; 4]>;

    fn dot_product(&self, factor: &V) -> ScalarResult<[T; 4]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = try!(v.dot_product(factor));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> PreciseDotProductOps<T, MatrixMxN<V, S, T>>
   for MatrixMxN<V, S, T>
    where V: PreciseDotProductOps<T, V, Output=ScalarResult<T>> {
    type Output = ScalarResult<Vec<T>>;

    fn dot_product_prec(&self, factor: &MatrixMxN<V, S, T>) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for (v, o) in self.rows().iter().zip(factor.rows()) {
            let res = try!(v.dot_product_prec(o));
            result.push(res);
        }

        Ok(result)
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> PreciseDotProductOps<T, Matrix2xN<V, S, T>>
   for Matrix2xN<V, S, T>
    where V: PreciseDotProductOps<T, V, Output=ScalarResult<T>> {
    type Output = ScalarResult<Vec<T>>;

    fn dot_product_prec(&self, factor: &Matrix2xN<V, S, T>) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for (v, o) in self.rows().iter().zip(factor.rows()) {
            let res = try!(v.dot_product_prec(o));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> PreciseDotProductOps<T, Matrix3xN<V, S, T>>
   for Matrix3xN<V, S, T>
    where V: PreciseDotProductOps<T, V, Output=ScalarResult<T>> {
    type Output = ScalarResult<Vec<T>>;

    fn dot_product_prec(&self, factor: &Matrix3xN<V, S, T>) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for (v, o) in self.rows().iter().zip(factor.rows()) {
            let res = try!(v.dot_product_prec(o));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> PreciseDotProductOps<T, Matrix4xN<V, S, T>>
   for Matrix4xN<V, S, T>
    where V: PreciseDotProductOps<T, V, Output=ScalarResult<T>> {
    type Output = ScalarResult<Vec<T>>;

    fn dot_product_prec(&self, factor: &Matrix4xN<V, S, T>) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for (v, o) in self.rows().iter().zip(factor.rows()) {
            let res = try!(v.dot_product_prec(o));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> PreciseDotProductOps<T, V> for MatrixMxN<V, S, T>
    where V: PreciseDotProductOps<T, V, Output = ScalarResult<T>>
{
    type Output = ScalarResult<Vec<T>>;

    fn dot_product_prec(&self, factor: &V) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = try!(v.dot_product_prec(factor));
            result.push(res);
        }

        Ok(result)
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> PreciseDotProductOps<T, V> for Matrix2xN<V, S, T>
    where V: PreciseDotProductOps<T, V, Output = ScalarResult<T>>
{
    type Output = ScalarResult<[T; 2]>;

    fn dot_product_prec(&self, factor: &V) -> ScalarResult<[T; 2]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = try!(v.dot_product_prec(factor));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> PreciseDotProductOps<T, V> for Matrix3xN<V, S, T>
    where V: PreciseDotProductOps<T, V, Output = ScalarResult<T>>
{
    type Output = ScalarResult<[T; 3]>;

    fn dot_product_prec(&self, factor: &V) -> ScalarResult<[T; 3]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = try!(v.dot_product_prec(factor));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> PreciseDotProductOps<T, V> for Matrix4xN<V, S, T>
    where V: PreciseDotProductOps<T, V, Output = ScalarResult<T>>
{
    type Output = ScalarResult<[T; 4]>;

    fn dot_product_prec(&self, factor: &V) -> ScalarResult<[T; 4]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = try!(v.dot_product_prec(factor));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, R: Send> MapAggregateOps<T, R>
    for MatrixMxN<V, S, T>
    where V: MapAggregateOps<T, R, Output = ScalarResult<R>>
{
    type Output = ScalarResult<Vec<R>>;

    fn map_aggregate<'a, A, FMap, FAggr>(&self,
                                         argument: A,
                                         map: FMap,
                                         aggregate: FAggr)
                                         -> ScalarResult<Vec<R>>
        where A: Sync + Copy + Send,
              FMap: Fn(T, usize, A) -> R + 'a + Sync,
              FAggr: Fn(R, R) -> R + 'a + Sync + Send,
              R: Send
    {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res =
                try!(v.map_aggregate(argument, &map, &aggregate));
            result.push(res);
        }

        Ok(result)
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, R: Send> MapAggregateOps<T, R>
    for Matrix2xN<V, S, T>
    where V: MapAggregateOps<T, R, Output = ScalarResult<R>>
{
    type Output = ScalarResult<[R; 2]>;

    fn map_aggregate<'a, A, FMap, FAggr>(&self,
                                         argument: A,
                                         map: FMap,
                                         aggregate: FAggr)
                                         -> ScalarResult<[R; 2]>
        where A: Sync + Copy + Send,
              FMap: Fn(T, usize, A) -> R + 'a + Sync,
              FAggr: Fn(R, R) -> R + 'a + Sync + Send,
              R: Send
    {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res =
                try!(v.map_aggregate(argument, &map, &aggregate));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, R: Send> MapAggregateOps<T, R>
    for Matrix3xN<V, S, T>
    where V: MapAggregateOps<T, R, Output = ScalarResult<R>>
{
    type Output = ScalarResult<[R; 3]>;

    fn map_aggregate<'a, A, FMap, FAggr>(&self,
                                         argument: A,
                                         map: FMap,
                                         aggregate: FAggr)
                                         -> ScalarResult<[R; 3]>
        where A: Sync + Copy + Send,
              FMap: Fn(T, usize, A) -> R + 'a + Sync,
              FAggr: Fn(R, R) -> R + 'a + Sync + Send,
              R: Send
    {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res =
                try!(v.map_aggregate(argument, &map, &aggregate));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, R: Send> MapAggregateOps<T, R>
    for Matrix4xN<V, S, T>
    where V: MapAggregateOps<T, R, Output = ScalarResult<R>>
{
    type Output = ScalarResult<[R; 4]>;

    fn map_aggregate<'a, A, FMap, FAggr>(&self,
                                         argument: A,
                                         map: FMap,
                                         aggregate: FAggr)
                                         -> ScalarResult<[R; 4]>
        where A: Sync + Copy + Send,
              FMap: Fn(T, usize, A) -> R + 'a + Sync,
              FAggr: Fn(R, R) -> R + 'a + Sync + Send,
              R: Send
    {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res =
                try!(v.map_aggregate(argument, &map, &aggregate));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> StatisticsOps<Vec<Statistics<T>>>
    for MatrixMxN<V, S, T>
    where V: StatisticsOps<Statistics<T>>
{
    fn statistics(&self) -> Vec<Statistics<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics();
            result.push(res);
        }

        result
    }

    fn statistics_splitted(&self, len: usize) -> Vec<Vec<Statistics<T>>> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics_splitted(len);
            result.push(res);
        }

        result
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> StatisticsOps<[Statistics<T>; 2]>
    for Matrix2xN<V, S, T>
    where V: StatisticsOps<Statistics<T>>
{
    fn statistics(&self) -> [Statistics<T>; 2] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics();
            result.push(res);
        }

        result.into_fixed_length()
    }

    fn statistics_splitted(&self, len: usize) -> Vec<[Statistics<T>; 2]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics_splitted(len);
            result.push(res.into_fixed_length());
        }

        result
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> StatisticsOps<[Statistics<T>; 3]>
    for Matrix3xN<V, S, T>
    where V: StatisticsOps<Statistics<T>>
{
    fn statistics(&self) -> [Statistics<T>; 3] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics();
            result.push(res);
        }

        result.into_fixed_length()
    }

    fn statistics_splitted(&self, len: usize) -> Vec<[Statistics<T>; 3]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics_splitted(len);
            result.push(res.into_fixed_length());
        }

        result
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> StatisticsOps<[Statistics<T>; 4]>
    for Matrix4xN<V, S, T>
    where V: StatisticsOps<Statistics<T>>
{
    fn statistics(&self) -> [Statistics<T>; 4] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics();
            result.push(res);
        }

        result.into_fixed_length()
    }

    fn statistics_splitted(&self, len: usize) -> Vec<[Statistics<T>; 4]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics_splitted(len);
            result.push(res.into_fixed_length());
        }

        result
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> SumOps<Vec<T>> for MatrixMxN<V, S, T>
    where V: SumOps<T>
{
    fn sum(&self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.sum();
            result.push(res);
        }

        result
    }

    fn sum_sq(&self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.sum_sq();
            result.push(res);
        }

        result
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> SumOps<[T; 2]> for Matrix2xN<V, S, T>
    where V: SumOps<T>
{
    fn sum(&self) -> [T; 2] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.sum();
            result.push(res);
        }

        result.into_fixed_length()
    }

    fn sum_sq(&self) -> [T; 2] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.sum_sq();
            result.push(res);
        }

        result.into_fixed_length()
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> SumOps<[T; 3]> for Matrix2xN<V, S, T>
    where V: SumOps<T>
{
    fn sum(&self) -> [T; 3] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.sum();
            result.push(res);
        }

        result.into_fixed_length()
    }

    fn sum_sq(&self) -> [T; 3] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.sum_sq();
            result.push(res);
        }

        result.into_fixed_length()
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> SumOps<[T; 4]> for Matrix2xN<V, S, T>
    where V: SumOps<T>
{
    fn sum(&self) -> [T; 4] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.sum();
            result.push(res);
        }

        result.into_fixed_length()
    }

    fn sum_sq(&self) -> [T; 4] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.sum_sq();
            result.push(res);
        }

        result.into_fixed_length()
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, O: RealNumber> 
    PreciseStatisticsOps<Vec<Statistics<O>>>
    for MatrixMxN<V, S, T>
    where V: PreciseStatisticsOps<Statistics<O>>
{
    fn statistics_prec(&self) -> Vec<Statistics<O>> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics_prec();
            result.push(res);
        }

        result
    }

    fn statistics_splitted_prec(&self, len: usize) -> Vec<Vec<Statistics<O>>> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics_splitted_prec(len);
            result.push(res);
        }

        result
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, O: RealNumber> 
    PreciseStatisticsOps<[Statistics<O>; 2]>
    for Matrix2xN<V, S, T>
    where V: PreciseStatisticsOps<Statistics<O>>
{
    fn statistics_prec(&self) -> [Statistics<O>; 2] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics_prec();
            result.push(res);
        }

        result.into_fixed_length()
    }

    fn statistics_splitted_prec(&self, len: usize) -> Vec<[Statistics<O>; 2]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics_splitted_prec(len);
            result.push(res.into_fixed_length());
        }

        result
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, O: RealNumber> 
    PreciseStatisticsOps<[Statistics<O>; 3]>
    for Matrix3xN<V, S, T>
    where V: PreciseStatisticsOps<Statistics<O>>
{
    fn statistics_prec(&self) -> [Statistics<O>; 3] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics_prec();
            result.push(res);
        }

        result.into_fixed_length()
    }

    fn statistics_splitted_prec(&self, len: usize) -> Vec<[Statistics<O>; 3]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics_splitted_prec(len);
            result.push(res.into_fixed_length());
        }

        result
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, O: RealNumber> 
    PreciseStatisticsOps<[Statistics<O>; 4]>
    for Matrix4xN<V, S, T>
    where V: PreciseStatisticsOps<Statistics<O>>
{
    fn statistics_prec(&self) -> [Statistics<O>; 4] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics_prec();
            result.push(res);
        }

        result.into_fixed_length()
    }

    fn statistics_splitted_prec(&self, len: usize) -> Vec<[Statistics<O>; 4]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics_splitted_prec(len);
            result.push(res.into_fixed_length());
        }

        result
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, O: RealNumber> PreciseSumOps<Vec<O>> 
    for MatrixMxN<V, S, T>
    where V: PreciseSumOps<O>
{
    fn sum_prec(&self) -> Vec<O> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.sum_prec();
            result.push(res);
        }

        result
    }

    fn sum_sq_prec(&self) -> Vec<O> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.sum_sq_prec();
            result.push(res);
        }

        result
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, O: RealNumber> PreciseSumOps<[O; 2]> 
    for Matrix2xN<V, S, T>
    where V: PreciseSumOps<O>
{
    fn sum_prec(&self) -> [O; 2] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.sum_prec();
            result.push(res);
        }

        result.into_fixed_length()
    }

    fn sum_sq_prec(&self) -> [O; 2] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.sum_sq_prec();
            result.push(res);
        }

        result.into_fixed_length()
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, O: RealNumber> PreciseSumOps<[O; 3]> 
    for Matrix3xN<V, S, T>
    where V: PreciseSumOps<O>
{
    fn sum_prec(&self) -> [O; 3] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.sum_prec();
            result.push(res);
        }

        result.into_fixed_length()
    }

    fn sum_sq_prec(&self) -> [O; 3] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.sum_sq_prec();
            result.push(res);
        }

        result.into_fixed_length()
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, O: RealNumber> PreciseSumOps<[O; 4]> 
    for Matrix4xN<V, S, T>
    where V: PreciseSumOps<O>
{
    fn sum_prec(&self) -> [O; 4] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.sum_prec();
            result.push(res);
        }

        result.into_fixed_length()
    }

    fn sum_sq_prec(&self) -> [O; 4] {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.sum_sq_prec();
            result.push(res);
        }

        result.into_fixed_length()
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use basic_dsp_vector::*;

    #[test]
    fn sum_test() {
        let mat = vec![vec![0.0, 1.0], vec![2.0, 3.0]].to_real_time_mat();
        let sum = mat.sum();
        assert_eq!(sum, &[1.0, 5.0]);
    }

    #[test]
    fn add_mat() {
        let mut mat1 = vec![vec![0.0, 1.0], vec![2.0, 3.0]].to_real_time_mat();
        let mat2 = vec![vec![3.0, 7.0], vec![-1.0, 4.0]].to_real_time_mat();
        mat1.add(&mat2).unwrap();
        assert_eq!(&mat1.rows[0][..], &[3.0, 8.0]);
        assert_eq!(&mat1.rows[1][..], &[1.0, 7.0]);
    }


    #[test]
    fn add_vec() {
        let mut mat = vec![vec![0.0, 1.0], vec![2.0, 3.0]].to_real_time_mat();
        let vec = vec![3.0, 7.0].to_real_time_vec();
        mat.add(&vec).unwrap();
        assert_eq!(&mat.rows[0][..], &[3.0, 8.0]);
        assert_eq!(&mat.rows[1][..], &[5.0, 10.0]);
    }
}
