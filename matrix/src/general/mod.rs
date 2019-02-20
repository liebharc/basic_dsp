use super::*;
use crate::IntoFixedLength;

mod elementary;
pub use self::elementary::*;
mod statistics;
pub use self::statistics::*;

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, N: NumberSpace, D: Domain>
    DotProductOps<MatrixMxN<V, S, T>, T, T, N, D> for MatrixMxN<V, S, T>
where
    V: DotProductOps<V, T, T, N, D, Output = ScalarResult<T>> + GetMetaData<T, N, D>,
{
    type Output = ScalarResult<Vec<T>>;

    fn dot_product(&self, factor: &MatrixMxN<V, S, T>) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for (v, o) in self.rows().iter().zip(factor.rows()) {
            let res = r#try!(v.dot_product(o));
            result.push(res);
        }

        Ok(result)
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, N: NumberSpace, D: Domain>
    DotProductOps<Matrix2xN<V, S, T>, T, T, N, D> for Matrix2xN<V, S, T>
where
    V: DotProductOps<V, T, T, N, D, Output = ScalarResult<T>> + GetMetaData<T, N, D>,
{
    type Output = ScalarResult<Vec<T>>;

    fn dot_product(&self, factor: &Matrix2xN<V, S, T>) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for (v, o) in self.rows().iter().zip(factor.rows()) {
            let res = r#try!(v.dot_product(o));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, N: NumberSpace, D: Domain>
    DotProductOps<Matrix3xN<V, S, T>, T, T, N, D> for Matrix3xN<V, S, T>
where
    V: DotProductOps<V, T, T, N, D, Output = ScalarResult<T>> + GetMetaData<T, N, D>,
{
    type Output = ScalarResult<Vec<T>>;

    fn dot_product(&self, factor: &Matrix3xN<V, S, T>) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for (v, o) in self.rows().iter().zip(factor.rows()) {
            let res = r#try!(v.dot_product(o));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, N: NumberSpace, D: Domain>
    DotProductOps<Matrix4xN<V, S, T>, T, T, N, D> for Matrix4xN<V, S, T>
where
    V: DotProductOps<V, T, T, N, D, Output = ScalarResult<T>> + GetMetaData<T, N, D>,
{
    type Output = ScalarResult<Vec<T>>;

    fn dot_product(&self, factor: &Matrix4xN<V, S, T>) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for (v, o) in self.rows().iter().zip(factor.rows()) {
            let res = r#try!(v.dot_product(o));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, N: NumberSpace, D: Domain>
    DotProductOps<V, T, T, N, D> for MatrixMxN<V, S, T>
where
    V: DotProductOps<V, T, T, N, D, Output = ScalarResult<T>> + GetMetaData<T, N, D>,
{
    type Output = ScalarResult<Vec<T>>;

    fn dot_product(&self, factor: &V) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = r#try!(v.dot_product(factor));
            result.push(res);
        }

        Ok(result)
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, N: NumberSpace, D: Domain>
    DotProductOps<V, T, T, N, D> for Matrix2xN<V, S, T>
where
    V: DotProductOps<V, T, T, N, D, Output = ScalarResult<T>> + GetMetaData<T, N, D>,
{
    type Output = ScalarResult<[T; 2]>;

    fn dot_product(&self, factor: &V) -> ScalarResult<[T; 2]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = r#try!(v.dot_product(factor));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, N: NumberSpace, D: Domain>
    DotProductOps<V, T, T, N, D> for Matrix3xN<V, S, T>
where
    V: DotProductOps<V, T, T, N, D, Output = ScalarResult<T>> + GetMetaData<T, N, D>,
{
    type Output = ScalarResult<[T; 3]>;

    fn dot_product(&self, factor: &V) -> ScalarResult<[T; 3]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = r#try!(v.dot_product(factor));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, N: NumberSpace, D: Domain>
    DotProductOps<V, T, T, N, D> for Matrix4xN<V, S, T>
where
    V: DotProductOps<V, T, T, N, D, Output = ScalarResult<T>> + GetMetaData<T, N, D>,
{
    type Output = ScalarResult<[T; 4]>;

    fn dot_product(&self, factor: &V) -> ScalarResult<[T; 4]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = r#try!(v.dot_product(factor));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, N: NumberSpace, D: Domain>
    PreciseDotProductOps<MatrixMxN<V, S, T>, T, T, N, D> for MatrixMxN<V, S, T>
where
    V: PreciseDotProductOps<V, T, T, N, D, Output = ScalarResult<T>> + GetMetaData<T, N, D>,
{
    type Output = ScalarResult<Vec<T>>;

    fn dot_product_prec(&self, factor: &MatrixMxN<V, S, T>) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for (v, o) in self.rows().iter().zip(factor.rows()) {
            let res = r#try!(v.dot_product_prec(o));
            result.push(res);
        }

        Ok(result)
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, N: NumberSpace, D: Domain>
    PreciseDotProductOps<Matrix2xN<V, S, T>, T, T, N, D> for Matrix2xN<V, S, T>
where
    V: PreciseDotProductOps<V, T, T, N, D, Output = ScalarResult<T>> + GetMetaData<T, N, D>,
{
    type Output = ScalarResult<Vec<T>>;

    fn dot_product_prec(&self, factor: &Matrix2xN<V, S, T>) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for (v, o) in self.rows().iter().zip(factor.rows()) {
            let res = r#try!(v.dot_product_prec(o));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, N: NumberSpace, D: Domain>
    PreciseDotProductOps<Matrix3xN<V, S, T>, T, T, N, D> for Matrix3xN<V, S, T>
where
    V: PreciseDotProductOps<V, T, T, N, D, Output = ScalarResult<T>> + GetMetaData<T, N, D>,
{
    type Output = ScalarResult<Vec<T>>;

    fn dot_product_prec(&self, factor: &Matrix3xN<V, S, T>) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for (v, o) in self.rows().iter().zip(factor.rows()) {
            let res = r#try!(v.dot_product_prec(o));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, N: NumberSpace, D: Domain>
    PreciseDotProductOps<Matrix4xN<V, S, T>, T, T, N, D> for Matrix4xN<V, S, T>
where
    V: PreciseDotProductOps<V, T, T, N, D, Output = ScalarResult<T>> + GetMetaData<T, N, D>,
{
    type Output = ScalarResult<Vec<T>>;

    fn dot_product_prec(&self, factor: &Matrix4xN<V, S, T>) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for (v, o) in self.rows().iter().zip(factor.rows()) {
            let res = r#try!(v.dot_product_prec(o));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, N: NumberSpace, D: Domain>
    PreciseDotProductOps<V, T, T, N, D> for MatrixMxN<V, S, T>
where
    V: PreciseDotProductOps<V, T, T, N, D, Output = ScalarResult<T>> + GetMetaData<T, N, D>,
{
    type Output = ScalarResult<Vec<T>>;

    fn dot_product_prec(&self, factor: &V) -> ScalarResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = r#try!(v.dot_product_prec(factor));
            result.push(res);
        }

        Ok(result)
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, N: NumberSpace, D: Domain>
    PreciseDotProductOps<V, T, T, N, D> for Matrix2xN<V, S, T>
where
    V: PreciseDotProductOps<V, T, T, N, D, Output = ScalarResult<T>> + GetMetaData<T, N, D>,
{
    type Output = ScalarResult<[T; 2]>;

    fn dot_product_prec(&self, factor: &V) -> ScalarResult<[T; 2]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = r#try!(v.dot_product_prec(factor));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, N: NumberSpace, D: Domain>
    PreciseDotProductOps<V, T, T, N, D> for Matrix3xN<V, S, T>
where
    V: PreciseDotProductOps<V, T, T, N, D, Output = ScalarResult<T>> + GetMetaData<T, N, D>,
{
    type Output = ScalarResult<[T; 3]>;

    fn dot_product_prec(&self, factor: &V) -> ScalarResult<[T; 3]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = r#try!(v.dot_product_prec(factor));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, N: NumberSpace, D: Domain>
    PreciseDotProductOps<V, T, T, N, D> for Matrix4xN<V, S, T>
where
    V: PreciseDotProductOps<V, T, T, N, D, Output = ScalarResult<T>> + GetMetaData<T, N, D>,
{
    type Output = ScalarResult<[T; 4]>;

    fn dot_product_prec(&self, factor: &V) -> ScalarResult<[T; 4]> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = r#try!(v.dot_product_prec(factor));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, R: Send> MapAggregateOps<T, R>
    for MatrixMxN<V, S, T>
where
    V: MapAggregateOps<T, R, Output = ScalarResult<R>>,
{
    type Output = ScalarResult<Vec<R>>;

    fn map_aggregate<'a, A, FMap, FAggr>(
        &self,
        argument: A,
        map: &FMap,
        aggregate: &FAggr,
    ) -> ScalarResult<Vec<R>>
    where
        A: Sync + Copy + Send,
        FMap: Fn(T, usize, A) -> R + 'a + Sync,
        FAggr: Fn(R, R) -> R + 'a + Sync + Send,
        R: Send,
    {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = r#try!(v.map_aggregate(argument, &map, &aggregate));
            result.push(res);
        }

        Ok(result)
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, R: Send> MapAggregateOps<T, R>
    for Matrix2xN<V, S, T>
where
    V: MapAggregateOps<T, R, Output = ScalarResult<R>>,
{
    type Output = ScalarResult<[R; 2]>;

    fn map_aggregate<'a, A, FMap, FAggr>(
        &self,
        argument: A,
        map: &FMap,
        aggregate: &FAggr,
    ) -> ScalarResult<[R; 2]>
    where
        A: Sync + Copy + Send,
        FMap: Fn(T, usize, A) -> R + 'a + Sync,
        FAggr: Fn(R, R) -> R + 'a + Sync + Send,
        R: Send,
    {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = r#try!(v.map_aggregate(argument, &map, &aggregate));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, R: Send> MapAggregateOps<T, R>
    for Matrix3xN<V, S, T>
where
    V: MapAggregateOps<T, R, Output = ScalarResult<R>>,
{
    type Output = ScalarResult<[R; 3]>;

    fn map_aggregate<'a, A, FMap, FAggr>(
        &self,
        argument: A,
        map: &FMap,
        aggregate: &FAggr,
    ) -> ScalarResult<[R; 3]>
    where
        A: Sync + Copy + Send,
        FMap: Fn(T, usize, A) -> R + 'a + Sync,
        FAggr: Fn(R, R) -> R + 'a + Sync + Send,
        R: Send,
    {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = r#try!(v.map_aggregate(argument, &map, &aggregate));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, R: Send> MapAggregateOps<T, R>
    for Matrix4xN<V, S, T>
where
    V: MapAggregateOps<T, R, Output = ScalarResult<R>>,
{
    type Output = ScalarResult<[R; 4]>;

    fn map_aggregate<'a, A, FMap, FAggr>(
        &self,
        argument: A,
        map: &FMap,
        aggregate: &FAggr,
    ) -> ScalarResult<[R; 4]>
    where
        A: Sync + Copy + Send,
        FMap: Fn(T, usize, A) -> R + 'a + Sync,
        FAggr: Fn(R, R) -> R + 'a + Sync + Send,
        R: Send,
    {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = r#try!(v.map_aggregate(argument, &map, &aggregate));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

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
        assert_eq!(mat1.rows[0].data(..), &[3.0, 8.0]);
        assert_eq!(mat1.rows[1].data(..), &[1.0, 7.0]);
    }

    #[test]
    fn add_vec() {
        let mut mat = vec![vec![0.0, 1.0], vec![2.0, 3.0]].to_real_time_mat();
        let vec = vec![3.0, 7.0].to_real_time_vec();
        mat.add(&vec).unwrap();
        assert_eq!(mat.rows[0].data(..), &[3.0, 8.0]);
        assert_eq!(mat.rows[1].data(..), &[5.0, 10.0]);
    }
}
