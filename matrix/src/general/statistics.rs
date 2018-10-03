use super::*;
use basic_dsp_vector::numbers::*;
use basic_dsp_vector::*;
use IntoFixedLength;

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> StatisticsOps<T> for MatrixMxN<V, S, T>
where
    V: StatisticsOps<Statistics<T>, Result = Statistics<T>>,
{
    type Result = Vec<Statistics<T>>;

    fn statistics(&self) -> Vec<Statistics<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics();
            result.push(res);
        }

        result
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> StatisticsSplitOps<T> for MatrixMxN<V, S, T>
where
    V: StatisticsSplitOps<Statistics<T>, Result = StatsVec<Statistics<T>>>,
{
    type Result = Vec<StatsVec<Statistics<T>>>;

    fn statistics_split(&self, len: usize) -> ScalarResult<Vec<StatsVec<Statistics<T>>>> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = try!(v.statistics_split(len));
            result.push(res);
        }

        Ok(result)
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> StatisticsOps<T> for Matrix2xN<V, S, T>
where
    V: StatisticsOps<Statistics<T>, Result = Statistics<T>>,
{
    type Result = [Statistics<T>; 2];

    fn statistics(&self) -> Self::Result {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics();
            result.push(res);
        }

        result.into_fixed_length()
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> StatisticsSplitOps<T> for Matrix2xN<V, S, T>
where
    V: StatisticsSplitOps<Statistics<T>, Result = StatsVec<Statistics<T>>>,
{
    type Result = [StatsVec<Statistics<T>>; 2];

    fn statistics_split(&self, len: usize) -> ScalarResult<Self::Result> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = try!(v.statistics_split(len));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> StatisticsOps<T> for Matrix3xN<V, S, T>
where
    V: StatisticsOps<Statistics<T>, Result = Statistics<T>>,
{
    type Result = [Statistics<T>; 3];

    fn statistics(&self) -> Self::Result {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics();
            result.push(res);
        }

        result.into_fixed_length()
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> StatisticsSplitOps<T> for Matrix3xN<V, S, T>
where
    V: StatisticsSplitOps<Statistics<T>, Result = StatsVec<Statistics<T>>>,
{
    type Result = [StatsVec<Statistics<T>>; 3];

    fn statistics_split(&self, len: usize) -> ScalarResult<Self::Result> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = try!(v.statistics_split(len));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> StatisticsOps<T> for Matrix4xN<V, S, T>
where
    V: StatisticsOps<Statistics<T>, Result = Statistics<T>>,
{
    type Result = [Statistics<T>; 4];

    fn statistics(&self) -> Self::Result {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics();
            result.push(res);
        }

        result.into_fixed_length()
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> StatisticsSplitOps<T> for Matrix4xN<V, S, T>
where
    V: StatisticsSplitOps<Statistics<T>, Result = StatsVec<Statistics<T>>>,
{
    type Result = [StatsVec<Statistics<T>>; 4];

    fn statistics_split(&self, len: usize) -> ScalarResult<Self::Result> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = try!(v.statistics_split(len));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> SumOps<Vec<T>> for MatrixMxN<V, S, T>
where
    V: SumOps<T>,
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
where
    V: SumOps<T>,
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
where
    V: SumOps<T>,
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
where
    V: SumOps<T>,
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

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> PreciseStatisticsOps<T> for MatrixMxN<V, S, T>
where
    V: PreciseStatisticsOps<Statistics<T>, Result = Statistics<T>>,
{
    type Result = Vec<Statistics<T>>;

    fn statistics_prec(&self) -> Vec<Statistics<T>> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics_prec();
            result.push(res);
        }

        result
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> PreciseStatisticsSplitOps<T> for MatrixMxN<V, S, T>
where
    V: PreciseStatisticsSplitOps<Statistics<T>, Result = StatsVec<Statistics<T>>>,
{
    type Result = Vec<StatsVec<Statistics<T>>>;

    fn statistics_split_prec(&self, len: usize) -> ScalarResult<Vec<StatsVec<Statistics<T>>>> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = try!(v.statistics_split_prec(len));
            result.push(res);
        }

        Ok(result)
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> PreciseStatisticsOps<T> for Matrix2xN<V, S, T>
where
    V: PreciseStatisticsOps<Statistics<T>, Result = Statistics<T>>,
{
    type Result = [Statistics<T>; 2];

    fn statistics_prec(&self) -> Self::Result {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics_prec();
            result.push(res);
        }

        result.into_fixed_length()
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> PreciseStatisticsSplitOps<T> for Matrix2xN<V, S, T>
where
    V: PreciseStatisticsSplitOps<Statistics<T>, Result = StatsVec<Statistics<T>>>,
{
    type Result = [StatsVec<Statistics<T>>; 2];

    fn statistics_split_prec(&self, len: usize) -> ScalarResult<Self::Result> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = try!(v.statistics_split_prec(len));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> PreciseStatisticsOps<T> for Matrix3xN<V, S, T>
where
    V: PreciseStatisticsOps<Statistics<T>, Result = Statistics<T>>,
{
    type Result = [Statistics<T>; 3];

    fn statistics_prec(&self) -> Self::Result {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics_prec();
            result.push(res);
        }

        result.into_fixed_length()
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> PreciseStatisticsSplitOps<T> for Matrix3xN<V, S, T>
where
    V: PreciseStatisticsSplitOps<Statistics<T>, Result = StatsVec<Statistics<T>>>,
{
    type Result = [StatsVec<Statistics<T>>; 3];

    fn statistics_split_prec(&self, len: usize) -> ScalarResult<Self::Result> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = try!(v.statistics_split_prec(len));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> PreciseStatisticsOps<T> for Matrix4xN<V, S, T>
where
    V: PreciseStatisticsOps<Statistics<T>, Result = Statistics<T>>,
{
    type Result = [Statistics<T>; 4];

    fn statistics_prec(&self) -> Self::Result {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = v.statistics_prec();
            result.push(res);
        }

        result.into_fixed_length()
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber> PreciseStatisticsSplitOps<T> for Matrix4xN<V, S, T>
where
    V: PreciseStatisticsSplitOps<Statistics<T>, Result = StatsVec<Statistics<T>>>,
{
    type Result = [StatsVec<Statistics<T>>; 4];

    fn statistics_split_prec(&self, len: usize) -> ScalarResult<Self::Result> {
        let mut result = Vec::with_capacity(self.col_len());
        for v in self.rows() {
            let res = try!(v.statistics_split_prec(len));
            result.push(res);
        }

        Ok(result.into_fixed_length())
    }
}

impl<S: ToSlice<T>, V: Vector<T>, T: RealNumber, O: RealNumber> PreciseSumOps<Vec<O>>
    for MatrixMxN<V, S, T>
where
    V: PreciseSumOps<O>,
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
where
    V: PreciseSumOps<O>,
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
where
    V: PreciseSumOps<O>,
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
where
    V: PreciseSumOps<O>,
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
