use super::super::{
    ComplexNumberSpace, Domain, DspVec, ErrorReason, NumberSpace, RealNumberSpace, ScalarResult,
    ToSlice, Vector,
};
use array_to_complex;
use arrayvec::ArrayVec;
use multicore_support::*;
use numbers::*;
use simd_extensions::*;

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Debug)]
/// Statistics about numeric data
pub struct Statistics<T> {
    /// Sum of all values
    pub sum: T,
    /// How many numbers have been considered for the stats
    pub count: usize,
    /// Average value
    pub average: T,
    /// Root-mean-square or rms over all values.
    pub rms: T,
    /// The smallest value.
    pub min: T,
    /// The index of the smallest value.
    pub min_index: usize,
    /// The largest value.
    pub max: T,
    /// The index of the largest value.
    pub max_index: usize,
}

/// The maximum `len` for any of the `*split` methods.
pub const STATS_VEC_CAPACTIY: usize = 16;

/// Alias for a vector of any statistical information.
pub type StatsVec<T> = ArrayVec<[T; STATS_VEC_CAPACTIY]>;

/// This trait offers operations to calculate statistics about the data in a type.
pub trait StatisticsOps<T> {
    type Result;

    /// Calculates the statistics of the data.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// # use num_complex::Complex32;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).to_complex_time_vec();
    /// let result = vector.statistics();
    /// assert_eq!(result.sum, Complex32::new(9.0, 12.0));
    /// assert_eq!(result.count, 3);
    /// assert_eq!(result.average, Complex32::new(3.0, 4.0));
    /// assert!((result.rms - Complex32::new(3.4027193, 4.3102784)).norm() < 1e-4);
    /// assert_eq!(result.min, Complex32::new(1.0, 2.0));
    /// assert_eq!(result.min_index, 0);
    /// assert_eq!(result.max, Complex32::new(5.0, 6.0));
    /// assert_eq!(result.max_index, 2);
    /// }
    /// ```
    fn statistics(&self) -> Self::Result;
}

/// This trait offers operations to calculate statistics about the data in a type.
pub trait StatisticsSplitOps<T> {
    type Result;

    /// Calculates the statistics of the data contained in the vector as if the vector would
    /// have been split into `len` pieces. `self.len` should be dividable by
    /// `len` without a remainder,
    /// but this isn't enforced by the implementation.
    /// For implementation reasons `len <= 16` must be true.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// # use num_complex::Complex32;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).to_complex_time_vec();
    /// let result = vector.statistics_split(2).expect("Ignoring error handling in examples");
    /// assert_eq!(result[0].sum, Complex32::new(6.0, 8.0));
    /// assert_eq!(result[1].sum, Complex32::new(3.0, 4.0));
    /// }
    /// ```
    fn statistics_split(&self, len: usize) -> ScalarResult<Self::Result>;
}

/// Offers operations to calculate the sum or the sum of squares.
pub trait SumOps<T>: Sized
where
    T: Sized,
{
    /// Calculates the sum of the data contained in the vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// # use num_complex::Complex32;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).to_complex_time_vec();
    /// let result = vector.sum();
    /// assert_eq!(result, Complex32::new(9.0, 12.0));
    /// }
    /// ```
    fn sum(&self) -> T;

    /// Calculates the sum of the squared data contained in the vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// # use num_complex::Complex64;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).to_complex_time_vec();
    /// let result = vector.sum_sq();
    /// assert_eq!(result, Complex64::new(-21.0, 88.0));
    /// }
    /// ```
    fn sum_sq(&self) -> T;
}

/// Operations on statistics.
pub trait Stats<T>: Sized {
    /// Creates an empty statistics struct.
    fn empty() -> Self;
    /// Creates a vector of empty statistics structs.
    fn empty_vec(len: usize) -> StatsVec<Self>;
    /// Creates a statistics struct which resembles an invalid result.
    fn invalid() -> Self;
    /// Merges several statistics into one.
    fn merge(stats: &[Self]) -> Self;
    /// Merges several vectors of statistics into one vector.
    fn merge_cols(stats: &[StatsVec<Self>]) -> StatsVec<Self>;
    /// Adds a new value to the statistics, all statistic fields get updated.
    fn add(&mut self, elem: T, index: usize);
}

macro_rules! impl_common_stats {
    () => {
        fn merge_cols(stats: &[StatsVec<Self>]) -> StatsVec<Self> {
            if stats.is_empty() {
                return StatsVec::new();
            }
        
            let len = stats[0].len();
            let mut results = StatsVec::new();
            for i in 0..len {
                let mut reordered = StatsVec::new();
                for s in stats.iter() {
                    reordered.push(s[i]);
                }
        
                let merged = Statistics::merge(&reordered);
                results.push(merged);
            }
            results
        }
        
        fn empty_vec(len: usize) -> StatsVec<Self> {
            let mut results = StatsVec::new();
            for _ in 0..len {
                results.push(Statistics::empty());
            }
            results
        }
    };
}

impl<T> Stats<T> for Statistics<T>
where
    T: RealNumber,
{
    fn empty() -> Self {
        Statistics {
            sum: T::zero(),
            count: 0,
            average: T::zero(),
            min: T::infinity(),
            max: T::neg_infinity(),
            rms: T::zero(), // this field therefore has a different meaning inside this function
            min_index: 0,
            max_index: 0,
        }
    }

    fn invalid() -> Self {
        Statistics {
            sum: T::zero(),
            count: 0,
            average: T::nan(),
            min: T::nan(),
            max: T::nan(),
            rms: T::nan(),
            min_index: 0,
            max_index: 0,
        }
    }

    fn merge(stats: &[Statistics<T>]) -> Statistics<T> {
        if stats.is_empty() {
            return Statistics::<T>::invalid();
        }

        let mut sum = T::zero();
        let mut max = stats[0].max;
        let mut min = stats[0].min;
        let mut max_index = stats[0].max_index;
        let mut min_index = stats[0].min_index;
        let mut sum_squared = T::zero();
        let mut len = 0;
        for stat in stats {
            sum = sum + stat.sum;
            len += stat.count;
            sum_squared = sum_squared + stat.rms; // We stored sum_squared in the field rms
            if stat.max > max {
                max = stat.max;
                max_index = stat.max_index;
            } else if stat.min < min {
                min = stat.min;
                min_index = stat.min_index;
            }
        }

        Statistics {
            sum,
            count: len,
            average: sum / (T::from(len).unwrap()),
            min,
            max,
            rms: (sum_squared / (T::from(len).unwrap())).sqrt(),
            min_index,
            max_index,
        }
    }

    impl_common_stats!();

    #[inline]
    fn add(&mut self, elem: T, index: usize) {
        self.sum = self.sum + elem;
        self.count += 1;
        self.rms = self.rms + elem * elem;
        if elem > self.max {
            self.max = elem;
            self.max_index = index;
        }
        if elem < self.min {
            self.min = elem;
            self.min_index = index;
        }
    }
}

impl<T> Stats<Complex<T>> for Statistics<Complex<T>>
where
    T: RealNumber,
{
    fn empty() -> Self {
        Statistics {
            sum: Complex::<T>::new(T::zero(), T::zero()),
            count: 0,
            average: Complex::<T>::new(T::zero(), T::zero()),
            min: Complex::<T>::new(T::infinity(), T::infinity()),
            max: Complex::<T>::new(T::zero(), T::zero()),
            // the rms field has a different meaning inside this function
            rms: Complex::<T>::new(T::zero(), T::zero()),
            min_index: 0,
            max_index: 0,
        }
    }

    fn invalid() -> Self {
        Statistics {
            sum: Complex::<T>::new(T::zero(), T::zero()),
            count: 0,
            average: Complex::<T>::new(T::nan(), T::nan()),
            min: Complex::<T>::new(T::nan(), T::nan()),
            max: Complex::<T>::new(T::nan(), T::nan()),
            rms: Complex::<T>::new(T::nan(), T::nan()),
            min_index: 0,
            max_index: 0,
        }
    }

    fn merge(stats: &[Statistics<Complex<T>>]) -> Statistics<Complex<T>> {
        if stats.is_empty() {
            return Statistics::<Complex<T>>::invalid();
        }

        let mut sum = Complex::<T>::new(T::zero(), T::zero());
        let mut max = stats[0].max;
        let mut min = stats[0].min;
        let mut count = 0;
        let mut max_index = stats[0].max_index;
        let mut min_index = stats[0].min_index;
        let mut max_norm = max.norm();
        let mut min_norm = min.norm();
        let mut sum_squared = Complex::<T>::new(T::zero(), T::zero());
        for stat in stats {
            sum = sum + stat.sum;
            count += stat.count;
            sum_squared = sum_squared + stat.rms; // We stored sum_squared in the field rms
            if stat.max.norm() > max_norm {
                max = stat.max;
                max_norm = max.norm();
                max_index = stat.max_index;
            } else if stat.min.norm() < min_norm {
                min = stat.min;
                min_norm = min.norm();
                min_index = stat.min_index;
            }
        }

        Statistics {
            sum,
            count,
            average: sum / (T::from(count).unwrap()),
            min,
            max,
            rms: (sum_squared / (T::from(count).unwrap())).sqrt(),
            min_index,
            max_index,
        }
    }

    impl_common_stats!();

    #[inline]
    fn add(&mut self, elem: Complex<T>, index: usize) {
        self.sum = self.sum + elem;
        self.count += 1;
        self.rms = self.rms + elem * elem;
        if elem.norm() > self.max.norm() {
            self.max = elem;
            self.max_index = index;
        }
        if elem.norm() < self.min.norm() {
            self.min = elem;
            self.min_index = index;
        }
    }
}

impl<S, T, N, D> StatisticsOps<T> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: RealNumberSpace,
    D: Domain,
{
    type Result = Statistics<T>;

    fn statistics(&self) -> Statistics<T> {
        let data_length = self.len();
        let array = self.data.to_slice();
        let chunks = Chunk::get_chunked_results(
            Complexity::Small,
            self.multicore_settings,
            &array[0..data_length],
            1,
            (),
            |array, range, _arg| {
                let mut stats = Statistics::empty();
                let mut j = range.start;
                for num in array {
                    stats.add(*num, j);
                    j += 1;
                }
                stats
            },
        );

        Statistics::merge(&chunks[..])
    }
}

impl<S, T, N, D> StatisticsSplitOps<T> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: RealNumberSpace,
    D: Domain,
{
    type Result = StatsVec<Statistics<T>>;

    fn statistics_split(&self, len: usize) -> ScalarResult<StatsVec<Statistics<T>>> {
        if len == 0 {
            return Ok(StatsVec::new());
        }

        if len > STATS_VEC_CAPACTIY {
            return Err(ErrorReason::InvalidArgumentLength);
        }

        let data_length = self.len();
        let array = self.data.to_slice();
        let chunks = Chunk::get_chunked_results(
            Complexity::Small,
            self.multicore_settings,
            &array[0..data_length],
            1,
            len,
            |array, range, len| {
                let mut results = Statistics::empty_vec(len);
                let mut j = range.start;
                for num in array {
                    let stats = &mut results[j % len];
                    stats.add(*num, j / len);
                    j += 1;
                }

                results
            },
        );

        Ok(Statistics::merge_cols(&chunks[..]))
    }
}

impl<S, T, N, D> DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn sum_real<Reg: SimdGeneric<T>>(&self, _: RegType<Reg>) -> T {
        let data_length = self.len();
        let array = self.data.to_slice();
        let partition = Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let sum = {
            let chunks = Chunk::get_chunked_results(
                Complexity::Small,
                self.multicore_settings,
                partition.center(array),
                Reg::LEN,
                (),
                move |array, _, _| {
                    let array = Reg::array_to_regs(array);
                    let mut sum = Reg::splat(T::zero());
                    for reg in array {
                        sum = sum + *reg;
                    }
                    sum
                },
            );
            (&chunks[..])
                .iter()
                .map(|v| v.sum_real())
                .fold(T::zero(), |a, b| a + b)
        };

        sum +
            partition.edge_iter(array).fold(T::zero(), |sum, x| sum + *x)
    }

    fn sum_sq_real<Reg: SimdGeneric<T>>(&self, _: RegType<Reg>) -> T {
        let data_length = self.len();
        let array = self.data.to_slice();
        let partition = Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let sum =  {
            let chunks = Chunk::get_chunked_results(
                Complexity::Small,
                self.multicore_settings,
                partition.center(array),
                Reg::LEN,
                (),
                move |array, _, _| {
                    let array = Reg::array_to_regs(array);
                    let mut sum = Reg::splat(T::zero());
                    for reg in array {
                        sum = sum + *reg * *reg;
                    }
                    sum
                },
            );
            (&chunks[..])
                .iter()
                .map(|v| v.sum_real())
                .fold(T::zero(), |a, b| a + b)
        };

        sum +
            partition.edge_iter(array).fold(T::zero(), |sum, x| sum + *x * *x)
    }

    fn sum_complex<Reg: SimdGeneric<T>>(&self, _: RegType<Reg>) -> Complex<T> {
        let data_length = self.len();
        let array = self.data.to_slice();
        let partition = Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let sum = {
            let chunks = Chunk::get_chunked_results(
                Complexity::Small,
                self.multicore_settings,
                partition.center(array),
                Reg::LEN,
                (),
                move |array, _, _| {
                    let array = Reg::array_to_regs(array);
                    let mut sum = Reg::splat(T::zero());
                    for reg in array {
                        sum = sum + *reg;
                    }
                    sum
                },
            );
            (&chunks[..])
                .iter()
                .map(|v| v.sum_complex())
                .fold(Complex::<T>::new(T::zero(), T::zero()), |acc, x| acc + x)
        };

        sum +
            partition.cedge_iter(array_to_complex(array)).fold(Complex::<T>::zero(), |sum, x| sum + *x)
    }

    fn sum_sq_complex<Reg: SimdGeneric<T>>(&self, _: RegType<Reg>) -> Complex<T> {
        let data_length = self.len();
        let array = self.data.to_slice();
        let partition = Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let sum = {
            let chunks = Chunk::get_chunked_results(
                Complexity::Small,
                self.multicore_settings,
                partition.center(array),
                Reg::LEN,
                (),
                move |array, _, _| {
                    let array = Reg::array_to_regs(array);
                    let mut sum = Reg::splat(T::zero());
                    for reg in array {
                        sum = sum + reg.mul_complex(*reg);
                    }
                    sum
                },
            );
            (&chunks[..])
                .iter()
                .map(|v| v.sum_complex())
                .fold(Complex::<T>::new(T::zero(), T::zero()), |acc, x| acc + x)
        };

        sum +
            partition.cedge_iter(array_to_complex(array)).fold(Complex::<T>::zero(), |sum, x| sum + *x * *x)
    }
}

impl<S, T, N, D> SumOps<T> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: RealNumberSpace,
    D: Domain,
{
    fn sum(&self) -> T {
        sel_reg!(self.sum_real::<T>())
    }

    fn sum_sq(&self) -> T {
        sel_reg!(self.sum_sq_real::<T>())
    }
}

impl<S, T, N, D> StatisticsOps<Complex<T>> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
{
    type Result = Statistics<Complex<T>>;

    fn statistics(&self) -> Statistics<Complex<T>> {
        let data_length = self.len();
        let array = self.data.to_slice();
        let chunks = Chunk::get_chunked_results(
            Complexity::Small,
            self.multicore_settings,
            &array[0..data_length],
            2,
            (),
            |array, range, _arg| {
                let mut stat = Statistics::<Complex<T>>::empty();
                let mut j = range.start / 2;
                let array = array_to_complex(array);
                for num in array {
                    stat.add(*num, j);
                    j += 1;
                }
                stat
            },
        );

        Statistics::merge(&chunks[..])
    }
}

impl<S, T, N, D> StatisticsSplitOps<Complex<T>> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
{
    type Result = StatsVec<Statistics<Complex<T>>>;

    fn statistics_split(&self, len: usize) -> ScalarResult<StatsVec<Statistics<Complex<T>>>> {
        if len == 0 {
            return Ok(StatsVec::new());
        }

        if len > STATS_VEC_CAPACTIY {
            return Err(ErrorReason::InvalidArgumentLength);
        }

        let data_length = self.len();
        let array = self.data.to_slice();
        let chunks = Chunk::get_chunked_results(
            Complexity::Small,
            self.multicore_settings,
            &array[0..data_length],
            2,
            len,
            |array, range, len| {
                let mut results = Statistics::<Complex<T>>::empty_vec(len);
                let mut j = range.start / 2;
                let array = array_to_complex(array);
                for num in array {
                    let stat = &mut results[j % len];
                    stat.add(*num, j / len);
                    j += 1;
                }

                results
            },
        );

        Ok(Statistics::merge_cols(&chunks[..]))
    }
}

impl<S, T, N, D> SumOps<Complex<T>> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
{
    fn sum(&self) -> Complex<T> {
        sel_reg!(self.sum_complex::<T>())
    }

    fn sum_sq(&self) -> Complex<T> {
        sel_reg!(self.sum_sq_complex::<T>())
    }
}
