use RealNumber;
use num::Complex;
use multicore_support::*;
use simd_extensions::*;
use super::super::{array_to_complex, Vector, DspVec, ToSlice, Domain, RealNumberSpace,
                   ComplexNumberSpace};

#[repr(C)]
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
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

/// This trait offers operations to calculate statistics about the data in a type.
pub trait StatisticsOps<T>: Sized
    where T: Sized
{
    /// Calculates the statistics of the data.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// # use num::complex::Complex32;
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
    fn statistics(&self) -> T;


    /// Calculates the statistics of the data contained in the vector as if the vector would
    /// have been split into `len` pieces. `self.len` should be dividable by
    /// `len` without a remainder,
    /// but this isn't enforced by the implementation.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// # use num::complex::Complex32;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).to_complex_time_vec();
    /// let result = vector.statistics_split(2);
    /// assert_eq!(result[0].sum, Complex32::new(6.0, 8.0));
    /// assert_eq!(result[1].sum, Complex32::new(3.0, 4.0));
    /// }
    /// ```
    fn statistics_split(&self, len: usize) -> Vec<T>;

    #[deprecated(since="0.4.1", note="Use `statistics_split` which is the same method but comes with correct spelling")]
    fn statistics_splitted(&self, len: usize) -> Vec<T>;
}

/// Offers operations to calculate the sum or the sum of squares.
pub trait SumOps<T>: Sized
    where T: Sized
{
    /// Calculates the sum of the data contained in the vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// # use num::complex::Complex32;
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
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// # use num::complex::Complex64;
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
    fn empty_vec(len: usize) -> Vec<Self>;
    /// Creates a statistics struct which resembles an invalid result.
    fn invalid() -> Self;
    /// Merges several statistics into one.
    fn merge(stats: &[Self]) -> Self;
    /// Merges several vectors of statistics into one vector.
    fn merge_cols(stats: &[Vec<Self>]) -> Vec<Self>;
    /// Adds a new value to the statistics, all statistic fields get updated.
    fn add(&mut self, elem: T, index: usize);
}

macro_rules! impl_common_stats {
    () => {
        fn merge_cols(stats: &[Vec<Self>]) -> Vec<Self> {
            if stats.is_empty() {
                return Vec::new();
            }

            let len = stats[0].len();
            let mut results = Vec::with_capacity(len);
            for i in 0..len {
                let mut reordered = Vec::with_capacity(stats.len());
                for s in stats.iter()
                {
                    reordered.push(s[i]);
                }

                let merged = Statistics::merge(&reordered);
                results.push(merged);
            }
            results
        }

        fn empty_vec(len: usize) -> Vec<Self> {
            let mut results = Vec::with_capacity(len);
            for _ in 0..len {
                results.push(Statistics::empty());
            }
            results
        }
    }
}

impl<T> Stats<T> for Statistics<T>
    where T: RealNumber
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
        if stats.len() == 0 {
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
            sum: sum,
            count: len,
            average: sum / (T::from(len).unwrap()),
            min: min,
            max: max,
            rms: (sum_squared / (T::from(len).unwrap())).sqrt(),
            min_index: min_index,
            max_index: max_index,
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
    where T: RealNumber
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
        if stats.len() == 0 {
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
            sum: sum,
            count: count,
            average: sum / (T::from(count).unwrap()),
            min: min,
            max: max,
            rms: (sum_squared / (T::from(count).unwrap())).sqrt(),
            min_index: min_index,
            max_index: max_index,
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

impl<S, T, N, D> StatisticsOps<Statistics<T>> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain
{
    fn statistics(&self) -> Statistics<T> {
        let data_length = self.len();
        let array = self.data.to_slice();
        let chunks = Chunk::get_chunked_results(Complexity::Small,
                                                &self.multicore_settings,
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
        });

        Statistics::merge(&chunks)
    }

    fn statistics_split(&self, len: usize) -> Vec<Statistics<T>> {
        if len == 0 {
            return Vec::new();
        }

        let data_length = self.len();
        let array = self.data.to_slice();
        let chunks = Chunk::get_chunked_results(Complexity::Small,
                                                &self.multicore_settings,
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
        });

        Statistics::merge_cols(&chunks)
    }

    fn statistics_splitted(&self, len: usize) -> Vec<Statistics<T>> {
        self.statistics_split(len)
    }
}

impl<S, T, N, D> SumOps<T> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain
{
    fn sum(&self) -> T {
        let data_length = self.len();
        let array = self.data.to_slice();
        let (scalar_left, scalar_right, vectorization_length) =
            T::Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let mut sum = if vectorization_length > 0 {
            let chunks = Chunk::get_chunked_results(Complexity::Small,
                                                    &self.multicore_settings,
                                                    &array[scalar_left..vectorization_length],
                                                    T::Reg::len(),
                                                    (),
                                                    move |array, _, _| {
                let array = T::Reg::array_to_regs(array);
                let mut sum = T::Reg::splat(T::zero());
                for reg in array {
                    sum = sum + *reg;
                }
                sum
            });
            chunks.iter()
                .map(|v| v.sum_real())
                .fold(T::zero(), |a, b| a + b)
        } else {
            T::zero()
        };
        for num in &array[0..scalar_left] {
            sum = sum + *num;
        }
        for num in &array[scalar_right..data_length] {
            sum = sum + *num;
        }
        sum
    }

    fn sum_sq(&self) -> T {
        let data_length = self.len();
        let array = self.data.to_slice();
        let (scalar_left, scalar_right, vectorization_length) =
            T::Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let mut sum = if vectorization_length > 0 {
            let chunks = Chunk::get_chunked_results(Complexity::Small,
                                                    &self.multicore_settings,
                                                    &array[scalar_left..vectorization_length],
                                                    T::Reg::len(),
                                                    (),
                                                    move |array, _, _| {
                let array = T::Reg::array_to_regs(array);
                let mut sum = T::Reg::splat(T::zero());
                for reg in array {
                    sum = sum + *reg * *reg;
                }
                sum
            });
            chunks.iter()
                .map(|v| v.sum_real())
                .fold(T::zero(), |a, b| a + b)
        } else {
            T::zero()
        };
        for num in &array[0..scalar_left] {
            sum = sum + *num * *num;
        }
        for num in &array[scalar_right..data_length] {
            sum = sum + *num * *num;
        }
        sum
    }
}

impl<S, T, N, D> StatisticsOps<Statistics<Complex<T>>> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: Domain
{
    fn statistics(&self) -> Statistics<Complex<T>> {
        let data_length = self.len();
        let array = self.data.to_slice();
        let chunks = Chunk::get_chunked_results(Complexity::Small,
                                                &self.multicore_settings,
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
        });

        Statistics::merge(&chunks)
    }

    fn statistics_split(&self, len: usize) -> Vec<Statistics<Complex<T>>> {
        if len == 0 {
            return Vec::new();
        }

        let data_length = self.len();
        let array = self.data.to_slice();
        let chunks = Chunk::get_chunked_results(Complexity::Small,
                                                &self.multicore_settings,
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
        });

        Statistics::merge_cols(&chunks)
    }

    fn statistics_splitted(&self, len: usize) -> Vec<Statistics<Complex<T>>> {
        self.statistics_split(len)
    }
}

impl<S, T, N, D> SumOps<Complex<T>> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: Domain
{
    fn sum(&self) -> Complex<T> {
        let data_length = self.len();
        let array = self.data.to_slice();
        let (scalar_left, scalar_right, vectorization_length) =
            T::Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let mut sum = if vectorization_length > 0 {
            let chunks = Chunk::get_chunked_results(Complexity::Small,
                                                    &self.multicore_settings,
                                                    &array[scalar_left..vectorization_length],
                                                    T::Reg::len(),
                                                    (),
                                                    move |array, _, _| {
                let array = T::Reg::array_to_regs(array);
                let mut sum = T::Reg::splat(T::zero());
                for reg in array {
                    sum = sum + *reg;
                }
                sum
            });
            chunks.iter()
                .map(|v| v.sum_complex())
                .fold(Complex::<T>::new(T::zero(), T::zero()), |acc, x| acc + x)
        } else {
            Complex::<T>::new(T::zero(), T::zero())
        };
        for num in array_to_complex(&array[0..scalar_left]) {
            sum = sum + *num;
        }
        for num in array_to_complex(&array[scalar_right..data_length]) {
            sum = sum + *num;
        }
        sum
    }

    fn sum_sq(&self) -> Complex<T> {
        let data_length = self.len();
        let array = self.data.to_slice();
        let (scalar_left, scalar_right, vectorization_length) =
            T::Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let mut sum = if vectorization_length > 0 {
            let chunks = Chunk::get_chunked_results(Complexity::Small,
                                                    &self.multicore_settings,
                                                    &array[scalar_left..vectorization_length],
                                                    T::Reg::len(),
                                                    (),
                                                    move |array, _, _| {
                let array = T::Reg::array_to_regs(array);
                let mut sum = T::Reg::splat(T::zero());
                for reg in array {
                    sum = sum + reg.mul_complex(*reg);
                }
                sum
            });
            chunks.iter()
                .map(|v| v.sum_complex())
                .fold(Complex::<T>::new(T::zero(), T::zero()), |acc, x| acc + x)
        } else {
            Complex::<T>::new(T::zero(), T::zero())
        };
        for num in array_to_complex(&array[0..scalar_left]) {
            sum = sum + *num * *num;
        }
        for num in array_to_complex(&array[scalar_right..data_length]) {
            sum = sum + *num * *num;
        }
        sum
    }
}
