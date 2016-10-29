use RealNumber;
use num::{Complex, Zero};
use num::complex::{Complex64};
use multicore_support::*;
use super::super::{array_to_complex, Vector, DspVec, ToSlice, Domain, RealNumberSpace,
                   ComplexNumberSpace, Statistics, Stats};

pub trait PreciseStatisticsOps<T>: Sized
    where T: Sized
{
    /// Calculates the statistics of the data contained in the vector.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// # use num::complex::Complex64;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).to_complex_time_vec();
    /// let result = vector.statistics_prec();
    /// assert_eq!(result.sum, Complex64::new(9.0, 12.0));
    /// assert_eq!(result.count, 3);
    /// assert_eq!(result.average, Complex64::new(3.0, 4.0));
    /// assert!((result.rms - Complex64::new(3.4027193, 4.3102784)).norm() < 1e-4);
    /// assert_eq!(result.min, Complex64::new(1.0, 2.0));
    /// assert_eq!(result.min_index, 0);
    /// assert_eq!(result.max, Complex64::new(5.0, 6.0));
    /// assert_eq!(result.max_index, 2);
    /// }
    /// ```
    fn statistics_prec(&self) -> T;

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
    /// # use num::complex::Complex64;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).to_complex_time_vec();
    /// let result = vector.statistics_splitted_prec(2);
    /// assert_eq!(result[0].sum, Complex64::new(6.0, 8.0));
    /// assert_eq!(result[1].sum, Complex64::new(3.0, 4.0));
    /// }
    /// ```
    fn statistics_splitted_prec(&self, len: usize) -> Vec<T>;
}

pub trait PreciseSumOps<T>: Sized
    where T: Sized
{
    /// Calculates the sum of the data contained in the vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp_vector;
    /// # use num::complex::Complex64;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).to_complex_time_vec();
    /// let result = vector.sum_prec();
    /// assert_eq!(result, Complex64::new(9.0, 12.0));
    /// }
    /// ```
    fn sum_prec(&self) -> T;

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
    /// let result = vector.sum_sq_prec();
    /// assert_eq!(result, Complex64::new(-21.0, 88.0));
    /// }
    /// ```
    fn sum_sq_prec(&self) -> T;
}

pub trait PreciseStats<T>: Sized {
    fn merge_prec(stats: &[Self]) -> Self;
    fn merge_cols_prec(stats: &[Vec<Self>]) -> Vec<Self>;
    fn add_prec(&mut self, elem: T, index: usize);
}

impl<S, N, D> PreciseStatisticsOps<Statistics<f64>> for DspVec<S, f32, N, D>
    where S: ToSlice<f32>,
          N: RealNumberSpace,
          D: Domain
{
    fn statistics_prec(&self) -> Statistics<f64> {
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
                stats.add(*num as f64, j);
                j += 1;
            }
            stats
        });

        Statistics::merge_prec(&chunks)
    }

    fn statistics_splitted_prec(&self, len: usize) -> Vec<Statistics<f64>> {
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
                stats.add_prec(*num as f64, j / len);
                j += 1;
            }

            results
        });

        Statistics::merge_cols_prec(&chunks)
    }
}

impl<S, N, D> PreciseSumOps<f64> for DspVec<S, f32, N, D>
    where S: ToSlice<f32>,
          N: RealNumberSpace,
          D: Domain
{
    fn sum_prec(&self) -> f64 {
        let data_length = self.len();
        let array = self.data.to_slice();
        let chunks = Chunk::get_chunked_results(Complexity::Small,
                                                &self.multicore_settings,
                                                &array[0..data_length],
                                                1,
                                                (),
                                                move |array, _, _| {
            let mut sum = 0.0;
            for n in array {
                sum = sum + *n as f64;
            }
            sum
        });
        chunks.iter()
            .fold(0.0, |a, b| a + b)
    }

    fn sum_sq_prec(&self) -> f64 {
        let data_length = self.len();
        let array = self.data.to_slice();
        let chunks = Chunk::get_chunked_results(Complexity::Small,
                                                &self.multicore_settings,
                                                &array[0..data_length],
                                                1,
                                                (),
                                                move |array, _, _| {
            let mut sum = 0.0;
            for n in array {
                let t = *n as f64;
                sum = sum + t * t;
            }
            sum
        });
        chunks.iter()
            .fold(0.0, |a, b| a + b)
    }
}

impl<S, T, N, D> PreciseStatisticsOps<Statistics<Complex<T>>> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: Domain
{
    fn statistics_prec(&self) -> Statistics<Complex<T>> {
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

        Statistics::merge_prec(&chunks)
    }

    fn statistics_splitted_prec(&self, len: usize) -> Vec<Statistics<Complex<T>>> {
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

        Statistics::merge_cols_prec(&chunks)
    }
}

impl<S, N, D> PreciseSumOps<Complex<f64>> for DspVec<S, f32, N, D>
    where S: ToSlice<f32>,
          N: ComplexNumberSpace,
          D: Domain
{
    fn sum_prec(&self) -> Complex<f64> {
        let data_length = self.len();
        let array = self.data.to_slice();
        let chunks = Chunk::get_chunked_results(Complexity::Small,
                                                &self.multicore_settings,
                                                &array[0..data_length],
                                                2,
                                                (),
                                                move |array, _, _| {
            let mut sum = Complex64::zero();
            let array = array_to_complex(&array[0..array.len()]);
            for n in array {
                sum = sum + Complex64::new(n.re as f64, n.im as f64);
            }
            sum
        });
        chunks.iter()
            .fold(Complex64::zero(), |a, b| a + b)
    }

    fn sum_sq_prec(&self) -> Complex<f64> {
        let data_length = self.len();
        let array = self.data.to_slice();
        let chunks = Chunk::get_chunked_results(Complexity::Small,
                                                &self.multicore_settings,
                                                &array[0..data_length],
                                                2,
                                                (),
                                                move |array, _, _| {
            let mut sum = Complex::<f64>::zero();
            let array = array_to_complex(&array[0..array.len()]);
            for n in array {
                let t = Complex64::new(n.re as f64, n.im as f64);
                sum = sum + t * t;
            }
            sum
        });
        chunks.iter()
            .fold(Complex64::zero(), |a, b| a + b)
    }
}

macro_rules! impl_common_stats {
    () => {
        fn merge_cols_prec(stats: &[Vec<Self>]) -> Vec<Self> {
            if stats.len() == 0 {
                return Vec::new();
            }

            let len = stats[0].len();
            let mut results = Vec::with_capacity(len);
            for i in 0..len {
                let mut reordered = Vec::with_capacity(stats.len());
                for j in 0..stats.len()
                {
                    reordered.push(stats[j][i]);
                }

                let merged = Statistics::merge_prec(&reordered);
                results.push(merged);
            }
            results
        }
    }
}

impl<T> PreciseStats<T> for Statistics<T>
    where T: RealNumber
{
    fn merge_prec(stats: &[Statistics<T>]) -> Statistics<T> {
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
    fn add_prec(&mut self, elem: T, index: usize) {
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

impl<T> PreciseStats<Complex<T>> for Statistics<Complex<T>>
    where T: RealNumber
{
    fn merge_prec(stats: &[Statistics<Complex<T>>]) -> Statistics<Complex<T>> {
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
            count = count + stat.count;
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
    fn add_prec(&mut self, elem: Complex<T>, index: usize) {
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