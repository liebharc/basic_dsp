//! Types around a convolution, see also <https://en.wikipedia.org/wiki/Convolution>.
//!
//! Convolutions in this library can be defined in time or frequency domain. In
//! frequency domain the convolution is automatically transformed into a multiplication
//! which is the analog operation to a convolution in time domain.
use super::FixedLenBuffer;
use crate::inline_vector::InlineVector;
use crate::numbers::*;
use crate::vector_types::*;
use num_complex::{Complex32, Complex64};
use std::marker::PhantomData;
use std::ops::*;

/// A convolution function in time domain and real number space
pub trait RealImpulseResponse<T>: Sync
where
    T: RealNumber,
{
    /// Indicates whether this function is symmetric around 0 or not.
    /// Symmetry is defined as `self.calc(x) == self.calc(-x)`.
    fn is_symmetric(&self) -> bool;

    /// Calculates the convolution for a data point
    fn calc(&self, x: T) -> T;
}

/// A convolution function in frequency domain and real number space
pub trait RealFrequencyResponse<T>: Sync
where
    T: RealNumber,
{
    /// Indicates whether this function is symmetric around 0 or not.
    /// Symmetry is defined as `self.calc(x) == self.calc(-x)`.
    fn is_symmetric(&self) -> bool;

    /// Calculates the convolution for a data point
    fn calc(&self, x: T) -> T;
}

/// A convolution function in time domain and complex number space
pub trait ComplexImpulseResponse<T>: Sync
where
    T: RealNumber,
{
    /// Indicates whether this function is symmetric around 0 or not.
    /// Symmetry is defined as `self.calc(x) == self.calc(-x)`.
    fn is_symmetric(&self) -> bool;

    /// Calculates the convolution for a data point
    fn calc(&self, x: T) -> Complex<T>;
}

/// A convolution function in frequency domain and complex number space
pub trait ComplexFrequencyResponse<T>: Sync
where
    T: RealNumber,
{
    /// Indicates whether this function is symmetric around 0 or not.
    /// Symmetry is defined as `self.calc(x) == self.calc(-x)`.
    fn is_symmetric(&self) -> bool;

    /// Calculates the convolution for a data point
    fn calc(&self, x: T) -> Complex<T>;
}

macro_rules! define_real_lookup_table {
    ($($name: ident);*) => {
        $(
            /// Allows to create a lookup table with linear interpolation between table points.
            /// This usually speeds up a convolution and sacrifices accuracy.
            pub struct $name<T>
                where T: RealNumber {
                table: InlineVector<T>,
                delta: T,
                is_symmetric: bool
            }

            impl<T> $name<T>
                where T: RealNumber {

                /// Allows to inspect the generated lookup table
                pub fn table(&self) -> &[T] {
                    &self.table[..]
                }

                /// Gets the delta value which determines the resolution
                pub fn delta(&self) -> T {
                    self.delta
                }
            }
        )*
    }
}
define_real_lookup_table!(RealTimeLinearTableLookup; RealFrequencyLinearTableLookup);

macro_rules! define_complex_lookup_table {
    ($($name: ident);*) => {
        $(
            /// Allows to create a lookup table with linear interpolation between table points.
            /// This usually speeds up a convolution and sacrifices accuracy.
            pub struct $name<T>
                where T: RealNumber {
                table: InlineVector<Complex<T>>,
                delta: T,
                is_symmetric: bool
            }

            impl<T> $name<T>
                where T: RealNumber {

                /// Allows to inspect the generated lookup table
                pub fn table(&self) -> &[Complex<T>] {
                    &self.table[..]
                }

                /// Gets the delta value which determines the resolution
                pub fn delta(&self) -> T {
                    self.delta
                }
            }
        )*
    }
}
define_complex_lookup_table!(ComplexTimeLinearTableLookup; ComplexFrequencyLinearTableLookup);

/// Linear interpolation between two points.
fn linear_interpolation_between_bins<T: RealNumber, C>(x: T, round: usize, table: &[C]) -> C
where
    C: Mul<C, Output = C> + Add<C, Output = C> + Sub<C, Output = C> + Mul<T, Output = C> + Copy,
{
    let round_float = T::from(round).unwrap();
    let len = table.len();
    if x > round_float {
        let other = round + 1;
        if other >= len {
            return table[round];
        }
        let y0 = table[round];
        let x0 = round_float;
        let y1 = table[other];
        y0 + (y1 - y0) * (x - x0)
    } else {
        if round == 0 {
            return table[round];
        }
        let other = round - 1;
        let y0 = table[round];
        let x0 = round_float;
        let y1 = table[other];
        y0 + (y1 - y0) * (x0 - x)
    }
}

macro_rules! add_linear_table_lookup_impl {
    ($($name: ident: $conv_type: ident, $($data_type: ident, $result_type:ident),*);*) => {
        $(
            $(
                impl $conv_type<$data_type> for $name<$data_type> {
                    fn is_symmetric(&self) -> bool {
                        self.is_symmetric
                    }

                    fn calc(&self, x: $data_type) -> $result_type {
                        let len = self.table.len();
                        let center = len / 2;
                        let center_float = center as $data_type;
                        let x = x / self.delta + center_float;
                        let round_float = x.round();
                        let round = round_float as usize;
                        if round >= len {
                            return $result_type::zero();
                        }

                        let round_tolerance = 1e-6;
                        let is_exactly_at_bin = (x - round_float).abs() < round_tolerance;
                        if is_exactly_at_bin {
                            return self.table[round];
                        }

                        linear_interpolation_between_bins(x, round, &self.table[..])
                    }
                }

                impl $name<$data_type> {
                    /// Creates a lookup table by putting the pieces together.
                    pub fn from_raw_parts(table: &[$result_type],
                                          delta: $data_type,
                                          is_symmetric: bool) -> Self {
                        let mut owned_table = InlineVector::with_capacity(table.len());
                        for n in &table[..] {
                            owned_table.push(*n);
                        }
                        $name { table: owned_table, delta: delta, is_symmetric: is_symmetric }
                    }

                    /// Creates a lookup table from another convolution function. The `delta` argument
                    /// can be used to balance performance vs. accuracy.
                    pub fn from_conv_function(other: &dyn $conv_type<$data_type>,
                                              delta: $data_type,
                                              len: usize) -> Self {
                        let center = len as isize;
                        let len = 2 * len + 1;
                        let is_symmetric = other.is_symmetric();
                        let mut table = InlineVector::of_size($result_type::zero(), len);
                        let mut i = -center;
                        for n in &mut table[..] {
                            *n = other.calc((i as $data_type) * delta);
                            i += 1;
                        }
                        $name { table: table, delta: delta, is_symmetric: is_symmetric }
                    }
                }
            )*
        )*
    }
}
add_linear_table_lookup_impl!(
    RealTimeLinearTableLookup: RealImpulseResponse, f32, f32, f64, f64;
    RealFrequencyLinearTableLookup: RealFrequencyResponse, f32, f32, f64, f64;
    ComplexTimeLinearTableLookup: ComplexImpulseResponse, f32, Complex32, f64, Complex64;
    ComplexFrequencyLinearTableLookup: ComplexFrequencyResponse, f32, Complex32, f64, Complex64);

macro_rules! add_real_linear_table_impl {
    ($($name: ident, $complex: ident, $($data_type: ident),*);*) => {
        $(
            $(
                impl $name<$data_type> {
                    /// Convert the lookup table into complex number space
                    pub fn to_complex(&self) -> $complex<$data_type> {
                        let len = self.table.len();
                        let vector = InlineVector::of_size($data_type::zero(), 2 * len);
                        let mut vector = vector.to_real_time_vec();
                        vector.resize(len).expect("shrinking should always succeed");
                        vector.data_mut(0.. len).copy_from_slice(&self.table[..]);
                        vector.set_delta(self.delta);
                        let mut buffer = FixedLenBuffer::new(InlineVector::of_size($data_type::zero(), 2 * vector.len()));
                        let complex = vector.to_complex_b(&mut buffer);
                        let complex = complex.datac(..);
                        let is_symmetric = self.is_symmetric;
                        let mut table = InlineVector::with_capacity(complex.len());
                        for n in complex {
                            table.push(*n);
                        }
                        $complex { table: table, delta: self.delta, is_symmetric: is_symmetric }
                    }
                }
            )*
        )*
    }
}
add_real_linear_table_impl!(
    RealTimeLinearTableLookup, ComplexTimeLinearTableLookup, f32, f64;
    RealFrequencyLinearTableLookup, ComplexFrequencyLinearTableLookup, f32, f64);

macro_rules! add_complex_linear_table_impl {
    ($($name: ident, $real: ident, $($data_type: ident),*);*) => {
        $(
            $(
                impl $name<$data_type> {
                    /// Convert the lookup table into real number space
                    pub fn to_real(&self) -> $real<$data_type> {
                        let complex = &self.table[..];
                        let mut interleaved = InlineVector::with_capacity(2 * complex.len());
                        for n in complex {
                            interleaved.push(n.re);
                            interleaved.push(n.im);
                        }
                        let mut vector = interleaved.to_complex_time_vec();
                        vector.set_delta(self.delta);
                        let mut buffer = FixedLenBuffer::new(InlineVector::of_size($data_type::zero(), complex.len()));
                        let real = vector.to_real_b(&mut buffer);
                        let real = real.data(..);
                        let is_symmetric = self.is_symmetric;
                        let mut table = InlineVector::with_capacity(real.len());
                        for n in real {
                            table.push(*n);
                        }
                        $real { table: table, delta: self.delta, is_symmetric: is_symmetric }
                    }
                }
            )*
        )*
    }
}
add_complex_linear_table_impl!(
    ComplexTimeLinearTableLookup, RealTimeLinearTableLookup, f32, f64;
    ComplexFrequencyLinearTableLookup, RealFrequencyLinearTableLookup, f32, f64);

macro_rules! add_complex_time_linear_table_impl {
    ($($data_type: ident),*) => {
        $(
            impl ComplexTimeLinearTableLookup<$data_type> {
                /// Convert the lookup table into frequency domain
                pub fn fft(self) -> ComplexFrequencyLinearTableLookup<$data_type> {
                    let complex = &self.table[..];
                    let mut interleaved = InlineVector::with_capacity(2 * complex.len());
                    for n in complex {
                        interleaved.push(n.re);
                        interleaved.push(n.im);
                    }
                    let mut vector = interleaved.to_complex_time_vec();
                    vector.set_delta(self.delta);
                    let mut buffer = FixedLenBuffer::new(InlineVector::of_size($data_type::zero(), vector.len()));
                    let freq = vector.fft(&mut buffer);
                    let delta = freq.delta();
                    let freq = freq.datac(..);
                    let is_symmetric = self.is_symmetric;
                    let mut table = InlineVector::with_capacity(freq.len());
                    for n in freq {
                        table.push(*n);
                    }
                    ComplexFrequencyLinearTableLookup {
                        table: table,
                        delta: delta,
                        is_symmetric: is_symmetric }
                }
            }
        )*
    }
}
add_complex_time_linear_table_impl!(f32, f64);

macro_rules! add_real_time_linear_table_impl {
    ($($data_type: ident),*) => {
        $(
            impl RealTimeLinearTableLookup<$data_type> {
                /// Convert the lookup table into a magnitude spectrum
                pub fn fft(self) -> RealFrequencyLinearTableLookup<$data_type> {
                    let len = self.table.len();
                    let vector = InlineVector::of_size($data_type::zero(), 2 * len);
                    let mut vector = vector.to_real_time_vec();
                    vector.resize(len).expect("shrinking should always succeed");
                    vector.data_mut(0.. len).copy_from_slice(&self.table[..]);
                    vector.set_delta(self.delta);
                    let mut buffer = FixedLenBuffer::new(InlineVector::of_size($data_type::zero(), 2 * len));
                    let freq = vector.fft(&mut buffer);
                    let freq = freq.magnitude();
                    let is_symmetric = self.is_symmetric;
                    let delta = freq.delta();
                    let freq = freq.data(..);
                    let mut table = InlineVector::with_capacity(freq.len());
                    for n in freq {
                        table.push(*n);
                    }
                    RealFrequencyLinearTableLookup {
                        table: table,
                        delta: delta,
                        is_symmetric: is_symmetric }
                }
            }
        )*
    }
}
add_real_time_linear_table_impl!(f32, f64);

macro_rules! add_complex_frequency_linear_table_impl {
    ($($data_type: ident),*) => {
        $(
            impl ComplexFrequencyLinearTableLookup<$data_type> {
                /// Convert the lookup table into time domain
                pub fn ifft(self) -> ComplexTimeLinearTableLookup<$data_type> {
                    let complex = &self.table[..];
                    let mut interleaved = InlineVector::with_capacity(2 * complex.len());
                    for n in complex {
                        interleaved.push(n.re);
                        interleaved.push(n.im);
                    }
                    let mut vector = interleaved.to_complex_freq_vec();
                    vector.set_delta(self.delta);
                    let mut buffer = FixedLenBuffer::new(InlineVector::of_size($data_type::zero(), vector.len()));
                    let time = vector.ifft(&mut buffer);
                    let delta = time.delta();
                    let time = time.datac(..);
                    let is_symmetric = self.is_symmetric;
                    let mut table = InlineVector::with_capacity(time.len());
                    for n in time {
                        table.push(*n);
                    }
                    ComplexTimeLinearTableLookup {
                        table: table,
                        delta: delta,
                        is_symmetric: is_symmetric }
                }
            }
        )*
    }
}
add_complex_frequency_linear_table_impl!(f32, f64);

/// Raised cosine function according to <https://en.wikipedia.org/wiki/Raised-cosine_filter>
pub struct RaisedCosineFunction<T>
where
    T: RealNumber,
{
    rolloff: T,
}

impl<T> RealImpulseResponse<T> for RaisedCosineFunction<T>
where
    T: RealNumber,
{
    fn is_symmetric(&self) -> bool {
        true
    }

    fn calc(&self, x: T) -> T {
        if x == T::zero() {
            return T::one();
        }

        let one = T::one();
        let two = T::from(2.0).unwrap();
        let pi = T::PI();
        let four = two * two;
        if x.abs() == one / (two * self.rolloff) {
            let arg = pi / two / self.rolloff;
            return (arg).sin() / arg * pi / four;
        }

        let pi_x = pi * x;
        let arg = two * self.rolloff * x;
        pi_x.sin() * (pi_x * self.rolloff).cos() / pi_x / (one - (arg * arg))
    }
}

impl<T> RealFrequencyResponse<T> for RaisedCosineFunction<T>
where
    T: RealNumber,
{
    fn is_symmetric(&self) -> bool {
        true
    }

    fn calc(&self, x: T) -> T {
        // assume x_delta = 1.0
        let one = T::one();
        let two = T::from(2.0).unwrap();
        let pi = T::PI();
        if x.abs() <= (one - self.rolloff) {
            return one;
        }

        if ((one - self.rolloff) < x.abs()) && (x.abs() <= (one + self.rolloff)) {
            return one / two
                * (one + (pi / self.rolloff * (x.abs() - (one - self.rolloff)) / two).cos());
        }

        T::zero()
    }
}

impl<T> RaisedCosineFunction<T>
where
    T: RealNumber,
{
    /// Creates a raised cosine function.
    pub fn new(rolloff: T) -> Self {
        RaisedCosineFunction { rolloff }
    }
}

/// Sinc function according to <https://en.wikipedia.org/wiki/Sinc_function>
#[derive(Default)]
pub struct SincFunction<T>
where
    T: RealNumber,
{
    _ghost: PhantomData<T>,
}

impl<T> RealImpulseResponse<T> for SincFunction<T>
where
    T: RealNumber,
{
    fn is_symmetric(&self) -> bool {
        true
    }

    fn calc(&self, x: T) -> T {
        if x == T::zero() {
            return T::one();
        }

        let pi = T::PI();
        let pi_x = pi * x;
        pi_x.sin() / pi_x
    }
}

impl<T> RealFrequencyResponse<T> for SincFunction<T>
where
    T: RealNumber,
{
    fn is_symmetric(&self) -> bool {
        true
    }

    fn calc(&self, x: T) -> T {
        let one = T::one();
        if x.abs() <= one {
            return one;
        }

        T::zero()
    }
}

impl<T> SincFunction<T>
where
    T: RealNumber,
{
    /// Creates a sinc function.
    pub fn new() -> Self {
        SincFunction {
            _ghost: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;

    fn conv_test<T, C>(conv: C, expected: &[T], step: T, tolerance: T)
    where
        T: RealNumber + Debug,
        C: RealImpulseResponse<T>,
    {
        let mut result = vec![T::zero(); expected.len()];
        let mut j = -(expected.len() as isize / 2);
        for i in 0..result.len() {
            result[i] = conv.calc(T::from(j).unwrap() * step);
            j += 1;
        }

        for i in 0..result.len() {
            if (result[i] - expected[i]).abs() > tolerance {
                panic!("assertion failed: {:?} != {:?}", result, expected);
            }
        }
    }

    fn complex_conv_test<T, C>(conv: C, expected: &[T], step: T, tolerance: T)
    where
        T: RealNumber + Debug,
        C: ComplexImpulseResponse<T>,
    {
        let mut result = vec![Complex::<T>::zero(); expected.len()];
        let mut j = -(expected.len() as isize / 2);
        for i in 0..result.len() {
            result[i] = conv.calc(T::from(j).unwrap() * step);
            j += 1;
        }

        for i in 0..result.len() {
            if (result[i].norm() - expected[i]).abs() > tolerance {
                panic!("assertion failed: {:?} != {:?}", result, expected);
            }
        }
    }

    fn real_freq_conv_test<T, C>(conv: C, expected: &[T], step: T, tolerance: T)
    where
        T: RealNumber + Debug,
        C: RealFrequencyResponse<T>,
    {
        let mut result = vec![T::zero(); expected.len()];
        let mut j = -(expected.len() as isize / 2);
        for i in 0..result.len() {
            result[i] = conv.calc(T::from(j).unwrap() * step);
            j += 1;
        }

        for i in 0..result.len() {
            if (result[i] - expected[i]).abs() > tolerance {
                panic!("assertion failed: {:?} != {:?}", result, expected);
            }
        }
    }

    #[test]
    fn raised_cosine_test() {
        let rc = RaisedCosineFunction::new(0.35);
        let expected = [
            0.0,
            0.2171850639713355,
            0.4840621929215732,
            0.7430526238101408,
            0.9312114164253432,
            1.0,
            0.9312114164253432,
            0.7430526238101408,
            0.4840621929215732,
            0.2171850639713355,
        ];
        conv_test(rc, &expected, 0.2, 1e-4);
    }

    #[test]
    fn sinc_test() {
        let rc = SincFunction::<f32>::new();
        let expected = [
            0.1273, -0.0000, -0.2122, 0.0000, 0.6366, 1.0000, 0.6366, 0.0000, -0.2122, -0.0000,
        ];
        conv_test(rc, &expected, 0.5, 1e-4);
    }

    #[test]
    fn sinc_freq_test() {
        let rc = SincFunction::<f32>::new();
        let expected = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        real_freq_conv_test(rc, &expected, 0.5, 1e-4);
    }

    #[test]
    fn lookup_table_test() {
        let rc = RaisedCosineFunction::new(0.35);
        let table = RealTimeLinearTableLookup::<f64>::from_conv_function(&rc, 0.2, 5);
        let expected = [
            0.0,
            0.2171850639713355,
            0.4840621929215732,
            0.7430526238101408,
            0.9312114164253432,
            1.0,
            0.9312114164253432,
            0.7430526238101408,
            0.4840621929215732,
            0.2171850639713355,
        ];
        conv_test(table, &expected, 0.2, 1e-4);
    }

    #[test]
    fn linear_interpolation_lookup_table_test() {
        let rc = RaisedCosineFunction::new(0.35);
        let table = RealTimeLinearTableLookup::<f64>::from_conv_function(&rc, 0.4, 5);
        let expected = [
            0.0,
            0.2171850639713355,
            0.4840621929215732,
            0.7430526238101408,
            0.9312114164253432,
            1.0,
            0.9312114164253432,
            0.7430526238101408,
            0.4840621929215732,
            0.2171850639713355,
        ];
        conv_test(table, &expected, 0.2, 0.1);
    }

    #[test]
    fn to_complex_test() {
        let rc = RaisedCosineFunction::new(0.35);
        let table = RealTimeLinearTableLookup::<f64>::from_conv_function(&rc, 0.4, 5);
        let complex = table.to_complex();
        let expected = [
            0.0,
            0.2171850639713355,
            0.4840621929215732,
            0.7430526238101408,
            0.9312114164253432,
            1.0,
            0.9312114164253432,
            0.7430526238101408,
            0.4840621929215732,
            0.2171850639713355,
        ];
        complex_conv_test(complex, &expected, 0.2, 0.1);
    }

    #[test]
    fn fft_test() {
        let rc = RaisedCosineFunction::new(0.5);
        let table = RealTimeLinearTableLookup::<f64>::from_conv_function(&rc, 0.2, 5);
        let freq = table.fft();
        assert_eq!(freq.delta(), 2.2);
        let expected = [
            0.0078, 0.0269, 0.0602, 0.1311, 2.7701, 5.6396, 2.7701, 0.1311, 0.0602, 0.0269, 0.0078,
        ];
        real_freq_conv_test(freq, &expected, 2.2, 0.1);
    }

    #[test]
    fn freq_test() {
        let rc = RaisedCosineFunction::new(0.5);
        let expected = [
            0.0,
            0.0,
            0.20610737385376332,
            0.7938926261462365,
            1.0,
            1.0,
            1.0,
            0.7938926261462365,
            0.20610737385376332,
            0.0,
        ];
        real_freq_conv_test(rc, &expected, 0.4, 0.1);
    }
}
