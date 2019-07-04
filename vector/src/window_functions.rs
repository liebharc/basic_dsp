//! This mod contains a definition for window functions and provides implementations for a
//! few standard windows. See the `WindowFunction` type for more information.
use crate::numbers::*;

/// A window function for FFT windows. See `https://en.wikipedia.org/wiki/Window_function`
/// for details. Window functions should document if they aren't applicable for
/// Inverse Fourier Transformations.
///
/// The contract for window functions is as follows:
///
/// 1. The second argument is of the function is always `vector.points()` and the possible values
///    for the first argument ranges from `0..vector.points()`.
/// 2. All real return values are allowed
pub trait WindowFunction<T>: Sync
where
    T: RealNumber,
{
    /// Indicates whether this function is symmetric around the y axis or not.
    /// Symmetry is defined as `self.window(x) == self.window(-x)`.
    fn is_symmetric(&self) -> bool;

    /// Calculates a point of the window function. Callers will ensure that `n <= length`.
    fn window(&self, n: usize, length: usize) -> T;
}

/// A triangular window: `https://en.wikipedia.org/wiki/Window_function#Triangular_window`
pub struct TriangularWindow;
impl<T> WindowFunction<T> for TriangularWindow
where
    T: RealNumber,
{
    fn is_symmetric(&self) -> bool {
        true
    }

    fn window(&self, n: usize, length: usize) -> T {
        let one = T::one();
        let two = T::from(2.0).unwrap();
        let n = T::from(n).unwrap();
        let length = T::from(length).unwrap();
        one - ((n - (length - one) / two) / (length / two)).abs()
    }
}

/// A generalized Hamming window: `https://en.wikipedia.org/wiki/Window_function#Hamming_window`
pub struct HammingWindow<T>
where
    T: RealNumber,
{
    alpha: T,
    beta: T,
}

impl<T> HammingWindow<T>
where
    T: RealNumber,
{
    /// Creates a new Hamming window
    pub fn new(alpha: T) -> Self {
        HammingWindow {
            alpha,
            beta: (T::one() - alpha),
        }
    }

    /// Creates the default Hamming window as defined in GNU Octave.
    pub fn default() -> Self {
        Self::new(T::from(0.54).unwrap())
    }
}

impl<T> WindowFunction<T> for HammingWindow<T>
where
    T: RealNumber,
{
    fn is_symmetric(&self) -> bool {
        true
    }

    fn window(&self, n: usize, length: usize) -> T {
        let one = T::one();
        let two = T::from(2.0).unwrap();
        let pi = T::PI();
        let n = T::from(n).unwrap();
        let length = T::from(length).unwrap();
        self.alpha - self.beta * (two * pi * n / (length - one)).cos()
    }
}

/// A Blackman-Harris Window: `https://en.wikipedia.org/wiki/Window_function#Blackman-Harris_window`
pub struct BlackmanHarrisWindow;
impl<T> WindowFunction<T> for BlackmanHarrisWindow
where
    T: RealNumber,
{
   fn is_symmetric(&self) -> bool {
       true
   }

   fn window(&self, n: usize, length: usize) -> T {
        let one = T::one();
        let two = T::from(2.0).unwrap();
        let four = T::from(4.0).unwrap();
        let six = T::from(6.0).unwrap();
        let pi = T::PI();
        let a_naught = T::from(0.35875).unwrap();
        let a_one = T::from(0.48829).unwrap();
        let a_two = T::from(0.14128).unwrap();
        let a_three = T::from(0.01168).unwrap();
        let n = T::from(n).unwrap();
        let length = T::from(length).unwrap();
        a_naught - a_one * (two * pi * n / (length - one)).cos() + a_two * (four * pi * n / (length - one)).cos() - a_three * (six * pi * n / (length - one)).cos()
   }
}

/// A rectangular window: `https://en.wikipedia.org/wiki/Window_function#Rectangular_window`
pub struct RectangularWindow;
impl<T> WindowFunction<T> for RectangularWindow
where
    T: RealNumber,
{
    fn is_symmetric(&self) -> bool {
        true
    }

    fn window(&self, _n: usize, _length: usize) -> T {
        let one = T::one();
        one
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;

    fn window_test<T, W>(window: W, expected: &[T])
    where
        T: RealNumber + Debug,
        W: WindowFunction<T>,
    {
        let mut result = vec![T::zero(); expected.len()];
        for i in 0..result.len() {
            result[i] = window.window(i, result.len());
        }

        for i in 0..result.len() {
            if (result[i] - expected[i]).abs() > T::from(1e-4).unwrap() {
                panic!("assertion failed: {:?} != {:?}", result, expected);
            }
        }
    }

    #[test]
    fn triangular_window32_test() {
        let window = TriangularWindow;
        let expected = [0.2, 0.6, 1.0, 0.6, 0.2];
        window_test(window, &expected);
    }

    #[test]
    fn hamming_window32_test() {
        let hamming = HammingWindow::<f32>::default();
        let expected = [0.08, 0.54, 1.0, 0.54, 0.08];
        window_test(hamming, &expected);
    }

    #[test]
    fn blackmanharris_window32_test() {
        let blackmanharris = BlackmanHarrisWindow;
        let expected = [0.0001, 0.2175, 1.0000, 0.2175, 0.0001];
        window_test(blackmanharris, &expected);
    }

    #[test]
    fn rectangular_window32_test() {
        let rectangular = RectangularWindow;
        let expected = [1.0, 1.0, 1.0, 1.0, 1.0];
        window_test(rectangular, &expected);
    }
}
