//! This mod contains a definition for window functions and provides implementations for a
//! few standard windows. See the `WindowFunction` type for more information.
use RealNumber;
use std::os::raw::c_void;
use std::mem;

/// A window function for FFT windows. See https://en.wikipedia.org/wiki/Window_function
/// for details. Window functions should document if they aren't applicable for
/// Inverse Fourier Transformations.
///
/// The contract for window functions is as follows:
///
/// 1. The second argument is of the function is always `self.points()` and the possible values for the first argument ranges from `0..self.points()`.
/// 2. A window function must be symmetric about the y-axis.
/// 3. All real return values are allowed
pub trait WindowFunction<T>: Sync
    where T: RealNumber
{
    /// Indicates whether this function is symmetric around 0 or not.
    /// Symmetry is defined as `self.window(x) == self.window(-x)`.
    fn is_symmetric(&self) -> bool;
    /// Calculates a point of the window function
    fn window(&self, n: usize, length: usize) -> T;
}

/// A triangular window: https://en.wikipedia.org/wiki/Window_function#Triangular_window
pub struct TriangularWindow;
impl<T> WindowFunction<T> for TriangularWindow
    where T: RealNumber
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

/// A generalized Hamming window: https://en.wikipedia.org/wiki/Window_function#Hamming_window
pub struct HammingWindow<T>
    where T: RealNumber
{
    alpha: T,
    beta: T,
}

impl<T> HammingWindow<T>
    where T: RealNumber
{
    /// Creates a new Hamming window
    pub fn new(alpha: T) -> Self {
        HammingWindow {
            alpha: alpha,
            beta: (T::one() - alpha),
        }
    }

    /// Creates the default Hamming window as defined in GNU Octave.
    pub fn default() -> Self {
        Self::new(T::from(0.54).unwrap())
    }
}

impl<T> WindowFunction<T> for HammingWindow<T>
    where T: RealNumber
{
    fn is_symmetric(&self) -> bool {
        true
    }

    fn window(&self, n: usize, length: usize) -> T {
        let one = T::one();
        let two = T::from(2.0).unwrap();
        let pi = two * one.asin();
        let n = T::from(n).unwrap();
        let length = T::from(length).unwrap();
        self.alpha - self.beta * (two * pi * n / (length - one)).cos()
    }
}

/// A window function which can be constructed outside this crate.
pub struct ForeignWindowFunction<T>
    where T: RealNumber
{
    /// The window function
    pub window_function: extern "C" fn(*const c_void, usize, usize) -> T,

    /// The data which is passed to the window function
    ///
    /// Actual data type is a `const* c_void`, but Rust doesn't allow that because it's unsafe so we store
    /// it as `usize` and transmute it when necessary. Callers should make very sure safety is guaranteed.
    pub window_data: usize,

    /// Indicates whether this function is symmetric around 0 or not.
    /// Symmetry is defined as `self.window(x) == self.window(-x)`.
    pub is_symmetric: bool,
}

impl<T> ForeignWindowFunction<T>
    where T: RealNumber
{
    /// Creates a new window function
    pub fn new(window: extern "C" fn(*const c_void, usize, usize) -> T,
               window_data: *const c_void,
               is_symmetric: bool)
               -> Self {
        unsafe {
            ForeignWindowFunction {
                window_function: window,
                window_data: mem::transmute(window_data),
                is_symmetric: is_symmetric,
            }
        }
    }
}

impl<T> WindowFunction<T> for ForeignWindowFunction<T>
    where T: RealNumber
{
    fn is_symmetric(&self) -> bool {
        self.is_symmetric
    }

    fn window(&self, idx: usize, points: usize) -> T {
        let fun = self.window_function;
        unsafe { fun(mem::transmute(self.window_data), idx, points) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::RealNumber;
    use std::fmt::Debug;

    fn window_test<T, W>(window: W, expected: &[T])
        where T: RealNumber + Debug,
              W: WindowFunction<T>
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
}
