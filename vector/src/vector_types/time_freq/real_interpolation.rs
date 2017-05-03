use numbers::*;
use super::super::{DspVec, Domain,
                   ToSliceMut, Buffer, BufferBorrow, Vector,
                   RealNumberSpace, MetaData};

/// Provides interpolation operations which are only applicable for real data vectors.
/// # Failures
/// All operations in this trait fail with `VectorMustBeReal` if the vector isn't in the
/// real number space.
pub trait RealInterpolationOps<S, T>
    where S: ToSliceMut<T>,
          T: RealNumber
{
    /// Piecewise cubic hermite interpolation between samples.
    /// # Unstable
    /// Algorithm might need to be revised.
    /// This operation and `interpolate_lin` might be merged into one function with an
    /// additional argument in future.
    fn interpolate_hermite<B>(&mut self, buffer: &mut B, interpolation_factor: T, delay: T)
        where B: for<'a> Buffer<'a, S, T>;

    /// Linear interpolation between samples.
    /// # Unstable
    /// This operation and `interpolate_hermite` might be merged into one function with an
    /// additional argument in future.
    fn interpolate_lin<B>(&mut self, buffer: &mut B, interpolation_factor: T, delay: T)
        where B: for<'a> Buffer<'a, S, T>;
}

impl<S, T, N, D> RealInterpolationOps<S, T> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain
{
    fn interpolate_lin<B>(&mut self, buffer: &mut B, interpolation_factor: T, delay: T)
        where B: for<'a> Buffer<'a, S, T>
    {
        let data_len = self.len();
        let dest_len = (T::from(data_len - 1).unwrap() * interpolation_factor)
            .round()
            .to_usize()
            .unwrap() + 1;
        let mut temp = buffer.borrow(dest_len);
        {
            if self.is_complex() {
                self.valid_len = 0;
                return;
            }
            let data = self.data.to_slice();
            let mut temp = temp.to_slice_mut();
            let data = &data[0..data_len];
            let dest = &mut temp[0..dest_len];
            let mut i = T::zero();

            for num in &mut dest[0..dest_len - 1] {
                let rounded = i / interpolation_factor + delay;
                let beforef = rounded.floor();
                let before = beforef.to_usize().unwrap();
                let next = before + 1;
                let x0 = beforef;
                let y0 = data[before];
                let y1 = data[next];
                let x = rounded;
                (*num) = y0 + (y1 - y0) * (x - x0);
                i = i + T::one();
            }
            dest[dest_len - 1] = data[data_len - 1];
            self.valid_len = dest_len;
        }
        temp.trade(&mut self.data);
    }

    fn interpolate_hermite<B>(&mut self, buffer: &mut B, interpolation_factor: T, delay: T)
        where B: for<'a> Buffer<'a, S, T>
    {
        let data_len = self.len();
        let dest_len = (T::from(data_len - 1).unwrap() * interpolation_factor)
            .round()
            .to_usize()
            .unwrap() + 1;
        let mut temp = buffer.borrow(dest_len);
        {
            if self.is_complex() {
                self.valid_len = 0;
                return;
            }
            let data = self.data.to_slice();
            let mut temp = temp.to_slice_mut();
            // Literature: http://paulbourke.net/miscellaneous/interpolation/
            let data = &data[0..data_len];
            let dest = &mut temp[0..dest_len];
            let mut i = T::zero();
            let start = ((T::one() - delay) * interpolation_factor).ceil().to_usize().unwrap();
            let end = start + 1;
            let half = T::from(0.5).unwrap();
            let one_point_five = T::from(1.5).unwrap();
            let two = T::from(2.0).unwrap();
            let two_point_five = T::from(2.5).unwrap();
            for num in &mut dest[0..start] {
                let rounded = i / interpolation_factor + delay;
                let beforef = rounded.floor();
                let before = beforef.to_usize().unwrap();
                assert_eq!(before, 0);
                let next = before + 1;
                let next_next = next + 1;
                let y1 = data[before];
                let y2 = data[next];
                let y3 = data[next_next];
                let x = rounded - beforef;
                let y0 = y1 - (y2 - y1);
                let x2 = x * x;
                let a0 = -half * y0 + one_point_five * y1 - one_point_five * y2 + half * y3;
                let a1 = y0 - two_point_five * y1 + two * y2 - half * y3;
                let a2 = -half * y0 + half * y2;
                let a3 = y1;

                (*num) = (a0 * x * x2) + (a1 * x2) + (a2 * x) + a3;
                i = i + T::one();
            }

            for num in &mut dest[start..dest_len - end] {
                let rounded = i / interpolation_factor + delay;
                let beforef = rounded.floor();
                let before = beforef.to_usize().unwrap();
                let before_before = before - 1;
                let next = before + 1;
                let next_next = next + 1;
                let y0 = data[before_before];
                let y1 = data[before];
                let y2 = data[next];
                let y3 = data[next_next];
                let x = rounded - beforef;
                let x2 = x * x;
                let a0 = -half * y0 + one_point_five * y1 - one_point_five * y2 + half * y3;
                let a1 = y0 - two_point_five * y1 + two * y2 - half * y3;
                let a2 = -half * y0 + half * y2;
                let a3 = y1;

                (*num) = (a0 * x * x2) + (a1 * x2) + (a2 * x) + a3;
                i = i + T::one();
            }

            for num in &mut dest[dest_len - end..dest_len] {
                let rounded = i / interpolation_factor + delay;
                let beforef = rounded.floor();
                let before = beforef.to_usize().unwrap();
                assert!(before + 2 >= data_len);
                let before_before = before - 1;
                let y0 = data[before_before];
                let y1 = data[before];
                let y2 = if before < data_len - 1 {
                    data[before + 1]
                } else {
                    y1 + (y1 - y0)
                };
                let y3 = if before < data_len - 2 {
                    data[before + 2]
                } else {
                    y2 + (y2 - y1)
                };
                let x = rounded - beforef;
                let x2 = x * x;
                let a0 = -half * y0 + one_point_five * y1 - one_point_five * y2 + half * y3;
                let a1 = y0 - two_point_five * y1 + two * y2 - half * y3;
                let a2 = -half * y0 + half * y2;
                let a3 = y1;

                (*num) = (a0 * x * x2) + (a1 * x2) + (a2 * x) + a3;
                i = i + T::one();
            }
        }
        self.valid_len = dest_len;
        temp.trade(&mut self.data);
    }
}
#[cfg(test)]
mod tests {
    use conv_types::*;
    use super::super::super::*;

    fn assert_eq_tol<T>(left: &[T], right: &[T], tol: T)
        where T: RealNumber
    {
        assert_eq!(left.len(), right.len());
        for i in 0..left.len() {
            if (left[i] - right[i]).abs() > tol {
                panic!("assertion failed: {:?} != {:?} at index {}", left, right, i);
            }
        }
    }

    #[test]
    fn hermit_spline_test() {
        let mut time = vec![-1.0, -2.0, -1.0, 0.0, 1.0, 3.0, 4.0].to_real_freq_vec();
        let mut buffer = SingleBuffer::new();
        time.interpolate_hermite(&mut buffer, 4.0, 0.0);
        let expected = [-1.0000, -1.4375, -1.7500, -1.9375, -2.0000, -1.8906, -1.6250, -1.2969,
                        -1.0000, -0.7500, -0.5000, -0.2500, 0.0, 0.2344, 0.4583, 0.7031, 1.0000,
                        1.4375, 2.0000, 2.5625, 3.0000, 3.3203, 3.6042, 3.8359, 4.0];
        assert_eq_tol(&time[4..expected.len() - 4],
                      &expected[4..expected.len() - 4],
                      6e-2);
    }

    #[test]
    fn hermit_spline_test_linear_increment() {
        let mut time = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0].to_real_freq_vec();
        let mut buffer = SingleBuffer::new();
        time.interpolate_hermite(&mut buffer, 3.0, 0.0);
        let expected = [-3.0, -2.666, -2.333, -2.0, -1.666, -1.333, -1.0, -0.666, -0.333, 0.0,
                        0.333, 0.666, 1.0, 1.333, 1.666, 2.0, 2.333, 2.666, 3.0];
        assert_eq_tol(&time[..], &expected, 5e-3);
    }

    #[test]
    fn linear_test() {
        let mut time = vec![-1.0, -2.0, -1.0, 0.0, 1.0, 3.0, 4.0].to_real_freq_vec();
        let mut buffer = SingleBuffer::new();
        time.interpolate_lin(&mut buffer, 4.0, 0.0);
        let expected = [-1.0000, -1.2500, -1.5000, -1.7500, -2.0000, -1.7500, -1.5000, -1.2500,
                        -1.0000, -0.7500, -0.5000, -0.2500, 0.0, 0.2500, 0.5000, 0.7500, 1.0000,
                        1.5000, 2.0000, 2.5000, 3.0000, 3.2500, 3.5000, 3.7500, 4.0];
        assert_eq_tol(&time[..], &expected, 0.1);
    }
}
