use super::*;

/// A trait for matrix types. In this lib a matrix is simply a collection of
/// vectors. The idea is that the matrix types can be used to reduce the size
/// of a large matrix and that the return types are basic enough
/// so that other specialized matrix libs can do the rest of the work, e.g.
/// inverting the resulting matrix.
pub trait Matrix<V, T>: MetaData + ResizeOps
where
    V: Vector<T>,
    T: RealNumber,
{
    /// The x-axis delta. If `domain` is time domain then `delta` is in `[s]`,
    /// in frequency domain `delta` is in `[Hz]`.
    fn delta(&self) -> T;

    /// Sets the x-axis delta. If `domain` is time domain then `delta` is in `[s]`,
    /// in frequency domain `delta` is in `[Hz]`.
    fn set_delta(&mut self, delta: T);

    /// The number of valid elements in each row of the matrix. This can be changed
    /// with the `Resize` trait.
    fn row_len(&self) -> usize;

    /// The number of valid points in a row. If the matrix is complex then every valid point
    /// consists of two floating point numbers,
    /// while for real vectors every point only consists of one floating point number.
    fn row_points(&self) -> usize;

    /// The number of columns in the matrix.
    fn col_len(&self) -> usize;

    /// Gets the rows as vectors.
    fn rows(&self) -> &[V];

    /// Gets the rows as mutable vectors.
    fn rows_mut(&mut self) -> &mut [V];
}

macro_rules! add_mat_impl {
    ($($matrix:ident);*) => {
        $(
			impl<V, S, T> MetaData for $matrix<V, S, T>
			    where T: RealNumber,
					  S: ToSlice<T>,
                  	  V: Vector<T> {
                  fn domain(&self) -> DataDomain {
					  if self.rows.len() == 0 {
						  return DataDomain::Time;
					  }

                      self.rows[0].domain()
                  }

                  fn is_complex(&self) -> bool {
					  if self.rows.len() == 0 {
					  	return false;
					  }

                      self.rows[0].is_complex()
                  }
			}

			impl<V, S, T> ResizeOps for $matrix<V, S, T>
			    where T: RealNumber,
					  S: ToSlice<T>,
                  	  V: Vector<T> {
			      fn resize(&mut self, len: usize) -> VoidResult {
					  for v in &mut self.rows[..] {
						  try!(v.resize(len));
					  }

			          Ok(())
			      }
			}

            impl<V, S, T> Matrix<V, T> for $matrix<V, S, T>
                where T: RealNumber,
					  S: ToSlice<T>,
                  	  V: Vector<T> {
                fn delta(&self) -> T {
					if self.rows.len() == 0 {
					  return T::zero();
					}

                    self.rows[0].delta()
                }

				fn set_delta(&mut self, delta: T) {
					for v in &mut self.rows[..] {
						v.set_delta(delta);
					}
                }

				fn row_len(&self) -> usize {
					if self.rows.len() == 0 {
					  return 0;
					}

                    self.rows[0].len()
                }

				fn row_points(&self) -> usize {
					if self.rows.len() == 0 {
					  return 0;
					}

                    self.rows[0].points()
                }

				fn col_len(&self) -> usize {
                    self.rows.len()
                }

				fn rows(&self) -> &[V] {
                    &self.rows[..]
                }

				fn rows_mut(&mut self) -> &mut [V] {
                    &mut self.rows[..]
                }
            }

            impl<V, S, T, N, D> GetMetaData<T, N, D> for $matrix<V, S, T>
                where T: RealNumber,
					  S: ToSlice<T>,
                  	  V: Vector<T> + GetMetaData<T, N, D>,
                      N: NumberSpace,
                      D: Domain {
                 fn get_meta_data(&self) -> TypeMetaData<T, N, D> {
                    self.rows[0].get_meta_data()
                 }
            }
        )*
    }
}

add_mat_impl!(MatrixMxN; Matrix2xN; Matrix3xN; Matrix4xN);
