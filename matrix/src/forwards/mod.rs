use basic_dsp_vector::*;
use super::*;

macro_rules! add_mat_impl {
    ($($matrix:ident);*) => {
        $(
			impl<V: Vector<T> + ScaleOps<T>, T: RealNumber> ScaleOps<T> for $matrix<V, T> {
				fn scale(&mut self, factor: T) {
					for v in self.rows_mut() {
						v.scale(factor);
					}
				}
			}
		)*
	}
}

add_mat_impl!(MatrixMxN; Matrix2xN; Matrix3xN; Matrix4xN);
