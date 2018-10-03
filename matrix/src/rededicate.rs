use super::*;
use std::marker;
use TransformContent;

macro_rules! add_mat_impl {
    ($($matrix:ident);*) => {
        $(
			impl<V, O, S: ToSlice<T>, T: RealNumber> RededicateForceOps<$matrix<O, S, T>>
                for $matrix<V, S, T>
                where V: RededicateForceOps<O> + Vector<T>,
                      T: RealNumber,
                      O: Vector<T> {

                fn rededicate_from_force(origin: $matrix<O, S, T>) -> Self {
					let rows = origin.rows.transform(V::rededicate_from_force);
					$matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
                }

                fn rededicate_with_runtime_data(
                        origin: $matrix<O, S, T>,
                        is_complex: bool,
                        domain: DataDomain) -> Self {
					let rows =
                        origin.rows.transform(
                            |v|V::rededicate_with_runtime_data(v, is_complex, domain));
					$matrix {
                        rows: rows,
                        storage_type: marker::PhantomData,
                	  	number_type: marker::PhantomData
                    }
                }
            }
		)*
	}
}

add_mat_impl!(MatrixMxN; Matrix2xN; Matrix3xN; Matrix4xN);
