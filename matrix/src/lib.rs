extern crate basic_dsp_vector;

mod definitions;

use std::result::Result;
use basic_dsp_vector::*;

pub use definitions::*;

pub struct MatrixMxN<V, T>
    where T: RealNumber,
          V: Vector<T> {
  rows: Vec<V>,
  number_type: std::marker::PhantomData<T>
}

pub struct Matrix2xN<V, T>
    where T: RealNumber,
          V: Vector<T> {
  rows: [V; 2],
  number_type: std::marker::PhantomData<T>
}

pub struct Matrix3xN<V, T>
    where T: RealNumber,
          V: Vector<T> {
  rows: [V; 3],
  number_type: std::marker::PhantomData<T>
}

pub struct Matrix4xN<V, T>
    where T: RealNumber,
          V: Vector<T> {
  rows: [V; 4],
  number_type: std::marker::PhantomData<T>
}

pub type Matrix32xN = MatrixMxN<GenDspVec32, f32>;
pub type Matrix64xN = MatrixMxN<GenDspVec64, f32>;
pub type RealTimeMatrix32xN = MatrixMxN<RealTimeVec32, f32>;
pub type RealTimeMatrix64xN = MatrixMxN<RealTimeVec64, f64>;
pub type ComplexTimeMatrix32xN = MatrixMxN<ComplexTimeVec32, f32>;
pub type ComplexTimeMatrix64xN = MatrixMxN<ComplexTimeVec64, f64>;
pub type RealFreqMatrix32xN = MatrixMxN<RealFreqVec32, f32>;
pub type RealFreqMatrix64xN = MatrixMxN<RealFreqVec64, f64>;
pub type ComplexFreqMatrix32xN = MatrixMxN<ComplexFreqVec32, f32>;
pub type ComplexFreqMatrix64xN = MatrixMxN<ComplexFreqVec64, f64>;

pub type Matrix32x2 = Matrix2xN<GenDspVec32, f32>;
pub type Matrix64x2 = Matrix2xN<GenDspVec64, f32>;
pub type RealTimeMatrix32x2 = Matrix2xN<RealTimeVec32, f32>;
pub type RealTimeMatrix64x2 = Matrix2xN<RealTimeVec64, f64>;
pub type ComplexTimeMatrix32x2 = Matrix2xN<ComplexTimeVec32, f32>;
pub type ComplexTimeMatrix64x2 = Matrix2xN<ComplexTimeVec64, f64>;
pub type RealFreqMatrix32x2 = Matrix2xN<RealFreqVec32, f32>;
pub type RealFreqMatrix64x2 = Matrix2xN<RealFreqVec64, f64>;
pub type ComplexFreqMatrix32x2 = Matrix2xN<ComplexFreqVec32, f32>;
pub type ComplexFreqMatrix64x2 = Matrix2xN<ComplexFreqVec64, f64>;

pub type Matrix32x3 = Matrix3xN<GenDspVec32, f32>;
pub type Matrix64x3 = Matrix3xN<GenDspVec64, f32>;
pub type RealTimeMatrix32x3 = Matrix3xN<RealTimeVec32, f32>;
pub type RealTimeMatrix64x3 = Matrix3xN<RealTimeVec64, f64>;
pub type ComplexTimeMatrix32x3 = Matrix3xN<ComplexTimeVec32, f32>;
pub type ComplexTimeMatrix64x3 = Matrix3xN<ComplexTimeVec64, f64>;
pub type RealFreqMatrix32x3 = Matrix3xN<RealFreqVec32, f32>;
pub type RealFreqMatrix64x3 = Matrix3xN<RealFreqVec64, f64>;
pub type ComplexFreqMatrix32x3 = Matrix3xN<ComplexFreqVec32, f32>;
pub type ComplexFreqMatrix64x3 = Matrix3xN<ComplexFreqVec64, f64>;

pub type Matrix32x4 = Matrix4xN<GenDspVec32, f32>;
pub type Matrix64x4 = Matrix4xN<GenDspVec64, f32>;
pub type RealTimeMatrix32x4 = Matrix4xN<RealTimeVec32, f32>;
pub type RealTimeMatrix64x4 = Matrix4xN<RealTimeVec64, f64>;
pub type ComplexTimeMatrix32x4 = Matrix4xN<ComplexTimeVec32, f32>;
pub type ComplexTimeMatrix64x4 = Matrix4xN<ComplexTimeVec64, f64>;
pub type RealFreqMatrix32x4 = Matrix4xN<RealFreqVec32, f32>;
pub type RealFreqMatrix64x4 = Matrix4xN<RealFreqVec64, f64>;
pub type ComplexFreqMatrix32x4 = Matrix4xN<ComplexFreqVec32, f32>;
pub type ComplexFreqMatrix64x4 = Matrix4xN<ComplexFreqVec64, f64>;
/*
impl<V, T> MatrixMxN<V, T>
    where T: RealNumber,
      V: DataVec<T> {
    pub fn new(rows: Vec<V>) -> Result<Self, ErrorReason> {
        if rows.len() < 1 {
            return Err(ErrorReason::InvalidArgumentLength);
        }

        let first_len = rows[0].len();
        let first_complex = rows[0].is_complex();
        let first_domain = rows[0].domain();
        for v in &rows[1..rows.len()] {
            if v.len() != first_len {
                return Err(ErrorReason::InputMustHaveTheSameSize);
            }

            if v.is_complex() != first_complex {
                return Err(ErrorReason::InputMetaDataMustAgree);
            }

            if v.domain() != first_domain {
                return Err(ErrorReason::InputMetaDataMustAgree);
            }
        }

        Ok(MatrixMxN { rows: rows, number_type: std::marker::PhantomData })
    }

    pub fn decompose(self) -> Vec<V> {
        self.rows
    }
}

impl<V, T> Matrix2xN<V, T>
    where T: RealNumber,
      V: DataVec<T> {
    pub fn new(rows: [V; 2]) -> Result<Self, ErrorReason> {
        let first_len = rows[0].len();
        let first_complex = rows[0].is_complex();
        let first_domain = rows[0].domain();
        for v in &rows[1..rows.len()] {
            if v.len() != first_len {
                return Err(ErrorReason::InputMustHaveTheSameSize);
            }

            if v.is_complex() != first_complex {
                return Err(ErrorReason::InputMetaDataMustAgree);
            }

            if v.domain() != first_domain {
                return Err(ErrorReason::InputMetaDataMustAgree);
            }
        }

        Ok(Matrix2xN { rows: rows, number_type: std::marker::PhantomData })
    }

    pub fn decompose(self) -> [V; 2] {
        self.rows
    }
}

impl<V, T> Matrix3xN<V, T>
    where T: RealNumber,
      V: DataVec<T> {
    pub fn new(rows: [V; 3]) -> Result<Self, ErrorReason> {
        let first_len = rows[0].len();
        let first_complex = rows[0].is_complex();
        let first_domain = rows[0].domain();
        for v in &rows[1..rows.len()] {
            if v.len() != first_len {
                return Err(ErrorReason::InputMustHaveTheSameSize);
            }

            if v.is_complex() != first_complex {
                return Err(ErrorReason::InputMetaDataMustAgree);
            }

            if v.domain() != first_domain {
                return Err(ErrorReason::InputMetaDataMustAgree);
            }
        }

        Ok(Matrix3xN { rows: rows, number_type: std::marker::PhantomData })
    }

    pub fn decompose(self) -> [V; 3] {
        self.rows
    }
}

impl<V, T> Matrix4xN<V, T>
    where T: RealNumber,
      V: DataVec<T> {
    pub fn new(rows: [V; 4]) -> Result<Self, ErrorReason> {
        let first_len = rows[0].len();
        let first_complex = rows[0].is_complex();
        let first_domain = rows[0].domain();
        for v in &rows[1..rows.len()] {
            if v.len() != first_len {
                return Err(ErrorReason::InputMustHaveTheSameSize);
            }

            if v.is_complex() != first_complex {
                return Err(ErrorReason::InputMetaDataMustAgree);
            }

            if v.domain() != first_domain {
                return Err(ErrorReason::InputMetaDataMustAgree);
            }
        }

        Ok(Matrix4xN { rows: rows, number_type: std::marker::PhantomData })
    }

    pub fn decompose(self) -> [V; 4] {
        self.rows
    }
}

macro_rules! add_basic_impl {
    ($($matrix:ident);*) => {
        $(
            impl<V, T> DataMatrix<T> for $matrix<V, T>
                where T: RealNumber,
                  V: DataVec<T> {

                fn delta(&self) -> T {
                    self.rows[0].delta()
                }

                fn domain(&self) -> DataVecDomain {
                    self.rows[0].domain()
                }

                fn is_complex(&self) -> bool {
                    self.rows[0].is_complex()
                }
            }
        )*
    }
}

add_basic_impl!(MatrixMxN; Matrix2xN; Matrix3xN; Matrix4xN);

#[cfg(test)]
mod tests {
    use super::*;
    use basic_dsp_vector::*;

    #[test]
    fn new_decompose() {
        let array1 = [1.0, 2.0, 3.0, 4.0];
        let vector1 = DataVec32::from_array(false, DataVecDomain::Time, &array1);
        let array2 = [4.0, 3.0, 2.0, 5.0];
        let vector2 = DataVec32::from_array(false, DataVecDomain::Time, &array2);
        let mat = Matrix2xN::new([vector1, vector2]).unwrap();
        let pair = mat.decompose();
        assert_eq!(pair[0].real(0..), array1);
        assert_eq!(pair[1].real(0..), array2);
    }
}*/
