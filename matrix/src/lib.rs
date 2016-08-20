extern crate basic_dsp_vector;

mod definitions;

use std::result::Result;
use std::ops::{Range, Index, IndexMut};
use basic_dsp_vector::*;

pub use definitions::*;

pub struct MatrixMxN<V, T>
    where T: RealNumber,
          V: DataVector<T> {
  rows: Vec<V>,
  number_type: std::marker::PhantomData<T>
}

pub struct Matrix2xN<V, T>
    where T: RealNumber,
          V: DataVector<T> {
  rows: [V; 2],
  number_type: std::marker::PhantomData<T>
}

pub struct Matrix3xN<V, T>
    where T: RealNumber,
          V: DataVector<T> {
  rows: [V; 3],
  number_type: std::marker::PhantomData<T>
}

pub struct Matrix4xN<V, T>
    where T: RealNumber,
          V: DataVector<T> {
  rows: [V; 4],
  number_type: std::marker::PhantomData<T>
}

pub type Matrix32xN = MatrixMxN<DataVector32, f32>;
pub type Matrix64xN = MatrixMxN<DataVector64, f32>;
pub type RealTimeMatrix32xN = MatrixMxN<RealTimeVector32, f32>;
pub type RealTimeMatrix64xN = MatrixMxN<RealTimeVector64, f64>;
pub type ComplexTimeMatrix32xN = MatrixMxN<ComplexTimeVector32, f32>;
pub type ComplexTimeMatrix64xN = MatrixMxN<ComplexTimeVector64, f64>;
pub type RealFreqMatrix32xN = MatrixMxN<RealFreqVector32, f32>;
pub type RealFreqMatrix64xN = MatrixMxN<RealFreqVector64, f64>;
pub type ComplexFreqMatrix32xN = MatrixMxN<ComplexFreqVector32, f32>;
pub type ComplexFreqMatrix64xN = MatrixMxN<ComplexFreqVector64, f64>;

pub type Matrix32x2 = Matrix2xN<DataVector32, f32>;
pub type Matrix64x2 = Matrix2xN<DataVector64, f32>;
pub type RealTimeMatrix32x2 = Matrix2xN<RealTimeVector32, f32>;
pub type RealTimeMatrix64x2 = Matrix2xN<RealTimeVector64, f64>;
pub type ComplexTimeMatrix32x2 = Matrix2xN<ComplexTimeVector32, f32>;
pub type ComplexTimeMatrix64x2 = Matrix2xN<ComplexTimeVector64, f64>;
pub type RealFreqMatrix32x2 = Matrix2xN<RealFreqVector32, f32>;
pub type RealFreqMatrix64x2 = Matrix2xN<RealFreqVector64, f64>;
pub type ComplexFreqMatrix32x2 = Matrix2xN<ComplexFreqVector32, f32>;
pub type ComplexFreqMatrix64x2 = Matrix2xN<ComplexFreqVector64, f64>;

pub type Matrix32x3 = Matrix3xN<DataVector32, f32>;
pub type Matrix64x3 = Matrix3xN<DataVector64, f32>;
pub type RealTimeMatrix32x3 = Matrix3xN<RealTimeVector32, f32>;
pub type RealTimeMatrix64x3 = Matrix3xN<RealTimeVector64, f64>;
pub type ComplexTimeMatrix32x3 = Matrix3xN<ComplexTimeVector32, f32>;
pub type ComplexTimeMatrix64x3 = Matrix3xN<ComplexTimeVector64, f64>;
pub type RealFreqMatrix32x3 = Matrix3xN<RealFreqVector32, f32>;
pub type RealFreqMatrix64x3 = Matrix3xN<RealFreqVector64, f64>;
pub type ComplexFreqMatrix32x3 = Matrix3xN<ComplexFreqVector32, f32>;
pub type ComplexFreqMatrix64x3 = Matrix3xN<ComplexFreqVector64, f64>;

pub type Matrix32x4 = Matrix4xN<DataVector32, f32>;
pub type Matrix64x4 = Matrix4xN<DataVector64, f32>;
pub type RealTimeMatrix32x4 = Matrix4xN<RealTimeVector32, f32>;
pub type RealTimeMatrix64x4 = Matrix4xN<RealTimeVector64, f64>;
pub type ComplexTimeMatrix32x4 = Matrix4xN<ComplexTimeVector32, f32>;
pub type ComplexTimeMatrix64x4 = Matrix4xN<ComplexTimeVector64, f64>;
pub type RealFreqMatrix32x4 = Matrix4xN<RealFreqVector32, f32>;
pub type RealFreqMatrix64x4 = Matrix4xN<RealFreqVector64, f64>;
pub type ComplexFreqMatrix32x4 = Matrix4xN<ComplexFreqVector32, f32>;
pub type ComplexFreqMatrix64x4 = Matrix4xN<ComplexFreqVector64, f64>;

impl<V, T> MatrixMxN<V, T>
    where T: RealNumber,
      V: DataVector<T> {
    pub fn new(rows: Vec<V>) -> Result<Self, ErrorReason> {
        if rows.len() < 1 {
            return Err(ErrorReason::InvalidArgumentLength);
        }

        let first_len = rows[0].len();
        let first_complex = rows[0].is_complex();
        let first_domain = rows[0].domain();
        for v in &rows[1..rows.len()] {
            if v.len() != first_len {
                return Err(ErrorReason::VectorsMustHaveTheSameSize);
            }

            if v.is_complex() != first_complex {
                return Err(ErrorReason::VectorMetaDataMustAgree);
            }

            if v.domain() != first_domain {
                return Err(ErrorReason::VectorMetaDataMustAgree);
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
      V: DataVector<T> {
    pub fn new(rows: [V; 2]) -> Result<Self, ErrorReason> {
        let first_len = rows[0].len();
        let first_complex = rows[0].is_complex();
        let first_domain = rows[0].domain();
        for v in &rows[1..rows.len()] {
            if v.len() != first_len {
                return Err(ErrorReason::VectorsMustHaveTheSameSize);
            }

            if v.is_complex() != first_complex {
                return Err(ErrorReason::VectorMetaDataMustAgree);
            }

            if v.domain() != first_domain {
                return Err(ErrorReason::VectorMetaDataMustAgree);
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
      V: DataVector<T> {
    pub fn new(rows: [V; 3]) -> Result<Self, ErrorReason> {
        let first_len = rows[0].len();
        let first_complex = rows[0].is_complex();
        let first_domain = rows[0].domain();
        for v in &rows[1..rows.len()] {
            if v.len() != first_len {
                return Err(ErrorReason::VectorsMustHaveTheSameSize);
            }

            if v.is_complex() != first_complex {
                return Err(ErrorReason::VectorMetaDataMustAgree);
            }

            if v.domain() != first_domain {
                return Err(ErrorReason::VectorMetaDataMustAgree);
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
      V: DataVector<T> {
    pub fn new(rows: [V; 4]) -> Result<Self, ErrorReason> {
        let first_len = rows[0].len();
        let first_complex = rows[0].is_complex();
        let first_domain = rows[0].domain();
        for v in &rows[1..rows.len()] {
            if v.len() != first_len {
                return Err(ErrorReason::VectorsMustHaveTheSameSize);
            }

            if v.is_complex() != first_complex {
                return Err(ErrorReason::VectorMetaDataMustAgree);
            }

            if v.domain() != first_domain {
                return Err(ErrorReason::VectorMetaDataMustAgree);
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
                  V: DataVector<T> {
                fn get_row(&self, row: usize) -> &[T] {
                    self.rows[row].data()
                }

                fn delta(&self) -> T {
                    self.rows[0].delta()
                }

                fn domain(&self) -> DataVectorDomain {
                    self.rows[0].domain()
                }

                fn is_complex(&self) -> bool {
                    self.rows[0].is_complex()
                }
            }

            impl<V, T> Index<usize> for $matrix<V, T>
                where T: RealNumber,
                  V: DataVector<T> {
                type Output = [T];

                fn index(&self, index: usize) -> &[T] {
                    self.get_row(index)
                }
            }

            impl<V, T> Index<(usize, usize)> for $matrix<V, T>
                where T: RealNumber,
                  V: DataVector<T> {
                type Output = T;

                fn index(&self, index: (usize, usize)) -> &T {
                    let vec = self.get_row(index.0);
                    &vec[index.1]
                }
            }

            impl<V, T> IndexMut<usize> for $matrix<V, T>
                where T: RealNumber,
                  V: DataVector<T> + IndexMut<Range<usize>, Output=[T]> {
                fn index_mut(&mut self, index: usize) -> &mut [T] {
                    let vec = &mut self.rows[index];
                    let len = vec.len();
                    &mut vec[0..len]
                }
            }

            impl<V, T> IndexMut<(usize, usize)> for $matrix<V, T>
                where T: RealNumber,
                  V: DataVector<T> + IndexMut<usize, Output=T> {
                fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
                    let vec = &mut self.rows[index.0];
                    &mut vec[index.1]
                }
            }
        )*
    }
}

add_basic_impl!(MatrixMxN; Matrix2xN; Matrix3xN; Matrix4xN);
