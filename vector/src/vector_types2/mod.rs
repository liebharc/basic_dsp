use std::ops::*;
use RealNumber;

mod requirements;
pub use self::requirements::*;
mod to_from_vec_conversions;
pub use self::to_from_vec_conversions::*;
mod vec_trait_and_indexers;
pub use self::vec_trait_and_indexers::*;
use vector_types::{
    DataVecDomain};
use multicore_support::{
    Chunk,
    Complexity,
    MultiCoreSettings};

macro_rules! define_vector_struct {
    ($($name:ident),*) => {
        $(
            #[derive(Debug)]
            /// A 1xN (one times N elements) or Nx1 data vector as used for most digital signal processing (DSP) operations.
            /// All data vector operations consume the vector they operate on and return a new vector. A consumed vector
            /// must not be accessed again.
            ///
            /// Vectors come in different flavors:
            ///
            /// 1. Time or Frequency domain
            /// 2. Real or Complex numbers
            /// 3. 32bit or 64bit floating point numbers
            ///
            /// The first two flavors define meta information about the vector and provide compile time information what
            /// operations are available with the given vector and how this will transform the vector. This makes sure that
            /// some invalid operations are already discovered at compile time. In case that this isn't desired or the information
            /// about the vector isn't known at compile time there are the generic [`DataVec32`](type.DataVec32.html) and [`DataVec64`](type.DataVec64.html) vectors
            /// available.
            ///
            /// 32bit and 64bit flavors trade performance and memory consumption against accuracy. 32bit vectors are roughly
            /// two times faster than 64bit vectors for most operations. But remember that you should benchmark first
            /// before you give away accuracy for performance unless however you are sure that 32bit accuracy is certainly good
            /// enough.
            pub struct $name<S, T>
                where S: ToSlice<T>,
                      T: RealNumber {
                data: S,
                delta: T,
                domain: DataVecDomain,
                is_complex: bool,
                valid_len: usize,
                multicore_settings: MultiCoreSettings
            }
        )*
    }
}
define_vector_struct!(
    GenDspVec,
    RealTimeVec, RealFreqVec,
    ComplexTimeVec, ComplexFreqVec);

/*
pub type DspVec32 = DspVec<Vec<f32>, f32>;

pub type DspSlice32<'a> = DspVec<&'a [f32], f32>;

pub type DspSliceMut32<'a> = DspVec<&'a mut [f32], f32>;

pub trait ReadOnlyOps<T>
    where T: RealNumber {
    fn sum(&self) -> T;
}

impl<S, T> ReadOnlyOps<T> for DspVec<S, T>
     where S: ToSlice<T>,
           T: RealNumber {
    fn sum(&self) -> T {
        let mut sum = T::zero();
        for i in self.data.to_slice() {
            sum = sum + *i;
        }
        sum
    }
}

pub trait MutOps<T>
    where T: RealNumber {
    fn add(&mut self, value: T);
}

impl<S, T> MutOps<T> for DspVec<S, T>
     where S: ToSliceMut<T>,
           T: RealNumber {
    fn add(&mut self, value: T) {
        for i in self.data.to_slice_mut() {
            *i = *i + value;
        }
    }
}

pub trait TransOps<T>
    where T: RealNumber {
    fn mag(self) -> Self;
}

impl<T> TransOps<T> for DspVec<Vec<T>, T>
    where T: RealNumber {
    fn mag(mut self) -> Self {
        for i in &mut self.data[..] {
            *i = i.abs();
        }
        self
    }
}

impl<S, T> Index<RangeFull> for DspVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber {
    type Output = [T];

    fn index(&self, _index: RangeFull) -> &[T] {
        self.data.to_slice()
    }
}
*/
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn len_of_vec() {
        let vec: Vec<f32> = vec!(1.0, 2.0, 3.0);
        let dsp = DspVec32::new(vec);
        assert_eq!(dsp.len(), 3);
    }

    #[test]
    fn len_of_slice() {
        let slice = [1.0, 5.0, 4.0];
        let dsp = DspSlice32::new(&slice);
        assert_eq!(dsp.len(), 3);
    }

    #[test]
    fn len_of_slice_mut() {
        let mut slice = [1.0, 5.0, 4.0];
        let dsp = DspSliceMut32::new(&mut slice);
        assert_eq!(dsp.len(), 3);
    }

    #[test]
    fn sum_of_vec() {
        let vec: Vec<f32> = vec!(1.0, 2.0, 3.0);
        let dsp = DspVec32::new(vec);
        let sum = dsp.sum();
        assert_eq!(sum, 6.0);
        assert_eq!(dsp.data, vec!(1.0, 2.0, 3.0));
    }

    #[test]
    fn sum_of_slice() {
        let slice = [1.0, 5.0, 4.0];
        let dsp = DspSlice32::new(&slice);
        let sum = dsp.sum();
        assert_eq!(sum, 10.0);
        assert_eq!(dsp.data, slice);
    }

    #[test]
    fn sum_of_slice_mut() {
        let mut slice = [1.0, 5.0, 4.0];
        let dsp = DspSliceMut32::new(&mut slice);
        let sum = dsp.sum();
        assert_eq!(sum, 10.0);
    }

    #[test]
    fn add_of_vec() {
        let vec: Vec<f32> = vec!(1.0, 2.0, 3.0);
        let mut dsp = DspVec32::new(vec);
        dsp.add(3.0);
        assert_eq!(dsp.data, vec!(4.0, 5.0, 6.0));
    }

    #[test]
    fn add_of_slice_mut() {
        let mut slice = [1.0, 5.0, 4.0];
        let mut dsp = DspSliceMut32::new(&mut slice);
        dsp.add(3.0);
        assert_eq!(dsp.data, [4.0, 8.0, 7.0]);
    }
    #[test]
    fn trans_of_vec() {
        let vec: Vec<f32> = vec!(1.0, -2.0, 3.0);
        let dsp = DspVec32::new(vec);
        let dsp = dsp.mag();
        assert_eq!(dsp.data, vec!(1.0, 2.0, 3.0));
    }

    #[test]
    fn index_of_vec() {
        let vec: Vec<f32> = vec!(1.0, 2.0, 3.0);
        let dsp = DspVec32::new(vec);
        assert_eq!(dsp[..], [1.0, 2.0, 3.0]);
    }

    #[test]
    fn index_of_slice() {
        let slice = [1.0, 5.0, 4.0];
        let dsp = DspSlice32::new(&slice);
        let dsp2 = DspSlice32::new(&dsp[..]);
        assert_eq!(dsp2[..], slice);
    }
}
