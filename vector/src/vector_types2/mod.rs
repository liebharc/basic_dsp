use std::marker::PhantomData;
use std::ops::*;
use RealNumber;

mod requirements;
pub use self::requirements::*;

pub struct DspVec<S, T>
    where S: ToSlice<T>,
          T: RealNumber {
    data: S,
    _data_type: PhantomData<T>
}

pub type DspVec32 = DspVec<Vec<f32>, f32>;

pub type DspSlice32<'a> = DspVec<&'a [f32], f32>;

pub type DspSliceMut32<'a> = DspVec<&'a mut [f32], f32>;

impl DspVec32 {
    pub fn new(values: Vec<f32>) -> Self {
        DspVec32 { data: values, _data_type: PhantomData }
    }
}

impl<'a> DspSlice32<'a> {
    pub fn new(values: &'a [f32]) -> Self {
        DspSlice32 { data: values, _data_type: PhantomData }
    }
}

impl<'a> DspSliceMut32<'a> {
    pub fn new(values: &'a mut [f32]) -> Self {
        DspSliceMut32 { data: values, _data_type: PhantomData }
    }
}

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
