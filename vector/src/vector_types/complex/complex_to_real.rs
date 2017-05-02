use multicore_support::*;
use simd_extensions::*;
use numbers::*;
use std::ops::*;
use super::super::{ToRealResult, ErrorReason, Buffer, Vector, Resize, MetaData,
                   DspVec, ToSliceMut, ToSlice, VoidResult, Domain, ComplexNumberSpace,
                   RededicateForceOps, RealNumberSpace, NumberSpace};

/// Defines transformations from complex to real number space.
///
/// # Failures
/// All operations in this trait set `self.len()` to `0` if the type isn't in
/// the complex number space.
pub trait ComplexToRealTransformsOps<T>: ToRealResult
    where T: RealNumber
{
    /// Gets the absolute value, magnitude or norm of all vector elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(3.0, -4.0, -3.0, 4.0).to_complex_time_vec();
    /// let result = vector.magnitude();
    /// assert_eq!([5.0, 5.0], result[0..]);
    /// # }
    /// ```
    fn magnitude(self) -> Self::RealResult;

    /// Gets the square root of the absolute value of all vector elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(3.0, -4.0, -3.0, 4.0).to_complex_time_vec();
    /// let result = vector.magnitude_squared();
    /// assert_eq!([25.0, 25.0], result[0..]);
    /// # }
    /// ```
    fn magnitude_squared(self) -> Self::RealResult;

    /// Gets all real elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(1.0, 2.0, 3.0, 4.0).to_complex_time_vec();
    /// let result = vector.to_real();
    /// assert_eq!([1.0, 3.0], result[0..]);
    /// # }
    /// ```
    fn to_real(self) -> Self::RealResult;

    /// Gets all imag elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(1.0, 2.0, 3.0, 4.0).to_complex_time_vec();
    /// let result = vector.to_imag();
    /// assert_eq!([2.0, 4.0], result[0..]);
    /// # }
    /// ```
    fn to_imag(self) -> Self::RealResult;

    /// Gets the phase of all elements in [rad].
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let data: Vec<f32> = vec!(1.0, 0.0, 0.0, 4.0, -2.0, 0.0, 0.0, -3.0, 1.0, 1.0);
    /// let vector = data.to_complex_time_vec();
    /// let result = vector.phase();
    /// assert_eq!([0.0, 1.5707964, 3.1415927, -1.5707964, 0.7853982], result[0..]);
    /// # }
    /// ```
    fn phase(self) -> Self::RealResult;
}

/// Defines transformations from complex to real number space.
///
/// # Failures
/// All operations in this trait set `self.len()` to `0` if the type isn't in
/// the complex number space.
pub trait ComplexToRealTransformsOpsBuffered<S, T>: ToRealResult
    where S: ToSliceMut<T>,
          T: RealNumber
{
    /// Gets the absolute value, magnitude or norm of all vector elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(3.0, -4.0, -3.0, 4.0).to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.magnitude_b(&mut buffer);
    /// assert_eq!([5.0, 5.0], result[0..]);
    /// # }
    /// ```
    fn magnitude_b<B>(self, buffer: &mut B) -> Self::RealResult where B: for<'a> Buffer<'a, S, T>;

    /// Gets the square root of the absolute value of all vector elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(3.0, -4.0, -3.0, 4.0).to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.magnitude_squared_b(&mut buffer);
    /// assert_eq!([25.0, 25.0], result[0..]);
    /// # }
    /// ```
    fn magnitude_squared_b<B>(self, buffer: &mut B) -> Self::RealResult where B: for<'a> Buffer<'a, S, T>;

    /// Gets all real elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(1.0, 2.0, 3.0, 4.0).to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.to_real_b(&mut buffer);
    /// assert_eq!([1.0, 3.0], result[0..]);
    /// # }
    /// ```
    fn to_real_b<B>(self, buffer: &mut B) -> Self::RealResult where B: for<'a> Buffer<'a, S, T>;

    /// Gets all imag elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(1.0, 2.0, 3.0, 4.0).to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.to_imag_b(&mut buffer);
    /// assert_eq!([2.0, 4.0], result[0..]);
    /// # }
    /// ```
    fn to_imag_b<B>(self, buffer: &mut B) -> Self::RealResult where B: for<'a> Buffer<'a, S, T>;

    /// Gets the phase of all elements in [rad].
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let data: Vec<f32> = vec!(1.0, 0.0, 0.0, 4.0, -2.0, 0.0, 0.0, -3.0, 1.0, 1.0);
    /// let vector = data.to_complex_time_vec();
    /// let mut buffer = SingleBuffer::new();
    /// let result = vector.phase_b(&mut buffer);
    /// assert_eq!([0.0, 1.5707964, 3.1415927, -1.5707964, 0.7853982], result[0..]);
    /// # }
    /// ```
    fn phase_b<B>(self, buffer: &mut B) -> Self::RealResult where B: for<'a> Buffer<'a, S, T>;
}


/// Defines getters to get real data from complex types.
///
/// # Failures
/// All operations in this trait set the arguments `len()` to `0` if the type isn't in
/// the complex number space.
pub trait ComplexToRealGetterOps<A, N, D>
    where N: NumberSpace,
          D: Domain,
          A: MetaData<N, D>
{
    /// Copies all real elements into the given vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(1.0, 2.0, 3.0, 4.0).to_complex_time_vec();
    /// let mut target = Vec::new().to_real_time_vec();
    /// vector.get_real(&mut target);
    /// assert_eq!([1.0, 3.0], target[..]);
    /// # }
    /// ```
    fn get_real(&self, destination: &mut A);

    /// Copies all imag elements into the given vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(1.0, 2.0, 3.0, 4.0).to_complex_time_vec();
    /// let mut target = Vec::new().to_real_time_vec();
    /// vector.get_imag(&mut target);
    /// assert_eq!([2.0, 4.0], target[..]);
    /// # }
    /// ```
    fn get_imag(&self, destination: &mut A);

    /// Copies the absolute value or magnitude of all vector elements into the given target vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(3.0, -4.0, -3.0, 4.0).to_complex_time_vec();
    /// let mut target = Vec::new().to_real_time_vec();
    /// vector.get_magnitude(&mut target);
    /// assert_eq!([5.0, 5.0], target[..]);
    /// # }
    /// ```
    fn get_magnitude(&self, destination: &mut A);

    /// Copies the absolute value squared or magnitude squared of all vector elements
    /// into the given target vector.
    /// # Example
    ///
    /// Copies the absolute value or magnitude of all vector elements into the
    /// given target vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector = vec!(3.0, -4.0, -3.0, 4.0).to_complex_time_vec();
    /// let mut target = Vec::new().to_real_time_vec();
    /// vector.get_magnitude_squared(&mut target);
    /// assert_eq!([25.0, 25.0], target[..]);
    /// # }
    /// ```
    fn get_magnitude_squared(&self, destination: &mut A);

    /// Copies the phase of all elements in [rad] into the given vector.
    /// # Example
    ///
    /// ```
    /// # use std::f64;
    /// # extern crate num_complex;
    /// # extern crate basic_dsp_vector;
    /// use basic_dsp_vector::*;
    /// # fn main() {
    /// let vector =
    ///     vec!(1.0, 0.0, 0.0, 4.0, -2.0, 0.0, 0.0, -3.0, 1.0, 1.0).to_complex_time_vec();
    /// let mut target = Vec::new().to_real_time_vec();
    /// vector.get_phase(&mut target);
    /// let actual = &target[..];
    /// let expected = &[0.0, 1.5707964, 3.1415927, -1.5707964, 0.7853982];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!(f64::abs(actual[i] - expected[i]) < 1e-4);
    /// }
    /// # }
    /// ```
    fn get_phase(&self, destination: &mut A);

    /// Gets the real and imaginary parts and stores them in the given vectors.
    /// See also  [`get_phase`](trait.ComplexVectorOps.html#tymethod.get_phase) and
    /// [`get_complex_abs`](trait.ComplexVectorOps.html#tymethod.get_complex_abs) for further
    /// information.
    fn get_real_imag(&self, real: &mut A, imag: &mut A);

    /// Gets the magnitude and phase and stores them in the given vectors.
    /// See also [`get_real`](trait.ComplexVectorOps.html#tymethod.get_real) and
    /// [`get_imag`](trait.ComplexVectorOps.html#tymethod.get_imag) for further
    /// information.
    fn get_mag_phase(&self, mag: &mut A, phase: &mut A);
}

/// Defines setters to create complex data from real data.
///
/// # Failures
/// All operations in this trait set `self.len()` to `0` if the type isn't in
/// the complex number space.
pub trait ComplexToRealSetterOps<A, N, D>
    where N: NumberSpace,
          D: Domain,
          A: MetaData<N, D>
{
    /// Overrides the `self` vectors data with the real and imaginary data in the given vectors.
    /// `real` and `imag` must have the same size.
    fn set_real_imag(&mut self, real: &A, imag: &A) -> VoidResult;

    /// Overrides the `self` vectors data with the magnitude and phase data in the given vectors.
    /// Note that `self` vector will immediately convert the data into a real and
    /// imaginary representation of the complex numbers which is its default format.
    /// `mag` and `phase` must have the same size.
    fn set_mag_phase(&mut self, mag: &A, phase: &A) -> VoidResult;
}

macro_rules! assert_complex {
    ($self_: ident) => {
        if !$self_.is_complex() {
            $self_.number_space.to_real();
            $self_.valid_len = 0;
            return Self::RealResult::rededicate_from_force($self_);
        }
    }
}

impl<S, T, N, D> ComplexToRealTransformsOps<T> for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: ToRealResult,
          <DspVec<S, T, N, D> as ToRealResult>::RealResult: RededicateForceOps<DspVec<S, T, N, D>>,
          S: ToSliceMut<T>,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: Domain
{
    fn magnitude(mut self) -> Self::RealResult {
        assert_complex!(self);
        self.pure_complex_to_real_operation_inplace(|x, _arg| x.norm(), ());
        self.number_space.to_real();
        Self::RealResult::rededicate_from_force(self)
    }

    fn magnitude_squared(mut self) -> Self::RealResult {
        assert_complex!(self);
        self.pure_complex_to_real_operation_inplace(|x, _arg| x.re * x.re + x.im * x.im, ());
        self.number_space.to_real();
        Self::RealResult::rededicate_from_force(self)
    }
    fn to_real(mut self) -> Self::RealResult {
        assert_complex!(self);
        self.pure_complex_to_real_operation_inplace(|x, _arg| x.re, ());
        self.number_space.to_real();
        Self::RealResult::rededicate_from_force(self)
    }
    fn to_imag(mut self) -> Self::RealResult {
        assert_complex!(self);
        self.pure_complex_to_real_operation_inplace(|x, _arg| x.im, ());
        self.number_space.to_real();
        Self::RealResult::rededicate_from_force(self)
    }

    fn phase(mut self) -> Self::RealResult {
        assert_complex!(self);
        self.pure_complex_to_real_operation_inplace(|x, _arg| x.arg(), ());
        self.number_space.to_real();
        Self::RealResult::rededicate_from_force(self)
    }
}

impl<S, T, N, D> ComplexToRealTransformsOpsBuffered<S, T> for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: ToRealResult,
          <DspVec<S, T, N, D> as ToRealResult>::RealResult: RededicateForceOps<DspVec<S, T, N, D>>,
          S: ToSliceMut<T>,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: Domain
{
    fn magnitude_b<B>(mut self, buffer: &mut B) -> Self::RealResult
        where B: for<'a> Buffer<'a, S, T>
    {
        assert_complex!(self);
        self.simd_complex_to_real_operation(buffer,
                                            |x, _arg| x.complex_abs(),
                                            |x, _arg| x.norm(),
                                            (),
                                            Complexity::Small);
        self.number_space.to_real();
        Self::RealResult::rededicate_from_force(self)
    }

    fn magnitude_squared_b<B>(mut self, buffer: &mut B) -> Self::RealResult
        where B: for<'a> Buffer<'a, S, T>
    {
        assert_complex!(self);
        self.simd_complex_to_real_operation(buffer,
                                            |x, _arg| x.complex_abs_squared(),
                                            |x, _arg| x.re * x.re + x.im * x.im,
                                            (),
                                            Complexity::Small);
        self.number_space.to_real();
        Self::RealResult::rededicate_from_force(self)
    }

    fn to_real_b<B>(mut self, buffer: &mut B) -> Self::RealResult
        where B: for<'a> Buffer<'a, S, T>
    {
        assert_complex!(self);
        self.pure_complex_to_real_operation(buffer, |x, _arg| x.re, (), Complexity::Small);
        self.number_space.to_real();
        Self::RealResult::rededicate_from_force(self)
    }

    fn to_imag_b<B>(mut self, buffer: &mut B) -> Self::RealResult
        where B: for<'a> Buffer<'a, S, T>
    {
        assert_complex!(self);
        self.pure_complex_to_real_operation(buffer, |x, _arg| x.im, (), Complexity::Small);
        self.number_space.to_real();
        Self::RealResult::rededicate_from_force(self)
    }

    fn phase_b<B>(mut self, buffer: &mut B) -> Self::RealResult
        where B: for<'a> Buffer<'a, S, T>
    {
        assert_complex!(self);
        self.pure_complex_to_real_operation(buffer, |x, _arg| x.arg(), (), Complexity::Small);
        self.number_space.to_real();
        Self::RealResult::rededicate_from_force(self)
    }
}

impl<S, T, N, D> DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: Domain
{
    #[inline]
    fn pure_complex_into_real_target_operation<A, F, V, NR>(&self,
                                                        destination: &mut V,
                                                        op: F,
                                                        argument: A,
                                                        complexity: Complexity)
        where A: Sync + Copy + Send,
              F: Fn(Complex<T>, A) -> T + 'static + Sync,
              V: Vector<T, NR, D> + Index<Range<usize>, Output = [T]> + IndexMut<Range<usize>>,
              NR: RealNumberSpace
    {
        let len = self.len();
        destination.resize(len / 2)
            .expect("Target should be real and thus all values for len / 2 should be valid");
        destination.set_delta(self.delta);
        let mut array = &mut destination[0..len / 2];
        let source = &self.data.to_slice();
        Chunk::from_src_to_dest(complexity,
                                &self.multicore_settings,
                                &source[0..len],
                                2,
                                array,
                                1,
                                argument,
                                move |original, range, target, argument| {
            let mut i = range.start;
            let mut j = 0;
            while j < target.len() {
                let complex = Complex::<T>::new(original[i], original[i + 1]);
                target[j] = op(complex, argument);
                i += 2;
                j += 1;
            }
        });
    }

    #[inline]
    fn simd_complex_into_real_target_operation<FSimd, F, V, NR>(&self,
                                                            destination: &mut V,
                                                            op_simd: FSimd,
                                                            op: F,
                                                            complexity: Complexity)
        where F: Fn(Complex<T>) -> T + 'static + Sync,
              FSimd: Fn(T::Reg) -> T::Reg + 'static + Sync,
              V: Vector<T, NR, D> + Index<Range<usize>, Output = [T]> + IndexMut<Range<usize>>,
              NR: RealNumberSpace
    {
        let data_length = self.len();
        destination.resize(data_length / 2)
            .expect("Target should be real and thus all values for len / 2 should be valid");;
        destination.set_delta(self.delta);
        let mut temp = &mut destination[0..data_length / 2];
        let (scalar_left, scalar_right, vectorization_length) =
            T::Reg::calc_data_alignment_reqs(temp);
        let array = &self.data.to_slice();
        if vectorization_length > 0 {
            Chunk::from_src_to_dest(complexity,
                                    &self.multicore_settings,
                                    &array[2 * scalar_left..2 * vectorization_length],
                                    T::Reg::len(),
                                    &mut temp[scalar_left..vectorization_length],
                                    T::Reg::len() / 2,
                                    (),
                                    move |array, range, target, _arg| {
                let mut i = 0;
                let array = T::Reg::array_to_regs(&array[range]);
                for n in array {
                    let result = op_simd(*n);
                    result.store_half_unchecked(target, i);
                    i += T::Reg::len() / 2;
                }
            });
        }

        let mut i = 0;
        while i < 2 * scalar_left {
            let c = Complex::new(array[i], array[i + 1]);
            temp[i / 2] = op(c);
            i += 2;
        }

        let mut i = 2 * scalar_right;
        while i < data_length {
            let c = Complex::new(array[i], array[i + 1]);
            temp[i / 2] = op(c);
            i += 2;
        }
    }
}

macro_rules! assert_self_complex_and_target_real {
    ($self_: ident, $target: ident) => {
        if !$self_.is_complex() || $target.is_complex() {
            $target.resize(0).unwrap();
            return;
        }
    }
}

macro_rules! assert_self_complex_and_targets_real {
    ($self_: ident, $real: ident, $imag: ident) => {
        if !$self_.is_complex() || $real.is_complex() || $imag.is_complex() {
            $real.resize(0).unwrap();
            $imag.resize(0).unwrap();
            return;
        }
    }
}

impl<S, T, N, NR, D, O> ComplexToRealGetterOps<O, NR, D> for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: ToRealResult,
          O:
            Vector<T, NR, D>
            + Index<Range<usize>, Output=[T]>
            + IndexMut<Range<usize>>,
          S: ToSlice<T>,
          T: RealNumber,
          N: ComplexNumberSpace,
          NR: RealNumberSpace,
          D: Domain {
    fn get_real(&self, destination: &mut O) {
        assert_self_complex_and_target_real!(self, destination);
        destination.resize(self.len()).expect("Target is real and thus all values are valid");
        self.pure_complex_into_real_target_operation(
            destination, |x,_arg|x.re, (), Complexity::Small);
    }

    fn get_imag(&self, destination: &mut O) {
        assert_self_complex_and_target_real!(self, destination);
        destination.resize(self.len()).expect("Target is real and thus all values are valid");
        self.pure_complex_into_real_target_operation(
            destination, |x,_arg|x.im, (), Complexity::Small);
    }

    fn get_magnitude(&self, destination: &mut O) {
        assert_self_complex_and_target_real!(self, destination);
        destination.resize(self.len()).expect("Target is real and thus all values are valid");
        self.simd_complex_into_real_target_operation(
            destination, |x|x.complex_abs(), |x|x.norm(), Complexity::Small);
    }

    fn get_magnitude_squared(&self, destination: &mut O) {
        assert_self_complex_and_target_real!(self, destination);
        destination.resize(self.len()).expect("Target is real and thus all values are valid");
        self.simd_complex_into_real_target_operation(
            destination,
            |x|x.complex_abs_squared(),
            |x|x.re * x.re + x.im * x.im,
            Complexity::Small);
    }

    fn get_phase(&self, destination: &mut O) {
        assert_self_complex_and_target_real!(self, destination);
        destination.resize(self.len()).expect("Target is real and thus all values are valid");
        self.pure_complex_into_real_target_operation(
            destination, |x,_arg|x.arg(), (), Complexity::Small)
    }

    fn get_real_imag(&self, real: &mut O, imag: &mut O) {
        assert_self_complex_and_targets_real!(self, real, imag);
        let data_length = self.len();
        real.resize(data_length / 2).expect(
            "Target should be real and thus all values for len / 2 should be valid");
        imag.resize(data_length / 2).expect(
            "Target should be real and thus all values for len / 2 should be valid");
        let real = &mut real[0..data_length / 2];
        let imag = &mut imag[0..data_length / 2];
        let data = self.data.to_slice();
        for (i, n) in (&data[0..data_length]).iter().enumerate() {
            if i % 2 == 0 {
                real[i / 2] = *n;
            } else {
                imag[i / 2] = *n;
            }
        }
    }

    fn get_mag_phase(&self, mag: &mut O, phase: &mut O) {
        assert_self_complex_and_targets_real!(self, mag, phase);
        let data_length = self.len();
        mag.resize(data_length / 2).expect(
            "Target should be real and thus all values for len / 2 should be valid");
        phase.resize(data_length / 2).expect(
            "Target should be real and thus all values for len / 2 should be valid");
        let mag = &mut mag[0..data_length / 2];
        let phase = &mut phase[0..data_length / 2];
        let data = self.data.to_slice();
        let mut i = 0;
        while i < data_length {
            let c = Complex::<T>::new(data[i], data[i + 1]);
            let (m, p) = c.to_polar();
            mag[i / 2] = m;
            phase[i / 2] = p;
            i += 2;
        }
    }
}

impl<S, T, N, NR, D, O> ComplexToRealSetterOps<O, NR, D> for DspVec<S, T, N, D>
    where DspVec<S, T, N, D>: ToRealResult,
          O:
            Index<Range<usize>, Output=[T]>
            + Vector<T, NR, D>,
          S: ToSliceMut<T> + Resize,
          T: RealNumber,
          N: ComplexNumberSpace,
          NR: RealNumberSpace,
          D: Domain {
    fn set_real_imag(&mut self, real: &O, imag: &O) -> VoidResult {
        {
            if real.len() != imag.len() {
                return Err(ErrorReason::InvalidArgumentLength);
            }
            let data_length = real.len() + imag.len();
            self.data.resize(data_length);
            self.valid_len = data_length;
            let data = self.data.to_slice_mut();
            let real = &real[0..real.len()];
            let imag = &imag[0..imag.len()];
            for (i, n) in (&mut data[0..data_length]).iter_mut().enumerate() {
                if i % 2 == 0 {
                    *n = real[i / 2];
                } else {
                    *n = imag[i / 2];
                }
            }
        }

        Ok(())
    }

    fn set_mag_phase(&mut self, mag: &O, phase: &O) -> VoidResult {
        {
            if mag.len() != phase.len() {
                return Err(ErrorReason::InvalidArgumentLength);
            }
            let data_length = mag.len() + phase.len();
            self.data.resize(data_length);
            self.valid_len = data_length;
            let data = self.data.to_slice_mut();
            let mag = &mag[0..mag.len()];
            let phase = &phase[0..phase.len()];
            let mut i = 0;
            while i < data_length {
                let c = Complex::<T>::from_polar(&mag[i / 2], &phase[i / 2]);
                data[i] = c.re;
                data[i + 1] = c.im;
                i += 2;
            }
        }

        Ok(())
    }
}
