use super::super::{
    Buffer, Domain, DspVec, ErrorReason, NumberSpace, RealOrComplexData, RededicateForceOps,
    TimeOrFrequencyData, ToSlice, ToSliceMut,
};
use inline_vector::InlineVector;
use numbers::*;
use std::result;

use super::{
    generic_vector_from_any_vector, Identifier, Operation, PreparedOperation1,
    PreparedOperation1Exec, PreparedOperation2, PreparedOperation2Exec,
};

/// A multi operation which holds a vector and records all changes
/// which need to be done to the vectors. By calling `get` on the struct
/// all operations will be executed in one run.
pub struct MultiOperation1<S, T, NO, DO>
where
    S: ToSlice<T>,
    T: RealNumber,
    NO: NumberSpace,
    DO: Domain,
{
    a: DspVec<S, T, RealOrComplexData, TimeOrFrequencyData>,
    prepared_ops: PreparedOperation1<T, RealOrComplexData, TimeOrFrequencyData, NO, DO>,
}

/// A multi operation which holds a vector and records all changes
/// which need to be done to the vectors. By calling `get` on the struct
/// all operations will be executed in one run.
pub struct MultiOperation2<S, T, NO1, DO1, NO2, DO2>
where
    S: ToSlice<T>,
    T: RealNumber,
    NO1: NumberSpace,
    DO1: Domain,
    NO2: NumberSpace,
    DO2: Domain,
{
    a: DspVec<S, T, RealOrComplexData, TimeOrFrequencyData>,
    b: DspVec<S, T, RealOrComplexData, TimeOrFrequencyData>,
    prepared_ops: PreparedOperation2<
        T,
        RealOrComplexData,
        TimeOrFrequencyData,
        RealOrComplexData,
        TimeOrFrequencyData,
        NO1,
        DO1,
        NO2,
        DO2,
    >,
}

/// Creates a new multi operation for one vectors.
pub fn multi_ops1<S, T, NI, DI>(vector: DspVec<S, T, NI, DI>) -> MultiOperation1<S, T, NI, DI>
where
    S: ToSliceMut<T>,
    DspVec<S, T, NI, DI>: RededicateForceOps<DspVec<S, T, RealOrComplexData, TimeOrFrequencyData>>,
    T: RealNumber,
    NI: NumberSpace,
    DI: Domain,
{
    let (number_space, domain, a) = generic_vector_from_any_vector(vector);
    let gen_number_space = RealOrComplexData {
        is_complex_current: number_space.is_complex(),
    };
    let gen_domain = TimeOrFrequencyData {
        domain_current: domain.domain(),
    };
    let ops: PreparedOperation1<T, RealOrComplexData, TimeOrFrequencyData, NI, DI> =
        PreparedOperation1 {
            number_space_in: gen_number_space,
            domain_in: gen_domain,
            number_space_out: number_space,
            domain_out: domain,
            ops: InlineVector::with_default_capcacity(),
        };
    MultiOperation1 {
        a,
        prepared_ops: ops,
    }
}

/// Creates a new multi operation for two vectors.
pub fn multi_ops2<S, T, NI1, DI1, NI2, DI2>(
    a: DspVec<S, T, NI1, DI1>,
    b: DspVec<S, T, NI2, DI2>,
) -> MultiOperation2<S, T, NI1, DI1, NI2, DI2>
where
    S: ToSliceMut<T>,
    DspVec<S, T, NI1, DI1>:
        RededicateForceOps<DspVec<S, T, RealOrComplexData, TimeOrFrequencyData>>,
    DspVec<S, T, NI2, DI2>:
        RededicateForceOps<DspVec<S, T, RealOrComplexData, TimeOrFrequencyData>>,
    T: RealNumber,
    NI1: NumberSpace,
    DI1: Domain,
    NI2: NumberSpace,
    DI2: Domain,
{
    let (number_space1, domain1, a) = generic_vector_from_any_vector(a);
    let gen_number_space1 = RealOrComplexData {
        is_complex_current: number_space1.is_complex(),
    };
    let gen_domain1 = TimeOrFrequencyData {
        domain_current: domain1.domain(),
    };
    let (number_space2, domain2, b) = generic_vector_from_any_vector(b);
    let gen_number_space2 = RealOrComplexData {
        is_complex_current: number_space2.is_complex(),
    };
    let gen_domain2 = TimeOrFrequencyData {
        domain_current: domain2.domain(),
    };
    let ops: PreparedOperation2<
        T,
        RealOrComplexData,
        TimeOrFrequencyData,
        RealOrComplexData,
        TimeOrFrequencyData,
        NI1,
        DI1,
        NI2,
        DI2,
    > = PreparedOperation2 {
        number_space_in1: gen_number_space1,
        domain_in1: gen_domain1,
        number_space_in2: gen_number_space2,
        domain_in2: gen_domain2,
        number_space_out1: number_space1,
        domain_out1: domain1,
        number_space_out2: number_space2,
        domain_out2: domain2,
        ops: InlineVector::with_default_capcacity(),
        swap: false,
    };
    MultiOperation2 {
        a,
        b,
        prepared_ops: ops,
    }
}

/// Holds two vectors and records all operations which shall be done on the
/// vectors. A call to `get` then runs all recorded operations on the vectors
/// and returns them. See the modules description for why this can be beneficial.
impl<S, T, NO, DO> MultiOperation1<S, T, NO, DO>
where
    S: ToSliceMut<T>,
    DspVec<S, T, NO, DO>: RededicateForceOps<DspVec<S, T, RealOrComplexData, TimeOrFrequencyData>>,
    T: RealNumber,
    NO: NumberSpace,
    DO: Domain,
{
    /// Extends the operation to operate on one more vector.
    pub fn extend<NI2, DI2>(
        self,
        vector: DspVec<S, T, NI2, DI2>,
    ) -> MultiOperation2<S, T, NO, DO, NI2, DI2>
    where
        NI2: NumberSpace,
        DI2: Domain,
    {
        let (number_space, domain, b) = generic_vector_from_any_vector(vector);
        let number_space_gen = RealOrComplexData {
            is_complex_current: number_space.is_complex(),
        };
        let domain_gen = TimeOrFrequencyData {
            domain_current: domain.domain(),
        };

        let a = self.a;
        let ops = self.prepared_ops;
        let extend = PreparedOperation2 {
            number_space_in1: ops.number_space_in,
            domain_in1: ops.domain_in,
            number_space_in2: number_space_gen,
            domain_in2: domain_gen,
            number_space_out1: ops.number_space_out,
            domain_out1: ops.domain_out,
            number_space_out2: number_space,
            domain_out2: domain,
            ops: ops.ops,
            swap: false,
        };

        MultiOperation2 {
            a,
            b,
            prepared_ops: extend,
        }
    }

    /// Executes all recorded operations on the stored vector.
    #[cfg_attr(feature = "cargo-clippy", allow(clippy::type_complexity))]
    pub fn get<B>(
        self,
        buffer: &mut B,
    ) -> result::Result<DspVec<S, T, NO, DO>, (ErrorReason, DspVec<S, T, NO, DO>)>
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        self.prepared_ops.exec(buffer, self.a)
    }

    /// Adds new operations which will be executed with the next call to `get`
    ///
    /// As a background: The function `operation` will be executed immediately.
    /// It only operated on `Identifier` types and these serve as
    /// placeholder for vectors. Every operation done to an `Identifier`
    /// is recorded and will be executed on vectors if `get` is called.
    pub fn add_ops<F, NT, DT>(self, operation: F) -> MultiOperation1<S, T, NT, DT>
    where
        F: Fn(Identifier<T, NO, DO>) -> Identifier<T, NT, DT>,
        DT: Domain,
        NT: NumberSpace,
    {
        let ops = self.prepared_ops.add_ops(operation);
        MultiOperation1 {
            a: self.a,
            prepared_ops: ops,
        }
    }

    /// Allows to directly push an `Operation` enum to a `MultiOperation1`.
    /// This mainly exists as interop between Rust and other languages.
    pub fn add_enum_op(&mut self, op: Operation<T>) {
        self.prepared_ops.add_enum_op(op);
    }
}

/// Holds two vectors and records all operations which shall be done on the
/// vectors. A call to `get` then runs all recorded operations on the vectors
/// and returns them. See the modules description for why this can be beneficial.
impl<S, T, NO1, DO1, NO2, DO2> MultiOperation2<S, T, NO1, DO1, NO2, DO2>
where
    S: ToSliceMut<T>,
    DspVec<S, T, NO1, DO1>:
        RededicateForceOps<DspVec<S, T, RealOrComplexData, TimeOrFrequencyData>>,
    DspVec<S, T, NO2, DO2>:
        RededicateForceOps<DspVec<S, T, RealOrComplexData, TimeOrFrequencyData>>,
    T: RealNumber,
    NO1: NumberSpace,
    DO1: Domain,
    NO2: NumberSpace,
    DO2: Domain,
{
    /// Executes all recorded operations on the stored vector.
    #[cfg_attr(feature = "cargo-clippy", allow(clippy::type_complexity))]
    pub fn get<B>(
        self,
        buffer: &mut B,
    ) -> result::Result<
        (DspVec<S, T, NO1, DO1>, DspVec<S, T, NO2, DO2>),
        (ErrorReason, DspVec<S, T, NO1, DO1>, DspVec<S, T, NO2, DO2>),
    >
    where
        B: for<'a> Buffer<'a, S, T>,
    {
        self.prepared_ops.exec(buffer, self.a, self.b)
    }

    /// Adds new operations which will be executed with the next call to `get`
    ///
    /// As a background: The function `operation` will be executed immediately.
    /// It only operated on `Identifier` types and these serve as
    /// placeholder for vectors. Every operation done to an `Identifier`
    /// is recorded and will be executed on vectors if `get` is called.
    pub fn add_ops<F, NT1, DT1, NT2, DT2>(
        self,
        operation: F,
    ) -> MultiOperation2<S, T, NT1, DT1, NT2, DT2>
    where
        F: Fn(Identifier<T, NO1, DO1>, Identifier<T, NO2, DO2>)
            -> (Identifier<T, NT1, DT1>, Identifier<T, NT2, DT2>),
        DT1: Domain,
        NT1: NumberSpace,
        DT2: Domain,
        NT2: NumberSpace,
    {
        let ops = self.prepared_ops.add_ops(operation);
        MultiOperation2 {
            a: self.a,
            b: self.b,
            prepared_ops: ops,
        }
    }

    /// Allows to directly push an `Operation` enum to a `MultiOperation1`.
    /// This mainly exists as interop between Rust and other languages.
    pub fn add_enum_op(&mut self, op: Operation<T>) {
        self.prepared_ops.add_enum_op(op);
    }
}
