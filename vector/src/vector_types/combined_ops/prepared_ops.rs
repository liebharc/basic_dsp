use traits::*;
use std::result;
use std::sync::{Arc, Mutex};
use super::{generic_vector_from_any_vector, generic_vector_back_to_vector, Identifier,
            PreparedOperation1, PreparedOperation2, Operation};
use super::super::{Domain, NumberSpace, ToSliceMut, DspVec, ErrorReason, RededicateForceOps,
                   Buffer, Owner, TimeOrFrequencyData, RealOrComplexData};

const ARGUMENT1: usize = 0;
const ARGUMENT2: usize = 1;

/// Executes the prepared operations to convert `S`to `D`.
pub trait PreparedOperation1Exec<B, S, D> {
    /// Executes the prepared operations to convert `S`to `D`.
    fn exec(&self, buffer: &mut B, source: S) -> result::Result<D, (ErrorReason, D)>;
}

/// Prepares an operation with one input and one output.
pub fn prepare32_1<N, D>(number_space: N, domain: D) -> PreparedOperation1<f32, N, D, N, D>
    where N: NumberSpace,
          D: Domain
{
    PreparedOperation1 {
        number_space_in: number_space.clone(),
        domain_in: domain.clone(),
        number_space_out: number_space,
        domain_out: domain,
        ops: Vec::new(),
    }
}

/// Prepares an operation with one input and one output.
pub fn prepare64_1<N, D>(number_space: N, domain: D) -> PreparedOperation1<f64, N, D, N, D>
    where N: NumberSpace,
          D: Domain
{
    PreparedOperation1 {
        number_space_in: number_space.clone(),
        domain_in: domain.clone(),
        number_space_out: number_space,
        domain_out: domain,
        ops: Vec::new(),
    }
}

/// Executes the prepared operations to convert `S1` and `S2` to `D1` and `D2`.
pub trait PreparedOperation2Exec<B, S1, S2, D1, D2> {
    /// Executes the prepared operations to convert `S1` and `S2` to `D1` and `D2`.
    fn exec(&self,
            buffer: &mut B,
            source1: S1,
            source2: S2)
            -> result::Result<(D1, D2), (ErrorReason, D1, D2)>;
}

/// Prepares an operation with one input and one output.
pub fn prepare32_2<N1, D1, N2, D2>(number_space1: N1,
                                   domain1: D1,
                                   number_space2: N2,
                                   domain2: D2)
                                   -> PreparedOperation2<f32, N1, D1, N2, D2, N1, D1, N2, D2>
    where N1: NumberSpace,
          D1: Domain,
          N2: NumberSpace,
          D2: Domain
{
    PreparedOperation2 {
        number_space_in1: number_space1.clone(),
        domain_in1: domain1.clone(),
        number_space_in2: number_space2.clone(),
        domain_in2: domain2.clone(),
        number_space_out1: number_space1,
        domain_out1: domain1,
        number_space_out2: number_space2,
        domain_out2: domain2,
        ops: Vec::new(),
        swap: false,
    }
}

/// Prepares an operation with one input and one output.
pub fn prepare64_2<N1, D1, N2, D2>(number_space1: N1,
                                   domain1: D1,
                                   number_space2: N2,
                                   domain2: D2)
                                   -> PreparedOperation2<f64, N1, D1, N2, D2, N1, D1, N2, D2>
    where N1: NumberSpace,
          D1: Domain,
          N2: NumberSpace,
          D2: Domain
{
    PreparedOperation2 {
        number_space_in1: number_space1.clone(),
        domain_in1: domain1.clone(),
        number_space_in2: number_space2.clone(),
        domain_in2: domain2.clone(),
        number_space_out1: number_space1,
        domain_out1: domain1,
        number_space_out2: number_space2,
        domain_out2: domain2,
        ops: Vec::new(),
        swap: false,
    }
}

/// An operation which can be prepared in advance and operates on one
/// input and produces one output
impl<T, NI, DI, NO, DO> PreparedOperation1<T, NI, DI, NO, DO>
    where T: RealNumber + 'static,
          NI: NumberSpace,
          DI: Domain,
          NO: NumberSpace,
          DO: Domain
{
    /// Extends the operation to operate on one more vector.
    pub fn extend<NI2, DI2>(self,
                            number_space: NI2,
                            domain: DI2)
                            -> PreparedOperation2<T, NI, DI, NI2, DI2, NO, DO, NI2, DI2>
        where NI2: NumberSpace,
              DI2: Domain
    {
        PreparedOperation2 {
            number_space_in1: self.number_space_in,
            domain_in1: self.domain_in,
            number_space_in2: number_space.clone(),
            domain_in2: domain.clone(),
            number_space_out1: self.number_space_out,
            domain_out1: self.domain_out,
            number_space_out2: number_space,
            domain_out2: domain,
            ops: self.ops,
            swap: false,
        }
    }

    /// Adds new operations which will be executed with the next call to `exec`
    ///
    /// As a background: The function `operation` will be executed immediately.
    /// It only operated on `Identifier` types and these serve as
    /// placeholder for vectors. Every operation done to an `Identifier`
    /// is recorded and will be executed on vectors if `exec` is called.
    pub fn add_ops<F, NT, DT>(self, operation: F) -> PreparedOperation1<T, NI, DI, NT, DT>
        where F: Fn(Identifier<T, NO, DO>) -> Identifier<T, NT, DT>,
              DT: Domain,
              NT: NumberSpace
    {
        let mut ops = self.ops;
        let (new_ops, domain_new, number_space_new) = {
            let counter = Arc::new(Mutex::new(0));
            let t1 = Identifier {
                arg: ARGUMENT1,
                ops: Vec::new(),
                counter: counter,
                domain: self.domain_out.clone(),
                number_space: self.number_space_out.clone(),
            };

            let r1 = operation(t1);
            (r1.ops, r1.domain, r1.number_space)
        };
        for v in new_ops {
            ops.push(v.1);
        }

        PreparedOperation1 {
            domain_in: self.domain_in,
            number_space_in: self.number_space_in,
            domain_out: domain_new,
            number_space_out: number_space_new,
            ops: ops,
        }
    }

    /// Allows to directly push an `Operation` enum to a `PreparedOperation1`.
    /// This mainly exists as interop between Rust and other languages.
    pub fn add_enum_op(&mut self, op: Operation<T>) {
        self.ops.push(op);
    }
}

impl<T, S, B, NI, DI, NO, DO> PreparedOperation1Exec<B, DspVec<S, T, NI, DI>, DspVec<S, T, NO, DO>>
    for PreparedOperation1<T, NI, DI, NO, DO>
	where T: RealNumber + 'static,
        S: ToSliceMut<T> + Owner,
        DspVec<S, T, NO, DO>: RededicateForceOps<DspVec<S, T, NI, DI>>,
		NI: NumberSpace, DI: Domain,
		NO: NumberSpace, DO: Domain,
        B: Buffer<S, T> {
	/// Executes all recorded operations on the input vectors.
	fn exec(&self, buffer: &mut B, a: DspVec<S, T, NI, DI>)
        -> result::Result<DspVec<S, T, NO, DO>, (ErrorReason, DspVec<S, T, NO, DO>)> {

		let mut vec = Vec::new();
		let (number_space, domain, gen) = generic_vector_from_any_vector(a);
		vec.push(gen);


		// at this point we would execute all ops and cast the result to the right types
		let result =
            DspVec::<S, T, RealOrComplexData, TimeOrFrequencyData>::perform_operations(
                buffer, vec, &self.ops);

		match result {
			Err((reason, mut vec)) => {
				let a = vec.pop().unwrap();
				let a = generic_vector_back_to_vector(number_space, domain, a);
				Err((
					reason,
					DspVec::<S, T, NO, DO>::rededicate_from_force(a)))
			},
			Ok(mut vec) => {
				let a = vec.pop().unwrap();
				let a = generic_vector_back_to_vector(number_space, domain, a);
				// Convert back
				Ok(DspVec::<S, T, NO, DO>::rededicate_from_force(a))
			}
		}
	}
}

/// An operation which can be prepared in advance and operates on one
/// input and produces one output
impl<T, NI1, DI1, NI2, DI2, NO1, DO1, NO2, DO2> PreparedOperation2<T,
                                                                   NI1,
                                                                   DI1,
                                                                   NI2,
                                                                   DI2,
                                                                   NO1,
                                                                   DO1,
                                                                   NO2,
                                                                   DO2>
    where T: RealNumber + 'static,
          NI1: NumberSpace,
          DI1: Domain,
          NI2: NumberSpace,
          DI2: Domain,
          NO1: NumberSpace,
          DO1: Domain,
          NO2: NumberSpace,
          DO2: Domain
{
    /// Adds new operations which will be executed with the next call to `exec`
    ///
    /// As a background: The function `operation` will be executed immediately.
    /// It only operated on `Identifier` types and these serve as
    /// placeholder for vectors. Every operation done to an `Identifier`
    /// is recorded and will be executed on vectors if `exec` is called.
    pub fn add_ops<F, NT1, DT1, NT2, DT2>
        (self,
         operation: F)
         -> PreparedOperation2<T, NI1, DI1, NI2, DI2, NT1, DT1, NT2, DT2>
        where F: Fn(Identifier<T, NO1, DO1>, Identifier<T, NO2, DO2>)
                    -> (Identifier<T, NT1, DT1>, Identifier<T, NT2, DT2>),
              DT1: Domain,
              NT1: NumberSpace,
              DT2: Domain,
              NT2: NumberSpace
    {
        let mut ops = self.ops;
        let mut new_ops = Vec::new();
        let (swap, domain_new1, number_space_new1, domain_new2, number_space_new2) = {
            let counter = Arc::new(Mutex::new(0));
            let t1 = Identifier {
                arg: if self.swap { ARGUMENT2 } else { ARGUMENT1 },
                ops: Vec::new(),
                counter: counter.clone(),
                domain: self.domain_out1.clone(),
                number_space: self.number_space_out1.clone(),
            };

            let t2 = Identifier {
                arg: if self.swap { ARGUMENT1 } else { ARGUMENT2 },
                ops: Vec::new(),
                counter: counter,
                domain: self.domain_out2.clone(),
                number_space: self.number_space_out2.clone(),
            };

            let (mut r1, mut r2) = operation(t1, t2);
            let r1arg = r1.arg;
            new_ops.append(&mut r1.ops);
            new_ops.append(&mut r2.ops);
            (r1arg == ARGUMENT2, r1.domain, r1.number_space, r2.domain, r2.number_space)
        };

        // In theory the sequence counter could overflow.
        // In practice if we assume that the counter is increment
        // every microsecond than the program needs to run for 500,000
        // years until an overflow occurs.
        new_ops.sort_by(|a, b| a.0.cmp(&b.0));
        for v in new_ops {
            ops.push(v.1);
        }

        PreparedOperation2 {
            domain_in1: self.domain_in1,
            number_space_in1: self.number_space_in1,
            domain_in2: self.domain_in2,
            number_space_in2: self.number_space_in2,
            domain_out1: domain_new1,
            number_space_out1: number_space_new1,
            domain_out2: domain_new2,
            number_space_out2: number_space_new2,
            ops: ops,
            swap: swap,
        }
    }

    /// Allows to directly push an `Operation` enum to a `PreparedOperation1`.
    /// This mainly exists as interop between Rust and other languages.
    pub fn add_enum_op(&mut self, op: Operation<T>) {
        self.ops.push(op);
    }
}

impl<T, S, B, NI1, DI1, NI2, DI2, NO1, DO1, NO2, DO2> PreparedOperation2Exec<
        B, DspVec<S, T, NI1, DI1>, DspVec<S, T, NI2, DI2>,
        DspVec<S, T, NO1, DO1>, DspVec<S, T, NO2, DO2>>
    for PreparedOperation2<T, NI1, DI1, NI2, DI2, NO1, DO1, NO2, DO2>
	where T: RealNumber + 'static,
        S: ToSliceMut<T> + Owner,
        DspVec<S, T, NO1, DO1>: RededicateForceOps<DspVec<S, T, NI1, DI1>>,
        DspVec<S, T, NO2, DO2>: RededicateForceOps<DspVec<S, T, NI2, DI2>>,
		NI1: NumberSpace, DI1: Domain,
        NI2: NumberSpace, DI2: Domain,
		NO1: NumberSpace, DO1: Domain,
        NO2: NumberSpace, DO2: Domain,
        B: Buffer<S, T> {

    /// Executes all recorded operations on the input vectors.
    fn exec(&self, buffer: &mut B, a: DspVec<S, T, NI1, DI1>, b: DspVec<S, T, NI2, DI2>)
        -> result::Result<
            (DspVec<S, T, NO1, DO1>, DspVec<S, T, NO2, DO2>),
            (ErrorReason, DspVec<S, T, NO1, DO1>, DspVec<S, T, NO2, DO2>)> {
        // First "cast" the vectors to generic vectors. This is done with the
        // the rededicate trait since in contrast to the to_gen method it
        // can be used in a generic context.

        let mut vec = Vec::new();
        let (number_space1, domain1, gen1) = generic_vector_from_any_vector(a);
		vec.push(gen1);
        let (number_space2, domain2, gen2) = generic_vector_from_any_vector(b);
		vec.push(gen2);

        // at this point we would execute all ops and cast the result to the right types
		let result =
            DspVec::<S, T, RealOrComplexData, TimeOrFrequencyData>::perform_operations(
                buffer, vec, &self.ops);

        match result {
			Err((reason, mut vec)) => {
                let b = vec.pop().unwrap();
				let b = generic_vector_back_to_vector(number_space2, domain2, b);
                let b = DspVec::<S, T, NO2, DO2>::rededicate_from_force(b);
				let a = vec.pop().unwrap();
				let a = generic_vector_back_to_vector(number_space1, domain1, a);
                let a = DspVec::<S, T, NO1, DO1>::rededicate_from_force(a);
				Err((reason, a, b))
			},
			Ok(mut vec) => {
                let b = vec.pop().unwrap();
				let b = generic_vector_back_to_vector(number_space2, domain2, b);
                let b = DspVec::<S, T, NO2, DO2>::rededicate_from_force(b);
                let a = vec.pop().unwrap();
				let a = generic_vector_back_to_vector(number_space1, domain1, a);
                let a = DspVec::<S, T, NO1, DO1>::rededicate_from_force(a);
				Ok((a, b))
			}
		}
    }
}
