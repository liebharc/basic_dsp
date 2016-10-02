use RealNumber;
use std::result;
use std::sync::{Arc, Mutex};
use super::Identifier;
use super::super::{
    Domain, NumberSpace, ToSliceMut, DspVec, ErrorReason, RededicateForceOps,
	Buffer, Owner, TimeOrFrequencyData, RealOrComplexData, Vector,
	DataDomain
};
use super::operations_enum::*;

const ARGUMENT1: usize = 0;
const ARGUMENT2: usize = 1;

/// An operation on one data vector which has been prepared in
/// advance.
///
/// The type arguments can be read like a function definition: TI1 -> TO1
#[derive(Clone)]
pub struct PreparedOperation1<T, NI, DI, NO, DO>
    where T: RealNumber,
		NI: NumberSpace, DI: Domain,
		NO: NumberSpace, DO: Domain {
    number_space_in: NI,
	domain_in: DI,
	number_space_out: NO,
	domain_out: DO,
    ops: Vec<Operation<T>>
}

/// Executes the prepared operations to convert `S`to `D`,
pub trait PreparedOperation1Exec<B, S, D> {
    /// Executes the prepared operations to convert `S`to `D`,
    fn exec(&self, buffer: &mut B, source: S) -> result::Result<D, (ErrorReason, D)>;
}

/// Prepares an operation with one input and one output.
pub fn prepare32_1<N, D>(number_space: N, domain: D)
    -> PreparedOperation1<f32, N, D, N, D>
    where
        N: NumberSpace,
		D: Domain {
    PreparedOperation1
         {
			number_space_in: number_space.clone(),
 			domain_in: domain.clone(),
 			number_space_out: number_space,
 			domain_out: domain,
            ops: Vec::new()
         }
}

/// Prepares an operation with one input and one output.
pub fn prepare64_1<N, D>(number_space: N, domain: D)
    -> PreparedOperation1<f64, N, D, N, D>
    where
        N: NumberSpace,
		D: Domain {
    PreparedOperation1
         {
			number_space_in: number_space.clone(),
 			domain_in: domain.clone(),
 			number_space_out: number_space,
 			domain_out: domain,
            ops: Vec::new()
         }
}

/// An operation on two data vectors which has been prepared in
/// advance.
///
/// The type arguments can be read like a function definition: (TI1, TI2) -> (TO1, TO2)
#[derive(Clone)]
pub struct PreparedOperation2<T, NI1, DI1, NI2, DI2, NO1, DO1, NO2, DO2>
    where T: RealNumber,
		NI1: NumberSpace, DI1: Domain, NI2: NumberSpace, DI2: Domain,
		NO1: NumberSpace, DO1: Domain, NO2: NumberSpace, DO2: Domain {
	number_space_in1: NI1,
	domain_in1: DI1,
	number_space_in2: NI2,
	domain_in2: DI2,
	number_space_out1: NO1,
	domain_out1: DO1,
	number_space_out2: NO2,
	domain_out2: DO2,
    ops: Vec<Operation<T>>,
    swap: bool
}
/*
/// A multi operation which holds a vector and records all changes
/// which need to be done to the vectors. By calling `get` on the struct
/// all operations will be executed in one run.
pub struct MultiOperation1<S, T, NI, DI, NO, DO>
    where T: RealNumber, S: ToSlice<T>,
		NI: NumberSpace, DI: Domain,
		NO: NumberSpace, DO: Domain {
    a: DspVec<S, T, NI, DI>,
    prepared_ops: PreparedOperation1<S, T, NI, DI, NO, DO>
}

/// A multi operation which holds two vectors and records all changes
/// which need to be done to the vectors. By calling `get` on the struct
/// all operations will be executed in one run.
pub struct MultiOperation2<S, T, NI1, DI1, NI2, DI2, NO1, DO1, NO2, DO2>
    where T: RealNumber, S: ToSlice<T>,
		NI1: NumberSpace, DI1: Domain, NI2: NumberSpace, DI2: Domain,
		NO1: NumberSpace, DO1: Domain, NO2: NumberSpace, DO2: Domain {
    a: DspVec<S, T, NI2, DI1>,
    b: DspVec<S, T, NI2, DI2>,
    prepared_ops: PreparedOperation2<S, T, NI1, DI1, NI2, DI2, NO1, DO1, NO2, DO2>
}
*/
/// An operation which can be prepared in advance and operates on one
/// input and produces one output
impl<T, NI, DI, NO, DO> PreparedOperation1<T, NI, DI, NO, DO>
	where T: RealNumber + 'static,
		NI: NumberSpace, DI: Domain,
		NO: NumberSpace, DO: Domain {

	/// Extends the operation to operate on one more vector.
	pub fn extend<NI2, DI2>(self, number_space: NI2, domain: DI2)
			-> PreparedOperation2<T, NI, DI, NI2, DI2, NO, DO, NI2, DI2>
		where NI2: NumberSpace, DI2: Domain {
		PreparedOperation2
		{
			number_space_in1: self.number_space_in,
			domain_in1: self.domain_in,
			number_space_in2: number_space.clone(),
			domain_in2: domain.clone(),
			number_space_out1: self.number_space_out,
			domain_out1: self.domain_out,
			number_space_out2: number_space,
			domain_out2: domain,
			ops: self.ops,
			swap: false
		}
	}

	/// Adds new operations which will be executed with the next call to `exec`
	///
	/// As a background: The function `operation` will be executed immediately. It only operated on `Identifier` types and these serve as
	/// placeholder for vectors. Every operation done to an `Identifier`
	/// is recorded and will be executed on vectors if `exec` is called.
	pub fn add_ops<F, NT, DT>(self, operation: F)
		-> PreparedOperation1<T, NI, DI, NT, DT>
		where F: Fn(Identifier<T, NO, DO>) -> Identifier<T, NT, DT>,
		      DT: Domain,
			  NT: NumberSpace

	{
		let mut ops = self.ops;
		let (new_ops, domain_new, number_space_new) =
		{
			let counter = Arc::new(Mutex::new(0));
			let t1 = Identifier
				{
				    arg: ARGUMENT1,
				    ops: Vec::new(),
				    counter: counter,
				    domain: self.domain_out.clone(),
				    number_space: self.number_space_out.clone()
				};

			let r1 = operation(t1);
			(r1.ops, r1.domain, r1.number_space)
		};
		for v in new_ops {
			ops.push(v.1);
		}

		PreparedOperation1
		{
			domain_in: self.domain_in,
			number_space_in: self.number_space_in,
			domain_out: domain_new,
			number_space_out: number_space_new,
			ops: ops
		}
	}
}

fn generic_vector_from_any_vector<S, T, N, D>(vec: DspVec<S, T, N, D>) -> (N, D, DspVec<S, T, RealOrComplexData, TimeOrFrequencyData>)
    where T: RealNumber + 'static,
        S: ToSliceMut<T> + Owner,
        N: NumberSpace, D: Domain {
    let domain = vec.domain.clone();
    let number_space = vec.number_space.clone();
    let gen = DspVec {
            delta: vec.delta,
            domain: TimeOrFrequencyData { domain_current: vec.domain() },
            number_space: RealOrComplexData { is_complex_current: vec.is_complex() },
            valid_len: vec.valid_len,
            multicore_settings: vec.multicore_settings,
            data: vec.data
    };
    (number_space, domain, gen)
}

fn generic_vector_back_to_vector<S, T, N, D>(number_space: N, domain: D, vec: DspVec<S, T, RealOrComplexData, TimeOrFrequencyData>) -> DspVec<S, T, N, D>
    where T: RealNumber + 'static,
        S: ToSliceMut<T> + Owner,
        N: NumberSpace, D: Domain {
    let is_complex = vec.is_complex();
    let result_domain = vec.domain();
    let mut vec = DspVec {
            data: vec.data,
            delta: vec.delta,
            domain: domain,
            number_space: number_space,
            valid_len: vec.valid_len,
            multicore_settings: vec.multicore_settings
    };

    if is_complex {
        vec.number_space.to_complex();
    } else {
        vec.number_space.to_real();
    }

    if result_domain == DataDomain::Time {
        vec.domain.to_time();
    } else {
        vec.domain.to_freq();
    }

    vec
}

impl<T, S, B, NI, DI, NO, DO> PreparedOperation1Exec<B, DspVec<S, T, NI, DI>, DspVec<S, T, NO, DO>> for PreparedOperation1<T, NI, DI, NO, DO>
	where T: RealNumber + 'static,
        S: ToSliceMut<T> + Owner,
        DspVec<S, T, NO, DO>: RededicateForceOps<DspVec<S, T, NI, DI>>,
		NI: NumberSpace, DI: Domain,
		NO: NumberSpace, DO: Domain,
        B: Buffer<S, T> {
	/// Executes all recorded operations on the input vectors.
	fn exec(&self, buffer: &mut B, a: DspVec<S, T, NI, DI>) -> result::Result<DspVec<S, T, NO, DO>, (ErrorReason, DspVec<S, T, NO, DO>)> {
		let mut vec = Vec::new();
		let (number_space, domain, gen) = generic_vector_from_any_vector(a);
		vec.push(gen);


		// at this point we would execute all ops and cast the result to the right types
		let result = DspVec::<S, T, RealOrComplexData, TimeOrFrequencyData>::perform_operations(buffer, vec, &self.ops);

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
/*
impl<T, S, B, N, D> PreparedOperation1Exec<B, DspVec<S, T, N, D>, DspVec<S, T, N, D>> for PreparedOperation1<T, N, D, N, D>
	where T: RealNumber + 'static,
        S: ToSliceMut<T> + Owner,
		N: NumberSpace, D: Domain,
        B: Buffer<S, T> {
	/// Executes all recorded operations on the input vectors.
	fn exec(&self, buffer: &mut B, a: DspVec<S, T, N, D>) -> result::Result<DspVec<S, T, N, D>, (ErrorReason, DspVec<S, T, N, D>)> {
		let mut vec = Vec::new();
		let (number_space, domain, gen) = generic_vector_from_any_vector(a);
		vec.push(gen);


		// at this point we would execute all ops and cast the result to the right types
		let result = DspVec::<S, T, RealOrComplexData, TimeOrFrequencyData>::perform_operations(buffer, vec, &self.ops);

		match result {
			Err((reason, mut vec)) => {
				let a = vec.pop().unwrap();
				let a = generic_vector_back_to_vector(number_space, domain, a);
				Err((reason, a))
			},
			Ok(mut vec) => {
                let a = vec.pop().unwrap();
				let a = generic_vector_back_to_vector(number_space, domain, a);
				// Convert back
				Ok(a)
			}
		}
	}
}*/
