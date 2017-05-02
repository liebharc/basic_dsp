use numbers::*;
use std::result;
use inline_vector::InlineVector;
use super::{generic_vector_from_any_vector, generic_vector_back_to_vector, Identifier,
            PreparedOperation1, PreparedOperation2, Operation, OpsVec};
use super::super::{Domain, NumberSpace, ToSliceMut, DspVec, ErrorReason, RededicateForceOps,
                   Buffer, TimeOrFrequencyData, RealOrComplexData};

const ARGUMENT1: usize = 0;
const ARGUMENT2: usize = 1;

/// Executes the prepared operations to convert `A`to `D`.
pub trait PreparedOperation1Exec<S: ToSliceMut<T>, T: RealNumber, A, D> {
    /// Executes the prepared operations to convert `A`to `D`.
    fn exec<B>(&self, buffer: &mut B, source: A) -> result::Result<D, (ErrorReason, D)>
        where B: Buffer<S, T>;
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
        ops: InlineVector::with_default_capcacity(),
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
        ops: InlineVector::with_default_capcacity(),
    }
}

/// Executes the prepared operations to convert `A1` and `A2` to `D1` and `D2`.
pub trait PreparedOperation2Exec<S: ToSliceMut<T>, T: RealNumber, A1, A2, D1, D2> {
    /// Executes the prepared operations to convert `A1` and `A2` to `D1` and `D2`.
    fn exec<B>(&self,
            buffer: &mut B,
            source1: A1,
            source2: A2)
            -> result::Result<(D1, D2), (ErrorReason, D1, D2)>
        where B: Buffer<S, T>;
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
        ops: InlineVector::with_default_capcacity(),
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
        ops: InlineVector::with_default_capcacity(),
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
            let t1 = Identifier {
                arg: ARGUMENT1,
                ops: Vec::new(),
                counter: 0,
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

impl<T, S, NI, DI, NO, DO> PreparedOperation1Exec<S, T, DspVec<S, T, NI, DI>, DspVec<S, T, NO, DO>>
    for PreparedOperation1<T, NI, DI, NO, DO>
	where T: RealNumber + 'static,
        S: ToSliceMut<T>,
        DspVec<S, T, NO, DO>: RededicateForceOps<DspVec<S, T, NI, DI>>,
		NI: NumberSpace, DI: Domain,
		NO: NumberSpace, DO: Domain {
	/// Executes all recorded operations on the input vectors.
	fn exec<B>(&self, buffer: &mut B, a: DspVec<S, T, NI, DI>)
        -> result::Result<DspVec<S, T, NO, DO>, (ErrorReason, DspVec<S, T, NO, DO>)>
        where B: Buffer<S,T>  {

		let mut vec = Vec::new();
		let (number_space, domain, gen) = generic_vector_from_any_vector(a);
		vec.push(gen);


		// at this point we would execute all ops and cast the result to the right types
		let result =
            DspVec::<S, T, RealOrComplexData, TimeOrFrequencyData>::perform_operations(
                buffer, vec, &self.ops[..]);

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

/// Lists all operations which must be executed on on argument.
type ArgVec<T> = InlineVector<(Operation<T>, Option<usize>)>;

fn sort_by_arg<T: Clone>(ops1: &OpsVec<T>, ops2: &OpsVec<T>) -> (ArgVec<T>, ArgVec<T>) {
    let mut res1 = ArgVec::with_capacity(ops1.len() + ops2.len());
    let mut res2 = ArgVec::with_capacity(ops1.len() + ops2.len());
    for n in &ops1[..] {
        let id = n.0;
        let other_count =
            match n.2 {
                Some(other) => Some(other.1),
                None => None
            };
        if id.0 == 0 {
            res1.push((n.1.clone(), other_count));
        }
        else {
            res2.push((n.1.clone(), other_count));
        }
    }

    for n in &ops2[..] {
        let id = n.0;
        let other_count =
            match n.2 {
                Some(other) => Some(other.1),
                None => None
            };
        if id.0 == 0 {
            if id.1 < res1.len() {
                res1.insert(id.1, (n.1.clone(), other_count));
            }
            else {
                res1.push((n.1.clone(), other_count));
            }
        }
        else {
            if id.1 < res2.len() {
                res2.insert(id.1, (n.1.clone(), other_count));
            }
            else {
                res2.push((n.1.clone(), other_count));
            }
        }
    }

    (res1, res2)
}

/// Returns the first index bigger than `start` which holds a binary operation.
fn find_first_binary_op_pos<T>(ops: &ArgVec<T>, start: usize) -> Option<usize> {
    for i in start..ops.len() {
        if ops[i].1.is_some() {
            return Some(i);
        }
    }

    None
}

/// Returns the indices which must be processed first and then the binary op which must be handled next.
fn first_op<T> (
    ops1: &ArgVec<T>, pos1: Option<usize>,
    ops2: &ArgVec<T>, pos2: Option<usize>)
     -> (usize, usize, u32) {
    assert!(pos1.is_some() || pos2.is_some());
    if pos1.is_some() && pos2.is_some() {
        let op1 = &ops1[pos1.unwrap()];
        let other1 = op1.1.unwrap();
        let op2 = &ops2[pos2.unwrap()];
        let other2 = op2.1.unwrap();
        if pos1.unwrap() > other2
        {
            // `op1` follows after `op2`
            (other2, pos2.unwrap(), 1)
        }
        else if pos1.unwrap() < pos2.unwrap()
        {
            // `op2` follows after `op1`
            (pos1.unwrap(), other1, 0)
        }
        else {
            panic!("Failed to determine the sequence of operations");
        }
    }
    else if pos1.is_some() {
        let op = &ops1[pos1.unwrap()];
        let other = op.1.unwrap();
        (pos1.unwrap(), other, 0)
    }
    else {
        let op = &ops2[pos2.unwrap()];
        let other = op.1.unwrap();
        (other, pos2.unwrap(), 1)
    }
}

/// Merges two ops vectors in a correct order.
fn merge_operations<T: Clone>(ops1: &OpsVec<T>, ops2: &OpsVec<T>) -> InlineVector<Operation<T>> {
    let mut res = InlineVector::with_capacity(ops1.len() + ops2.len());
    let (arg1, arg2) =  sort_by_arg(ops1, ops2);
    let mut ops1pos = 0;
    let mut ops2pos = 0;
    let mut bin1 = find_first_binary_op_pos(&arg1, ops1pos);
    let mut bin2 = find_first_binary_op_pos(&arg2, ops2pos);
    while bin1.is_some() || bin2.is_some() {
        let (pos1, pos2, bin_op) = first_op(&arg1, bin1, &arg2, bin2);
        for i in ops1pos..pos1 {
            res.push(arg1[i].0.clone());
        }
        for i in ops2pos..pos2 {
            res.push(arg2[i].0.clone());
        }
        ops1pos = pos1;
        ops2pos = pos2;

        if bin_op == 0 {
            res.push(arg1[ops1pos].0.clone());
            ops1pos += 1;
        }
        else {
            res.push(arg2[ops2pos].0.clone());
            ops2pos += 1;
        }

        bin1 = find_first_binary_op_pos(&arg1, ops1pos);
        bin2 = find_first_binary_op_pos(&arg2, ops2pos);
    }

    // Handle remaining elements
    for i in ops1pos..arg1.len() {
        res.push(arg1[i].0.clone());
    }
    for i in ops2pos..arg2.len() {
        res.push(arg2[i].0.clone());
    }

    res
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
        let (swap, domain_new1, number_space_new1, domain_new2, number_space_new2) = {
            let t1 = Identifier {
                arg: if self.swap { ARGUMENT2 } else { ARGUMENT1 },
                ops: Vec::new(),
                counter: 0,
                domain: self.domain_out1.clone(),
                number_space: self.number_space_out1.clone(),
            };

            let t2 = Identifier {
                arg: if self.swap { ARGUMENT1 } else { ARGUMENT2 },
                ops: Vec::new(),
                counter: 0,
                domain: self.domain_out2.clone(),
                number_space: self.number_space_out2.clone(),
            };

            let (r1, r2) = operation(t1, t2);
            let r1arg = r1.arg;
            let mut new_ops = merge_operations(&r1.ops, &r2.ops);
            ops.append(&mut new_ops);
            (r1arg == ARGUMENT2, r1.domain, r1.number_space, r2.domain, r2.number_space)
        };

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

impl<T, S, NI1, DI1, NI2, DI2, NO1, DO1, NO2, DO2> PreparedOperation2Exec<
        S, T, DspVec<S, T, NI1, DI1>, DspVec<S, T, NI2, DI2>,
        DspVec<S, T, NO1, DO1>, DspVec<S, T, NO2, DO2>>
    for PreparedOperation2<T, NI1, DI1, NI2, DI2, NO1, DO1, NO2, DO2>
	where T: RealNumber + 'static,
        S: ToSliceMut<T>,
        DspVec<S, T, NO1, DO1>: RededicateForceOps<DspVec<S, T, NI1, DI1>>,
        DspVec<S, T, NO2, DO2>: RededicateForceOps<DspVec<S, T, NI2, DI2>>,
		NI1: NumberSpace, DI1: Domain,
        NI2: NumberSpace, DI2: Domain,
		NO1: NumberSpace, DO1: Domain,
        NO2: NumberSpace, DO2: Domain {

    /// Executes all recorded operations on the input vectors.
    fn exec<B>(&self, buffer: &mut B, a: DspVec<S, T, NI1, DI1>, b: DspVec<S, T, NI2, DI2>)
        -> result::Result<
            (DspVec<S, T, NO1, DO1>, DspVec<S, T, NO2, DO2>),
            (ErrorReason, DspVec<S, T, NO1, DO1>, DspVec<S, T, NO2, DO2>)>
            where B: Buffer<S, T> {
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
                buffer, vec, &self.ops[..]);

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
