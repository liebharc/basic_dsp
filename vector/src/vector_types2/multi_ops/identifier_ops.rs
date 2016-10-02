use super::{Identifier, Operation};
use RealNumber;
use super::super::{
		OffsetOps, NumberSpace, Domain
};

impl<T, N, D> OffsetOps<T> for Identifier<T, N, D>
 	where T: RealNumber,
		  N: NumberSpace,
		  D: Domain{
    fn offset(&mut self, offset: T) {
		let arg = self.arg;
		self.add_op(Operation::AddReal(arg, offset));
	}
}
