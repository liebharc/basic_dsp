/// Selects a SIMD register type and passes it as 2nd argument to a function.
/// The macro tries to mimic the Rust syntax of a method call.
macro_rules! sel_reg(
    ($self_:ident.$method: ident::<$type: ident>($($args: expr),*)) => {
		// TODO at some point in time it would be good to support ARM vectorization
        $self_.$method(RegType::<<$type as ToSimd>::RegFallback>::new(), $($args),*)
    };
    ($method: ident::<$type: ident>($($args: expr),*)) => {
        $method(RegType::<<$type as ToSimd>::RegFallback>::new(), $($args),*)
    };
);