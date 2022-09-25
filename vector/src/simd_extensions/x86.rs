/// Selects a SIMD register type and passes it as 2nd argument to a function.
/// The macro tries to mimic the Rust syntax of a method call.
macro_rules! sel_reg(
    ($self_:ident.$method: ident::<$type: ident>($($args: expr),*)) => {
		// TODO enable AVX512 detection again as soon as stdsimd is stable
        /*if is_x86_feature_detected!("avx512vl") && cfg!(feature="use_avx512") {
            $self_.$method(RegType::<<$type as ToSimd>::RegAvx512>::new(), $($args),*)
        } else*/ if cfg!(feature="use_avx2") && is_x86_feature_detected!("avx2") {
            $self_.$method(RegType::<<$type as ToSimd>::RegAvx>::new(), $($args),*)
        } else if cfg!(feature="use_sse2") && is_x86_feature_detected!("sse2") {
            $self_.$method(RegType::<<$type as ToSimd>::RegSse>::new(), $($args),*)
        } else {
            $self_.$method(RegType::<<$type as ToSimd>::RegFallback>::new(), $($args),*)
        }
    };
    ($method: ident::<$type: ident>($($args: expr),*)) => {
        /*if is_x86_feature_detected!("avx512vl") && cfg!(feature="use_avx512") {
            $method(RegType::<<$type as ToSimd>::RegAvx512>::new(), $($args),*)
        } else*/ if cfg!(feature="use_avx2") && cfg!(target_feature="avx2") && is_x86_feature_detected!("avx2") {
            $method(RegType::<<$type as ToSimd>::RegAvx>::new(), $($args),*)
        } else if cfg!(feature="use_sse2")&& cfg!(target_feature="sse2") && is_x86_feature_detected!("sse2") {
            $method(RegType::<<$type as ToSimd>::RegSse>::new(), $($args),*)
        } else {
            $method(RegType::<<$type as ToSimd>::RegFallback>::new(), $($args),*)
        }
    };
);