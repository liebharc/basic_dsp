#![cfg_attr(feature = "cargo-clippy", allow(clippy::excessive_precision))]
#![cfg_attr(feature = "cargo-clippy", allow(clippy::unreadable_literal))]

// This source code is a conversion from C to Rust with. The original C code
// can be found here https://github.com/RJVB/sse_mathfun
// The intrinsics are documented here: https://software.intel.com/sites/landingpage/IntrinsicsGuide/
//
// The C code is licensed as follows:
//
// Copyright (C) 2010,2011  RJVB - extensions
// Copyright (C) 2007  Julien Pommier
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 1. The origin of this software must not be misrepresented; you must not
// claim that you wrote the original software. If you use this software
// in a product, an acknowledgment in the product documentation would be
// appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
// misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
// (this is the zlib license)
//

use super::{Simd, SimdApproximations, SimdFrom};
use numbers::*;
#[cfg(feature = "use_avx")]
use simd::x86::avx::*;
#[cfg(feature = "use_sse")]
use simd::x86::sse2::*;
#[cfg(feature = "use_sse")]
use simd::*;
use std::mem;
use std::ops::*;
use Zero;

macro_rules! simd_approx_impl {
    ($data_type: ident,
     $bit_len: expr,
     $regf: ident,
     $regi: ident,
     $regu: ident) => {
        impl SimdApproximations<$data_type> for $regf {
            #[inline]
            fn ln_approx(self) -> Self {
                let x = self;
        
                // integer constants
                let (hex7fi, min_norm_pos, inv_mant_mask, mant_len) = if $bit_len == 32 {
                    (
                        $regi::splat(0x7f),
                        $regu::splat(1 << 23),
                        $regu::splat(!0x7f800000),
                        23,
                    )
                } else {
                    // We would get a warning for f32 with those constants, but that can be ignored
                    #[allow(overflowing_literals)]
                    {
                        (
                            $regi::splat(0x3ff),
                            $regu::splat(1 << ($bit_len - 12)),
                            $regu::splat(!0x7ff0000000000000),
                            52,
                        )
                    }
                };
        
                // floating point constants
                let one = $regf::splat(1.0);
                let onef_as_uint: $regu = unsafe { mem::transmute(one) };
                let half = $regf::splat(0.5);
                let sqrthf = $regf::splat(2.0.sqrt());
                let log_p0 = $regf::splat(7.0376836292E-2);
                let log_p1 = $regf::splat(-1.1514610310E-1);
                let log_p2 = $regf::splat(1.1676998740E-1);
                let log_p3 = $regf::splat(-1.2420140846E-1);
                let log_p4 = $regf::splat(1.4249322787E-1);
                let log_p5 = $regf::splat(-1.6668057665E-1);
                let log_p6 = $regf::splat(2.0000714765E-1);
                let log_p7 = $regf::splat(-2.4999993993E-1);
                let log_p8 = $regf::splat(3.3333331174E-1);
                let log_q1 = $regf::splat(-2.12194440e-4);
                let log_q2 = $regf::splat(0.693359375);
        
                let invalid_mask = x.le($regf::zero());
                let x = unsafe { Simd::<$data_type>::max(x, mem::transmute(min_norm_pos)) }; // cut off denormalized stuff
                let x: $regi = unsafe { mem::transmute(x) };
                let emm0 = x.shr(mant_len);
        
                // keep only the fractional part
                let x: $regu = unsafe { mem::transmute(x) };
                let x = x.bitand(inv_mant_mask);
                let x = x.bitor(unsafe { mem::transmute(half) });
        
                let emm0: $regi = emm0 - hex7fi;
                let e: $regf = $regf::regfrom(emm0);
                let e = e + one;
        
                let mask = unsafe { x.lt(mem::transmute(sqrthf)) };
                let tmp = unsafe { x.bitand(mem::transmute(mask)) };
                let x: $regf = unsafe { mem::transmute(x) };
                let x = x - one;
                let x: $regu = unsafe { mem::transmute(x) };
                let masked_one: $regf =
                    unsafe { mem::transmute(onef_as_uint.bitand(mem::transmute(mask))) };
                let e = e - masked_one;
                let tmp: $regf = unsafe { mem::transmute(tmp) };
                let x: $regf = unsafe { mem::transmute(x) };
                let x = x + tmp;
        
                let z = x * x;
        
                let y = log_p0;
                let y = y * x;
                let y = y + log_p1;
                let y = y * x;
                let y = y + log_p2;
                let y = y * x;
                let y = y + log_p3;
                let y = y * x;
                let y = y + log_p4;
                let y = y * x;
                let y = y + log_p5;
                let y = y * x;
                let y = y + log_p6;
                let y = y * x;
                let y = y + log_p7;
                let y = y * x;
                let y = y + log_p8;
                let y = y * x;
        
                let y = y * z;
                let tmp = e * log_q1;
                let y = y + tmp;
        
                let tmp = z * half;
                let y = y - tmp;
        
                let tmp = e * log_q2;
                let x = x + y;
                let x = x + tmp;
                let x: $regu = unsafe { mem::transmute(x) };
                let x = unsafe { x.bitor(mem::transmute(invalid_mask)) };
                let x: $regf = unsafe { mem::transmute(x) };
                x
            }
        
            #[inline]
            fn exp_approx(self) -> Self {
                let x = self;
        
                // integer constants
                let (hex7fi, mant_len) = if $bit_len == 32 {
                    ($regi::splat(0x7f), 23)
                } else {
                    ($regi::splat(0x3ff), 52)
                };
        
                // floating point constants
                let half = $regf::splat(0.5);
                let one = $regf::splat(1.0);
                let exp_hi = $regf::splat(88.3762626647949);
                let exp_lo = $regf::splat(-88.3762626647949);
                let log2ef = $regf::splat(1.44269504088896341);
                let exp_c1 = $regf::splat(0.693359375);
                let exp_c2 = $regf::splat(-2.12194440e-4);
                let exp_p0 = $regf::splat(1.9875691500E-4);
                let exp_p1 = $regf::splat(1.3981999507E-3);
                let exp_p2 = $regf::splat(8.3334519073E-3);
                let exp_p3 = $regf::splat(4.1665795894E-2);
                let exp_p4 = $regf::splat(1.6666665459E-1);
                let exp_p5 = $regf::splat(5.0000001201E-1);
        
                let x = Simd::<$data_type>::min(x, exp_hi);
                let x = Simd::<$data_type>::max(x, exp_lo);
        
                // express exp(x) as exp(g + n*log(2))
                let fx = x * log2ef + half;
        
                // how to perform a floorf with SSE: just below
                let emm0 = $regi::regfrom(fx);
                let tmp = $regf::regfrom(emm0);
        
                // if greater, substract 1
                let mask = tmp.gt(fx);
                let mask = mask.bitand(unsafe { mem::transmute(one) });
                let mask: $regf = unsafe { mem::transmute(mask) };
                let fx = tmp - mask;
        
                let tmp = fx * exp_c1;
                let z = fx * exp_c2;
                let x = x - tmp - z;
                let z = x * x;
        
                let y = exp_p0;
                let y = y * x;
                let y = y + exp_p1;
                let y = y * x;
                let y = y + exp_p2;
                let y = y * x;
                let y = y + exp_p3;
                let y = y * x;
                let y = y + exp_p4;
                let y = y * x;
                let y = y + exp_p5;
                let y = y * z;
                let y = y + x;
                let y = y + one;
        
                // build 2^n
                let emm0 = $regi::regfrom(fx);
                let emm0 = emm0 + hex7fi;
                let emm0: $regu = unsafe { mem::transmute(emm0) };
                let emm0: $regu = emm0.shl(mant_len);
                let pow2n: $regf = unsafe { mem::transmute(emm0) };
        
                y * pow2n
            }
        
            #[inline]
            fn sin_approx(self) -> Self {
                self.sin_cos_approx(true)
            }
        
            #[inline]
            fn cos_approx(self) -> Self {
                self.sin_cos_approx(false)
            }
        
            #[inline]
            fn sin_cos_approx(self, is_sin: bool) -> Self {
                let x = self;
        
                // integer constants
                let sign_mask = 1 << ($bit_len - 1);
                let inv_sign_mask = $regu::splat(!sign_mask);
                let sign_mask = $regu::splat(sign_mask);
                let one = $regi::splat(1);
                let inv_one = one.not();
                let two = $regi::splat(2);
                let four = $regi::splat(4);
        
                // floating point constants
                let half = $regf::splat(0.5);
                let fopi = $regf::splat(1.27323954473516); // 4 / M_PI
                let dp1 = $regf::splat(-0.78515625);
                let dp2 = $regf::splat(-2.4187564849853515625e-4);
                let dp3 = $regf::splat(-3.77489497744594108e-8);
                let sincof_p0 = $regf::splat(-1.9515295891E-4);
                let sincof_p1 = $regf::splat(8.3321608736E-3);
                let sincof_p2 = $regf::splat(-1.6666654611E-1);
                let coscof_p0 = $regf::splat(2.443315711809948E-005);
                let coscof_p1 = $regf::splat(-1.388731625493765E-003);
                let coscof_p2 = $regf::splat(4.166664568298827E-002);
        
                let x: $regu = unsafe { mem::transmute(x) };
        
                // extract the sign bit (upper one)
                let sign_bit = x.bitand(sign_mask); // Only used for `sin` implementation
                                                    // take the absolute value
                let x = x.bitand(inv_sign_mask);
        
                // scale by 4/Pi
                let x: $regf = unsafe { mem::transmute(x) };
                let y = x * fopi;
        
                // store the integer part of y in mm0
                let emm2 = $regi::regfrom(y);
                // j=(j+1) & (~1) (see the cephes sources)
                let emm2 = emm2 + one;
                let mut emm2 = emm2.bitand(inv_one);
                let y = $regf::regfrom(emm2);
                if !is_sin {
                    emm2 = emm2 - two;
                }
        
                // get the swap sign flag
                let emm0 = if is_sin {
                    emm2.bitand(four)
                } else {
                    emm2.not().bitand(four)
                };
                let emm0: $regu = unsafe { mem::transmute(emm0) };
                let emm0 = emm0.shl($bit_len - 3);
                // get the polynom selection mask
                // there is one polynom for 0 <= x <= Pi/4
                // and another one for Pi/4<x<=Pi/2
                //
                // Both branches will be computed.
                let emm2 = emm2.bitand(two);
                let emm2 = emm2.eq($regi::splat(0));
        
                let poly_mask = emm2;
                let sign_bit = if is_sin { sign_bit.bitxor(emm0) } else { emm0 };
        
                // The magic pass: "Extended precision modular arithmetic"
                // x = ((x - y * DP1) - y * DP2) - y * DP3;
                let xmm1 = y * dp1;
                let xmm2 = y * dp2;
                let xmm3 = y * dp3;
                let x = x + xmm1;
                let x = x + xmm2;
                let x = x + xmm3;
        
                // Evaluate the first polynom  (0 <= x <= Pi/4)
                let y = coscof_p0;
                let z = x * x;
        
                let y = y * z;
                let y = y + coscof_p1;
                let y = y * z;
                let y = y + coscof_p2;
                let y = y * z * z;
                let tmp = z * half;
                let y = y - tmp;
                let y = y + $regf::splat(1.0);
        
                // Evaluate the second polynom  (Pi/4 <= x <= 0)
                let y2 = sincof_p0;
                let y2 = y2 * z;
                let y2 = y2 + sincof_p1;
                let y2 = y2 * z;
                let y2 = y2 + sincof_p2;
                let y2 = y2 * z * x;
                let y2 = y2 + x;
        
                // select the correct result from the two polynoms
                let xmm3: $regu = unsafe { mem::transmute(poly_mask) };
                let y2: $regu = unsafe { mem::transmute(y2) };
                let y: $regu = unsafe { mem::transmute(y) };
                let y2 = xmm3.bitand(y2);
                let y = xmm3.not().bitand(y);
                let y2: $regf = unsafe { mem::transmute(y2) };
                let y: $regf = unsafe { mem::transmute(y) };
                let y = y + y2;
        
                // update the sign
                let y: $regu = unsafe { mem::transmute(y) };
                let y = y.bitxor(sign_bit);
                let y: $regf = unsafe { mem::transmute(y) };
                y
            }
        }
    };
}

#[cfg(feature = "use_sse")]
simd_approx_impl!(f32, 32, f32x4, i32x4, u32x4);
#[cfg(feature = "use_sse")]
simd_approx_impl!(f64, 64, f64x2, i64x2, u64x2);
#[cfg(feature = "use_avx")]
simd_approx_impl!(f32, 32, f32x8, i32x8, u32x8);
#[cfg(feature = "use_avx")]
simd_approx_impl!(f64, 64, f64x4, i64x4, u64x4);

#[cfg(test)]
#[cfg(feature = "use_sse")]
mod tests {
    use super::super::*;
    use simd::f32x4;
    use simd::x86::sse2::f64x2;
    use RealNumber;

    fn assert_eq_tol<T>(left: T, right: T, tol: T)
    where
        T: RealNumber,
    {
        let diff = (left - right).abs();
        if diff > tol {
            panic!("assertion failed: {:?} != {:?}", left, right);
        }
        if diff.is_nan() {
            panic!("assertion failed: {:?} != {:?}", left, right);
        }
    }

    #[test]
    fn ln_approx_test5() {
        let value = 5.0;
        let reg = f32x4::splat(value);
        let res = reg.ln_approx();
        assert_eq_tol(res.extract(0), value.ln(), 1e-9);
    }

    #[test]
    fn ln_approx_test1e8() {
        let value = 1e8;
        let reg = f32x4::splat(value);
        let res = reg.ln_approx();
        assert_eq_tol(res.extract(0), value.ln(), 1e-9);
    }

    #[test]
    fn ln_approx_test_small_value() {
        let value = 1e-8;
        let reg = f32x4::splat(value);
        let res = reg.ln_approx();
        assert_eq_tol(res.extract(0), value.ln(), 1e-9);
    }

    #[test]
    fn ln_approx_test_zero() {
        let reg = f32x4::splat(0.0);
        let res = reg.ln_approx();
        assert!(res.extract(0).is_nan());
    }

    #[test]
    fn ln_approx_test_neg() {
        let reg = f32x4::splat(-5.0);
        let res = reg.ln_approx();
        assert!(res.extract(0).is_nan());
    }

    #[test]
    fn ln_approx_test_f64() {
        let value = 5.0;
        let reg = f64x2::splat(value);
        let res = reg.ln_approx();
        assert_eq_tol(res.extract(0), value.ln(), 1e-9);
    }

    #[test]
    fn exp_approx_test5() {
        let value = 5.0;
        let reg = f32x4::splat(value);
        let res = reg.exp_approx();
        assert_eq_tol(res.extract(0), value.exp(), 1e-9);
    }

    #[test]
    fn exp_approx_test_f64() {
        let value = 5.0;
        let reg = f64x2::splat(value);
        let res = reg.exp_approx();
        assert_eq_tol(res.extract(0), value.exp(), 1e-6);
    }

    #[test]
    fn sin_approx_test5() {
        let value = 5.0;
        let reg = f32x4::splat(value);
        let res = reg.sin_approx();
        assert_eq_tol(res.extract(0), value.sin(), 1e-9);
    }

    #[test]
    fn sin_approx_test_f64() {
        let value = 5.0;
        let reg = f64x2::splat(value);
        let res = reg.sin_approx();
        assert_eq_tol(res.extract(0), value.sin(), 1e-9);
    }

    #[test]
    fn cos_approx_test5() {
        let value = 5.0;
        let reg = f32x4::splat(value);
        let res = reg.cos_approx();
        assert_eq_tol(res.extract(0), value.cos(), 1e-7);
    }

    #[test]
    fn cos_approx_testf64() {
        let value = 5.0;
        let reg = f64x2::splat(value);
        let res = reg.cos_approx();
        assert_eq_tol(res.extract(0), value.cos(), 1e-7);
    }
}
