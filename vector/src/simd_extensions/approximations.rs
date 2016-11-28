/* This source code is a conversion from C to Rust with. The original C code
  can be found here https://github.com/RJVB/sse_mathfun
  The intrinsics are documented here: https://software.intel.com/sites/landingpage/IntrinsicsGuide/
  
  The C code is licensed as follows:
  
  Copyright (C) 2010,2011  RJVB - extensions 
  Copyright (C) 2007  Julien Pommier
  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.
  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:
  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
  (this is the zlib license)
*/

use simd::{f32x4, u32x4, i32x4};
use super::Simd;
use std::mem;
use std::ops::*;
use num::Float;
use Zero;
type Reg32 = f32x4;

pub trait SimdApproximations<T> : Simd<T>
    where T: Sized + Sync + Send {
    fn ln_approx(self) -> Self;

    fn exp_approx(self) -> Self;

    fn sin_approx(self) -> Self;
    
    fn cos_approx(self) -> Self;
}

macro_rules! simd_approx_impl {
    ($data_type:ident, $reg:ident)
    =>
    {
        impl SimdApproximations<$data_type> for $reg {
            #[inline]
            fn ln_approx(self) -> Self {
                let x = self;
                
                // integer constants
                // let onei = u32x4::splat(1);
                //let negonei = u32x4::splat(!1);
                //let twoi = u32x4::splat(2);
                //let fouri = u32x4::splat(4);
                let hex7fi = i32x4::splat(0x7f);
                let min_norm_pos = u32x4::splat(0x00800000);
                //let mant_mask = u32x4::splat(0x7f800000);
                let inv_mant_mask = u32x4::splat(!0x7f800000);
                //let sign_mask = u32x4::splat(0x80000000);
                //let inv_sign_mask = u32x4::splat(!0x80000000);
                
                 // floating point constants
                let one = $reg::splat(1.0);
                let onef_as_uint: u32x4 = unsafe { mem::transmute(one) };
                //let negone = $reg::splat(-1.0);
                let half = $reg::splat(0.5);
                let sqrthf = $reg::splat(2.0.sqrt());
                let log_p0 = $reg::splat(7.0376836292E-2);
                let log_p1 = $reg::splat(-1.1514610310E-1);
                let log_p2 = $reg::splat(1.1676998740E-1);
                let log_p3 = $reg::splat(-1.2420140846E-1);
                let log_p4 = $reg::splat(1.4249322787E-1);
                let log_p5 = $reg::splat(-1.6668057665E-1);
                let log_p6 = $reg::splat(2.0000714765E-1);
                let log_p7 = $reg::splat(-2.4999993993E-1);
                let log_p8 = $reg::splat(3.3333331174E-1);
                let log_q1 = $reg::splat(-2.12194440e-4);
                let log_q2 = $reg::splat(0.693359375);
                
                let invalid_mask = x.le($reg::zero());
                let x = unsafe { x.max(mem::transmute(min_norm_pos)) }; // cut off denormalized stuff
                let x: i32x4 = unsafe { mem::transmute(x) };
                let emm0 = x.shr(23);
                
                // keep only the fractional part
                let x: u32x4 = unsafe { mem::transmute(x) };
                let x = x.bitand(inv_mant_mask);
                let x = x.bitor(unsafe { mem::transmute(half) });
                
                let emm0: i32x4 = emm0 - hex7fi;
                let e = emm0.to_f32();
                let e = e + one;
                
                let mask = unsafe { x.lt(mem::transmute(sqrthf)) };
                let tmp = unsafe { x.bitand(mem::transmute(mask)) };
                let x: f32x4 = unsafe { mem::transmute(x) };
                let x = x - one;
                let x: u32x4 = unsafe { mem::transmute(x) };
                let e: f32x4 = unsafe { mem::transmute(e) };
                let masked_one: f32x4 = unsafe { mem::transmute(onef_as_uint.bitand(mem::transmute(mask))) };
                let e = e - masked_one;
                let tmp: f32x4 = unsafe { mem::transmute(tmp) };
                let x: f32x4 = unsafe { mem::transmute(x) };
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
                let x: u32x4 = unsafe { mem::transmute(x) };
                let x = unsafe { x.bitor(mem::transmute(invalid_mask)) };
                let x: $reg = unsafe { mem::transmute(x) };
                x
            }

            #[inline]
            fn exp_approx(self) -> Self {
                panic!("TODO")
            }

            #[inline]
            fn sin_approx(self) -> Self {
                panic!("TODO")
            }

            #[inline]
            fn cos_approx(self) -> Self {
                panic!("TODO")
            }
        }
    }
}

simd_approx_impl!(f32, Reg32);

#[cfg(test)]
mod tests {
    use super::super::*;
    use RealNumber;

    fn assert_eq_tol<T>(left: T, right: T, tol: T)
        where T: RealNumber
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
        let reg = Reg32::splat(value);
        let res = reg.ln_approx();
        assert_eq_tol(res.extract(0), value.ln(), 1e-15);
    }

    #[test]
    fn ln_approx_test1e8() {
        let value = 1e8;
        let reg = Reg32::splat(value);
        let res = reg.ln_approx();
        assert_eq_tol(res.extract(0), value.ln(), 1e-15);
    }
    
    #[test]
    fn ln_approx_test_small_value() {
        let value = 1e-8;
        let reg = Reg32::splat(value);
        let res = reg.ln_approx();
        assert_eq_tol(res.extract(0), value.ln(), 1e-15);
    }

    #[test]
    fn ln_approx_test_zero() {
        let reg = Reg32::splat(0.0);
        let res = reg.ln_approx();
        assert!(res.extract(0).is_nan());
    }

    #[test]
    fn ln_approx_test_neg() {
        let reg = Reg32::splat(-5.0);
        let res = reg.ln_approx();
        assert!(res.extract(0).is_nan());
    }
}