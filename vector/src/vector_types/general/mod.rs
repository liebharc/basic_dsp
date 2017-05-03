mod elementary;
pub use self::elementary::*;
mod trigonometry_and_powers;
pub use self::trigonometry_and_powers::*;
mod data_reorganization;
pub use self::data_reorganization::*;
mod diff_sum;
pub use self::diff_sum::*;
mod dot_products;
pub use self::dot_products::*;
mod mapping;
pub use self::mapping::*;
mod statistics;
pub use self::statistics::*;
mod precise_stats;
pub use self::precise_stats::*;

use std::ops::{Add, Sub};
use numbers::*;

/// Sums up the given values using a more precise
/// summation algorithm: `https://en.wikipedia.org/wiki/Kahan_summation_algorithm`
fn kahan_sum<I, T>(values: I) -> T
    where I: Iterator<Item = T>,
          T: Add<Output = T> + Sub<Output = T> + Zero + Copy
{
    let mut sum = T::zero();
    let mut c = T::zero();
    for n in values {
        let y = n - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Sums up the given values using a more precise
/// summation algorithm: `https://en.wikipedia.org/wiki/Kahan_summation_algorithm`
fn kahan_sumb<'a, I, T>(values: I) -> T
    where I: Iterator<Item = &'a T>,
          T: 'a + Add<Output = T> + Sub<Output = T> + Zero + Copy
{
    let mut sum = T::zero();
    let mut c = T::zero();
    for n in values {
        let y = *n - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}
