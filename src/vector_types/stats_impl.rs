use super::definitions::Statistics;
use std::{f32, f64};
use num::Complex;

pub trait Stats<T> {
    fn empty() -> Self;
    fn invalid() -> Self;
}

macro_rules! impl_stat_trait {
	($($data_type:ident),*)
	 =>
	 {
         $(
            impl Stats<$data_type> for Statistics<$data_type> {
                fn empty() -> Self {
                    Statistics {
                        sum: 0.0,
                        count: 0,
                        average: 0.0, 
                        min: $data_type::INFINITY,
                        max: $data_type::NEG_INFINITY, 
                        rms: 0.0, // this field therefore has a different meaning inside this function
                        min_index: 0,
                        max_index: 0
                    }
                }
                
                fn invalid() -> Self {
                    Statistics {
                        sum: 0.0,
                        count: 0,
                        average: $data_type::NAN,
                        min: $data_type::NAN,
                        max: $data_type::NAN,
                        rms: $data_type::NAN,
                        min_index: 0,
                        max_index: 0,
                    }
                }
            }
            
            impl Stats<Complex<$data_type>> for Statistics<Complex<$data_type>> {
                fn empty() -> Self {
                    Statistics {
                        sum: Complex::<$data_type>::new(0.0, 0.0),
                        count: 0,
                        average: Complex::<$data_type>::new(0.0, 0.0),
                        min: Complex::<$data_type>::new($data_type::INFINITY, $data_type::INFINITY),
                        max: Complex::<$data_type>::new(0.0, 0.0),
                        rms: Complex::<$data_type>::new(0.0, 0.0), // this field therefore has a different meaning inside this function
                        min_index: 0,
                        max_index: 0
                    }
                }
                
                fn invalid() -> Self {
                    Statistics {
                        sum: Complex::<$data_type>::new(0.0, 0.0),
                        count: 0,
                        average: Complex::<$data_type>::new($data_type::NAN, $data_type::NAN),
                        min: Complex::<$data_type>::new($data_type::NAN, $data_type::NAN),
                        max: Complex::<$data_type>::new($data_type::NAN, $data_type::NAN),
                        rms: Complex::<$data_type>::new($data_type::NAN, $data_type::NAN),
                        min_index: 0,
                        max_index: 0,
                    }
                }
            }
         )*
     }
}

impl_stat_trait!(f32, f64);