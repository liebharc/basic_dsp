use super::definitions::{
    RededicateOps,
    DataVecDomain};
use vector_types::{
    GenericDataVec,
    RealTimeVector,
    ComplexTimeVector,
    RealFreqVector,
    ComplexFreqVector};
use RealNumber;    

macro_rules! add_rededicate_impl {
    ($(struct $name:ident, $is_complex:ident, $domain:expr);*) => {
    $(
        impl<T> RededicateOps<ComplexTimeVector<T>> for $name<T>
            where T: RealNumber {
            fn rededicate(self) -> ComplexTimeVector<T> {
                ComplexTimeVector {
                    delta: self.delta,
                    is_complex: true,
                    valid_len: 0,
                    domain: DataVecDomain::Time,
                    data: self.data,
                    temp: self.temp,
                    multicore_settings: self.multicore_settings
                }
            }

            fn rededicate_from(other: ComplexTimeVector<T>) -> Self {
                other.rededicate()
            }
        }

        impl<T> RededicateOps<ComplexFreqVector<T>> for $name<T>
            where T: RealNumber {
            fn rededicate(self) -> ComplexFreqVector<T> {
                ComplexFreqVector {
                    delta: self.delta,
                    is_complex: true,
                    valid_len: 0,
                    domain: DataVecDomain::Frequency,
                    data: self.data,
                    temp: self.temp,
                    multicore_settings: self.multicore_settings
                }
            }

            fn rededicate_from(other: ComplexFreqVector<T>) -> Self {
                other.rededicate()
            }
        }

        impl<T> RededicateOps<RealTimeVector<T>> for $name<T>
            where T: RealNumber {
            fn rededicate(self) -> RealTimeVector<T> {
                RealTimeVector {
                    delta: self.delta,
                    is_complex: false,
                    valid_len: 0,
                    domain: DataVecDomain::Time,
                    data: self.data,
                    temp: self.temp,
                    multicore_settings: self.multicore_settings
                }
            }

            fn rededicate_from(other: RealTimeVector<T>) -> Self {
                other.rededicate()
            }
        }

        impl<T> RededicateOps<RealFreqVector<T>> for $name<T>
            where T: RealNumber {
            fn rededicate(self) -> RealFreqVector<T> {
                RealFreqVector {
                    delta: self.delta,
                    is_complex: false,
                    valid_len: 0,
                    domain: DataVecDomain::Frequency,
                    data: self.data,
                    temp: self.temp,
                    multicore_settings: self.multicore_settings
                }
            }

            fn rededicate_from(other: RealFreqVector<T>) -> Self {
                other.rededicate()
            }
        }

        impl<T> RededicateOps<GenericDataVec<T>> for $name<T>
            where T: RealNumber {
            fn rededicate(self) -> GenericDataVec<T> {
                GenericDataVec {
                    delta: self.delta,
                    is_complex: self.is_complex,
                    valid_len: self.valid_len,
                    domain: self.domain,
                    data: self.data,
                    temp: self.temp,
                    multicore_settings: self.multicore_settings
                }
            }

            fn rededicate_from(other: GenericDataVec<T>) -> Self {
                other.rededicate()
            }
        }
      )*
    }
}

add_rededicate_impl!(struct RealTimeVector, false, DataVecDomain::Time);
add_rededicate_impl!(struct ComplexTimeVector, true, DataVecDomain::Time);
add_rededicate_impl!(struct RealFreqVector, false, DataVecDomain::Frequency);
add_rededicate_impl!(struct ComplexFreqVector, true, DataVecDomain::Frequency);

// Conversions to `GenericDataVec` are always valid therefore `GenericDataVec`
// needs to have a specific `RededicateOps` implementation

impl<T> RededicateOps<ComplexTimeVector<T>> for GenericDataVec<T>
        where T: RealNumber {
    fn rededicate(self) -> ComplexTimeVector<T> {
        let valid_len = 
            if self.is_complex == true
                && self.domain == DataVecDomain::Time {
                self.valid_len
            } else {
                0
            };
    
        ComplexTimeVector {
            delta: self.delta,
            is_complex: true,
            valid_len: valid_len,
            domain: DataVecDomain::Time,
            data: self.data,
            temp: self.temp,
            multicore_settings: self.multicore_settings
        }
    }

    fn rededicate_from(other: ComplexTimeVector<T>) -> Self {
        GenericDataVec {
            delta: other.delta,
            is_complex: other.is_complex,
            valid_len: other.valid_len,
            domain: other.domain,
            data: other.data,
            temp: other.temp,
            multicore_settings: other.multicore_settings
        }
    }
}

impl<T> RededicateOps<ComplexFreqVector<T>> for GenericDataVec<T>
    where T: RealNumber {
    fn rededicate(self) -> ComplexFreqVector<T> {
        let valid_len = 
            if self.is_complex == true
                && self.domain == DataVecDomain::Frequency {
                self.valid_len
            } else {
                0
            };
    
        ComplexFreqVector {
            delta: self.delta,
            is_complex: true,
            valid_len: valid_len,
            domain: DataVecDomain::Frequency,
            data: self.data,
            temp: self.temp,
            multicore_settings: self.multicore_settings
        }
    }

    fn rededicate_from(other: ComplexFreqVector<T>) -> Self {
        GenericDataVec {
            delta: other.delta,
            is_complex: other.is_complex,
            valid_len: other.valid_len,
            domain: other.domain,
            data: other.data,
            temp: other.temp,
            multicore_settings: other.multicore_settings
        }
    }
}

impl<T> RededicateOps<RealTimeVector<T>> for GenericDataVec<T>
    where T: RealNumber {
    fn rededicate(self) -> RealTimeVector<T> {
        let valid_len = 
            if self.is_complex == false
                && self.domain == DataVecDomain::Time {
                self.valid_len
            } else {
                0
            };
    
        RealTimeVector {
            delta: self.delta,
            is_complex: false,
            valid_len: valid_len,
            domain: DataVecDomain::Time,
            data: self.data,
            temp: self.temp,
            multicore_settings: self.multicore_settings
        }
    }

    fn rededicate_from(other: RealTimeVector<T>) -> Self {
        GenericDataVec {
            delta: other.delta,
            is_complex: other.is_complex,
            valid_len: other.valid_len,
            domain: other.domain,
            data: other.data,
            temp: other.temp,
            multicore_settings: other.multicore_settings
        }
    }
}

impl<T> RededicateOps<RealFreqVector<T>> for GenericDataVec<T>
    where T: RealNumber {
    fn rededicate(self) -> RealFreqVector<T> {
        let valid_len = 
            if self.is_complex == false
                && self.domain == DataVecDomain::Frequency {
                self.valid_len
            } else {
                0
            };
        
        RealFreqVector {
            delta: self.delta,
            is_complex: false,
            valid_len: valid_len,
            domain: DataVecDomain::Frequency,
            data: self.data,
            temp: self.temp,
            multicore_settings: self.multicore_settings
        }
    }

    fn rededicate_from(other: RealFreqVector<T>) -> Self {
        GenericDataVec {
            delta: other.delta,
            is_complex: other.is_complex,
            valid_len: other.valid_len,
            domain: other.domain,
            data: other.data,
            temp: other.temp,
            multicore_settings: other.multicore_settings
        }
    }
}

impl<T> RededicateOps<GenericDataVec<T>> for GenericDataVec<T>
    where T: RealNumber {
    fn rededicate(self) -> GenericDataVec<T> {
        self
    }

    fn rededicate_from(other: GenericDataVec<T>) -> Self {
        other
    }
}