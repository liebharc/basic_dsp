use super::definitions::{
    RededicateVector,
    DataVectorDomain};
use vector_types::{
    GenericDataVector,
    RealTimeVector,
    ComplexTimeVector,
    RealFreqVector,
    ComplexFreqVector};
use RealNumber;    

macro_rules! add_rededicate_impl {
    ($(struct $name:ident, $is_complex:ident, $domain:expr);*) => {
    $(
        impl<T> RededicateVector<ComplexTimeVector<T>> for $name<T>
            where T: RealNumber {
            fn rededicate(self) -> ComplexTimeVector<T> {
                ComplexTimeVector {
                    delta: self.delta,
                    is_complex: true,
                    valid_len: 0,
                    domain: DataVectorDomain::Time,
                    data: self.data,
                    temp: self.temp,
                    multicore_settings: self.multicore_settings
                }
            }

            fn rededicate_from(other: ComplexTimeVector<T>) -> Self {
                other.rededicate()
            }
        }

        impl<T> RededicateVector<ComplexFreqVector<T>> for $name<T>
            where T: RealNumber {
            fn rededicate(self) -> ComplexFreqVector<T> {
                ComplexFreqVector {
                    delta: self.delta,
                    is_complex: true,
                    valid_len: 0,
                    domain: DataVectorDomain::Frequency,
                    data: self.data,
                    temp: self.temp,
                    multicore_settings: self.multicore_settings
                }
            }

            fn rededicate_from(other: ComplexFreqVector<T>) -> Self {
                other.rededicate()
            }
        }

        impl<T> RededicateVector<RealTimeVector<T>> for $name<T>
            where T: RealNumber {
            fn rededicate(self) -> RealTimeVector<T> {
                RealTimeVector {
                    delta: self.delta,
                    is_complex: false,
                    valid_len: 0,
                    domain: DataVectorDomain::Time,
                    data: self.data,
                    temp: self.temp,
                    multicore_settings: self.multicore_settings
                }
            }

            fn rededicate_from(other: RealTimeVector<T>) -> Self {
                other.rededicate()
            }
        }

        impl<T> RededicateVector<RealFreqVector<T>> for $name<T>
            where T: RealNumber {
            fn rededicate(self) -> RealFreqVector<T> {
                RealFreqVector {
                    delta: self.delta,
                    is_complex: false,
                    valid_len: 0,
                    domain: DataVectorDomain::Frequency,
                    data: self.data,
                    temp: self.temp,
                    multicore_settings: self.multicore_settings
                }
            }

            fn rededicate_from(other: RealFreqVector<T>) -> Self {
                other.rededicate()
            }
        }

        impl<T> RededicateVector<GenericDataVector<T>> for $name<T>
            where T: RealNumber {
            fn rededicate(self) -> GenericDataVector<T> {
                GenericDataVector {
                    delta: self.delta,
                    is_complex: self.is_complex,
                    valid_len: self.valid_len,
                    domain: self.domain,
                    data: self.data,
                    temp: self.temp,
                    multicore_settings: self.multicore_settings
                }
            }

            fn rededicate_from(other: GenericDataVector<T>) -> Self {
                other.rededicate()
            }
        }
      )*
    }
}

add_rededicate_impl!(struct RealTimeVector, false, DataVectorDomain::Time);
add_rededicate_impl!(struct ComplexTimeVector, true, DataVectorDomain::Time);
add_rededicate_impl!(struct RealFreqVector, false, DataVectorDomain::Frequency);
add_rededicate_impl!(struct ComplexFreqVector, true, DataVectorDomain::Frequency);

// Conversions to `GenericDataVector` are always valid therefore `GenericDataVector`
// needs to have a specific `RededicateVector` implementation

impl<T> RededicateVector<ComplexTimeVector<T>> for GenericDataVector<T>
        where T: RealNumber {
    fn rededicate(self) -> ComplexTimeVector<T> {
        let valid_len = 
            if self.is_complex == true
                && self.domain == DataVectorDomain::Time {
                self.valid_len
            } else {
                0
            };
    
        ComplexTimeVector {
            delta: self.delta,
            is_complex: true,
            valid_len: valid_len,
            domain: DataVectorDomain::Time,
            data: self.data,
            temp: self.temp,
            multicore_settings: self.multicore_settings
        }
    }

    fn rededicate_from(other: ComplexTimeVector<T>) -> Self {
        GenericDataVector {
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

impl<T> RededicateVector<ComplexFreqVector<T>> for GenericDataVector<T>
    where T: RealNumber {
    fn rededicate(self) -> ComplexFreqVector<T> {
        let valid_len = 
            if self.is_complex == true
                && self.domain == DataVectorDomain::Frequency {
                self.valid_len
            } else {
                0
            };
    
        ComplexFreqVector {
            delta: self.delta,
            is_complex: true,
            valid_len: valid_len,
            domain: DataVectorDomain::Frequency,
            data: self.data,
            temp: self.temp,
            multicore_settings: self.multicore_settings
        }
    }

    fn rededicate_from(other: ComplexFreqVector<T>) -> Self {
        GenericDataVector {
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

impl<T> RededicateVector<RealTimeVector<T>> for GenericDataVector<T>
    where T: RealNumber {
    fn rededicate(self) -> RealTimeVector<T> {
        let valid_len = 
            if self.is_complex == false
                && self.domain == DataVectorDomain::Time {
                self.valid_len
            } else {
                0
            };
    
        RealTimeVector {
            delta: self.delta,
            is_complex: false,
            valid_len: valid_len,
            domain: DataVectorDomain::Time,
            data: self.data,
            temp: self.temp,
            multicore_settings: self.multicore_settings
        }
    }

    fn rededicate_from(other: RealTimeVector<T>) -> Self {
        GenericDataVector {
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

impl<T> RededicateVector<RealFreqVector<T>> for GenericDataVector<T>
    where T: RealNumber {
    fn rededicate(self) -> RealFreqVector<T> {
        let valid_len = 
            if self.is_complex == false
                && self.domain == DataVectorDomain::Frequency {
                self.valid_len
            } else {
                0
            };
        
        RealFreqVector {
            delta: self.delta,
            is_complex: false,
            valid_len: valid_len,
            domain: DataVectorDomain::Frequency,
            data: self.data,
            temp: self.temp,
            multicore_settings: self.multicore_settings
        }
    }

    fn rededicate_from(other: RealFreqVector<T>) -> Self {
        GenericDataVector {
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

impl<T> RededicateVector<GenericDataVector<T>> for GenericDataVector<T>
    where T: RealNumber {
    fn rededicate(self) -> GenericDataVector<T> {
        self
    }

    fn rededicate_from(other: GenericDataVector<T>) -> Self {
        other
    }
}