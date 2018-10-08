use super::vector_types::*;

/// Marker for types containing real data.
#[derive(Debug, Clone, PartialEq)]
pub struct Real;
impl NumberSpace for Real {
    fn is_complex(&self) -> bool {
        false
    }
    fn to_complex(&mut self) {}
    fn to_real(&mut self) {}
}
impl RealNumberSpace for Real {}

/// Marker for types containing complex data.
#[derive(Debug, Clone, PartialEq)]
pub struct Complex;
impl NumberSpace for Complex {
    fn is_complex(&self) -> bool {
        true
    }
    fn to_complex(&mut self) {}
    fn to_real(&mut self) {}
}
impl ComplexNumberSpace for Complex {}

/// Marker for types containing real or complex data.
#[derive(Debug, Clone, PartialEq)]
pub struct RealOrComplex {
    pub is_complex_current: bool,
}
impl NumberSpace for RealOrComplex {
    fn is_complex(&self) -> bool {
        self.is_complex_current
    }

    fn to_complex(&mut self) {
        self.is_complex_current = true;
    }

    fn to_real(&mut self) {
        self.is_complex_current = false;
    }
}
impl RealNumberSpace for RealOrComplex {}
impl ComplexNumberSpace for RealOrComplex {}

/// Marker for types containing time data.
#[derive(Debug, Clone, PartialEq)]
pub struct Time;
impl Domain for Time {
    fn domain(&self) -> DataDomain {
        DataDomain::Time
    }
    fn to_freq(&mut self) {}
    fn to_time(&mut self) {}
}
impl TimeDomain for Time {}

/// Marker for types containing frequency data.
#[derive(Debug, Clone, PartialEq)]
pub struct Freq;
impl Domain for Freq {
    fn domain(&self) -> DataDomain {
        DataDomain::Frequency
    }
    fn to_time(&mut self) {}
    fn to_freq(&mut self) {}
}
impl FrequencyDomain for Freq {}

/// Marker for types containing time or frequency data.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeOrFreq {
    pub domain_current: DataDomain,
}
impl Domain for TimeOrFreq {
    fn domain(&self) -> DataDomain {
        self.domain_current
    }

    fn to_freq(&mut self) {
        self.domain_current = DataDomain::Frequency;
    }

    fn to_time(&mut self) {
        self.domain_current = DataDomain::Time;
    }
}

impl TimeDomain for TimeOrFreq {}
impl FrequencyDomain for TimeOrFreq {}

impl PosEq<Real> for Real {}
impl PosEq<RealOrComplex> for Real {}
impl PosEq<Real> for RealOrComplex {}
impl PosEq<Complex> for RealOrComplex {}
impl PosEq<RealOrComplex> for RealOrComplex {}
impl PosEq<Complex> for Complex {}
impl PosEq<RealOrComplex> for Complex {}

impl PosEq<Time> for Time {}
impl PosEq<TimeOrFreq> for Time {}
impl PosEq<Time> for TimeOrFreq {}
impl PosEq<Freq> for TimeOrFreq {}
impl PosEq<TimeOrFreq> for TimeOrFreq {}
impl PosEq<Freq> for Freq {}
impl PosEq<TimeOrFreq> for Freq {}