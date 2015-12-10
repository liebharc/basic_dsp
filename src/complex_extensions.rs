use num::complex::Complex;
use num::traits::Float;

pub trait ComplexExtensions<T>
    where T: Float
{
    fn cos(&self) -> Self;
    fn sin(&self) -> Self;
    fn sqrt(&self) -> Self;
    fn powf(&self, value: T) -> Self;
    fn ln(&self) -> Self;
    fn expn(&self) -> Self;
    fn log_base(&self, value: T) -> Self;
    fn exp_base(&self, value: T) -> Self;
}

#[allow(unused)]
impl<T> ComplexExtensions<T> for Complex<T>
    where T: Float
{   
    fn sin(&self) -> Self {
        Complex::new(self.re.sin() * self.im.cosh(), self.re.cos() * self.im.sinh())
    }
    
    fn cos(&self) -> Self {
        Complex::new(self.re.cos() * self.im.cosh(), -self.re.sin() * self.im.sinh())
    }
       
    fn sqrt(&self) -> Self {
        let two = T::one() + T::one();
        let (r, theta) = self.to_polar();
        Complex::from_polar(&(r.sqrt()), &(theta/two))
    }
    
    fn powf(&self, value: T) -> Self {
        let (r, theta) = self.to_polar();
        Complex::from_polar(&(r.powf(value)), &(theta*value))
    }
    
    fn ln(&self) -> Self {
        let (r, theta) = self.to_polar();
        Complex::new(r.ln(), theta)
    }
    
    fn expn(&self) -> Self {
        let exp = self.re.exp();
        // Equation is taken from http://mathfaculty.fullerton.edu/mathews/c2003/ComplexFunExponentialMod.html
        Complex::new(exp * self.im.cos(), exp * self.im.sin())
    }
    
    fn log_base(&self, value: T) -> Self {
        let (r, theta) = self.to_polar();
        let e = T::one().exp(); // there is for sure a faster way to get the constant, but I don't know how to do that in this generic context 
        Complex::new(r.log(value), theta * e.log(value))
    }
    
    // Same as value.powf(self) but I would prefer to have this version
    // around too
    fn exp_base(&self, value: T) -> Self {
        let ln = value.ln();
        let exp = (self.re * ln).exp();
        let im = self.im * ln; 
        Complex::new(exp * im.cos(), exp * im.sin())
    }
}

#[cfg(test)]
mod tests {
	use super::*;
	use num::complex::Complex32;
    
    fn assert_complex(left: Complex32, right: Complex32) {
        assert!((left.re - right.re).abs() < 1e-4);
        assert!((left.im - right.im).abs() < 1e-4);
    }
	
    #[test]
	fn powf_test()
	{
		let c = Complex32::new(2.0, -1.0);
        let r = c.powf(3.5);
        assert_complex(r, Complex32::new(-0.8684746, -16.695934));
	}
    
    #[test]
	fn logn_test()
	{
		let c = Complex32::new(2.0, -1.0);
        let r = c.ln();
        assert_complex(r, Complex32::new(0.804719, -0.4636476));
	}
    
    #[test]
	fn expn_test()
	{
		let c = Complex32::new(2.0, -1.0);
        let r = c.expn();
        assert_complex(r, Complex32::new(3.9923239, -6.217676));
	}
    
    #[test]
	fn log_test()
	{
		let c = Complex32::new(2.0, -1.0);
        let r = c.log_base(10.0);
        assert_complex(r, Complex32::new(0.349485, -0.20135958));
	}
    
    #[test]
	fn exp_test()
	{
		let c = Complex32::new(2.0, -1.0);
        let r = c.exp_base(10.0);
        assert_complex(r, Complex32::new(-66.82015, -74.39803));
	}
}