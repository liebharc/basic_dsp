macro_rules! define_generic_operations_forward {
    (from: $name:ident, to: $gen_type:ident, $($data_type:ident),*)
     =>
     {
         $(
            impl GenericVectorOps<$data_type> for $name<$data_type>
            {
                fn add_vector(self, summand: &Self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().add_vector(&summand.to_gen_borrow()))
                }
                
                fn add_smaller_vector(self, summand: &Self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().add_smaller_vector(&summand.to_gen_borrow()))
                }
        
                fn subtract_vector(self, subtrahend: &Self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().subtract_vector(&subtrahend.to_gen_borrow()))
                }
                
                fn subtract_smaller_vector(self, subtrahend: &Self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().subtract_smaller_vector(&subtrahend.to_gen_borrow()))
                }
                
                fn multiply_vector(self, factor: &Self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().multiply_vector(&factor.to_gen_borrow()))
                }
                
                fn multiply_smaller_vector(self, factor: &Self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().multiply_smaller_vector(&factor.to_gen_borrow()))
                }
                
                fn divide_vector(self, divisor: &Self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().divide_vector(&divisor.to_gen_borrow()))
                }
                
                fn divide_smaller_vector(self, divisor: &Self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().divide_smaller_vector(&divisor.to_gen_borrow()))
                }
                
                fn zero_pad(self, points: usize, option: PaddingOption) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().zero_pad(points, option))
                }
                
                fn reverse(self) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().reverse())
                }
                
                fn zero_interleave(self, factor: u32) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().zero_interleave(factor))
                }
                
                fn diff(self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().diff())
                }
                
                fn diff_with_start(self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().diff_with_start())
                }
                
                fn cum_sum(self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().cum_sum())
                }
                
                fn sqrt(self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().sqrt()) 
                }
                
                fn square(self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().square()) 
                }
                
                fn root(self, degree: $data_type) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().root(degree)) 
                }
                
                fn powf(self, exponent: $data_type) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().powf(exponent)) 
                }
                
                fn ln(self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().ln()) 
                }
                
                fn exp(self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().exp()) 
                }
            
                fn log(self, base: $data_type) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().log(base)) 
                }
                
                fn sin(self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().sin()) 
                }
                
                fn cos(self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().cos()) 
                }
    
                fn tan(self) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().tan()) 
                }
                
                fn asin(self) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().asin()) 
                }
               
                fn acos(self) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().acos()) 
                }
                
                fn atan(self) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().atan()) 
                }
                
                fn sinh(self) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().sinh()) 
                }
                
                fn cosh(self) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().cosh()) 
                }
                
                fn tanh(self) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().tanh()) 
                }
                
                fn asinh(self) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().asinh()) 
                }
                
                fn acosh(self) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().acosh()) 
                }
                
                fn atanh(self) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().atanh()) 
                }
                
                fn swap_halves(self) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().swap_halves()) 
                }
                
                fn expf(self, base: $data_type) -> TransRes<Self>
                {
                    Self::from_genres(self.to_gen().expf(base)) 
                }
                
                fn override_data(self, data: &[$data_type]) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().override_data(data))
                }
                
                fn split_into(&self, targets: &mut [Box<Self>]) -> VoidResult {
                    unsafe { 
                        self.to_gen_borrow().split_into(mem::transmute(targets))
                    }
                }
                
                fn merge(self, sources: &[Box<Self>]) -> TransRes<Self> {
                    unsafe { 
                        Self::from_genres(self.to_gen().merge(mem::transmute(sources)))
                    }
                }
            }
       )*
    }    
}