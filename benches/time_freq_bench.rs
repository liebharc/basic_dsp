#![feature(test)]
#![feature(box_syntax)] 
extern crate test;
extern crate basic_dsp;
extern crate num;

pub mod tools;

#[cfg(test)]
mod bench {
    use test::Bencher;
    use basic_dsp::{
        DataVector,
        TimeDomainOperations,
        FrequencyDomainOperations,
        DataVector32,
        DataVectorDomain};
    use tools::VectorBox;
    use basic_dsp::window_functions::TriangularWindow;
    
    #[bench]
    fn plain_fft_ifft_32t_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<DataVector32>::with_size(true, 2000);
        b.iter(|| {
            vector.execute_res(|v|  
            { 
                if v.domain() == DataVectorDomain::Time {
                    v.plain_fft()
                } else {
                    v.plain_ifft()
                }
            } )
        });
    }
    
    #[bench]
    fn window_32t_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<DataVector32>::with_size(true, 2000);
        b.iter(|| {
            vector.execute_res(|v|  
            { 
                let triag = TriangularWindow;
                v.apply_window(&triag) 
            } )
        });
    }
    
    #[bench]
    fn fft_ifft_32t_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<DataVector32>::with_size(true, 2000);
        b.iter(|| {
            vector.execute_res(|v|  
            { 
                if v.domain() == DataVectorDomain::Time {
                    v.fft()
                } else {
                    v.ifft()
                }
            } )
        });
    }
}