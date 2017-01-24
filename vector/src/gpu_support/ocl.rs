use ocl::*;
use ocl::builders::ProgramBuilder;
use ocl::aliases::{ClFloat4, ClDouble2};
use ocl::flags::DeviceType;
use ocl::traits::{OclVec, OclPrm};
use ocl::enums::*;
use std::mem;
use super::GpuSupport;
use RealNumber;

pub type Gpu32 = ClFloat4;

pub type Gpu64 = ClDouble2;

pub trait GpuRegTrait : OclPrm + OclVec {}

impl<T> GpuRegTrait for T
    where T: OclPrm + OclVec {}

const KERNEL_SRC_32: &'static str = r#"
    #define mulc32(a,b) ((float4)((float)mad(-(a).y, (b).y, (a).x * (b).x), (float)mad((a).y, (b).x, (a).x * (b).y), (float)mad(-(a).z, (b).z, (a).w * (b).w), (float)mad((a).z, (b).w, (a).w * (b).z)))

    __kernel
    void conv_vecs32r(
                __global float4 const* const src,
                __constant float4 const* const conv,
                __global float4* const res)
    {
        ulong const idx = get_global_id(0);
        ulong const data_length = get_global_size(0);

        float4 sum1 = (float4)0.0;
        float4 sum2 = (float4)0.0;
        float4 sum3 = (float4)0.0;
        float4 sum4 = (float4)0.0;
        long start = max((long)(idx - FILTER_LENGTH / 2), 0L);
        long end = min(start + FILTER_LENGTH, (long)data_length);
        long j = 0;
        for (long i = start; i < end; i++) {
            float4 current = src[i];
            sum1 += current * conv[j];
            j++;
            sum2 += current * conv[j];
            j++;
            sum3 += current * conv[j];
            j++;
            sum4 += current * conv[j];
            j++;
        }

        res[idx] = (float4)
                   (sum1.x+sum1.y+sum1.z+sum1.w,
                    sum2.x+sum2.y+sum2.z+sum2.w,
                    sum3.x+sum3.y+sum3.z+sum3.w,
                    sum4.x+sum4.y+sum4.z+sum4.w);
    }

    __kernel
    void conv_vecs32c(
                __global float4 const* const src,
                __constant float4 const* const conv,
                __global float4* const res)
    {
        ulong const idx = get_global_id(0);
        ulong const data_length = get_global_size(0);

        float4 sum1 = (float4)0.0;
        float4 sum2 = (float4)0.0;
        long start = max((long)(idx - FILTER_LENGTH / 2), 0L);
        long end = min(start + FILTER_LENGTH, (long)data_length);
        long j = 0;
        for (long i = start; i < end; i+=2) {
            float4 current = src[i];
            sum1 += mulc32(current, conv[j]);
            j+=2;
            sum2 += mulc32(current, conv[j]);
            j+=2;
        }

        res[idx] = (float4)
                   (sum1.x+sum1.w,
                    sum1.y+sum1.z,
                    sum2.x+sum2.w,
                    sum2.y+sum2.z);
    }
"#;

const KERNEL_SRC_64: &'static str = r#"
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #define mulc64(a,b) ((float2)((float)mad(-(a).y, (b).y, (a).x * (b).x), (float)mad((a).y, (b).x, (a).x * (b).y))

    __kernel
    void conv_vecs64r(
                __global double2 const* const src,
                __constant double2 const* const conv,
                __global double2* const res)
    {
        ulong const idx = get_global_id(0);
        ulong const data_length = get_global_size(0);

        double2 sum1 = (double2)0.0;
        double2 sum2 = (double2)0.0;
        long start = max((long)(idx - FILTER_LENGTH / 2), 0L);
        long end = min(start + FILTER_LENGTH, (long)data_length);
        long j = 0;
        for (long i = start; i < end; i++) {
            double2 current = src[i];
            sum1 += current * conv[j];
            j++;
            sum2 += current * conv[j];
            j++;
        }

        res[idx] = (double2)
                   (sum1.x+sum1.y,
                    sum2.x+sum2.y);
    }
"#;


fn has_f64_support(device: Device) -> bool {
    const F64_SUPPORT: &'static str = "cl_khr_fp64";
    match device.info(DeviceInfo::Extensions) {
        DeviceInfoResult::Extensions(ext) => ext.contains(F64_SUPPORT),
        _ => false
    }
}

/// Returns an indicator of how powerful the device is. More powerful
/// devices should get the calculations done faster. The higher the
/// returned value, the higher the device is rated.
///
/// For now we only look at the number of computational units. Likely this
/// should be a good enough indication for normal consumer PCs which come
/// with up to two GPUs: One on the CPU and one dedicated GPU. The dedicated
/// GPU is likely the better choice in most cases for large data sets and it should
/// have more computational units.
///
/// A potential optimization might be to prefer GPU integrated into the CPU for
/// small data sets since the latency is much higher (since we don't need t ogo over
/// the PCI bus).
fn determine_processing_power(device: Device) -> u32 {
    match device.info(DeviceInfo::MaxComputeUnits) {
        DeviceInfoResult::MaxComputeUnits(units) => units,
        _ => 0
    }
}

fn find_gpu_device(require_f64_support: bool) -> Option<(Platform, Device)> {
    let mut result: Option<(Platform, Device)> = None;
    for p in Platform::list() {
        let devices = Device::list(&p, DeviceType::from_bits(ffi::CL_DEVICE_TYPE_GPU));
        if devices.is_ok() {
              for d in devices.unwrap() {
                  if !require_f64_support || has_f64_support(d) {
                      result = match result {
                          Some((cp, cd)) if determine_processing_power(d) < determine_processing_power(cd)
                            => Some((cp, cd)),
                          _ => Some((p, d))
                      }
                  }
              }
        }
     }

     return result;
}

fn array_to_gpu_simd<T, R>(array: &[T]) -> &[R] {
    unsafe {
        let is_f64 = mem::size_of::<T>() == 8;
        let vec_len = if is_f64 { 2 } else { 4 };
        let len = array.len();
        if len % vec_len != 0 {
            panic!("Argument must have an even length");
        }
        let trans: &[R] = mem::transmute(array);
        &trans[0..len / vec_len]
    }
}

fn array_to_gpu_simd_mut<T, R>(array: &mut [T]) -> &mut [R] {
    unsafe {
        let is_f64 = mem::size_of::<T>() == 8;
        let vec_len = if is_f64 { 2 } else { 4 };
        let len = array.len();
        if len % vec_len != 0 {
            panic!("Argument must have an even length");
        }
        let trans: &mut [R] = mem::transmute(array);
        &mut trans[0..len / vec_len]
    }
}

impl<T> GpuSupport<T> for T
    where T: RealNumber {
    fn has_gpu_support() -> bool {
        find_gpu_device(mem::size_of::<T>() == 8).is_some()
    }

    // TODO: f64 support
    // TODO: Caller needs to make sure that edges are calculated
    fn gpu_convolve_vector(is_complex: bool, source: &[T], target: &mut [T], imp_resp: &[T]) {
        assert!(target.len() >= source.len());
        let data_set_size = source.len();
        let conv_size = imp_resp.len();
        let is_f64 = mem::size_of::<T>() == 8;
        let vec_len = if is_f64 { 2 } else { 4 };

        let padding = vec_len;
        let conv_size_rounded =
            (conv_size as f32 / vec_len as f32).ceil() as usize * vec_len;
        let conv_size_padded = conv_size_rounded + padding;
        let num_conv_vectors = conv_size_padded / vec_len;
        let phase = match conv_size % (2 * vec_len) {
            0 => 0,
            x => vec_len - x / 2
        };

        let (platform, device) = find_gpu_device(is_f64)
            .expect("No GPU device available which supports this data type");

        let kernel_src = if is_f64 { KERNEL_SRC_64 } else { KERNEL_SRC_32 };
        // Create an all-in-one context, program, and command queue:
        let prog_bldr = ProgramBuilder::new()
            .cmplr_def("FILTER_LENGTH", num_conv_vectors as i32)
            .cmplr_opt("-cl-fast-relaxed-math -DMAC")
            .src(kernel_src);
        let ocl_pq = ProQue::builder()
            .prog_bldr(prog_bldr)
            .platform(platform)
            .device(device)
            .dims([data_set_size / vec_len - 1])
            .build()
            .expect("Building ProQue");

        // Prepare impulse response
        let mut imp_vec_padded = vec!(T::zero(); vec_len * conv_size_padded);
        for (n, j) in imp_resp.iter().rev().zip(0..) {
            for i in 0..vec_len {
                let p = j + i;
                let tuple_pos = p % vec_len;
                let tuple = ((p - tuple_pos) + i) * vec_len;
                imp_vec_padded[tuple + tuple_pos] = *n;
            }
        }

        // Create buffers
        let in_buffer =
            Buffer::new(
                ocl_pq.queue().clone(),
                Some(core::MEM_READ_ONLY |
                     core::MEM_COPY_HOST_PTR),
                ocl_pq.dims().clone(),
                Some(array_to_gpu_simd::<T, T::GpuReg>(&source[phase..data_set_size-vec_len+phase]))).unwrap();

       let imp_buffer =
            Buffer::new(
                ocl_pq.queue().clone(),
                Some(core::MEM_READ_ONLY |
                     core::MEM_COPY_HOST_PTR),
                [conv_size_padded],
                Some(array_to_gpu_simd::<T, T::GpuReg>(&imp_vec_padded))).unwrap();


       let res_buffer =
            Buffer::<T::GpuReg>::new(
                ocl_pq.queue().clone(),
                Some(core::MEM_WRITE_ONLY),
                ocl_pq.dims().clone(),
                None).unwrap();

       let kenel_name =
            if is_f64 { if is_complex { "conv_vecs64c" } else { "conv_vecs64r" }}
            else      { if is_complex { "conv_vecs32r" } else { "conv_vecs32r" }};
       // Kernel compilation
       let kernel = ocl_pq.create_kernel(kenel_name).expect("ocl program build")
            .gws([data_set_size / vec_len - 1])
            .arg_buf_named("src", Some(&in_buffer))
            .arg_buf_named("conv", Some(&imp_buffer))
            .arg_buf(&res_buffer);

       kernel.enq().expect("Running kernel");
       ocl_pq.queue().finish();
       res_buffer.cmd().read(array_to_gpu_simd_mut::<T, T::GpuReg>(&mut target[0..data_set_size-vec_len])).enq().expect("Transferring res_vec");
    }
}

/// These testa are only compiled&run with the feature flag `gpu_support`.
/// The tests assume that the machine running the tests has a GPU which at least supports
/// 32bit floating point numbers. However the library can be compiled with enabled GPU support
/// even if the machine doesn't have a suitable GPU.
mod tests {
    use super::super::GpuSupport;
    use super::super::super::*;
    use std::fmt::Debug;

    fn assert_eq_tol<T>(left: &[T], right: &[T], tol: T)
        where T: RealNumber + Debug
    {
        assert_eq!(left.len(), right.len());
        for i in 0..left.len() {
            if (left[i] - right[i]).abs() > tol {
                panic!("assertion failed: {:?} != {:?}", left, right);
            }
        }
    }

    #[test]
    fn gpu_real_convolution32() {
        assert!(f32::has_gpu_support());

        let source: Vec<f32> = vec![0.2; 1000];
        let mut target = vec![0.0; 1000];
        let imp_resp = vec![0.1; 64];
        let mut source_vec = source.clone().to_real_time_vec();
        let imp_resp_vec = imp_resp.clone().to_real_time_vec();
        let mut buffer = SingleBuffer::new();
        source_vec.convolve_vector(&mut buffer, &imp_resp_vec).unwrap();
        f32::gpu_convolve_vector(false,&source[..], &mut target[..], &imp_resp[..]);
        assert_eq_tol(&target[100..900], &source_vec[100..900], 1e-6);
    }

    #[test]
    fn gpu_real_convolution64() {
        if !f64::has_gpu_support() {
            // Allow to skip tests on a host without GPU for f64
            return;
        }

        let source: Vec<f64> = vec![0.2; 1000];
        let mut target = vec![0.0; 1000];
        let imp_resp = vec![0.1; 64];
        let mut source_vec = source.clone().to_real_time_vec();
        let imp_resp_vec = imp_resp.clone().to_real_time_vec();
        let mut buffer = SingleBuffer::new();
        source_vec.convolve_vector(&mut buffer, &imp_resp_vec).unwrap();
        f64::gpu_convolve_vector(false,&source[..], &mut target[..], &imp_resp[..]);
        assert_eq_tol(&target[100..900], &source_vec[100..900], 1e-6);
    }
}
