use ocl::*;
use ocl::builders::ProgramBuilder;
use ocl::aliases::{ClFloat4, ClDouble2};
use ocl::flags::DeviceType;
use ocl::traits::{OclVec, OclPrm};
use ocl::enums::*;
use std::mem;
use std::cmp;
use std::ops::Range;
use super::GpuSupport;
use {RealNumber, array_to_complex, array_to_complex_mut};
use num;
use clfft::{ClFftPrm, builder, Precision, Layout, Direction};

pub type Gpu32 = ClFloat4;

pub type Gpu64 = ClDouble2;

pub trait GpuRegTrait : OclPrm + OclVec {}

pub trait GpuFloat : ClFftPrm {}

impl<T> GpuRegTrait for T
    where T: OclPrm + OclVec {}

impl<T> GpuFloat for T
    where T: ClFftPrm {}

const KERNEL_SRC_32: &'static str = r#"
    #define mulc32(a,b) ((float4)((float)mad(-(a).y, (b).y, (a).x * (b).x), (float)mad((a).y, (b).x, (a).x * (b).y), (float)mad(-(a).z, (b).z, (a).w * (b).w), (float)mad((a).z, (b).w, (a).w * (b).z)))

    __kernel
    void conv_vecs_r(
                __global float4 const* const src,
                __global float4 const* const conv,
                __global float4* const res)
    {
        ulong const idx = get_global_id(0);

        float4 sum1 = (float4)0.0;
        float4 sum2 = (float4)0.0;
        float4 sum3 = (float4)0.0;
        float4 sum4 = (float4)0.0;
        long start = idx - FILTER_LENGTH / 2;
        long end = start + FILTER_LENGTH;
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
    void conv_vecs_c(
                __global float4 const* const src,
                __global float4 const* const conv,
                __global float4* const res)
    {
        ulong const idx = get_global_id(0);

        float4 sum1 = (float4)0.0;
        float4 sum2 = (float4)0.0;
        long start = idx - FILTER_LENGTH / 2;
        long end = start + FILTER_LENGTH;
        long j = 0;
        for (long i = start; i < end; i++) {
            float4 current = src[i];
            sum1 += mulc32(current, conv[j]);
            j++;
            sum2 += mulc32(current, conv[j]);
            j++;
        }

        // float4 order is: xyzw
        res[idx] = (float4)
                   (sum1.x+sum1.z,
                    sum1.y+sum1.w,
                    sum2.x+sum2.z,
                    sum2.y+sum2.w);
    }
"#;

const KERNEL_SRC_64: &'static str = r#"
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #define mulc64(a,b) ((double2)((double)mad(-(a).y, (b).y, (a).x * (b).x), (double)mad((a).y, (b).x, (a).x * (b).y)))

    __kernel
    void conv_vecs_r(
                __global double2 const* const src,
                __global double2 const* const conv,
                __global double2* const res)
    {
        ulong const idx = get_global_id(0);

        double2 sum1 = (double2)0.0;
        double2 sum2 = (double2)0.0;
        long start = idx - FILTER_LENGTH / 2;
        long end = start + FILTER_LENGTH;
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


    __kernel
    void conv_vecs_c(
                __global double2 const* const src,
                __global double2 const* const conv,
                __global double2* const res)
    {
        ulong const idx = get_global_id(0);

        double2 sum = (double2)0.0;
        long start = idx - FILTER_LENGTH / 2;
        long end = start + FILTER_LENGTH;
        long j = 0;
        for (long i = start; i < end; i++) {
            double2 current = src[i];
            sum += mulc64(current, conv[j]);
            j++;
        }

        res[idx] = sum;
    }
"#;


fn has_f64_support(device: Device) -> bool {
    const F64_SUPPORT: &'static str = "cl_khr_fp64";
    match device.info(DeviceInfo::Extensions) {
        DeviceInfoResult::Extensions(ext) => ext.contains(F64_SUPPORT),
        _ => false
    }
}

/// Integrated GPUs are typically less powerful but data transfer from memory to the
/// integrated device can be faster.
fn is_integrated_gpu(device: Device) -> bool {
    const INTEL_VENDOR_ID: &'static str = "8086";
    match device.info(DeviceInfo::VendorId) {
        DeviceInfoResult::Extensions(vid) => vid == INTEL_VENDOR_ID,
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
/// As an optimization the integrated GPU is preffered for
/// small data sets since the latency is much lower (since we don't need t ogo over
/// the PCI bus).
fn determine_processing_power(device: Device, data_length: usize) -> u32 {
    if data_length < 50000 && is_integrated_gpu(device) {
        return u32::max_value();
    }

    match device.info(DeviceInfo::MaxComputeUnits) {
        DeviceInfoResult::MaxComputeUnits(units) => units,
        _ => 0
    }
}

fn find_gpu_device(require_f64_support: bool, data_length: usize) -> Option<(Platform, Device)> {
    let mut result: Option<(Platform, Device)> = None;
    for p in Platform::list() {
        let devices_op = Device::list(&p, DeviceType::from_bits(ffi::CL_DEVICE_TYPE_GPU));
        match devices_op {
            Ok(devices) => {
                for d in devices {
                    if !require_f64_support || has_f64_support(d) {
                        result = match result {
                            Some((cp, cd))
                                if determine_processing_power(d, data_length)
                                    < determine_processing_power(cd, data_length)
                              => Some((cp, cd)),
                            _ => Some((p, d))
                        }
                    }
                };
            },
            Err(_) => ()
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
            panic!("Argument must fit into an integer number of GPU registers");
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
            panic!("Argument must fit into an integer number of GPU registers");
        }
        let trans: &mut [R] = mem::transmute(array);
        &mut trans[0..len / vec_len]
    }
}

/// Prepare impulse response
///
/// The data is layout so that it's easier/faster for the kernel to go through the
/// coefficients.
///
/// An example for the data layout can be found in the unit test section.
fn prepare_impulse_response<T: Clone + Copy + num::Zero>(
        imp_resp: &[T],
        destination: &mut [T],
        vec_len: usize) {
    for (n, j) in imp_resp.iter().rev().zip(0..) {
        for i in 0..vec_len {
            let p = j + i;
            let tuple_pos = p % vec_len;
            let tuple = ((p - tuple_pos) + i) * vec_len;
            destination[tuple + tuple_pos] = *n;
        }
    }
}

impl<T> GpuSupport<T> for T
    where T: RealNumber {
    fn has_gpu_support() -> bool {
        find_gpu_device(mem::size_of::<T>() == 8, 0).is_some()
    }

    fn gpu_convolve_vector(is_complex: bool, source: &[T], target: &mut [T], imp_resp: &[T])
            -> Option<Range<usize>> {
        assert!(target.len() >= source.len());
        let is_f64 = mem::size_of::<T>() == 8;
        let vec_len = if is_f64 { 2 } else { 4 };
        let data_set_size = (source.len() / vec_len) * vec_len;
        let conv_size = imp_resp.len();

        let conv_size_rounded =
            (conv_size as f32 / vec_len as f32).ceil() as usize * vec_len;
        let conv_size_padded = conv_size_rounded + vec_len;

        if conv_size_padded >= data_set_size {
            return None;
        }

        let num_conv_vectors = conv_size_padded / vec_len;
        let phase = match conv_size % (2 * vec_len) {
            0 => 0,
            x => vec_len - x / 2
        };

        let (platform, device) = find_gpu_device(is_f64, data_set_size)
            .expect("No GPU device available which supports this data type");

        let kernel_src = if is_f64 { KERNEL_SRC_64 } else { KERNEL_SRC_32 };
        // Create an all-in-one context, program, and command queue:
        let prog_bldr = ProgramBuilder::new()
            .cmplr_def("FILTER_LENGTH", num_conv_vectors as i32)
            .cmplr_opt("-cl-fast-relaxed-math -DMAC")
            .src(kernel_src);
        let source = array_to_gpu_simd::<T, T::GpuReg>(&source[phase..data_set_size-vec_len+phase]);
        let ocl_pq = ProQue::builder()
            .prog_bldr(prog_bldr)
            .platform(platform)
            .device(device)
            .dims([source.len()])
            .build()
            .expect("Building ProQue");

        let step_size = if is_complex { 2 } else { 1 };
        let mut imp_vec_padded = vec!(T::zero(); vec_len * conv_size_padded / step_size);
        if is_complex {
            let complex = array_to_complex(&imp_resp);
            let complex_dest = array_to_complex_mut(&mut imp_vec_padded);
            prepare_impulse_response(complex, complex_dest, vec_len / 2);
        }
        else {
            prepare_impulse_response(imp_resp, &mut imp_vec_padded, vec_len);
        }

        // Create buffers
        let in_buffer =
            Buffer::new(
                ocl_pq.queue().clone(),
                Some(core::MEM_READ_ONLY |
                     core::MEM_COPY_HOST_PTR),
                ocl_pq.dims().clone(),
                Some(source))
                .expect("Failed to create GPU input buffer");

       let imp_vec_padded = array_to_gpu_simd::<T, T::GpuReg>(&imp_vec_padded);
       let imp_buffer =
            Buffer::new(
                ocl_pq.queue().clone(),
                Some(core::MEM_READ_ONLY |
                     core::MEM_COPY_HOST_PTR),
                [imp_vec_padded.len()],
                Some(&imp_vec_padded))
                .expect("Failed to create GPU impulse response buffer");;

       let res_buffer =
            Buffer::<T::GpuReg>::new(
                ocl_pq.queue().clone(),
                Some(core::MEM_WRITE_ONLY),
                ocl_pq.dims().clone(),
                None)
                .expect("Failed to create GPU result buffer");;

       let kenel_name = if is_complex { "conv_vecs_c" } else { "conv_vecs_r" };

       // Compile the kernel
       let kernel = ocl_pq.create_kernel(kenel_name).expect("ocl program build")
            .arg_buf_named("src", Some(&in_buffer))
            .arg_buf_named("conv", Some(&imp_buffer))
            .arg_buf(&res_buffer);

       // Execute kernel, do this in chunks so that the GPU watchdog isn't
       // terminating our kernel
       let gws_total = (data_set_size - conv_size_padded) / vec_len;
       let chunk_size = 100000;
       let mut chunk = conv_size_padded / vec_len;
       while chunk < gws_total {
           let current_size = cmp::min(chunk_size, gws_total - chunk);
           kernel
            .cmd()
            .gwo([chunk]) // Offset
            .gws([current_size])
            .enq()
            .expect(&format!("Running kernel (Params: {}, {}) failed:",
                                chunk, gws_total));
            chunk += chunk_size;
        }

       // Wait for all kernels to finish
       ocl_pq.queue().finish();
       res_buffer.cmd()
        .read(array_to_gpu_simd_mut::<T, T::GpuReg>(&mut target[0..data_set_size-vec_len]))
        .enq()
        .expect("Transferring result vector from the GPU back to memory failed");

       Some(Range { start: conv_size_padded, end: data_set_size - conv_size_padded })
    }

    fn is_supported_fft_len(is_complex: bool, len: usize) -> bool {
        if !is_complex { false }
        else if len == 0 { false }
        else if len == 1 { true }
        else if len % 2 == 0 { Self::is_supported_fft_len(is_complex, len / 2) }
        else if len % 3 == 0 { Self::is_supported_fft_len(is_complex, len / 3) }
        else if len % 5 == 0 { Self::is_supported_fft_len(is_complex, len / 5) }
        else if len % 7 == 0 { Self::is_supported_fft_len(is_complex, len / 7) }
        else if len % 11 == 0 { Self::is_supported_fft_len(is_complex, len / 11) }
        else if len % 13 == 0 { Self::is_supported_fft_len(is_complex, len / 13) }
        else { false }
    }

    fn fft(
        _: bool,
        source: &[T],
        target: &mut [T],
        reverse: bool) {
        let len = source.len();
        // Build ocl ProQue
        let prog_bldr = ProgramBuilder::new();
        // clFFT sometimes fails if we try to force it to use a certain device. Therefore
        // we don't set a device in the builder.
        let ocl_pq = ProQue::builder()
            .prog_bldr(prog_bldr)
            .dims([source.len()])
            .build()
            .expect("Building ProQue");

        // Create buffers
        let mut in_buffer =
            Buffer::new(
                ocl_pq.queue().clone(),
                Some(flags::MEM_READ_ONLY |
                     flags::MEM_COPY_HOST_PTR),
                ocl_pq.dims().clone(),
                Some(&source))
                .expect("Failed to create GPU input buffer");

        let mut res_buffer =
            Buffer::<T>::new(
                ocl_pq.queue().clone(),
                Some(flags::MEM_WRITE_ONLY),
                ocl_pq.dims().clone(),
                None)
                .expect("Failed to create GPU result buffer");

        // Make a plan
        let plan =
            builder::<T>()
            .precision(Precision::Precise)
            .dims([len / 2])
            .input_layout(Layout::ComplexInterleaved)
            .output_layout(Layout::ComplexInterleaved)
            .bake_out_of_place_plan(&ocl_pq).unwrap();

        let direction = if reverse { Direction::Backward } else { Direction::Forward };
        // Execute plan
        plan.enqueue(direction, &mut in_buffer, &mut res_buffer).unwrap();

        // Wait for calculation to finish and read results
        res_buffer.cmd()
            .read(target)
            .enq()
            .expect("Transferring result vector from the GPU back to memory failed");

        ocl_pq.queue().finish();
    }
}

/// These testa are only compiled&run with the feature flag `gpu_support`.
/// The tests assume that the machine running the tests has a GPU which at least supports
/// 32bit floating point numbers. However the library can be compiled with enabled GPU support
/// even if the machine doesn't have a suitable GPU.
#[cfg(test)]
mod tests {
    use super::super::GpuSupport;
    use super::super::super::*;
    use super::prepare_impulse_response;
    use {array_to_complex, array_to_complex_mut};
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
        let range = f32::gpu_convolve_vector(false,&source[..], &mut target[..], &imp_resp[..]).unwrap();
        assert_eq_tol(&target[range.clone()], &source_vec[range.clone()], 1e-6);
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
        let range = f64::gpu_convolve_vector(false,&source[..], &mut target[..], &imp_resp[..]).unwrap();
        assert_eq_tol(&target[range.clone()], &source_vec[range.clone()], 1e-6);
    }

    #[test]
    fn gpu_complex_convolution32() {
        assert!(f32::has_gpu_support());

        let source = vec![0.2; 1000];
        let mut target = vec![0.0; 1000];
        let imp_resp = vec![0.1; 64];
        let mut source_vec = source.clone().to_complex_time_vec();
        let imp_resp_vec = imp_resp.clone().to_complex_time_vec();
        let mut buffer = SingleBuffer::new();
        source_vec.convolve_vector(&mut buffer, &imp_resp_vec).unwrap();
        let range = f32::gpu_convolve_vector(true, &source[..], &mut target[..], &imp_resp[..]).unwrap();
        assert_eq_tol(&target[range.clone()], &source_vec[range.clone()], 1e-6);
    }

    #[test]
    fn gpu_complex_convolution64() {
        if !f64::has_gpu_support() {
            // Allow to skip tests on a host without GPU for f64
            return;
        }

        let source: Vec<f64> = vec![0.2; 1000];
        let mut target = vec![0.0; 1000];
        let imp_resp = vec![0.1; 64];
        let mut source_vec = source.clone().to_complex_time_vec();
        let imp_resp_vec = imp_resp.clone().to_complex_time_vec();
        let mut buffer = SingleBuffer::new();
        source_vec.convolve_vector(&mut buffer, &imp_resp_vec).unwrap();
        let range = f64::gpu_convolve_vector(true, &source[..], &mut target[..], &imp_resp[..]).unwrap();
        assert_eq_tol(&target[range.clone()], &source_vec[range.clone()], 1e-6);
    }

    #[test]
    fn gpu_prepare_real_impulse_response() {
        let imp_resp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut padded = vec!(0.0; 8 * 2);
        prepare_impulse_response(&imp_resp, &mut padded, 2);
        // Explanation of the result:
        // The data is chunked in pairs since `vec_len` is 2. So that means the data is
        // put into a format so that if the GPU loads a vector of size 2 it always
        // loads a correct value. If there is not enough data to form a pair then a zero is added.
        //
        // The convolution requires that the we iterate through the impulse response in
        // reversed order. However for some systems its faster to access data in forward order
        // (e.g. because of cache prediction). So that the reason why the result here is already
        // inverted.
        //
        // Finally every second pair is shifted by one byte. That's because the previous steps
        // mean that we can only access the `vec_len` samples at once, but to calculate
        // the convolution we need to access every sample. The shifted version give us that
        let expected = [7.0, 6.0, 0.0, 7.0, 5.0, 4.0, 6.0, 5.0,
                        3.0, 2.0, 4.0, 3.0, 1.0, 0.0, 2.0, 1.0];
        assert_eq_tol(&padded, &expected, 1e-6);
    }

    #[test]
    fn gpu_prepare_complex_impulse_response() {
        let imp_resp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut padded = vec!(0.0; 8 * 2);
        prepare_impulse_response(array_to_complex(&imp_resp), array_to_complex_mut(&mut padded), 2);
        let expected = [5.0, 6.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0,
                        1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 1.0, 2.0];
        assert_eq_tol(&padded, &expected, 1e-6);
    }
}
