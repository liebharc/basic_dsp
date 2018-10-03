pub const CONV_KERNEL: &'static str = r#"
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

pub static MUL_KERNEL: &'static str = r#"
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #define mulc64(a,b) ((double2)((double)mad(-(a).y, (b).y, (a).x * (b).x), (double)mad((a).y, (b).x, (a).x * (b).y)))

    __kernel void multiply_vector(
                __global double2 const* const coeff,
                __global double2* const srcres)
    {
        uint const idx = get_global_id(0);
        srcres[idx] = mulc64(srcres[idx], coeff[idx]);
    }
"#;
