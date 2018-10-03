pub const CONV_KERNEL: &'static str = r#"
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

pub static MUL_KERNEL: &'static str = r#"
    #define mulc32(a,b) ((float2)((float)mad(-(a).y, (b).y, (a).x * (b).x), (float)mad((a).y, (b).x, (a).x * (b).y)))

    __kernel void multiply_vector(
                __global float2 const* const coeff,
                __global float2* const srcres)
    {
        uint const idx = get_global_id(0);
        srcres[idx] = mulc32(srcres[idx], coeff[idx]);
    }
"#;
