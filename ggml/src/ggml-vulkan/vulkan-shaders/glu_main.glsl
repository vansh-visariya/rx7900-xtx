float glu_value(const uint i) {

    const uint i23 = fastdiv(i, p.ne2_012mp, p.ne2_012L);
    const uint i23_offset = i23 * p.ne22*p.ne21*p.ne20;
    const uint i22 = fastdiv(i - i23_offset, p.ne2_01mp, p.ne2_01L);
    const uint i22_offset = i22*p.ne21*p.ne20;
    const uint i21 = fastdiv(i - i23_offset - i22_offset, p.ne2_0mp, p.ne2_0L);
    const uint i20 = i - i23_offset - i22_offset - i21*p.ne20;

    const uint src_idx_a = get_aoffset() + i23 * p.nb03 + i22 * p.nb02 + i21 * p.nb01 + i20 * p.nb00;
    const uint src_idx_b = get_boffset() + i23 * p.nb13 + i22 * p.nb12 + i21 * p.nb11 + i20 * p.nb10;
    const uint dst_idx = get_doffset() + i23 * p.nb23 + i22 * p.nb22 + i21 * p.nb21 + i20 * p.nb20;

    if (p.mode == 0) {
        // Default
        const uint offset = (p.ne00 / 2) * p.nb00;
        const uint idx = src_idx_a;

        const float r = op(float(data_a[idx]), float(data_a[idx + offset]));
        data_d[dst_idx] = D_TYPE(r);
        return r;
    } else if (p.mode == 1) {
        // Swapped
        const uint offset = (p.ne00 / 2) * p.nb00;
        const uint idx = src_idx_a;

        const float r = op(float(data_a[idx + offset]), float(data_a[idx]));
        data_d[dst_idx] = D_TYPE(r);
        return r;
    } else {
        // Split
        const float r = op(float(data_a[src_idx_a]), float(data_b[src_idx_b]));
        data_d[dst_idx] = D_TYPE(r);
        return r;
    }
}

#if GLU_Q8_1
shared vec4 q8_sh[128];
#endif

void main() {
    const uint i = gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;

#if GLU_Q8_1
    // each workgroup holds 512 consecutive outputs (16 q8_1 blocks): stage them, then quantize a vec4 per
    // invocation like quantize_q8_1 (8 invocations per block)
    const uint lid = gl_LocalInvocationID.x;
    const float r = i < p.N ? glu_value(i) : 0.0;
    q8_sh[lid / 4][lid % 4] = r;
    barrier();
    if (lid < 128) {
        const uint e = i - lid + 4 * lid;
        if (e < p.N) {
            const vec4 vals = q8_sh[lid];
            const vec4 abs_vals = abs(vals);
            const float amax = subgroupClusteredMax(max(max(abs_vals.x, abs_vals.y), max(abs_vals.z, abs_vals.w)), 8);
            const float d = amax / 127.0;
            const float d_inv = d != 0.0 ? 1.0 / d : 0.0;
            const vec4 q = round(vals * d_inv);
            const float sum = subgroupClusteredAdd(q.x + q.y + q.z + q.w, 8);
            const uint ib = e / 32;
            data_q[ib / 4].qs[(ib % 4) * 8 + (e % 32) / 4] = pack32(i8vec4(q));
            if (e % 32 == 0) {
                data_q[ib / 4].ds[ib % 4] = f16vec2(vec2(d, sum * d));
            }
        }
    }
#else
    if (i >= p.N) {
        return;
    }
    glu_value(i);
#endif
}
