#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include "types.glsl"

#if defined(DATA_A_Q2_0)
FLOAT_TYPE get_dm(uint ib) {
    return FLOAT_TYPE(data_a[ib / 2].d);
}
#elif defined(DATA_A_Q4_0) || defined(DATA_A_Q5_0) || defined(DATA_A_Q8_0) || defined(DATA_A_IQ1_S) || defined(DATA_A_IQ2_XXS) || defined(DATA_A_IQ2_XS) || defined(DATA_A_IQ2_S) || defined(DATA_A_IQ3_XXS) || defined(DATA_A_IQ3_S) || defined(DATA_A_IQ4_XS) || defined(DATA_A_IQ4_NL)
FLOAT_TYPE get_dm(uint ib) {
    return FLOAT_TYPE(data_a[ib].d);
}
#endif

#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1)
FLOAT_TYPEV2 get_dm(uint ib) {
    return FLOAT_TYPEV2(data_a_packed32[ib].dm);
}
#endif

#if defined(DATA_A_MXFP4)
FLOAT_TYPE get_dm(uint ib) {
    return FLOAT_TYPE(e8m0_to_fp32(data_a[ib].e));
}
#endif

#if defined(DATA_A_Q2_K)
FLOAT_TYPEV2 get_dm(uint ib) {
    const uint ib_k = ib / 8;
    return FLOAT_TYPEV2(data_a_packed32[ib_k].dm);
}
#endif

// Each iqs value maps to a 32-bit integer
#if defined(DATA_A_Q2_0)
uint unpack_q2_0(uint bits) {
    // Move bit pairs [1:0], [3:2], [5:4], [7:6] to [1:0], [9:8], [17:16], [25:24].
    bits &= 0xffu;
    bits = (bits | (bits << 12u)) & 0x000f000fu;
    return (bits | (bits << 6u)) & 0x03030303u;
}

i32vec4 repack4(uint ib, uint iqs) {
    const uint qs_idx = (ib & 1u) * 4u + iqs * 2u;
    const uint bits = pack32(u16vec2(data_a_packed16[ib / 2].qs[qs_idx],
                                     data_a_packed16[ib / 2].qs[qs_idx + 1]));
    return i32vec4(unpack_q2_0(bits), unpack_q2_0(bits >> 8u),
                   unpack_q2_0(bits >> 16u), unpack_q2_0(bits >> 24u));
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(da * (float(q_sum) * dsb.x - dsb.y / float(sum_divisor)));
}
#endif

#if defined(DATA_A_Q4_0)
// 2-byte loads for Q4_0 blocks (18 bytes)
i32vec2 repack(uint ib, uint iqs) {
    const u16vec2 quants = u16vec2(data_a_packed16[ib].qs[iqs * 2    ],
                                   data_a_packed16[ib].qs[iqs * 2 + 1]);
    const uint32_t vui = pack32(quants);
    return i32vec2( vui       & 0x0F0F0F0F,
                   (vui >> 4) & 0x0F0F0F0F);
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(da * (float(q_sum) * dsb.x - (8 / sum_divisor) * dsb.y));
}
#endif

#if defined(DATA_A_Q4_1)
// 4-byte loads for Q4_1 blocks (20 bytes)
i32vec2 repack(uint ib, uint iqs) {
    const uint32_t vui = data_a_packed32[ib].qs[iqs];
    return i32vec2( vui       & 0x0F0F0F0F,
                   (vui >> 4) & 0x0F0F0F0F);
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const vec2 dma, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(float(q_sum) * dma.x * dsb.x + dma.y * dsb.y / sum_divisor);
}
#endif

#if defined(DATA_A_Q5_0)
// 2-byte loads for Q5_0 blocks (22 bytes)
i32vec2 repack(uint ib, uint iqs) {
    const u16vec2 quants = u16vec2(data_a_packed16[ib].qs[iqs * 2    ],
                                   data_a_packed16[ib].qs[iqs * 2 + 1]);
    const uint32_t vui = pack32(quants);
    const int32_t qh = int32_t((uint32_t(data_a_packed16[ib].qh[1]) << 16 | data_a_packed16[ib].qh[0]) >> (4 * iqs));
    const int32_t v0 = int32_t(vui & 0x0F0F0F0F)
                     | ((qh & 0xF) * 0x02040810) & 0x10101010; // (0,1,2,3) -> (4,12,20,28)

    const int32_t v1 = int32_t((vui >> 4) & 0x0F0F0F0F)
                     | (((qh >> 16) & 0xF) * 0x02040810) & 0x10101010; // (16,17,18,19) -> (4,12,20,28)

    return i32vec2(v0, v1);
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(da * (float(q_sum) * dsb.x - (16 / sum_divisor) * dsb.y));
}
#endif

#if defined(DATA_A_Q5_1)
// 4-byte loads for Q5_1 blocks (24 bytes)
i32vec2 repack(uint ib, uint iqs) {
    const u16vec2 quants = u16vec2(data_a_packed16[ib].qs[iqs * 2    ],
                                   data_a_packed16[ib].qs[iqs * 2 + 1]);
    const uint32_t vui = pack32(quants);
    const int32_t qh = int32_t(data_a_packed32[ib].qh >> (4 * iqs));
    const int32_t v0 = int32_t(vui & 0x0F0F0F0F)
                     | ((qh & 0xF) * 0x02040810) & 0x10101010; // (0,1,2,3) -> (4,12,20,28)

    const int32_t v1 = int32_t((vui >> 4) & 0x0F0F0F0F)
                     | (((qh >> 16) & 0xF) * 0x02040810) & 0x10101010; // (16,17,18,19) -> (4,12,20,28)

    return i32vec2(v0, v1);
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const vec2 dma, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(float(q_sum) * dma.x * dsb.x + dma.y * dsb.y / sum_divisor);
}
#endif

#if defined(DATA_A_Q8_0)
// 2-byte loads for Q8_0 blocks (34 bytes)
int32_t repack(uint ib, uint iqs) {
    return pack32(i16vec2(data_a_packed16[ib].qs[iqs * 2    ],
                          data_a_packed16[ib].qs[iqs * 2 + 1]));
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(float(q_sum) * da * dsb.x);
}
#endif

#if defined(DATA_A_MXFP4)
// 1-byte loads for mxfp4 blocks (17 bytes)
i32vec2 repack(uint ib, uint iqs) {
    const uint32_t qs = pack32(u8vec4(data_a[ib].qs[iqs * 4    ],
                                      data_a[ib].qs[iqs * 4 + 1],
                                      data_a[ib].qs[iqs * 4 + 2],
                                      data_a[ib].qs[iqs * 4 + 3]));

    const u8vec4 i_a0 = unpack8( qs       & 0x0F0F0F0F);
    const u8vec4 i_a1 = unpack8((qs >> 4) & 0x0F0F0F0F);

    return i32vec2(pack32(i8vec4(kvalues_mxfp4[i_a0.x], kvalues_mxfp4[i_a0.y], kvalues_mxfp4[i_a0.z], kvalues_mxfp4[i_a0.w])),
                   pack32(i8vec4(kvalues_mxfp4[i_a1.x], kvalues_mxfp4[i_a1.y], kvalues_mxfp4[i_a1.z], kvalues_mxfp4[i_a1.w])));
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(da * dsb.x * float(q_sum) * 0.5);
}
#endif

#if defined(DATA_A_Q2_0)
FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t q_sum = 0;
    const i32vec4 qs_a = repack4(ib_a, iqs);
    q_sum += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]);

    // 16 quants per call => divide sums by 32/16 = 2
    return mul_q8_1(q_sum, get_dm(ib_a), cache_b_ds, 2);
}
#elif defined(DATA_A_QUANT_LEGACY) || defined(DATA_A_MXFP4)
FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t q_sum = 0;
#if QUANT_R == 2
    const i32vec2 data_a_qs = repack(ib_a, iqs);
    q_sum += dotPacked4x8EXT(data_a_qs.x,
                             cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(data_a_qs.y,
                             cache_b_qs[1]);
#else
    int32_t data_a_qs = repack(ib_a, iqs * 2);
    q_sum += dotPacked4x8EXT(data_a_qs,
                             cache_b_qs[0]);
    data_a_qs = repack(ib_a, iqs * 2 + 1);
    q_sum += dotPacked4x8EXT(data_a_qs,
                             cache_b_qs[1]);
#endif

    // 2 quants per call => divide sums by 8/2 = 4
    return mul_q8_1(q_sum, get_dm(ib_a), cache_b_ds, 4);
}
#endif

#if defined(DATA_A_Q2_K)
// 4-byte loads for Q2_K blocks (84 bytes)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;

    return i32vec4((data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x03030303,
                   (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x03030303,
                   (data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x03030303,
                   (data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x03030303);
}

uint8_t get_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    return data_a[ib_k].scales[iqs_k / 4];
}

FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t sum_d = 0;
    int32_t sum_m = 0;

    const i32vec4 qs_a = repack4(ib_a, iqs * 4);
    const uint8_t scale = get_scale(ib_a, iqs * 4);
    const vec2 dm = vec2(get_dm(ib_a));
    const int32_t scale_m = int32_t(scale >> 4) * 0x01010101; // Duplicate 8-bit value across 32-bits.

    sum_d += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]) * (scale & 0xF);
    sum_m += dotPacked4x8EXT(scale_m, cache_b_qs[0]);

    sum_d += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]) * (scale & 0xF);
    sum_m += dotPacked4x8EXT(scale_m, cache_b_qs[1]);

    sum_d += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]) * (scale & 0xF);
    sum_m += dotPacked4x8EXT(scale_m, cache_b_qs[2]);

    sum_d += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]) * (scale & 0xF);
    sum_m += dotPacked4x8EXT(scale_m, cache_b_qs[3]);

    return FLOAT_TYPE(float(cache_b_ds.x) * (float(dm.x) * float(sum_d) - float(dm.y) * float(sum_m)));
}
#endif

#if defined(DATA_A_Q3_K)
// 2-byte loads for Q3_K blocks (110 bytes)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;
    const uint hm_shift = iqs_k / 8;

    const uvec4 qs = uvec4( uint32_t(data_a_packed16[ib_k].qs[qs_idx * 2    ]) |
                           (uint32_t(data_a_packed16[ib_k].qs[qs_idx * 2 + 1]) << 16),
                            uint32_t(data_a_packed16[ib_k].qs[qs_idx * 2 + 2]) |
                           (uint32_t(data_a_packed16[ib_k].qs[qs_idx * 2 + 3]) << 16),
                            uint32_t(data_a_packed16[ib_k].qs[qs_idx * 2 + 4]) |
                           (uint32_t(data_a_packed16[ib_k].qs[qs_idx * 2 + 5]) << 16),
                            uint32_t(data_a_packed16[ib_k].qs[qs_idx * 2 + 6]) |
                           (uint32_t(data_a_packed16[ib_k].qs[qs_idx * 2 + 7]) << 16));

    const uvec4 hmask = uvec4( uint32_t(data_a_packed16[ib_k].hmask[iqs * 2    ]) |
                              (uint32_t(data_a_packed16[ib_k].hmask[iqs * 2 + 1]) << 16),
                               uint32_t(data_a_packed16[ib_k].hmask[iqs * 2 + 2]) |
                              (uint32_t(data_a_packed16[ib_k].hmask[iqs * 2 + 3]) << 16),
                               uint32_t(data_a_packed16[ib_k].hmask[iqs * 2 + 4]) |
                              (uint32_t(data_a_packed16[ib_k].hmask[iqs * 2 + 5]) << 16),
                               uint32_t(data_a_packed16[ib_k].hmask[iqs * 2 + 6]) |
                              (uint32_t(data_a_packed16[ib_k].hmask[iqs * 2 + 7]) << 16));

    // bitwise OR to add 4 if hmask is set, subtract later
    const uint vals0 = ((    qs.x >> qs_shift) & 0x03030303) |
                       (((hmask.x >> hm_shift) & 0x01010101) << 2);
    const uint vals1 = ((    qs.y >> qs_shift) & 0x03030303) |
                       (((hmask.y >> hm_shift) & 0x01010101) << 2);
    const uint vals2 = ((    qs.z >> qs_shift) & 0x03030303) |
                       (((hmask.z >> hm_shift) & 0x01010101) << 2);
    const uint vals3 = ((    qs.w >> qs_shift) & 0x03030303) |
                       (((hmask.w >> hm_shift) & 0x01010101) << 2);

    // Subtract 4 by twiddling bits rather than using re-packing as mesa
    // compiles repacking poorly.
    return i32vec4(int32_t(((vals0 ^ 0x80808080) - 0x04040404) ^ 0x80808080),
                   int32_t(((vals1 ^ 0x80808080) - 0x04040404) ^ 0x80808080),
                   int32_t(((vals2 ^ 0x80808080) - 0x04040404) ^ 0x80808080),
                   int32_t(((vals3 ^ 0x80808080) - 0x04040404) ^ 0x80808080));
}

float get_d_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint is = iqs_k / 4;

    const int8_t scale = int8_t(((data_a[ib_k].scales[is % 8      ] >> (4 * (is / 8))) & 0x0F0F) |
                               (((data_a[ib_k].scales[8 + (is % 4)] >> (2 * (is / 4))) & 0x0303) << 4));
    return float(data_a[ib_k].d) * float(scale - 32);
}

FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t q_sum = 0;

    const i32vec4 qs_a = repack4(ib_a, iqs * 4);
    const float d_scale = get_d_scale(ib_a, iqs * 4);

    q_sum += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]);

    return FLOAT_TYPE(float(cache_b_ds.x) * d_scale * float(q_sum));
}
#endif

#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
// 4-byte loads for Q4_K blocks (144 bytes) and Q5_K blocks (176 bytes)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 16) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 16) / 8) * 4;

#if defined(DATA_A_Q4_K)
    const uint32_t vals0 = (data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x0F0F0F0F;
    const uint32_t vals1 = (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x0F0F0F0F;
    const uint32_t vals2 = (data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x0F0F0F0F;
    const uint32_t vals3 = (data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x0F0F0F0F;

    return i32vec4(vals0, vals1, vals2, vals3);
#else // defined(DATA_A_Q5_K)
    const uint qh_idx = iqs;
    const uint qh_shift = iqs_k / 8;

    return i32vec4(((data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x0F0F0F0F) |
                  (((data_a_packed32[ib_k].qh[qh_idx    ] >> qh_shift) & 0x01010101) << 4),
                   ((data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x0F0F0F0F) |
                  (((data_a_packed32[ib_k].qh[qh_idx + 1] >> qh_shift) & 0x01010101) << 4),
                   ((data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x0F0F0F0F) |
                  (((data_a_packed32[ib_k].qh[qh_idx + 2] >> qh_shift) & 0x01010101) << 4),
                   ((data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x0F0F0F0F) |
                  (((data_a_packed32[ib_k].qh[qh_idx + 3] >> qh_shift) & 0x01010101) << 4));
#endif
}

vec2 get_dm_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint is = iqs_k / 8;

    const uvec3 scales = uvec3(data_a_packed32[ib_k].scales[0],
                               data_a_packed32[ib_k].scales[1],
                               data_a_packed32[ib_k].scales[2]);
    const uint scalesoffs = (is & 3) * 8;

    const uint scidx0 = (is < 4) ? 0 : 2;
    const uint scidxshift0 = scalesoffs;
    const uint scidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
    const uint mbidx0 = (is < 4) ? 1 : 2;
    const uint mbidxshift0 = (is < 4) ? scalesoffs : scalesoffs + 4;
    const uint mbidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;

    const uint8_t sc    = uint8_t(((scales[scidx0] >> scidxshift0) & 0xF) | ((scales[0] >> scidxshift1) & 0x30));
    const uint8_t mbyte = uint8_t(((scales[mbidx0] >> mbidxshift0) & 0xF) | ((scales[1] >> mbidxshift1) & 0x30));
    u8vec2 scale_dm = u8vec2(sc, mbyte);

    return FLOAT_TYPEV2(data_a_packed32[ib_k].dm) * FLOAT_TYPEV2(scale_dm);
}

#define MMVQ_UNPACK_A
#define MMVQ_UNPACK_A_MIN
// one 32-value sub-block (one q8_1 block): its 8 packed int8 words, and (d * scale, dmin * min), so the driver
// unpacks it once for every column of a multi-column (speculative verify) batch
void mmvq_unpack_a(const uint ib_a, out int32_t a_qs[8], out vec2 a_dm) {
    const uint ib_k = ib_a / 8;
    const uint is = ib_a % 8;
    const uint qs_base = (is / 2) * 8;
    const uint qs_shift = (is % 2) * 4;

    [[unroll]] for (uint w = 0; w < 8; ++w) {
#if defined(DATA_A_Q4_K)
        a_qs[w] = int32_t((data_a_packed32[ib_k].qs[qs_base + w] >> qs_shift) & 0x0F0F0F0F);
#else
        a_qs[w] = int32_t(((data_a_packed32[ib_k].qs[qs_base + w] >> qs_shift) & 0x0F0F0F0F) |
                          (((data_a_packed32[ib_k].qh[w] >> is) & 0x01010101) << 4));
#endif
    }

    a_dm = vec2(get_dm_scale(ib_a, 0));
}

FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t q_sum = 0;

    const i32vec4 qs_a = repack4(ib_a, iqs * 4);
    const vec2 dm_scale = get_dm_scale(ib_a, iqs * 4);

    q_sum += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]);

    return FLOAT_TYPE(float(cache_b_ds.x) * float(dm_scale.x) * float(q_sum) - float(dm_scale.y) * float(cache_b_ds.y / 2));
}
#endif

#if defined(DATA_A_Q6_K)
// 2-byte loads for Q6_K blocks (210 bytes)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint ql_idx = (iqs_k / 32) * 16 + iqs_k % 16;
    const uint ql_shift = ((iqs_k % 32) / 16) * 4;

    const uint qh_idx = (iqs_k / 32) * 8 + iqs;
    const uint qh_shift = ((iqs_k % 32) / 8) * 2;

    const uvec4 ql = uvec4( uint32_t(data_a_packed16[ib_k].ql[ql_idx * 2    ]) |
                           (uint32_t(data_a_packed16[ib_k].ql[ql_idx * 2 + 1]) << 16),
                            uint32_t(data_a_packed16[ib_k].ql[ql_idx * 2 + 2]) |
                           (uint32_t(data_a_packed16[ib_k].ql[ql_idx * 2 + 3]) << 16),
                            uint32_t(data_a_packed16[ib_k].ql[ql_idx * 2 + 4]) |
                           (uint32_t(data_a_packed16[ib_k].ql[ql_idx * 2 + 5]) << 16),
                            uint32_t(data_a_packed16[ib_k].ql[ql_idx * 2 + 6]) |
                           (uint32_t(data_a_packed16[ib_k].ql[ql_idx * 2 + 7]) << 16));

    const uvec4 qh = uvec4( uint32_t(data_a_packed16[ib_k].qh[qh_idx * 2    ]) |
                           (uint32_t(data_a_packed16[ib_k].qh[qh_idx * 2 + 1]) << 16),
                            uint32_t(data_a_packed16[ib_k].qh[qh_idx * 2 + 2]) |
                           (uint32_t(data_a_packed16[ib_k].qh[qh_idx * 2 + 3]) << 16),
                            uint32_t(data_a_packed16[ib_k].qh[qh_idx * 2 + 4]) |
                           (uint32_t(data_a_packed16[ib_k].qh[qh_idx * 2 + 5]) << 16),
                            uint32_t(data_a_packed16[ib_k].qh[qh_idx * 2 + 6]) |
                           (uint32_t(data_a_packed16[ib_k].qh[qh_idx * 2 + 7]) << 16));

    const uint vals0 = (( ql.x >> ql_shift) & 0x0F0F0F0F) |
                       (((qh.x >> qh_shift) & 0x03030303) << 4);
    const uint vals1 = (( ql.y >> ql_shift) & 0x0F0F0F0F) |
                       (((qh.y >> qh_shift) & 0x03030303) << 4);
    const uint vals2 = (( ql.z >> ql_shift) & 0x0F0F0F0F) |
                       (((qh.z >> qh_shift) & 0x03030303) << 4);
    const uint vals3 = (( ql.w >> ql_shift) & 0x0F0F0F0F) |
                       (((qh.w >> qh_shift) & 0x03030303) << 4);

    // Subtract 32 by twiddling bits rather than using re-packing as mesa
    // compiles repacking poorly.
    return i32vec4(int32_t(((vals0 ^ 0x80808080) - 0x20202020) ^ 0x80808080),
                   int32_t(((vals1 ^ 0x80808080) - 0x20202020) ^ 0x80808080),
                   int32_t(((vals2 ^ 0x80808080) - 0x20202020) ^ 0x80808080),
                   int32_t(((vals3 ^ 0x80808080) - 0x20202020) ^ 0x80808080));
}

float get_d_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;
    return float(data_a[ib_k].d) * float(data_a[ib_k].scales[iqs_k / 4]);
}

FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t q_sum = 0;

    const i32vec4 qs_a = repack4(ib_a, iqs * 4);
    const float d_scale = get_d_scale(ib_a, iqs * 4);

    q_sum += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]);

    return FLOAT_TYPE(float(cache_b_ds.x) * float(d_scale) * float(q_sum));
}
#endif

#if defined(DATA_A_IQ2_XXS) || defined(DATA_A_IQ2_XS) || defined(DATA_A_IQ2_S) || defined(DATA_A_IQ3_XXS) || defined(DATA_A_IQ3_S)
#define MMVQ_SIGN_TABLE
#if defined(DATA_A_IQ2_XXS) || defined(DATA_A_IQ2_XS) || defined(DATA_A_IQ3_XXS)
// indexed by the 7 stored sign bits, the 8th sign is their parity
#define SIGN_TABLE_SIZE 128
#else
#define SIGN_TABLE_SIZE 256
#endif
// 8 sign bits -> two words with 0xFF in each byte whose value is negated
shared uvec2 sign_masks[SIGN_TABLE_SIZE];

uint spread_signs(const uint signs) {
    const uint s = (signs & 1) | ((signs & 2) << 7) | ((signs & 4) << 14) | ((signs & 8) << 21);
    return s * 0xFF;
}

// filled before init_iq_shmem, whose barrier covers both tables
void init_sign_table() {
    [[unroll]] for (uint i = 0; i < SIGN_TABLE_SIZE; i += gl_WorkGroupSize.x) {
        const uint idx = i + gl_LocalInvocationIndex.x;
        if (SIGN_TABLE_SIZE % gl_WorkGroupSize.x == 0 || idx < SIGN_TABLE_SIZE) {
#if SIGN_TABLE_SIZE == 128
            const uint signs = idx | ((bitCount(idx) & 1) << 7);
#else
            const uint signs = idx;
#endif
            sign_masks[idx] = uvec2(spread_signs(signs), spread_signs(signs >> 4));
        }
    }
}

// apply a sign mask to 4 packed grid bytes. the grid bytes are nonzero (IQ2 8..43, IQ3_XXS 4..62,
// IQ3_S 1..15), so each negated byte (v ^ 0xFF) + 1 stays below 256 and the packed add never
// carries into the next byte
int32_t grid_signed(const uint v, const uint mask) {
    return int32_t((v ^ mask) + (mask & 0x01010101));
}
#endif

#if defined(DATA_A_IQ2_XXS)
#define MMVQ_UNPACK_A
// 2-byte loads, IQ2_XXS blocks are 66 bytes. per 32-value sub-block: 4 grid indices, then one
// 32-bit word of four 7-bit sign groups (8th sign = parity) and a 4-bit scale
void mmvq_unpack_a(const uint ib_a, out int32_t a_qs[8], out float a_d) {
    const uint ib_k = ib_a / 8;
    const uint ib32 = ib_a % 8;

    const uint qs_all = pack32(u16vec2(
        data_a_packed16[ib_k].qs[4 * ib32],
        data_a_packed16[ib_k].qs[4 * ib32 + 1]));
    const uint signscale = pack32(u16vec2(
        data_a_packed16[ib_k].qs[4 * ib32 + 2],
        data_a_packed16[ib_k].qs[4 * ib32 + 3]));

    [[unroll]] for (uint g = 0; g < 4; ++g) {
        const uvec2 m = sign_masks[bitfieldExtract(signscale, 7 * int(g), 7)];
        const uvec2 grid = iq2xxs_grid[(qs_all >> (8 * g)) & 0xFF];

        a_qs[2 * g    ] = grid_signed(grid.x, m.x);
        a_qs[2 * g + 1] = grid_signed(grid.y, m.y);
    }

    a_d = float(data_a_packed16[ib_k].d) * 0.25 * (0.5 + (signscale >> 28));
}
#endif

#if defined(DATA_A_IQ2_XS)
#define MMVQ_UNPACK_A
#define MMVQ_UNPACK_A_HALF_SCALES
// IQ2_XS: per 32-value sub-block 4 x (9-bit grid index | 7-bit signs) and one 4-bit scale per
// 16 values, so the scale comes back per half sub-block
void mmvq_unpack_a(const uint ib_a, out int32_t a_qs[8], out vec2 a_d) {
    const uint ib_k = ib_a / 8;
    const uint ib32 = ib_a % 8;

    [[unroll]] for (uint g = 0; g < 4; ++g) {
        const uint qs    = uint(data_a_packed16[ib_k].qs[4 * ib32 + g]);
        const uvec2 m = sign_masks[qs >> 9];
        const uvec2 grid = iq2xs_grid[qs & 511];

        a_qs[2 * g    ] = grid_signed(grid.x, m.x);
        a_qs[2 * g + 1] = grid_signed(grid.y, m.y);
    }

    const float d = float(data_a[ib_k].d);
    const uint scale = uint(data_a[ib_k].scales[ib32]);
    a_d = vec2(d * 0.25 * (0.5 + (scale & 0xF)), d * 0.25 * (0.5 + (scale >> 4)));
}
#endif

#if defined(DATA_A_IQ2_S)
#define MMVQ_UNPACK_A
#define MMVQ_UNPACK_A_HALF_SCALES
// IQ2_S: per 32-value sub-block 4 grid index low bytes, 2 high bits each in qh, 4 sign bytes in
// the second half of qs, and one 4-bit scale per 16 values
void mmvq_unpack_a(const uint ib_a, out int32_t a_qs[8], out vec2 a_d) {
    const uint ib_k = ib_a / 8;
    const uint ib32 = ib_a % 8;

    const uint qs_all = pack32(u16vec2(
        data_a_packed16[ib_k].qs[2 * ib32],
        data_a_packed16[ib_k].qs[2 * ib32 + 1]));
    const uint signs_all = pack32(u16vec2(
        data_a_packed16[ib_k].qs[QUANT_K / 16 + 2 * ib32],
        data_a_packed16[ib_k].qs[QUANT_K / 16 + 2 * ib32 + 1]));
    const uint qh = uint(data_a[ib_k].qh[ib32]);

    [[unroll]] for (uint g = 0; g < 4; ++g) {
        const uint idx   = ((qs_all >> (8 * g)) & 0xFF) | (((qh >> (2 * g)) & 3) << 8);
        const uvec2 m = sign_masks[(signs_all >> (8 * g)) & 0xFF];
        const uvec2 grid = iq2s_grid[idx];

        a_qs[2 * g    ] = grid_signed(grid.x, m.x);
        a_qs[2 * g + 1] = grid_signed(grid.y, m.y);
    }

    const float d = float(data_a[ib_k].d);
    const uint scale = uint(data_a[ib_k].scales[ib32]);
    a_d = vec2(d * 0.25 * (0.5 + (scale & 0xF)), d * 0.25 * (0.5 + (scale >> 4)));
}
#endif

#if defined(DATA_A_IQ3_XXS)
#define MMVQ_UNPACK_A
// 2-byte loads, IQ3_XXS blocks are 98 bytes. per 32-value sub-block: 8 grid indices, then one
// 32-bit word of four 7-bit sign groups (8th sign = parity) and a 4-bit scale
void mmvq_unpack_a(const uint ib_a, out int32_t a_qs[8], out float a_d) {
    const uint ib_k = ib_a / 8;
    const uint ib32 = ib_a % 8;

    const uint signscale = pack32(u16vec2(
        data_a_packed16[ib_k].qs[QUANT_K / 8 + 2 * ib32],
        data_a_packed16[ib_k].qs[QUANT_K / 8 + 2 * ib32 + 1]));

    [[unroll]] for (uint g = 0; g < 4; ++g) {
        const uint qs    = uint(data_a_packed16[ib_k].qs[4 * ib32 + g]);
        const uvec2 m = sign_masks[bitfieldExtract(signscale, 7 * int(g), 7)];

        // grid holds 4 values, one per byte
        a_qs[2 * g    ] = grid_signed(iq3xxs_grid[qs & 0xFF], m.x);
        a_qs[2 * g + 1] = grid_signed(iq3xxs_grid[qs >> 8],   m.y);
    }

    a_d = float(data_a_packed16[ib_k].d) * 0.5 * (0.5 + (signscale >> 28));
}
#endif

#if defined(DATA_A_IQ3_S)
#define MMVQ_UNPACK_A
// 2-byte loads, IQ3_S blocks are 110 bytes. same unpacking as the IQ3_S MMQ block_a_to_shmem.
// unpacks one 32-value sub-block to 8 packed int8 words + its scale, so the driver can reuse it
// for every column of a multi-column (speculative verify) batch
void mmvq_unpack_a(const uint ib_a, out int32_t a_qs[8], out float a_d) {
    const uint ib_k = ib_a / 8;
    const uint ib32 = ib_a % 8;

    // high index bit pairs for the 8 grid indices of this sub-block, and its 32 sign bits
    const uint qh_all    = uint(data_a_packed16[ib_k].qh[ib32 / 2]) >> ((ib32 & 1) * 8);
    const uint signs_all = uint(data_a_packed16[ib_k].signs[ib32 * 2]) | (uint(data_a_packed16[ib_k].signs[ib32 * 2 + 1]) << 16);

    [[unroll]] for (uint l = 0; l < 4; ++l) {
        const uint qs    = uint(data_a_packed16[ib_k].qs[ib32 * 4 + l]);
        const uint qh    = qh_all >> (2 * l);
        const uvec2 m = sign_masks[(signs_all >> (8 * l)) & 0xFF];

        // grid holds 4 values of 1..15, one per byte
        a_qs[2 * l    ] = grid_signed(iq3s_grid[( qs       & 0xFF) | ((qh & 1) << 8)], m.x);
        a_qs[2 * l + 1] = grid_signed(iq3s_grid[((qs >> 8) & 0xFF) | ((qh & 2) << 7)], m.y);
    }

    const uint scale = (uint(data_a_packed16[ib_k].scales[ib32 / 4]) >> ((ib32 & 3) * 4)) & 0xF;
    a_d = float(data_a_packed16[ib_k].d) * float(1 + 2 * scale);
}
#endif

#if defined(DATA_A_IQ4_XS)
#define MMVQ_UNPACK_A
void mmvq_unpack_a(const uint ib_a, out int32_t a_qs[8], out float a_d) {
    const uint ib = ib_a / 8;
    const uint ib32 = ib_a % 8;

    [[unroll]] for (uint j = 0; j < 4; ++j) {
        const uint32_t vui = data_a_packed32[ib].qs[4 * ib32 + j];
        const i32vec2 qs_a = iq4nl_to_i8x8(vui);
        a_qs[j    ] = qs_a.x;
        a_qs[j + 4] = qs_a.y;
    }

    const uint sl = (data_a_packed32[ib].scales_l >> (4 * ib32)) & 0xF;
    const uint sh = (data_a_packed32[ib].scales_h >> (2 * ib32)) & 3;
    a_d = float(data_a[ib].d) * float(int(sl | (sh << 4)) - 32);
}

FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    const uint ib = ib_a / 8;
    const uint ib32 = ib_a % 8;

    int32_t q_sum = 0;
    [[unroll]] for (uint j = 0; j < 4; ++j) {
        const uint32_t vui = data_a_packed32[ib].qs[4 * ib32 + j];
        const i32vec2 qs_a = iq4nl_to_i8x8(vui);

        q_sum += dotPacked4x8EXT(qs_a.x, cache_b_qs[j]);
        q_sum += dotPacked4x8EXT(qs_a.y, cache_b_qs[j + 4]);
    }

    const uint sl = (data_a_packed32[ib].scales_l >> (4 * ib32)) & 0xF;
    const uint sh = (data_a_packed32[ib].scales_h >> (2 * ib32)) & 3;
    const float d = float(data_a[ib].d) * float(int(sl | (sh << 4)) - 32);

    return FLOAT_TYPE(float(cache_b_ds.x) * d * float(q_sum));
}
#endif

#if defined(DATA_A_IQ1_S)
void repack8(uint ib, uint iqs, out i32vec4 out0, out i32vec4 out1) {
    const uint ib32 = iqs / 32;

    const uint qh = data_a[ib].qh[ib32];

    const uint qs16_0 = data_a_packed16[ib].qs[(4 * ib32 + 0) / 2];
    const uint qs16_1 = data_a_packed16[ib].qs[(4 * ib32 + 2) / 2];

    const uint qs0 = qs16_0 & 0xFF;
    const uint qs1 = qs16_0 >> 8;
    const uint qs2 = qs16_1 & 0xFF;
    const uint qs3 = qs16_1 >> 8;

    const uint hi0 = bitfieldExtract(qh, 3 * int(0), 3);
    const uint hi1 = bitfieldExtract(qh, 3 * int(1), 3);
    const uint hi2 = bitfieldExtract(qh, 3 * int(2), 3);
    const uint hi3 = bitfieldExtract(qh, 3 * int(3), 3);

    const int32_t grid0 = int32_t(iq1s_grid_gpu[qs0 | (hi0 << 8)]);
    const int32_t grid1 = int32_t(iq1s_grid_gpu[qs1 | (hi1 << 8)]);
    const int32_t grid2 = int32_t(iq1s_grid_gpu[qs2 | (hi2 << 8)]);
    const int32_t grid3 = int32_t(iq1s_grid_gpu[qs3 | (hi3 << 8)]);

    out0 = i32vec4((grid0 >> 0) & 0x0F0F0F0F,
                   (grid0 >> 4) & 0x0F0F0F0F,
                   (grid1 >> 0) & 0x0F0F0F0F,
                   (grid1 >> 4) & 0x0F0F0F0F);
    out1 = i32vec4((grid2 >> 0) & 0x0F0F0F0F,
                   (grid2 >> 4) & 0x0F0F0F0F,
                   (grid3 >> 0) & 0x0F0F0F0F,
                   (grid3 >> 4) & 0x0F0F0F0F);
}

vec2 get_dm(uint ib, uint iqs) {
    const uint ib32 = iqs / 32;

    const uint qh = data_a[ib].qh[ib32];
    const float delta = ((qh & 0x8000) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;

    const float d = float(data_a[ib].d);
    const float dl = d * float(2 * bitfieldExtract(qh, 12, 3) + 1);

    // the -1 cancels out the bias in iq1s_grid_gpu
    return FLOAT_TYPEV2(dl, dl * (delta - 1));
}

FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t q_sum = 0;

    const uint ib_k = ib_a / 8;
    const uint iqs_k = (ib_a % 8) * 32 + iqs * 32;

    i32vec4 qs_a0;
    i32vec4 qs_a1;
    repack8(ib_k, iqs_k, qs_a0, qs_a1);

    const vec2 dm = get_dm(ib_k, iqs_k);

    q_sum += dotPacked4x8EXT(qs_a0.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a0.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a0.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a0.w, cache_b_qs[3]);
    q_sum += dotPacked4x8EXT(qs_a1.x, cache_b_qs[4]);
    q_sum += dotPacked4x8EXT(qs_a1.y, cache_b_qs[5]);
    q_sum += dotPacked4x8EXT(qs_a1.z, cache_b_qs[6]);
    q_sum += dotPacked4x8EXT(qs_a1.w, cache_b_qs[7]);

    return FLOAT_TYPE(float(cache_b_ds.x) * float(dm.x) * float(q_sum) + float(dm.y) * float(cache_b_ds.y));
}
#endif

#if defined(DATA_A_IQ1_M)
FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    const uint ib_k = ib_a / 8;
    const uint iqs_k = (ib_a % 8) * 32 + iqs * 32;

    const uint ib32 = iqs_k / 32;
    const uint ib64 = ib32 / 2;

    const uint16_t[4] scales = data_a[ib_k].scales;
    const u16vec4 s = u16vec4(scales[0], scales[1], scales[2], scales[3]) >> 12;
    const float d = float(unpackHalf2x16(s.x | (s.y << 4) | (s.z << 8) | (s.w << 12)).x);

    const uint qs32 = data_a_packed32[ib_k].qs[ib32];
    const uint qh16 = data_a_packed16[ib_k].qh[ib32];

    float sum = 0;
    const uint sc = data_a[ib_k].scales[ib64];
    [[unroll]] for (int l = 0; l < 4; ++l) {
        const uint ib16 = 2 * ib32 + l / 2;
        const float dl = d * (2 * bitfieldExtract(sc, 3 * int(ib16 & 3), 3) + 1);
        const uint qh = qh16 >> (4 * l);
        const uint qs = (qs32 >> (8 * l)) & 0xFF;
        const float delta = ((qh & 8) != 0) ? -IQ1M_DELTA : IQ1M_DELTA;

        const int32_t grid = int32_t(iq1s_grid_gpu[qs | ((qh & 7) << 8)]);

        int32_t q_sum = 0;
        q_sum += dotPacked4x8EXT((grid >> 0) & 0x0F0F0F0F, cache_b_qs[2 * l + 0]);
        q_sum += dotPacked4x8EXT((grid >> 4) & 0x0F0F0F0F, cache_b_qs[2 * l + 1]);

        int32_t y_sum = 0;
        y_sum += dotPacked4x8EXT(int(0x01010101), cache_b_qs[2 * l + 0]);
        y_sum += dotPacked4x8EXT(int(0x01010101), cache_b_qs[2 * l + 1]);

        // the -1 cancels out the bias in iq1s_grid_gpu
        sum += dl * (q_sum + y_sum * (delta - 1));
    }
    sum *= float(cache_b_ds.x);

    return sum;
}
#endif
