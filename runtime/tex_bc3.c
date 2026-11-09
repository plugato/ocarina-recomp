// BC3 (DXT5) from RGBA8 on the CPU, for ASTC decoded by tex_astc.c when the device has no ASTC and the GPU decoder is
// not used (an integrated GPU): the image is BC3 as on the GPU path (vk_astc.c bc3_encode), a quarter of RGBA8's
// memory. A 1365x1365x90 ASTC 8x8 array took 894 MB as RGBA8 and did not fit a 1.75 GiB Vega 11 (staging failed).
// ponytail: a bounding-box encoder (colour endpoints from the block's min and max, inset, one projection; alpha min and
// max, 8 levels), like stb_dxt's fast mode; a PCA/refining encoder if its quality shows.
#include <string.h>

#include "tex.h"

static uint16_t to565(int r, int g, int b) {
    return (uint16_t)(((r * 31 + 127) / 255) << 11 | ((g * 63 + 127) / 255) << 5 | ((b * 31 + 127) / 255));
}
static void from565(uint16_t c, int o[3]) {
    int r = c >> 11, g = (c >> 5) & 63, b = c & 31;
    o[0] = (r << 3) | (r >> 2), o[1] = (g << 2) | (g >> 4), o[2] = (b << 3) | (b >> 2);
}

// One 4x4 block, px: 16 RGBA texels in row order; out: 16 bytes.
void bc3_encode_block(const uint8_t px[64], uint8_t out[16]) {
    int amin = 255, amax = 0, mn[3] = {255, 255, 255}, mx[3] = {0, 0, 0};
    for (int i = 0; i < 16; i++) {
        const uint8_t *p = px + 4 * i;
        for (int c = 0; c < 3; c++) {
            if (p[c] < mn[c]) mn[c] = p[c];
            if (p[c] > mx[c]) mx[c] = p[c];
        }
        if (p[3] < amin) amin = p[3];
        if (p[3] > amax) amax = p[3];
    }
    // alpha: a0 = max > a1 = min, eight levels; index 0 is a0, 1 is a1, 2..7 step from a0 towards a1
    memset(out, 0, 16);
    out[0] = (uint8_t)amax, out[1] = (uint8_t)amin;
    if (amax > amin) {
        uint64_t bits = 0;
        int range = amax - amin;
        for (int i = 0; i < 16; i++) {
            int t = ((amax - px[4 * i + 3]) * 7 + range / 2) / range;   // 0 at a0 .. 7 at a1
            int idx = t == 0 ? 0 : t == 7 ? 1 : t + 1;
            bits |= (uint64_t)idx << (3 * i);
        }
        for (int i = 0; i < 6; i++) out[2 + i] = (uint8_t)(bits >> (8 * i));
    }
    // colour: endpoints inset by 1/16 of the range, c0 > c1 (four-colour mode); index 0 c0, 1 c1, 2 2/3 c0, 3 1/3 c0
    for (int c = 0; c < 3; c++) {
        int in = (mx[c] - mn[c]) >> 4;
        mx[c] -= in, mn[c] += in;
    }
    uint16_t c0 = to565(mx[0], mx[1], mx[2]), c1 = to565(mn[0], mn[1], mn[2]);
    uint32_t idx = 0;
    if (c0 < c1) { uint16_t t = c0; c0 = c1; c1 = t; }
    if (c0 != c1) {
        int e0[3], e1[3];
        from565(c0, e0), from565(c1, e1);
        int d[3] = {e0[0] - e1[0], e0[1] - e1[1], e0[2] - e1[2]};
        int dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        static const uint8_t code[4] = {1, 3, 2, 0};   // t in thirds from c1 to c0 -> index
        for (int i = 0; i < 16; i++) {
            const uint8_t *p = px + 4 * i;
            int dot = (p[0] - e1[0]) * d[0] + (p[1] - e1[1]) * d[1] + (p[2] - e1[2]) * d[2];
            int t = dd ? (dot * 3 + dd / 2) / dd : 0;
            t = t < 0 ? 0 : t > 3 ? 3 : t;
            idx |= (uint32_t)code[t] << (2 * i);
        }
    }
    out[8] = (uint8_t)c0, out[9] = (uint8_t)(c0 >> 8), out[10] = (uint8_t)c1, out[11] = (uint8_t)(c1 >> 8);
    memcpy(out + 12, &idx, 4);
}

// Block rows [j0, j1) of a width x height RGBA8 image to BC3 blocks, row-major (out: the image's first block); edge
// blocks repeat the last texel.
void bc3_encode_rows(const uint8_t *rgba, uint32_t width, uint32_t height, uint32_t j0, uint32_t j1, uint8_t *out) {
    uint8_t px[64];
    out += (size_t)j0 * ((width + 3) / 4) * 16;
    for (uint32_t by = j0; by < j1; by++)
        for (uint32_t bx = 0; bx < (width + 3) / 4; bx++) {
            for (int y = 0; y < 4; y++) {
                uint32_t sy = by * 4 + (uint32_t)y < height ? by * 4 + (uint32_t)y : height - 1;
                for (int x = 0; x < 4; x++) {
                    uint32_t sx = bx * 4 + (uint32_t)x < width ? bx * 4 + (uint32_t)x : width - 1;
                    memcpy(px + 4 * (4 * y + x), rgba + 4 * ((size_t)sy * width + sx), 4);
                }
            }
            bc3_encode_block(px, out), out += 16;
        }
}

void bc3_encode(const uint8_t *rgba, uint32_t width, uint32_t height, uint8_t *out) {
    bc3_encode_rows(rgba, width, height, 0, (height + 3) / 4, out);
}
