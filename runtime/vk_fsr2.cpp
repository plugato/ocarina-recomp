// LOCAL TEST (Ocarina, scene_upscaler = fsr2): AMD FidelityFX Super Resolution 2.2 in place of the game's upscaler
// (vk_exec_draw.c big_compute_blit). AMD's FSR 2.2 core and Vulkan backend (runtime/third_party/fsr2, MIT,
// unmodified) are compiled here; their shaders come from tools/build_fsr2_shaders.py (fsr2_shaders_gen.h) in place of
// AMD's FidelityFX_SC tables, for the one set of options the context below uses: LDR colour, render-resolution motion
// vectors that are not jittered, depth not inverted. The game does not jitter its camera, so the jitter is zero: FSR 2
// then works as a temporal anti-aliasing with an upscale, more stable than a spatial one but with no added detail.
// AMD's backend converts its resource names with std::wstring_convert, which libc++ hides as deprecated
#define _LIBCPP_ENABLE_CXX26_REMOVED_CODECVT
#define _LIBCPP_ENABLE_CXX26_REMOVED_WSTRING_CONVERT
#define _LIBCPP_ENABLE_CXX17_REMOVED_FEATURES
#include <codecvt>
#include <locale>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

#ifndef _WIN32   // Linux: what AMD's sources take from MSVC/MinGW (__declspec(dllexport), _countof, the array wcscpy_s)
#include <wchar.h>
#define FFX_GCC 1
#ifndef _countof
#define _countof(a) (sizeof(a) / sizeof(*(a)))
#endif
template <size_t N> static inline int wcscpy_s(wchar_t (&dst)[N], const wchar_t *src) {
    wcsncpy(dst, src, N - 1);
    dst[N - 1] = 0;
    return 0;
}
#endif

#include "third_party/fsr2/ffx_assert.cpp"
#include "third_party/fsr2/ffx_fsr2.cpp"
#include "third_party/fsr2/vk/ffx_fsr2_vk.cpp"
#include "third_party/fsr2/fsr2_shaders_gen.h"

extern "C" void rt_log(const char *fmt, ...);

// The blob the backend asks for, from the generated table (vk/shaders/ffx_fsr2_shaders_vk.h).
Fsr2ShaderBlobVK fsr2GetPermutationBlobByIndexVK(FfxFsr2Pass pass, uint32_t options) {
    static const char *const names[] = {"depth_clip", "reconstruct_previous_depth", "lock", "accumulate", "accumulate",
                                        "rcas", "compute_luminance_pyramid", "autogen_reactive", "tcr_autogen"};
    Fsr2ShaderBlobVK b = {};
    if ((unsigned)pass >= sizeof names / sizeof *names) return b;
    int lanczos = (options & FSR2_SHADER_PERMUTATION_REPROJECT_USE_LANCZOS_TYPE) != 0;
    int sharpen = pass == FFX_FSR2_PASS_ACCUMULATE_SHARPEN;
    const uint32_t unsupported = FSR2_SHADER_PERMUTATION_HDR_COLOR_INPUT | FSR2_SHADER_PERMUTATION_JITTER_MOTION_VECTORS |
                                 FSR2_SHADER_PERMUTATION_DEPTH_INVERTED;
    static int told;
    if ((options & unsupported) || !(options & FSR2_SHADER_PERMUTATION_LOW_RES_MOTION_VECTORS))
        if (!told++) rt_log("vk: FSR 2: options %#x asked for, not built (tools/build_fsr2_shaders.py)", options);
    for (const Fsr2GenBlob &g : fsr2_gen_blobs)
        if (!strcmp(g.pass, names[pass]) && g.lanczos == lanczos && g.sharpen == (pass == FFX_FSR2_PASS_ACCUMULATE ? 0 : sharpen)) {
            b.data = (const uint8_t *)g.spv;
            b.size = g.size;
            b.storageImageCount = g.nuav, b.sampledImageCount = g.nsrv, b.uniformBufferCount = g.ncb;
            b.boundStorageImageNames = g.uav_n, b.boundStorageImageBindings = g.uav_b;
            b.boundSampledImageNames = g.srv_n, b.boundSampledImageBindings = g.srv_b;
            b.boundUniformBufferNames = g.cb_n, b.boundUniformBufferBindings = g.cb_b;
            return b;
        }
    return b;
}

static struct {
    int tried, ok;
    FfxFsr2Context ctx;
    void *scratch;
    uint32_t rw, rh, ow, oh;
} S;

static void ffx_message(FfxFsr2MsgType type, const wchar_t *msg) {
    static int n;
    if (n++ < 16) rt_log("vk: FSR 2 %s: %ls", type == FFX_FSR2_MESSAGE_TYPE_ERROR ? "error" : "warning", msg);
}

extern "C" int vk_fsr2_upscale(VkPhysicalDevice phys, VkDevice dev, VkCommandBuffer cb, VkImage color, VkImageView color_view,
                               VkImage depth, VkImageView depth_view, VkImage mv, VkImageView mv_view, uint32_t rw, uint32_t rh,
                               VkImage out, VkImageView out_view, VkFormat out_fmt, uint32_t ow, uint32_t oh, float near_, float far_,
                               float fovy, int reset, float dt_ms, float sharpness) {
    if (S.tried && !S.ok) return 0;
    if (S.ok && (S.rw != rw || S.rh != rh || S.ow != ow || S.oh != oh)) {   // a size change: a new context
        vkDeviceWaitIdle(dev);   // ponytail: the caller's queue lock is not held; sizes change rarely
        ffxFsr2ContextDestroy(&S.ctx);
        S.ok = 0, S.tried = 0;
    }
    if (!S.tried) {
        S.tried = 1;
        size_t n = ffxFsr2GetScratchMemorySizeVK(phys);
        if (!S.scratch) S.scratch = malloc(n);
        FfxFsr2ContextDescription d = {};
        d.flags = FFX_FSR2_ENABLE_AUTO_EXPOSURE;
        d.maxRenderSize = {rw, rh};
        d.displaySize = {ow, oh};
        d.device = ffxGetDeviceVK(dev);
        d.fpMessage = ffx_message;
        if (!S.scratch || ffxFsr2GetInterfaceVK(&d.callbacks, S.scratch, n, phys, vkGetDeviceProcAddr) != FFX_OK ||
            ffxFsr2ContextCreate(&S.ctx, &d) != FFX_OK) {
            rt_log("vk: FSR 2 off: its context could not be created");
            return 0;
        }
        S.ok = 1, S.rw = rw, S.rh = rh, S.ow = ow, S.oh = oh;
        rt_log("vk: FSR 2.2 upscales the scene %ux%u to %ux%u (no jitter: temporal anti-aliasing with an upscale)", rw, rh, ow, oh);
    }
    FfxFsr2DispatchDescription dd = {};
    dd.commandList = ffxGetCommandListVK(cb);
    dd.color = ffxGetTextureResourceVK(&S.ctx, color, color_view, rw, rh, VK_FORMAT_R16G16B16A16_SFLOAT, L"color",
                                       FFX_RESOURCE_STATE_COMPUTE_READ);
    dd.depth = ffxGetTextureResourceVK(&S.ctx, depth, depth_view, rw, rh, VK_FORMAT_R32_SFLOAT, L"depth", FFX_RESOURCE_STATE_COMPUTE_READ);
    dd.motionVectors = ffxGetTextureResourceVK(&S.ctx, mv, mv_view, rw, rh, VK_FORMAT_R16G16_SFLOAT, L"mv", FFX_RESOURCE_STATE_COMPUTE_READ);
    dd.output = ffxGetTextureResourceVK(&S.ctx, out, out_view, ow, oh, out_fmt, L"out", FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    dd.jitterOffset = {0, 0};
    dd.motionVectorScale = {(float)rw, (float)rh};   // the vectors are in UV
    dd.renderSize = {rw, rh};
    dd.enableSharpening = sharpness > 0;
    dd.sharpness = sharpness;
    dd.frameTimeDelta = dt_ms;
    dd.preExposure = 1.0f;
    dd.reset = reset != 0;
    dd.cameraNear = near_;
    dd.cameraFar = far_;
    dd.cameraFovAngleVertical = fovy;
    dd.viewSpaceToMetersFactor = 1.0f;
    FfxErrorCode e = ffxFsr2ContextDispatch(&S.ctx, &dd);
    if (e != FFX_OK) {
        static int n;
        if (n++ < 4) rt_log("vk: FSR 2 dispatch failed (%d)", (int)e);
        return 0;
    }
    return 1;
}
