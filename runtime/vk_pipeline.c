// Graphics and compute pipelines and descriptor layouts of the Vulkan replay. The op-stream replay (vk_exec.c)
// collects NVN state into a VkPipeState and asks for the pipeline of each draw; this file maps that state to Vulkan,
// numbers the descriptor bindings and keeps the caches.
//
// Binding model. The translator (tools/xlate/Program.cs, Ryujinx 1.3.2's SPIR-V back end) decorates every resource of a
// shader with DescriptorSet 0 and numbers the bindings of each shader from 1 in the order it meets them (Accessor
// Create*Binding); binding 0 is Ryujinx's support buffer (ResourceManager: SupportBuffer.Binding, a std140 uniform
// buffer every shader declares). The numbering is per shader, so the vertex and fragment stages of one program collide.
// The scheme here:
//   - one descriptor set, set 0, per pipeline; no push constants (Ryujinx's SPIR-V uses none);
//   - binding 0: the support buffer, one uniform buffer shared by all stages of the pipeline;
//   - vertex stage: its own bindings 1..n unchanged; fragment stage: its bindings b >= 1 move to b + n (the SPIR-V
//     Binding decorations are rewritten when the module is created); compute: its own bindings;
//   - descriptor types from the reflection: constant buffers UNIFORM_BUFFER, storage buffers STORAGE_BUFFER, textures
//     COMBINED_IMAGE_SAMPLER (UNIFORM_TEXEL_BUFFER for buffer textures, SAMPLED_IMAGE for separate ones), images
//     STORAGE_IMAGE (STORAGE_TEXEL_BUFFER for buffer images), array lengths as descriptor counts;
//   - set 0 is a push-descriptor set (VK_KHR_push_descriptor) when its descriptors fit maxPushDescriptors (32 on the RX
//     9070; a pipeline layout may hold one push set only, which rules out a set per stage); larger ones (a vertex and a
//     fragment stage near the census maxima of 15 and 27) are ordinary sets the replay allocates (VkPipeInfo.push 0).
// A fixed layout per stage combination (Ryujinx's own back end: a binding per constant buffer, storage buffer, texture
// and image slot per stage) would hold about 150 descriptors and could never be pushed, so layouts are made per distinct
// binding list and shared by every pipeline with that list; VkPipeInfo.bindings tells the replay which NVN slot feeds
// each binding.
//
// Fragment inputs the vertex stage does not write (stages are translated separately; MNG nvn_mtl.m msl_varyings) get
// outputs added to the vertex module, as Vulkan rejects such pairs, initialised to (0, 0, 0, 1) as Maxwell reads an
// unwritten output (docs/maxwell.md 1.2). The unwritten components of an output the vertex stage does write
// start at (0, 0, 0, 1) in the translator (tools/xlate/patches/0006).
//
// Pipeline key. Everything Vulkan 1.3 can take as dynamic state is dynamic: viewports and scissors with count, depth
// bias, blend constants, stencil masks and references, cull mode, front face, topology (within its class), depth and
// stencil tests and ops, depth bias enable, rasterizer discard; vertex input (VK_EXT_vertex_input_dynamic_state, else
// binding strides only); and, where VK_EXT_extended_dynamic_state3 offers them, depth clamp, polygon mode, blend enables,
// equations, write masks and alpha to coverage. The key holds the rest: shaders, target formats, sample count, topology
// class, depth clip mode, logic op and what is not dynamic on the GPU. The VkPipelineCache persists to
// pipeline-cache.bin next to totk.exe (TOTK_PIPELINE_CACHE overrides the path).
//
// Provenance: the NVN -> Vulkan enum mappings (blend factors and equations, compare functions, stencil ops), the vertex
// format table with its 3-component fallback and the pipeline cache file handling follow LAS Recomp
// runtime/nvn/vulkan/exec.c and device.c @ da35b83a (MIT, Copyright (c) 2026 Peiikai); the pipeline state the key holds
// (program, target formats, blend and write masks, vertex layout, sample count) and the varying fix-up follow
// MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m pso_key_fill / pso_desc / msl_varyings and its vfmt
// table @ b75be8cf (MIT, Copyright (c) 2026 jhaizhou); the binding conventions are those of Ryujinx's shader translator
// (MIT, tools/vendor/ryujinx).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_hmap.h"
#include "platform/platform.h"
#include "tex.h"

#define LOG_ONCE(...) do { static volatile int32_t once_; if (!plat_xchg32(&once_, 1)) rt_log(__VA_ARGS__); } while (0)

enum { SPV_OP_ENTRY_POINT = 15, SPV_OP_TYPE_INT = 21, SPV_OP_TYPE_FLOAT = 22, SPV_OP_TYPE_VECTOR = 23,
       SPV_OP_TYPE_POINTER = 32, SPV_OP_CONSTANT = 43, SPV_OP_CONSTANT_COMPOSITE = 44,
       SPV_OP_CONSTANT_NULL = 46, SPV_OP_FUNCTION = 54, SPV_OP_VARIABLE = 59,
       SPV_OP_DECORATE = 71, SPV_DEC_BUILTIN = 11, SPV_DEC_LOCATION = 30, SPV_DEC_INDEX = 32, SPV_DEC_BINDING = 33, SPV_DEC_DESCRIPTOR_SET = 34,
       SPV_SC_OUTPUT = 3 };

#define MAX_BINDINGS 256   // LOCAL TEST (Ocarina): was 128; guarded vertex and fragment stages together passed it and the draws were lost
#define PIPE_BUCKETS 4096
#define LAYOUT_BUCKETS 256

// ---- device capabilities and the pipeline cache (set up on first use, under P.lock)

static struct {
    PlatMutex lock;
    volatile int32_t ready;
    VkDevice device;
    VkPipelineCache cache;
    uint32_t max_push;
    int indep;                         // stage-independent layouts (vk_caps.gpl): vertex set 0, fragment set 1
    int push_stage;                    // GFX_STAGE_* whose set is pushed with independent layouts
    int gpl;                           // draws whose pipeline is building get a fast-linked one (vk_caps.gpl)
    struct { uint64_t links, wlinks, dlinks, requeued, link_fail, defer, lib_n[4], lib_fail, swaps, slow, pre_used[4]; double link_ms, link_max, lib_ms[4], lib_max[4], dlink_ms, dlink_max; } g;
    int vs_outs_all;                   // TOTK_VS_OUTS: 0 union of paired fragment inputs, 1 (all) all 32 output locations
    int dyn_vi, dyn_clamp, dyn_poly, dyn_blend_en, dyn_blend_eq, dyn_wmask, dyn_a2c;
    int f_logic_op, f_depth_clamp, f_fill_non_solid, f_dual_src;
    int f_geometry, f_vertex_stores, f_gs_point_size, f_clip, f_cull;
    int pv_last;                       // provokingVertexLast enabled: NVN's default is the last vertex
    VkShaderStageFlags sg32_stages;    // stages that accept requiredSubgroupSize = 32 (0: not available)
    VkShaderStageFlags subgroup_stages;
    VkSubgroupFeatureFlags subgroup_operations;
    VkDynamicState dyn[32];
    uint32_t n_dyn;
    PFN_vkCmdPushDescriptorSetKHR push;
    int tpl;                           // descriptor update templates on (make_tpl)
    PFN_vkCreateDescriptorUpdateTemplate tpl_create;
    PFN_vkDestroyDescriptorUpdateTemplate tpl_destroy;
    PFN_vkUpdateDescriptorSetWithTemplate tpl_update;
    PFN_vkCmdPushDescriptorSetWithTemplateKHR tpl_push;
    PFN_vkCmdSetVertexInputEXT set_vi;
    PFN_vkCmdSetDepthClampEnableEXT set_clamp;
    PFN_vkCmdSetPolygonModeEXT set_poly;
    PFN_vkCmdSetColorBlendEnableEXT set_blend_en;
    PFN_vkCmdSetColorBlendEquationEXT set_blend_eq;
    PFN_vkCmdSetColorWriteMaskEXT set_wmask;
    PFN_vkCmdSetAlphaToCoverageEnableEXT set_a2c;
    char cache_path[PLAT_PATH_MAX * 2];
    VkPipeStats stats;
} P = {.lock = PLAT_MUTEX_INIT};

typedef struct Layout {
    struct Layout *next;
    uint64_t hash;
    VkDescriptorSetLayout set;
    VkPipelineLayout layout;
    int push;
    uint32_t n;
    VkDescriptorSetLayout set1;        // independent layouts: the fragment set (set 1); VK_NULL_HANDLE otherwise
    int push1;                         // set 1 is a push-descriptor set
    uint32_t n0;                       // independent layouts: b[0..n0) are set 0, the rest set 1
    VkPipelineLayout half[2];          // GPL libraries: {set 0, null}, {null, set 1} (made on first use)
    VkDescriptorUpdateTemplate tpl[2]; // per set (make_tpl); VK_NULL_HANDLE: written with write structs
    VkDescriptorSetLayoutBinding b[];
} Layout;

typedef struct Pipe {
    struct Pipe *next;
    uint64_t hash;
    VkPipeInfo info;
    VkPipeBinding *bindings;
    int building;                      // a background job owns info.pipeline (under P.lock)
    double one_shot_deadline_ms;       // bound the wait across retries while that job is still building
    struct Job *job;                   // that job while building
    const struct Layout *lay;          // graphics: the layout it was entered with
    VkPipeline fast;                   // GPL: the fast-linked pipeline (kept until shutdown: recorded draws use it)
    uint8_t fast_tried;
    uint8_t waited_out;                // a draw's wait for it ran out while it built: later draws skip at once (wait_once)
    size_t key_size;
    uint8_t key[];
} Pipe;

static Layout *layouts[LAYOUT_BUCKETS];
static Pipe *pipes[PIPE_BUCKETS];

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}
#define FNV0 0xcbf29ce484222325ull

// Hash of a pipeline table key, 8 bytes per step: a draw hashes its whole GKey, and byte-wise FNV over it cost more
// than the lookup. Only the in-memory table uses it; the key file keeps fnv.
static uint64_t key_hash(const void *k, size_t n) { return nvn_hm_hash(k, (uint32_t)n); }

// The cache file lives next to totk.exe, so every build dir (the main build and each private fork) has its own:
// forks that shared build\v121's cache broke each other's runs (2026-10-22).
static PlatMutex cache_access = PLAT_MUTEX_INIT;
// When both locks are needed, take cache_access before deferred_access.
static PlatMutex deferred_access = PLAT_MUTEX_INIT;
static VkPipelineCache deferred_cache;
static uint32_t deferred_count;
static uint64_t cache_updates;   // private cache results can arrive after the pipeline was counted

static VkPipelineCache cache_private(void) {
    VkPipelineCache own = VK_NULL_HANDLE;
    if (!P.cache) return own;
    VkPipelineCacheCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    if (vkCreatePipelineCache(P.device, &ci, NULL, &own) != VK_SUCCESS) return VK_NULL_HANDLE;
    // Seed the private build from the cache loaded at startup, including
    // compute pipelines saved by a previous process.
    plat_mutex_lock_shared(&cache_access);
    if (P.cache) vkMergePipelineCaches(P.device, own, 1, &P.cache);
    plat_mutex_unlock_shared(&cache_access);
    return own;
}

static void cache_retry_deferred(void) {
    if (!plat_mutex_trylock(&cache_access)) return;
    if (!plat_mutex_trylock(&deferred_access)) {
        plat_mutex_unlock(&cache_access);
        return;
    }
    if (deferred_cache && P.cache && vkMergePipelineCaches(P.device, P.cache, 1, &deferred_cache) == VK_SUCCESS) {
        vkDestroyPipelineCache(P.device, deferred_cache, NULL);
        deferred_cache = VK_NULL_HANDLE;
        rt_log("vk_pipeline: merged %u deferred pipeline caches", deferred_count);
        deferred_count = 0;
        __atomic_add_fetch(&cache_updates, 1, __ATOMIC_RELAXED);
    }
    plat_mutex_unlock(&deferred_access);
    plat_mutex_unlock(&cache_access);
}

// Takes ownership of a private cache once its pipeline has finished building.
static void cache_publish(VkPipelineCache source) {
    // A queued writer can block replay behind a slow driver compile. Keep the
    // compiled pipeline even if the cache stays busy (VUID 10202).
    const uint64_t deadline = plat_now_ns() + 50000000;
    do {
        if (plat_mutex_trylock(&cache_access)) {
            VkResult r = P.cache ? vkMergePipelineCaches(P.device, P.cache, 1, &source) : VK_ERROR_INITIALIZATION_FAILED;
            plat_mutex_unlock(&cache_access);
            if (r == VK_SUCCESS) {
                vkDestroyPipelineCache(P.device, source, NULL);
                __atomic_add_fetch(&cache_updates, 1, __ATOMIC_RELAXED);
                return;
            }
            rt_log("vk_pipeline: shared cache merge failed (%d), retaining private cache", (int)r);
            break;
        }
        plat_sleep_ms(5);
    } while (plat_now_ns() < deadline);

    plat_mutex_lock(&deferred_access);
    // Aggregate delayed results in one cache instead of retaining a seeded
    // copy of the shared cache for every pipeline that finishes.
    if (!deferred_cache) {
        deferred_cache = source;
        source = VK_NULL_HANDLE;
    } else if (vkMergePipelineCaches(P.device, deferred_cache, 1, &source) != VK_SUCCESS) {
        rt_log("vk_pipeline: could not retain a private pipeline cache");
        vkDestroyPipelineCache(P.device, source, NULL);
        plat_mutex_unlock(&deferred_access);
        return;
    }
    rt_log("vk_pipeline: private cache merge deferred (%u pending)", ++deferred_count);
    __atomic_add_fetch(&cache_updates, 1, __ATOMIC_RELAXED);
    plat_mutex_unlock(&deferred_access);
    if (source) vkDestroyPipelineCache(P.device, source, NULL);
}

static void cache_path_init(void) {
    const char *cfg_path = cfg_get("TOTK_PIPELINE_CACHE");
    if (cfg_path && *cfg_path && strlen(cfg_path) < sizeof P.cache_path) {
        strcpy(P.cache_path, cfg_path);
        return;
    }
    P.cache_path[0] = 0;
    char exe[PLAT_PATH_MAX * 2];
    int n = plat_exe_path(exe, sizeof exe);
    if (!n || n >= sizeof exe) return;
    char *cut = plat_last_sep(exe);
    if (!cut) return;
    *cut = 0;
    snprintf(P.cache_path, sizeof P.cache_path, "%s/pipeline-cache.bin", exe);
}

// Cache data whose header names another device or driver is dropped before Vulkan sees it.
static void cache_load(VkCtx *c) {
    cache_path_init();
    void *data = NULL;
    size_t size = 0;
    FILE *f = P.cache_path[0] ? fopen(P.cache_path, "rb") : NULL;
    if (f) {
        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (len > (long)sizeof(VkPipelineCacheHeaderVersionOne) && (data = malloc((size_t)len)) &&
            fread(data, 1, (size_t)len, f) == (size_t)len)
            size = (size_t)len;
        fclose(f);
        VkPipelineCacheHeaderVersionOne h;
        if (size) memcpy(&h, data, sizeof h);
        if (size && (h.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE || h.vendorID != c->props.vendorID ||
                     h.deviceID != c->props.deviceID || memcmp(h.pipelineCacheUUID, c->props.pipelineCacheUUID, VK_UUID_SIZE))) {
            rt_log("vk_pipeline: %s was written by another device or driver; starting empty", P.cache_path);
            size = 0;
        }
    }
    VkPipelineCacheCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, .initialDataSize = size,
                                    .pInitialData = size ? data : NULL};
    if (vkCreatePipelineCache(c->device, &ci, NULL, &P.cache) != VK_SUCCESS) {
        ci.initialDataSize = 0, ci.pInitialData = NULL;
        if (vkCreatePipelineCache(c->device, &ci, NULL, &P.cache) != VK_SUCCESS) P.cache = VK_NULL_HANDLE;
    }
    free(data);
}

static void keys_open(void);
static void prebuild_start(void);

static int ensure_init(void) {
    if (P.ready) return 1;
    if (!vk_init()) return 0;
    plat_mutex_lock(&P.lock);
    if (P.ready) {
        plat_mutex_unlock(&P.lock);
        return 1;
    }
    VkCtx *c = vk_ctx();
    P.device = c->device;
    VkPhysicalDevicePushDescriptorPropertiesKHR pd = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &pd};
    vkGetPhysicalDeviceProperties2(c->phys, &p2);
    P.max_push = pd.maxPushDescriptors;
    P.indep = vk_caps.gpl;
    P.gpl = vk_caps.gpl;
    rt_log(P.gpl ? "vk_pipeline: graphics pipeline library on: a draw whose pipeline is building gets a fast-linked one, "
                   "the optimised build replaces it when done"
                 : "vk_pipeline: graphics pipeline library off: draws wait for or skip pipelines that are building");
    const char *ps = cfg_get("TOTK_PUSH_STAGE"), *vo = cfg_get("TOTK_VS_OUTS");
    P.push_stage = ps && !strcmp(ps, "fs") ? GFX_STAGE_FS : GFX_STAGE_VS;
    P.vs_outs_all = vo && !strcmp(vo, "all");
    if (P.indep) rt_log("vk_pipeline: stage-independent layouts: vertex set 0, fragment set 1, %s set pushed; vertex outputs: %s",
                        P.push_stage == GFX_STAGE_VS ? "vertex" : "fragment", P.vs_outs_all ? "all 32 locations" : "union of paired fragment inputs");
    // vk_device.c enables every feature the GPU reports in this chain, so the query tells what is enabled.
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT e3 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
    VkPhysicalDeviceProvokingVertexFeaturesEXT pvf = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT,
                                                      .pNext = c->has_eds3 ? &e3 : NULL};
    VkPhysicalDeviceVulkan13Features v13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                            .pNext = c->has_provoking_vertex ? (void *)&pvf : (c->has_eds3 ? (void *)&e3 : NULL)};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v13};
    vkGetPhysicalDeviceFeatures2(c->phys, &f2);
    VkPhysicalDeviceSubgroupProperties subgroup = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 subgroup_properties = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &subgroup};
    vkGetPhysicalDeviceProperties2(c->phys, &subgroup_properties);
    P.subgroup_stages = subgroup.supportedStages;
    P.subgroup_operations = subgroup.supportedOperations;
    P.pv_last = c->has_provoking_vertex && pvf.provokingVertexLast;
    if (!P.pv_last) rt_log("vk_pipeline: provokingVertexLast unavailable: flat outputs come from the first vertex (NVN: last)");
    // Maxwell warps have 32 lanes; the translator assumes it (maxwell.md 3.4). Pin 32 where the driver allows it.
    if (v13.subgroupSizeControl) {
        VkPhysicalDeviceVulkan13Properties p13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
        VkPhysicalDeviceProperties2 q2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &p13};
        vkGetPhysicalDeviceProperties2(c->phys, &q2);
        P.sg32_stages = vk_caps.sg_pin_stages;   // vk_caps.c: 32 unless TOTK_VK_SUBGROUP; 0 without the pin
        rt_log("vk_pipeline: subgroup size %u-%u, requiredSubgroupSizeStages %#x, pinning %u on %#x", p13.minSubgroupSize,
               p13.maxSubgroupSize, (unsigned)p13.requiredSubgroupSizeStages, vk_caps.sg_size, (unsigned)P.sg32_stages);
    }
    P.f_logic_op = f2.features.logicOp;
    P.f_depth_clamp = f2.features.depthClamp;
    P.f_fill_non_solid = f2.features.fillModeNonSolid;
    P.f_dual_src = f2.features.dualSrcBlend;
    P.f_geometry = f2.features.geometryShader;
    P.f_vertex_stores = f2.features.vertexPipelineStoresAndAtomics;
    P.f_gs_point_size = f2.features.shaderTessellationAndGeometryPointSize;
    P.f_clip = f2.features.shaderClipDistance;
    P.f_cull = f2.features.shaderCullDistance;
#define PROC(field, name) (P.field = (void *)vkGetDeviceProcAddr(c->device, name))
    PROC(push, "vkCmdPushDescriptorSetKHR");
    // Descriptor update templates (core 1.1; the push variant comes with VK_KHR_push_descriptor on 1.1): the driver
    // walks a layout prepared when the layout is made instead of parsing write structs every draw (DXVK does so for the
    // same reason). TOTK_DESC_TEMPLATE=0 keeps the write structs.
    if (!cfg_bool("TOTK_DESC_TEMPLATE")) {
        rt_log("vk_pipeline: descriptor update templates off (TOTK_DESC_TEMPLATE=0)");
    } else if (c->props.apiVersion >= VK_API_VERSION_1_1 && PROC(tpl_create, "vkCreateDescriptorUpdateTemplate") &&
               PROC(tpl_destroy, "vkDestroyDescriptorUpdateTemplate") && PROC(tpl_update, "vkUpdateDescriptorSetWithTemplate")) {
        P.tpl = 1;
        if (!PROC(tpl_push, "vkCmdPushDescriptorSetWithTemplateKHR"))
            rt_log("vk_pipeline: vkCmdPushDescriptorSetWithTemplateKHR missing: push sets are written with write structs");
    } else {
        rt_log("vk_pipeline: descriptor update templates unavailable: sets are written with write structs");
    }
    // TOTK_VK_DYN_VERTEX=0/1 forces dynamic vertex input off or on. Default off on Intel's frozen Windows driver for its
    // older integrated GPUs (UHD 630, Ice Lake: Vulkan 1.3.215 and older), which draws nothing with it; LAS Recomp
    // turns it off there for the same reason.
    const char *dv = cfg_get("TOTK_VK_DYN_VERTEX");   // no fixed default: it depends on the GPU, see above
    int intel_legacy = c->props.vendorID == 0x8086 && c->props.apiVersion <= VK_MAKE_API_VERSION(0, 1, 3, 215);
    int dv_on = dv && *dv ? strcmp(dv, "0") != 0 : !intel_legacy;
    P.dyn_vi = dv_on && c->has_vertex_input_dynamic_state && PROC(set_vi, "vkCmdSetVertexInputEXT");
    if (!dv_on && c->has_vertex_input_dynamic_state)
        rt_log("vk_pipeline: dynamic vertex input off (%s)", dv && *dv ? "TOTK_VK_DYN_VERTEX=0" : "Intel's older Windows driver draws nothing with it");
    if (c->has_eds3) {
        P.dyn_clamp = e3.extendedDynamicState3DepthClampEnable && P.f_depth_clamp && PROC(set_clamp, "vkCmdSetDepthClampEnableEXT");
        P.dyn_poly = e3.extendedDynamicState3PolygonMode && PROC(set_poly, "vkCmdSetPolygonModeEXT");
        P.dyn_blend_en = e3.extendedDynamicState3ColorBlendEnable && PROC(set_blend_en, "vkCmdSetColorBlendEnableEXT");
        P.dyn_blend_eq = e3.extendedDynamicState3ColorBlendEquation && PROC(set_blend_eq, "vkCmdSetColorBlendEquationEXT");
        P.dyn_wmask = e3.extendedDynamicState3ColorWriteMask && PROC(set_wmask, "vkCmdSetColorWriteMaskEXT");
        P.dyn_a2c = e3.extendedDynamicState3AlphaToCoverageEnable && PROC(set_a2c, "vkCmdSetAlphaToCoverageEnableEXT");
    }
    // TOTK_VK_DYN_A2C=0/1 forces dynamic alpha to coverage off or on. Default off on AMD's Windows driver branch for
    // RDNA 2 and older (vk_caps.amd_sc), which applies alpha to coverage while the dynamic state says off: on an RX 590
    // (24.9.1) TotK, which never turns it on, drew Link and the NPCs with depth but no colour (black silhouettes) and
    // dithered other surfaces into checkerboards (issue #67, RX 470/570/580). Taken out of the dynamic state, it goes
    // into the pipeline key and the characters draw.
    const char *da = cfg_get("TOTK_VK_DYN_A2C");   // no fixed default: it depends on the driver, see above
    int da_on = da && *da ? strcmp(da, "0") != 0 : !vk_caps.amd_sc;
    if (!da_on && P.dyn_a2c) {
        P.dyn_a2c = 0;
        rt_log("vk_pipeline: dynamic alpha to coverage off (%s)", da && *da ? "TOTK_VK_DYN_A2C=0" : "AMD's driver for RDNA 2 and older");
    }
#undef PROC
    static const VkDynamicState core[] = {
        VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT, VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT, VK_DYNAMIC_STATE_DEPTH_BIAS,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE,
        VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP,
        VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE,
        VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE};
    P.n_dyn = 0;
    for (size_t i = 0; i < sizeof core / sizeof *core; i++) P.dyn[P.n_dyn++] = core[i];
    P.dyn[P.n_dyn++] = P.dyn_vi ? VK_DYNAMIC_STATE_VERTEX_INPUT_EXT : VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE;
    if (P.dyn_clamp) P.dyn[P.n_dyn++] = VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT;
    if (P.dyn_poly) P.dyn[P.n_dyn++] = VK_DYNAMIC_STATE_POLYGON_MODE_EXT;
    if (P.dyn_blend_en) P.dyn[P.n_dyn++] = VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT;
    if (P.dyn_blend_eq) P.dyn[P.n_dyn++] = VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT;
    if (P.dyn_wmask) P.dyn[P.n_dyn++] = VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT;
    if (P.dyn_a2c) P.dyn[P.n_dyn++] = VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT;
    cache_load(c);
    rt_log("vk_pipeline: max push descriptors %u; dynamic vertex input %d, depth clamp %d, polygon mode %d, blend enable %d, "
           "blend equation %d, write mask %d, alpha to coverage %d; cache %s",
           P.max_push, P.dyn_vi, P.dyn_clamp, P.dyn_poly, P.dyn_blend_en, P.dyn_blend_eq, P.dyn_wmask, P.dyn_a2c,
           P.cache_path[0] ? P.cache_path : "<memory only>");
    keys_open();
    plat_xchg32(&P.ready, 1);
    plat_mutex_unlock(&P.lock);
    prebuild_start();
    return 1;
}

// ---- NVN -> Vulkan mappings (LAS Recomp exec.c blend_factor, blend_op, compare_fn, stencil_op)

static VkBlendFactor blend_factor(unsigned f) {
    switch (f) {
    case 1: return VK_BLEND_FACTOR_ZERO;
    case 2: return VK_BLEND_FACTOR_ONE;
    case 3: return VK_BLEND_FACTOR_SRC_COLOR;
    case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 5: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 7: return VK_BLEND_FACTOR_DST_ALPHA;
    case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 9: return VK_BLEND_FACTOR_DST_COLOR;
    case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 0x10: return VK_BLEND_FACTOR_SRC1_COLOR;
    case 0x11: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case 0x12: return VK_BLEND_FACTOR_SRC1_ALPHA;
    case 0x13: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default: LOG_ONCE("vk_pipeline: unknown NVN blend factor %#x, using ONE", f); return VK_BLEND_FACTOR_ONE;
    }
}

static VkBlendOp blend_op(unsigned e) {
    switch (e) {
    case 2: return VK_BLEND_OP_SUBTRACT;
    case 3: return VK_BLEND_OP_REVERSE_SUBTRACT;
    case 4: return VK_BLEND_OP_MIN;
    case 5: return VK_BLEND_OP_MAX;
    default: return VK_BLEND_OP_ADD;
    }
}

// NVN_DEPTH_FUNC_* / NVN_STENCIL_FUNC_* are NEVER = 1 .. ALWAYS = 8 in VkCompareOp order.
static VkCompareOp compare_op(unsigned f) { return f >= 1 && f <= 8 ? (VkCompareOp)(f - 1) : VK_COMPARE_OP_ALWAYS; }
// NVN_STENCIL_OP_* are KEEP = 1 .. DECR_WRAP = 8 in VkStencilOp order.
static VkStencilOp stencil_op(unsigned o) { return o >= 1 && o <= 8 ? (VkStencilOp)(o - 1) : VK_STENCIL_OP_KEEP; }
// NVN_POLYGON_MODE_POINT/LINE/FILL = 0/1/2, VkPolygonMode FILL/LINE/POINT = 0/1/2.
static VkPolygonMode polygon_mode(unsigned m) {
    if (!P.f_fill_non_solid || m > 2) return VK_POLYGON_MODE_FILL;
    return (VkPolygonMode)(2 - m);
}

VkPrimitiveTopology vk_pipeline_topology(int prim) {
    switch (prim) {
    case NVN_DRAW_PRIMITIVE_POINTS: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case NVN_DRAW_PRIMITIVE_LINES: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case NVN_DRAW_PRIMITIVE_LINE_LOOP:
    case NVN_DRAW_PRIMITIVE_LINE_STRIP: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP:
    case NVN_DRAW_PRIMITIVE_QUAD_STRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;   // same triangles, same coverage
    case NVN_DRAW_PRIMITIVE_TRIANGLE_FAN:
    case NVN_DRAW_PRIMITIVE_POLYGON: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    case NVN_DRAW_PRIMITIVE_LINES_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_LINE_STRIP_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_TRIANGLES_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_PATCHES: return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;   // TRIANGLES; QUADS after the replay's index rewrite
    }
}

// Topology class (a pipeline's static topology only fixes the class): 0 point, 1 line, 2 triangle, 3 patch.
uint8_t vk_geometry_input(uint32_t primitive) {
    // Vulkan Geometry Shading: these list/strip forms have the same GS input
    // primitive, including adjacency. Rewritten NVN forms have no proven GS ABI.
    switch (primitive) {
    case NVN_DRAW_PRIMITIVE_POINTS: return GFX_GEOMETRY_POINTS;
    case NVN_DRAW_PRIMITIVE_LINES:
    case NVN_DRAW_PRIMITIVE_LINE_STRIP: return GFX_GEOMETRY_LINES;
    case NVN_DRAW_PRIMITIVE_LINES_ADJACENCY:
    case NVN_DRAW_PRIMITIVE_LINE_STRIP_ADJACENCY: return GFX_GEOMETRY_LINES_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_TRIANGLES:
    case NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP:
    case NVN_DRAW_PRIMITIVE_TRIANGLE_FAN: return GFX_GEOMETRY_TRIANGLES;
    case NVN_DRAW_PRIMITIVE_TRIANGLES_ADJACENCY:
    case NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP_ADJACENCY: return GFX_GEOMETRY_TRIANGLES_ADJACENCY;
    default: return GFX_GEOMETRY_NONE;
    }
}

static int topology_class(VkPrimitiveTopology t) {
    switch (t) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: return 0;
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY: return 1;
    case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST: return 3;
    default: return 2;
    }
}

int vk_pipeline_tex_type(int target) {
    switch (target) {
    case NVN_TEXTURE_TARGET_1D: return GFX_TEX_1D;
    case NVN_TEXTURE_TARGET_3D: return GFX_TEX_3D;
    case NVN_TEXTURE_TARGET_1D_ARRAY: return GFX_TEX_1D_ARRAY;
    case NVN_TEXTURE_TARGET_2D_ARRAY: return GFX_TEX_2D_ARRAY;
    case NVN_TEXTURE_TARGET_2D_MULTISAMPLE: return GFX_TEX_2D_MS;
    case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY: return GFX_TEX_2D_MS_ARRAY;
    case NVN_TEXTURE_TARGET_CUBEMAP: return GFX_TEX_CUBE;
    case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: return GFX_TEX_CUBE_ARRAY;
    case NVN_TEXTURE_TARGET_BUFFER: return GFX_TEX_BUFFER;
    default: return GFX_TEX_2D;   // 2D, RECTANGLE
    }
}

// Numeric type of a colour format as a shader sees it: GFX_TYPE_*.
static int format_type(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8_UINT: case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16_UINT: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_B8G8R8A8_UINT: case VK_FORMAT_S8_UINT:
        return GFX_TYPE_UINT;
    case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8_SINT: case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16_SINT: case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32_SINT: case VK_FORMAT_R32G32B32A32_SINT:
    case VK_FORMAT_A2B10G10R10_SINT_PACK32: case VK_FORMAT_B8G8R8A8_SINT:
        return GFX_TYPE_SINT;
    default: return GFX_TYPE_FLOAT;
    }
}

// Optimal-tiling (o = 1) or buffer (o = 0) features of a core format, cached.
static VkFormatFeatureFlags format_features(VkFormat f, int optimal) {
    static VkFormatFeatureFlags known[2][256];
    static uint8_t have[2][256];
    VkCtx *c = vk_ctx();
    if (!c) return 0;
    if ((unsigned)f < 256 && have[optimal][f]) return known[optimal][f];
    VkFormatProperties p;
    vkGetPhysicalDeviceFormatProperties(c->phys, f, &p);
    VkFormatFeatureFlags r = optimal ? p.optimalTilingFeatures : p.bufferFeatures;
    if ((unsigned)f < 256) known[optimal][f] = r, have[optimal][f] = 1;
    return r;
}

// Vertex attribute formats by NVNformat (LAS Recomp exec.c vertex_format; MNG nvn_mtl.m vfmt for the TotK additions).
typedef struct { VkFormat vk; uint8_t type; } VtxFmt;
#define F(v) {VK_FORMAT_##v, GFX_TYPE_FLOAT}
#define U(v) {VK_FORMAT_##v, GFX_TYPE_UINT}
#define I(v) {VK_FORMAT_##v, GFX_TYPE_SINT}
static const VtxFmt vtx_formats[0x9B] = {
    [0x01] = F(R8_UNORM), [0x02] = F(R8_SNORM), [0x03] = U(R8_UINT), [0x04] = I(R8_SINT), [0x05] = F(R16_SFLOAT),
    [0x06] = F(R16_UNORM), [0x07] = F(R16_SNORM), [0x08] = U(R16_UINT), [0x09] = I(R16_SINT), [0x0A] = F(R32_SFLOAT),
    [0x0B] = U(R32_UINT), [0x0C] = I(R32_SINT), [0x0D] = F(R8G8_UNORM), [0x0E] = F(R8G8_SNORM), [0x0F] = U(R8G8_UINT),
    [0x10] = I(R8G8_SINT), [0x11] = F(R16G16_SFLOAT), [0x12] = F(R16G16_UNORM), [0x13] = F(R16G16_SNORM),
    [0x14] = U(R16G16_UINT), [0x15] = I(R16G16_SINT), [0x16] = F(R32G32_SFLOAT), [0x17] = U(R32G32_UINT),
    [0x18] = I(R32G32_SINT), [0x19] = F(R8G8B8_UNORM), [0x1A] = F(R8G8B8_SNORM), [0x1B] = U(R8G8B8_UINT),
    [0x1C] = I(R8G8B8_SINT), [0x1D] = F(R16G16B16_SFLOAT), [0x1E] = F(R16G16B16_UNORM), [0x1F] = F(R16G16B16_SNORM),
    [0x20] = U(R16G16B16_UINT), [0x21] = I(R16G16B16_SINT), [0x22] = F(R32G32B32_SFLOAT), [0x23] = U(R32G32B32_UINT),
    [0x24] = I(R32G32B32_SINT), [0x25] = F(R8G8B8A8_UNORM), [0x26] = F(R8G8B8A8_SNORM), [0x27] = U(R8G8B8A8_UINT),
    [0x28] = I(R8G8B8A8_SINT), [0x29] = F(R16G16B16A16_SFLOAT), [0x2A] = F(R16G16B16A16_UNORM),
    [0x2B] = F(R16G16B16A16_SNORM), [0x2C] = U(R16G16B16A16_UINT), [0x2D] = I(R16G16B16A16_SINT),
    [0x2E] = F(R32G32B32A32_SFLOAT), [0x2F] = U(R32G32B32A32_UINT), [0x30] = I(R32G32B32A32_SINT),
    [0x3D] = F(A2B10G10R10_UNORM_PACK32), [0x3E] = U(A2B10G10R10_UINT_PACK32), [0x3F] = F(B10G11R11_UFLOAT_PACK32),
    // *_UI2F / *_I2F: integer data read as float; the 8- and 16-bit ones are Vulkan's SCALED formats
    [0x51] = F(R8_USCALED), [0x52] = F(R8_SSCALED), [0x53] = F(R16_USCALED), [0x54] = F(R16_SSCALED),
    [0x57] = F(R8G8_USCALED), [0x58] = F(R8G8_SSCALED), [0x59] = F(R16G16_USCALED), [0x5A] = F(R16G16_SSCALED),
    [0x5D] = F(R8G8B8_USCALED), [0x5E] = F(R8G8B8_SSCALED), [0x5F] = F(R16G16B16_USCALED), [0x60] = F(R16G16B16_SSCALED),
    [0x63] = F(R8G8B8A8_USCALED), [0x64] = F(R8G8B8A8_SSCALED), [0x65] = F(R16G16B16A16_USCALED),
    [0x66] = F(R16G16B16A16_SSCALED),
    // 32-bit ones have no SCALED format: read as integers (values are not converted; logged when used)
    [0x55] = U(R32_UINT), [0x56] = I(R32_SINT), [0x5B] = U(R32G32_UINT), [0x5C] = I(R32G32_SINT),
    [0x61] = U(R32G32B32_UINT), [0x62] = I(R32G32B32_SINT), [0x67] = U(R32G32B32A32_UINT), [0x68] = I(R32G32B32A32_SINT),
    [0x69] = F(A2B10G10R10_SNORM_PACK32), [0x6A] = I(A2B10G10R10_SINT_PACK32), [0x6B] = F(A2B10G10R10_USCALED_PACK32),
    [0x6C] = F(A2B10G10R10_SSCALED_PACK32), [0x9A] = F(B8G8R8A8_UNORM),
};
#undef F
#undef U
#undef I

// 3-component 8/16-bit formats are not vertex formats on every GPU: read 4 components (LAS vertex_format_supported).
static VkFormat widen(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8G8B8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_R8G8B8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
    case VK_FORMAT_R8G8B8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
    case VK_FORMAT_R8G8B8_SINT: return VK_FORMAT_R8G8B8A8_SINT;
    case VK_FORMAT_R8G8B8_USCALED: return VK_FORMAT_R8G8B8A8_USCALED;
    case VK_FORMAT_R8G8B8_SSCALED: return VK_FORMAT_R8G8B8A8_SSCALED;
    case VK_FORMAT_R16G16B16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case VK_FORMAT_R16G16B16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
    case VK_FORMAT_R16G16B16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
    case VK_FORMAT_R16G16B16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
    case VK_FORMAT_R16G16B16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
    case VK_FORMAT_R16G16B16_USCALED: return VK_FORMAT_R16G16B16A16_USCALED;
    case VK_FORMAT_R16G16B16_SSCALED: return VK_FORMAT_R16G16B16A16_SSCALED;
    default: return VK_FORMAT_UNDEFINED;
    }
}

VkFormat vk_vertex_format(int nvn_format, int *type) {
    if (type) *type = GFX_TYPE_FLOAT;
    if (nvn_format <= 0 || nvn_format >= (int)(sizeof vtx_formats / sizeof *vtx_formats)) return VK_FORMAT_UNDEFINED;
    const VtxFmt *v = &vtx_formats[nvn_format];
    if (v->vk == VK_FORMAT_UNDEFINED) return VK_FORMAT_UNDEFINED;
    if (type) *type = v->type;
    if ((nvn_format >= 0x55 && nvn_format <= 0x68) && v->type != GFX_TYPE_FLOAT)
        LOG_ONCE("vk_pipeline: 32-bit *_UI2F/*_I2F vertex formats are read as integers (first: %#x)", nvn_format);
    VkFormat f = v->vk;
    if (vk_ctx() && !(format_features(f, 0) & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT)) {
        VkFormat w = widen(f);
        if (w == VK_FORMAT_UNDEFINED || !(format_features(w, 0) & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT)) {
            LOG_ONCE("vk_pipeline: vertex format %d (NVN %#x) unsupported", (int)f, nvn_format);
            return VK_FORMAT_UNDEFINED;
        }
        LOG_ONCE("vk_pipeline: vertex format %d unsupported, read as 4 components (first: NVN %#x)", (int)f, nvn_format);
        f = w;
    }
    return f;
}

// Attachment format of an NVN format (tex_vk_format: the format the images are created with).
static VkFormat target_format(int nvn_format, int *depth, int *stencil) {
    TexVkFormat t;
    if (depth) *depth = 0;
    if (stencil) *stencil = 0;
    if (!nvn_format || !tex_vk_format(nvn_format, &t)) return VK_FORMAT_UNDEFINED;
    if (depth) *depth = t.depth;
    if (stencil) *stencil = t.stencil;
    if (nvn_format == 0x40 && t.vk == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && tex_rgb9e5_stand_in())
        return tex_rgb9e5_target();   // the render target's view format (vk_exec.c resolve_target)
    return t.vk;
}

void vk_pipe_state_defaults(VkPipeState *s) {
    memset(s, 0, sizeof *s);
    s->samples = 1;
    s->prim = NVN_DRAW_PRIMITIVE_TRIANGLES;
    for (int i = 0; i < 8; i++)
        s->blend[i] = (blend_st){(uint64_t)i, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE,
                                 NVN_BLEND_FUNC_ZERO, NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD};
    s->color = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS};
    s->chmask.mask = 0xffffffffu;
    s->ms.enable = 1;
    s->poly = (poly_st){NVN_FACE_NONE, NVN_FRONT_FACE_CCW, NVN_POLYGON_MODE_FILL, 0};
    s->ds = (ds_st){0, 1, NVN_DEPTH_FUNC_LESS, 0, NVN_STENCIL_FUNC_ALWAYS, NVN_STENCIL_FUNC_ALWAYS,
                    NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP,
                    NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP};
}

void vk_pipeline_answers(const VkPipeState *s, GfxShaderAnswers *vs, GfxShaderAnswers *fs) {
    if (vs)
        for (int i = 0; i < 16; i++) {
            va_st a;
            memcpy(&a, &s->vattr[i], sizeof a);
            int type = GFX_TYPE_FLOAT;
            if (a.fmt) vk_vertex_format((int)a.fmt, &type);
            vs->attr_type[i] = (uint8_t)type;
        }
    if (fs)
        for (int i = 0; i < 8; i++) {
            VkFormat f = target_format(s->color_fmt[i], NULL, NULL);
            fs->frag_out_type[i] = (uint8_t)(f == VK_FORMAT_UNDEFINED ? GFX_TYPE_FLOAT : format_type(f));
        }
}

// ---- vertex input (pipeline state, or vkCmdSetVertexInputEXT)

typedef struct {
    uint32_t n_attr, n_bind;
    VkVertexInputAttributeDescription2EXT a[32];
    VkVertexInputBindingDescription2EXT b[17];
} VtxInput;

// An attribute the game left without a format is a fixed one on Maxwell, a single 32-bit component of 0 (deko3d
// gpu_3d_vbo.cpp binds unused attributes as IsFixed 1x32 float): it reads (0, 0, 0, 1), as Vulkan expands a
// one-component format. TOTK's foliage shaders scale wind sway by 1 - colour.w; shapes without a vertex colour
// (Brightbloom stem and leaves) must not sway, or they bend away from the parts that have one (the bulb).
static VkFormat zero_format(int type) {
    return type == GFX_TYPE_UINT ? VK_FORMAT_R32_UINT : type == GFX_TYPE_SINT ? VK_FORMAT_R32_SINT : VK_FORMAT_R32_SFLOAT;
}

// Attributes for the Locations the vertex shader reads; one without a usable NVN attribute reads VK_PIPE_ZERO_BINDING.
static void vertex_input(const VkPipeState *s, VtxInput *v) {
    GfxShaderIo io[32];
    uint32_t used = gfx_shader_inputs(s->vs, io), streams = 0;
    int zero = 0;
    memset(v, 0, sizeof *v);
    for (int l = 0; l < 32; l++) {
        if (!(used >> l & 1)) continue;
        VkVertexInputAttributeDescription2EXT *a = &v->a[v->n_attr++];
        *a = (VkVertexInputAttributeDescription2EXT){.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                                     .location = (uint32_t)l};
        va_st va = {0};
        if (l < 16) memcpy(&va, &s->vattr[l], sizeof va);
        VkFormat f = va.fmt ? vk_vertex_format((int)va.fmt, NULL) : VK_FORMAT_UNDEFINED;
        if (f != VK_FORMAT_UNDEFINED && va.stream < 16) {
            a->binding = va.stream, a->format = f, a->offset = va.offset;
            streams |= 1u << va.stream;
        } else {
            a->binding = VK_PIPE_ZERO_BINDING, a->format = zero_format(io[l].type);
            zero = 1;
        }
    }
    for (uint32_t b = 0; b < 16; b++) {
        if (!(streams >> b & 1)) continue;
        uint32_t div = s->vstream[b].divisor;
        if (div > 1) LOG_ONCE("vk_pipeline: instance divisor %u treated as 1 (no VK_EXT_vertex_attribute_divisor)", div);
        v->b[v->n_bind++] = (VkVertexInputBindingDescription2EXT){
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, .binding = b, .stride = s->vstream[b].stride,
            .inputRate = div ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX, .divisor = 1};
    }
    if (zero)
        v->b[v->n_bind++] = (VkVertexInputBindingDescription2EXT){
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, .binding = VK_PIPE_ZERO_BINDING,
            .stride = 0, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX, .divisor = 1};
}

int vk_pipeline_vertex_input_dynamic(void) { return ensure_init() && P.dyn_vi; }

PFN_vkCmdPushDescriptorSetKHR vk_pipeline_push_fn(void) { return ensure_init() ? P.push : NULL; }
PFN_vkCmdPushDescriptorSetWithTemplateKHR vk_pipeline_push_tpl_fn(void) { return ensure_init() && P.tpl ? P.tpl_push : NULL; }
PFN_vkUpdateDescriptorSetWithTemplate vk_pipeline_update_tpl_fn(void) { return ensure_init() && P.tpl ? P.tpl_update : NULL; }

int vk_pipeline_set_indep(int on) {
    if (!ensure_init()) return 0;
    plat_mutex_lock(&P.lock);
    int was = P.indep;
    P.indep = on && vk_caps.gpl;
    plat_mutex_unlock(&P.lock);
    return was;
}

// ---- SPIR-V rewrites

// Binding decorations b >= 1 become b + off (the fragment stage's share of set 0).
static uint32_t *spirv_rebind(const uint32_t *w, uint64_t n, uint32_t off) {
    uint32_t *o = malloc(n * 4);
    if (!o) return NULL;
    memcpy(o, w, n * 4);
    uint32_t fixed[16];   // LOCAL TEST (Ocarina): the variables of the shared set 2 keep their bindings
    int nfixed = 0;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = o[i] & 0xffff, len = o[i] >> 16;
        if (!len || i + len > n) break;
        if (op == SPV_OP_DECORATE && len >= 4 && o[i + 2] == SPV_DEC_DESCRIPTOR_SET && o[i + 3] == 2 && nfixed < 16) fixed[nfixed++] = o[i + 1];
        i += len;
    }
    for (uint64_t i = 5; i < n;) {
        uint32_t op = o[i] & 0xffff, len = o[i] >> 16;
        if (!len || i + len > n) break;
        int keep = 0;
        for (int k = 0; k < nfixed; k++) keep |= fixed[k] == o[i + 1];
        if (op == SPV_OP_DECORATE && len >= 4 && o[i + 2] == SPV_DEC_BINDING && o[i + 3] >= 1 && !keep) o[i + 3] += off;
        i += len;
    }
    return o;
}

// DescriptorSet decorations become set (the fragment stage's resources move to set 1 with independent layouts).
static uint32_t *spirv_set(const uint32_t *w, uint64_t n, uint32_t set) {
    uint32_t *o = malloc(n * 4);
    if (!o) return NULL;
    memcpy(o, w, n * 4);
    for (uint64_t i = 5; i < n;) {
        uint32_t op = o[i] & 0xffff, len = o[i] >> 16;
        if (!len || i + len > n) break;
        if (op == SPV_OP_DECORATE && len >= 4 && o[i + 2] == SPV_DEC_DESCRIPTOR_SET && o[i + 3] != 2) o[i + 3] = set;
        i += len;
    }
    return o;
}

// Dual-source blending (docs/maxwell.md 5.6): SRC1 blend factors read the fragment shader's colour output 1,
// which Vulkan takes from Location 0, Index 1. The Output variable at Location 1 moves there, as Ryujinx declares it
// with DualSourceBlendEnable (CodeGen/Spirv/Declarations.cs); the pipeline is keyed by GKey.dsb, so only pipelines
// that blend with SRC1 get this module. NULL when the module has no Output at Location 1.
static uint32_t *spirv_dual_source(const uint32_t *w, uint64_t n, uint64_t *out_n) {
    uint32_t outs[64];
    int nout = 0;
    uint64_t at = 0;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) return NULL;
        if (op == SPV_OP_VARIABLE && len >= 4 && w[i + 3] == SPV_SC_OUTPUT && nout < 64) outs[nout++] = w[i + 2];
        i += len;
    }
    for (uint64_t i = 5; i < n && !at;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == SPV_OP_DECORATE && len == 4 && w[i + 2] == SPV_DEC_LOCATION && w[i + 3] == 1)
            for (int k = 0; k < nout; k++)
                if (outs[k] == w[i + 1]) at = i;
        i += len;
    }
    if (!at) return NULL;
    uint32_t *o = malloc((n + 4) * 4);
    if (!o) return NULL;
    memcpy(o, w, (at + 4) * 4);
    o[at + 3] = 0;
    o[at + 4] = 4u << 16 | SPV_OP_DECORATE, o[at + 5] = w[at + 1], o[at + 6] = SPV_DEC_INDEX, o[at + 7] = 1;
    memcpy(o + at + 8, w + at + 4, (n - at - 4) * 4);
    *out_n = n + 4;
    return o;
}

// NVN's alpha test (nvnColorStateSetAlphaTest, nvnCommandBufferSetAlphaRef) has no Vulkan state. The fragment fails when
// colour output 0's alpha does not compare true with the reference, before blending; NaN fails. So before each return of
// the entry point this adds: load Location 0, compare its w, OpKill when false. The pipeline is keyed by GKey.alpha, so
// only draws with a test get this module. NULL when the module has no float vec4 Output at Location 0.
static uint32_t *spirv_alpha_test(const uint32_t *w, uint64_t n, unsigned func, float ref, uint64_t *out_n) {
    enum { OP_TYPE_BOOL = 20, OP_FUNCTION_END = 56, OP_LOAD = 61, OP_COMPOSITE_EXTRACT = 81, OP_SELECTION_MERGE = 247,
           OP_LABEL = 248, OP_BRANCH_CONDITIONAL = 250, OP_KILL = 252, OP_RETURN = 253, RETURN_WORDS = 26 };
    // By NVN_ALPHA_FUNC_*: FOrdLessThan (a < a is never true), FOrdLessThan, FOrdEqual, FOrdLessThanEqual,
    // FOrdGreaterThan, FOrdNotEqual, FOrdGreaterThanEqual
    static const uint16_t compare[8] = {0, 184, 184, 180, 188, 186, 182, 190};
    if (func < 1 || func > 7 || n < 5) return NULL;
    uint32_t bound = w[3], entry = 0, bool_type = 0, *loc0 = calloc(bound, 4), *type_of = calloc(bound, 4);
    uint32_t *comp = calloc(bound, 4), *count = calloc(bound, 4);
    uint32_t *f32 = calloc(bound, 4), *o = NULL;
    uint32_t out_var = 0, vec = 0, flt = 0, returns = 0, bad = !loc0 || !type_of || !comp || !count || !f32;
    uint64_t fn_at = 0;
    for (uint64_t i = 5; i < n && !bad;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) { bad = 1; break; }
        const uint32_t *a = w + i + 1;
        if (op == SPV_OP_ENTRY_POINT && len >= 3 && !entry) {
            if (a[0] != 4) bad = 1;   // ExecutionModel Fragment
            entry = a[1];
        } else if (op == SPV_OP_DECORATE && len == 4 && a[1] == SPV_DEC_LOCATION && a[2] == 0 && a[0] < bound) loc0[a[0]] = 1;
        else if (op == OP_TYPE_BOOL && len == 2) bool_type = a[0];
        else if (op == SPV_OP_TYPE_FLOAT && len >= 3 && a[0] < bound && a[1] == 32) f32[a[0]] = 1;
        else if (op == SPV_OP_TYPE_VECTOR && len >= 4 && a[0] < bound) comp[a[0]] = a[1], count[a[0]] = a[2];
        else if (op == SPV_OP_TYPE_POINTER && len >= 4 && a[0] < bound && a[1] == SPV_SC_OUTPUT) type_of[a[0]] = a[2];
        else if (op == SPV_OP_VARIABLE && len >= 4 && a[2] == SPV_SC_OUTPUT && a[1] < bound && loc0[a[1]] && a[0] < bound) {
            uint32_t t = type_of[a[0]];
            if (t < bound && count[t] == 4 && comp[t] < bound && f32[comp[t]]) out_var = a[1], vec = t, flt = comp[t];
        } else if (op == SPV_OP_FUNCTION) { fn_at = i; break; }
        i += len;
    }
    int in_entry = 0;
    for (uint64_t i = fn_at; fn_at && i < n && !bad;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) { bad = 1; break; }
        if (op == SPV_OP_FUNCTION) in_entry = len >= 3 && w[i + 2] == entry;
        else if (op == OP_FUNCTION_END) in_entry = 0;
        else if (op == OP_RETURN && in_entry) returns++;
        i += len;
    }
    if (!bad && out_var && entry && fn_at && returns) {
        uint32_t add = (bool_type ? 0 : 2) + 4;
        uint64_t total = n + add + (uint64_t)returns * RETURN_WORDS;
        o = malloc(total * 4);
        if (o) {
            uint32_t ref_id = bound++, bits;
            memcpy(&bits, &ref, 4);
            memcpy(o, w, fn_at * 4);
            uint64_t d = fn_at;
            if (!bool_type) bool_type = bound++, o[d++] = 2u << 16 | OP_TYPE_BOOL, o[d++] = bool_type;
            o[d++] = 4u << 16 | SPV_OP_CONSTANT, o[d++] = flt, o[d++] = ref_id, o[d++] = bits;
            in_entry = 0;
            for (uint64_t i = fn_at; i < n;) {
                uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
                if (op == SPV_OP_FUNCTION) in_entry = w[i + 2] == entry;
                else if (op == OP_FUNCTION_END) in_entry = 0;
                else if (op == OP_RETURN && in_entry) {
                    uint32_t v = bound++, a = bound++, c = bound++, kill = bound++, merge = bound++;
                    o[d++] = 4u << 16 | OP_LOAD, o[d++] = vec, o[d++] = v, o[d++] = out_var;
                    o[d++] = 5u << 16 | OP_COMPOSITE_EXTRACT, o[d++] = flt, o[d++] = a, o[d++] = v, o[d++] = 3;
                    o[d++] = 5u << 16 | compare[func], o[d++] = bool_type, o[d++] = c, o[d++] = a;
                    o[d++] = func == 1 ? a : ref_id;
                    o[d++] = 3u << 16 | OP_SELECTION_MERGE, o[d++] = merge, o[d++] = 0;
                    o[d++] = 4u << 16 | OP_BRANCH_CONDITIONAL, o[d++] = c, o[d++] = merge, o[d++] = kill;
                    o[d++] = 2u << 16 | OP_LABEL, o[d++] = kill;
                    o[d++] = 1u << 16 | OP_KILL;
                    o[d++] = 2u << 16 | OP_LABEL, o[d++] = merge;
                }
                memcpy(o + d, w + i, len * 4);
                d += len, i += len;
            }
            o[3] = bound;
            *out_n = d;
        }
    }
    free(loc0), free(type_of), free(comp), free(count), free(f32);
    return o;
}

// Adds an Output variable for each Location of `missing` (type and width from io) to a module, declared in the entry
// point interface and initialised as Maxwell reads an unwritten output: (0, 0, 0, 1), the w as the bits of 1.0f whatever
// the type (attribute memory is untyped); fewer than 4 components are all zero. Existing scalar, vector and pointer types
// are reused. NULL when the module does not have the expected layout.
static uint32_t *spirv_add_outputs(const uint32_t *w, uint64_t n, uint32_t missing, const GfxShaderIo io[32],
                                   uint64_t *out_n) {
    uint64_t ep_at = 0, ann_end = 0, fn_at = 0;
    uint32_t bound = w[3], scalar[3] = {0, 0, 0};          // GFX_TYPE_FLOAT, SINT, UINT
    struct { uint32_t id, comp, count; } vec[64];
    struct { uint32_t id, type; } ptr[64];
    int nvec = 0, nptr = 0;
    uint32_t mesh_verts = 0;   // MeshEXT entry point: its outputs are arrays of OutputVertices
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) return NULL;
        const uint32_t *a = w + i + 1;
        if (op == SPV_OP_ENTRY_POINT && !ep_at) ep_at = i, mesh_verts = a[0] == 5365 ? 1 : 0;
        else if (op == 16 && len >= 4 && a[1] == 26) mesh_verts = mesh_verts ? a[2] : 0;   // OpExecutionMode OutputVertices
        else if (op == SPV_OP_DECORATE || op == 72 || op == 73 || op == 74 || op == 75 || op == 332 || op == 5632 || op == 5633)
            ann_end = i + len;
        else if (op == SPV_OP_TYPE_FLOAT && len >= 3 && a[1] == 32) scalar[GFX_TYPE_FLOAT] = a[0];
        else if (op == SPV_OP_TYPE_INT && len >= 4 && a[1] == 32) scalar[a[2] ? GFX_TYPE_SINT : GFX_TYPE_UINT] = a[0];
        else if (op == SPV_OP_TYPE_VECTOR && len >= 4 && nvec < 64) vec[nvec].id = a[0], vec[nvec].comp = a[1], vec[nvec++].count = a[2];
        else if (op == SPV_OP_TYPE_POINTER && len >= 4 && a[1] == SPV_SC_OUTPUT && nptr < 64) ptr[nptr].id = a[0], ptr[nptr++].type = a[2];
        else if (op == SPV_OP_FUNCTION) { fn_at = i; break; }
        i += len;
    }
    if (!ep_at || !ann_end || !fn_at || ann_end < ep_at || fn_at < ann_end) return NULL;
    if (mesh_verts == 1 || mesh_verts > 256) return NULL;   // a mesh module without a vertex count
    uint32_t types[32 * 300], decs[32 * 4], vars[32], zero[3] = {0, 0, 0}, one[3] = {0, 0, 0}, vlen = 0;
    int nt = 0, nd = 0, nv = 0;
    if (mesh_verts) {   // the array length: a u32 constant
        if (!scalar[GFX_TYPE_UINT]) {
            scalar[GFX_TYPE_UINT] = bound++;
            types[nt++] = 4u << 16 | SPV_OP_TYPE_INT, types[nt++] = scalar[GFX_TYPE_UINT], types[nt++] = 32, types[nt++] = 0;
        }
        vlen = bound++;
        types[nt++] = 4u << 16 | SPV_OP_CONSTANT, types[nt++] = scalar[GFX_TYPE_UINT], types[nt++] = vlen, types[nt++] = mesh_verts;
    }
    for (int l = 0; l < 32; l++) {
        if (!(missing >> l & 1)) continue;
        int t = io[l].type <= GFX_TYPE_UINT ? io[l].type : GFX_TYPE_FLOAT;
        uint32_t comps = io[l].comps ? io[l].comps : 4;
        if (!scalar[t]) {
            scalar[t] = bound++;
            if (t == GFX_TYPE_FLOAT) types[nt++] = 3u << 16 | SPV_OP_TYPE_FLOAT, types[nt++] = scalar[t], types[nt++] = 32;
            else types[nt++] = 4u << 16 | SPV_OP_TYPE_INT, types[nt++] = scalar[t], types[nt++] = 32, types[nt++] = t == GFX_TYPE_SINT;
        }
        uint32_t type = scalar[t];
        if (comps > 1) {
            int k = 0;
            while (k < nvec && !(vec[k].comp == type && vec[k].count == comps)) k++;
            if (k == nvec) {
                if (nvec == 64) return NULL;
                vec[nvec].id = bound++, vec[nvec].comp = type, vec[nvec++].count = comps;
                types[nt++] = 4u << 16 | SPV_OP_TYPE_VECTOR, types[nt++] = vec[k].id, types[nt++] = type, types[nt++] = comps;
            }
            type = vec[k].id;
        }
        int k = 0;
        if (!mesh_verts) {
            while (k < nptr && ptr[k].type != type) k++;
            if (k == nptr) {
                if (nptr == 64) return NULL;
                ptr[nptr].id = bound++, ptr[nptr++].type = type;
                types[nt++] = 4u << 16 | SPV_OP_TYPE_POINTER, types[nt++] = ptr[k].id, types[nt++] = SPV_SC_OUTPUT, types[nt++] = type;
            }
        }
        uint32_t init = bound++, var = bound++;
        if (comps == 4) {
            if (!zero[t]) {
                zero[t] = bound++, one[t] = bound++;
                types[nt++] = 4u << 16 | SPV_OP_CONSTANT, types[nt++] = scalar[t], types[nt++] = zero[t], types[nt++] = 0;
                types[nt++] = 4u << 16 | SPV_OP_CONSTANT, types[nt++] = scalar[t], types[nt++] = one[t], types[nt++] = 0x3f800000;
            }
            types[nt++] = 7u << 16 | SPV_OP_CONSTANT_COMPOSITE, types[nt++] = type, types[nt++] = init;
            types[nt++] = zero[t], types[nt++] = zero[t], types[nt++] = zero[t], types[nt++] = one[t];
        } else {
            types[nt++] = 3u << 16 | SPV_OP_CONSTANT_NULL, types[nt++] = type, types[nt++] = init;
        }
        uint32_t var_ptr = ptr[k].id, var_init = init;
        if (mesh_verts) {   // MeshEXT outputs are per-vertex arrays: the initial value repeats for every vertex
            uint32_t arr = bound++, arr_ptr = bound++, arr_init = bound++;
            types[nt++] = 4u << 16 | 28 /* OpTypeArray */, types[nt++] = arr, types[nt++] = type, types[nt++] = vlen;
            types[nt++] = 4u << 16 | SPV_OP_TYPE_POINTER, types[nt++] = arr_ptr, types[nt++] = SPV_SC_OUTPUT, types[nt++] = arr;
            types[nt++] = (3u + mesh_verts) << 16 | SPV_OP_CONSTANT_COMPOSITE, types[nt++] = arr, types[nt++] = arr_init;
            for (uint32_t v = 0; v < mesh_verts; v++) types[nt++] = init;
            var_ptr = arr_ptr, var_init = arr_init;
        }
        types[nt++] = 5u << 16 | SPV_OP_VARIABLE, types[nt++] = var_ptr, types[nt++] = var, types[nt++] = SPV_SC_OUTPUT,
        types[nt++] = var_init;
        decs[nd++] = 4u << 16 | SPV_OP_DECORATE, decs[nd++] = var, decs[nd++] = SPV_DEC_LOCATION, decs[nd++] = (uint32_t)l;
        vars[nv++] = var;
    }
    uint64_t total = n + (uint64_t)(nt + nd + nv);
    uint32_t *o = malloc(total * 4), *p = o;
    if (!o) return NULL;
    uint32_t ep_len = w[ep_at] >> 16;
    memcpy(p, w, 5 * 4), p[3] = bound, p += 5;
    memcpy(p, w + 5, (ep_at - 5) * 4), p += ep_at - 5;
    *p++ = (ep_len + (uint32_t)nv) << 16 | SPV_OP_ENTRY_POINT;
    memcpy(p, w + ep_at + 1, (ep_len - 1) * 4), p += ep_len - 1;
    memcpy(p, vars, (size_t)nv * 4), p += nv;
    memcpy(p, w + ep_at + ep_len, (ann_end - ep_at - ep_len) * 4), p += ann_end - ep_at - ep_len;
    memcpy(p, decs, (size_t)nd * 4), p += nd;
    memcpy(p, w + ann_end, (fn_at - ann_end) * 4), p += fn_at - ann_end;
    memcpy(p, types, (size_t)nt * 4), p += nt;
    memcpy(p, w + fn_at, (n - fn_at) * 4), p += n - fn_at;
    *out_n = (uint64_t)(p - o);
    return o;
}

// LOCAL TEST (Ocarina): the Locations a mesh module writes per primitive (Output variables decorated PerPrimitiveEXT, 5271).
static uint32_t spirv_prim_mask(const uint32_t *w, uint64_t n) {
    uint32_t bound = w[3] < (1u << 22) ? w[3] : (1u << 22), mask = 0;
    uint8_t *prim = calloc(bound, 1);
    int32_t *loc = malloc((size_t)bound * 4);
    if (!prim || !loc) { free(prim), free(loc); return 0; }
    for (uint32_t i = 0; i < bound; i++) loc[i] = -1;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) break;
        if (op == SPV_OP_DECORATE && len >= 3 && w[i + 1] < bound) {
            if (w[i + 2] == 5271) prim[w[i + 1]] = 1;
            else if (w[i + 2] == SPV_DEC_LOCATION && len >= 4) loc[w[i + 1]] = (int32_t)w[i + 3];
        }
        if (op == SPV_OP_FUNCTION) break;
        i += len;
    }
    for (uint32_t id = 0; id < bound; id++)
        if (prim[id] && loc[id] >= 0 && loc[id] < 32) mask |= 1u << loc[id];
    free(prim), free(loc);
    return mask;
}

// LOCAL TEST (Ocarina): marks the Input variables of a fragment module at the Locations of `mask` PerPrimitiveEXT, with the
// MeshShadingEXT capability and SPV_EXT_mesh_shader (what a mesh stage's per-primitive outputs need on the consumer). NULL when
// nothing changes.
static uint32_t *spirv_mark_prim_inputs(const uint32_t *w, uint64_t n, uint32_t mask, uint64_t *out_n) {
    if (!mask || n < 6) return NULL;
    uint32_t bound = w[3] < (1u << 22) ? w[3] : (1u << 22);
    int32_t *loc = malloc((size_t)bound * 4);
    uint32_t vars[64];
    int nv = 0, has_cap = 0, has_ext = 0;
    uint64_t cap_end = 0, ext_end = 0, ann_end = 0;
    if (!loc) return NULL;
    for (uint32_t i = 0; i < bound; i++) loc[i] = -1;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) break;
        if (op == 17) { cap_end = i + len; if (w[i + 1] == 5283) has_cap = 1; }
        else if (op == 10) { ext_end = i + len; if (len >= 6 && !memcmp(&w[i + 1], "SPV_EXT_mesh_shader", 20)) has_ext = 1; }
        else if (op == SPV_OP_DECORATE || op == 72 || op == 73 || op == 74 || op == 75 || op == 332 || op == 5632 || op == 5633) {
            ann_end = i + len;
            if (op == SPV_OP_DECORATE && len >= 4 && w[i + 2] == SPV_DEC_LOCATION && w[i + 1] < bound) loc[w[i + 1]] = (int32_t)w[i + 3];
        } else if (op == 59 && len >= 4 && w[i + 3] == 1 && w[i + 2] < bound && loc[w[i + 2]] >= 0 && loc[w[i + 2]] < 32 &&
                   (mask >> loc[w[i + 2]] & 1) && nv < 64) vars[nv++] = w[i + 2];
        if (op == SPV_OP_FUNCTION) break;
        i += len;
    }
    free(loc);
    if (!nv || !cap_end || !ann_end) return NULL;
    if (!ext_end) ext_end = cap_end;
    uint32_t *o = malloc((n + 2 + 6 + 3 * 64) * 4), *p = o;
    if (!o) return NULL;
    memcpy(p, w, cap_end * 4), p += cap_end;
    if (!has_cap) *p++ = 2u << 16 | 17, *p++ = 5283;
    memcpy(p, w + cap_end, (ext_end - cap_end) * 4), p += ext_end - cap_end;
    if (!has_ext) {
        *p++ = 6u << 16 | 10;
        uint32_t name[5] = {0};
        memcpy(name, "SPV_EXT_mesh_shader", 19);
        memcpy(p, name, sizeof name), p += 5;
    }
    memcpy(p, w + ext_end, (ann_end - ext_end) * 4), p += ann_end - ext_end;
    for (int k = 0; k < nv; k++) *p++ = 3u << 16 | SPV_OP_DECORATE, *p++ = vars[k], *p++ = 5271;
    memcpy(p, w + ann_end, (n - ann_end) * 4), p += n - ann_end;
    *out_n = (uint64_t)(p - o);
    return o;
}

// Remove unused translator declarations (NULL: nothing to change):
//  - the fragment shader interlock capability and extension, which Ryujinx adds to every fragment shader when the host
//    reports interlock support (SpirvGenerator.cs) and vk_device.c does not enable; kept when the shader interlocks;
//  - the CullDistance[1] member of gl_PerVertex next to ClipDistance[8] (Declarations.cs CreatePerVertexStructType):
//    9 distances exceed maxCombinedClipAndCullDistances (8 on the RX 9070). Ryujinx never accesses CullDistance; the
//    member goes when it is the struct's last and no access chain selects it;
//  - a GS has input and output per-vertex blocks. Remove their unused trailing
//    ClipDistance/CullDistance members, retaining every referenced distance and
//    every surviving member index. Declaring ClipDistance[8] in both blocks alone
//    can exceed the entry-point combined distance limit even when unused.
static uint32_t *spirv_fixup_linked(const uint32_t *w, uint64_t n, uint64_t *out_n, int geometry_chain) {
    enum { OP_EXTENSION = 10, OP_EXT_INST = 12, OP_CAPABILITY = 17, OP_MEMBER_NAME = 6, OP_EXECUTION_MODE = 16, OP_TYPE_ARRAY = 28,
           OP_TYPE_STRUCT = 30, OP_CONSTANT = 43, OP_MEMBER_DECORATE = 72, OP_ACCESS_CHAIN = 65, OP_IN_BOUNDS_ACCESS_CHAIN = 66,
           OP_BEGIN_INTERLOCK = 5364, OP_END_INTERLOCK = 5365, CAP_SAMPLE_INTERLOCK = 5363, CAP_PIXEL_INTERLOCK = 5378,
           CAP_RATE_INTERLOCK = 5372, BUILTIN_CLIP = 3, BUILTIN_CULL = 4 };
    if (n < 5) return NULL;
    uint32_t bound = w[3] < (1u << 22) ? w[3] : (1u << 22);
    uint32_t *val = calloc(bound, sizeof *val), *elem = calloc(bound, sizeof *elem), *len = calloc(bound, sizeof *len),
             *pointee = calloc(bound, sizeof *pointee), *var_type = calloc(bound, sizeof *var_type);
    uint8_t *constant = calloc(bound, 1);
    typedef struct { uint32_t type, cull, clip, members, member_types[8]; int used, drop; } CullFix;
    CullFix *blocks = NULL;
    uint32_t nblocks = 0, *o = NULL;
    int interlock_used = 0, interlock_declared = 0, cull_ok = 0;
    uint32_t entry_points = 0, geometry_entries = 0, vertex_entries = 0;
    if (!val || !elem || !len || !pointee || !var_type || !constant) goto out;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, l = w[i] >> 16;
        if (!l || i + l > n) goto out;
        const uint32_t *a = w + i + 1;
        if (op == SPV_OP_ENTRY_POINT && l >= 3)
            entry_points++, geometry_entries += a[0] == 3, vertex_entries += a[0] == 0;
        else if (op == OP_BEGIN_INTERLOCK || op == OP_END_INTERLOCK) interlock_used = 1;
        else if (op == OP_EXECUTION_MODE && l >= 3 && a[1] >= 5366 && a[1] <= 5371) interlock_used = 1;
        else if (op == OP_CAPABILITY && l >= 2 && (a[0] == CAP_PIXEL_INTERLOCK || a[0] == CAP_SAMPLE_INTERLOCK || a[0] == CAP_RATE_INTERLOCK))
            interlock_declared = 1;
        else if (op == OP_CONSTANT && l == 4 && a[1] < bound) val[a[1]] = a[2], constant[a[1]] = 1;
        else if (op == OP_TYPE_ARRAY && l >= 4 && a[0] < bound) elem[a[0]] = a[1], len[a[0]] = a[2] < bound ? val[a[2]] : 0;
        else if (op == SPV_OP_TYPE_POINTER && l >= 4 && a[0] < bound) pointee[a[0]] = a[2];
        else if (op == SPV_OP_VARIABLE && l >= 4 && a[1] < bound) var_type[a[1]] = a[0] < bound ? pointee[a[0]] : 0;
        else if (op == OP_MEMBER_DECORATE && l >= 5 && a[2] == SPV_DEC_BUILTIN &&
                 (a[3] == BUILTIN_CULL || a[3] == BUILTIN_CLIP)) {
            uint32_t j = 0;
            while (j < nblocks && blocks[j].type != a[0]) j++;
            if (j == nblocks) {
                CullFix *p = realloc(blocks, (nblocks + 1) * sizeof *p);
                if (!p) goto out;
                blocks = p;
                blocks[nblocks++] = (CullFix){.type = a[0], .cull = ~0u, .clip = ~0u};
            }
            if (a[3] == BUILTIN_CULL) blocks[j].cull = a[1];
            else blocks[j].clip = a[1];
        }
        i += l;
    }
    // A GS has separate arrayed input and output per-vertex blocks. Treat each
    // independently; used bits 1/2 retain CullDistance/ClipDistance respectively.
    int in_function = 0;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, l = w[i] >> 16;
        const uint32_t *a = w + i + 1;
        if (op == SPV_OP_FUNCTION) in_function = 1;
        for (uint32_t j = 0; j < nblocks; j++) {
            CullFix *b = &blocks[j];
            if (op == OP_TYPE_STRUCT && l >= 2 && a[0] == b->type) {
                b->members = l - 2;
                for (uint32_t m = 0; m < b->members && m < 8; m++) b->member_types[m] = a[1 + m];
            } else if ((op == OP_ACCESS_CHAIN || op == OP_IN_BOUNDS_ACCESS_CHAIN) && l >= 5 && a[2] < bound) {
                uint32_t t = var_type[a[2]], k = 3;
                if (t < bound && t && elem[t] == b->type) k = 4;
                if (t == b->type || k == 4) {
                    if (k + 1 >= l || a[k] >= bound || !constant[a[k]]) b->used |= 3;
                    else {
                        if (val[a[k]] == b->cull) b->used |= 1;
                        if (val[a[k]] == b->clip) b->used |= 2;
                    }
                }
            }
            if (in_function) for (uint32_t k = 0; k + 1 < l; k++) {
                // OpExtInst's fourth word is the instruction number, a literal: GLSL.std.450 Fma is 50, the same
                // number as the gl_PerVertex variable in many vertex shaders, and read as an id it kept CullDistance.
                if (op == OP_EXT_INST && k == 3) continue;
                uint32_t t = a[k] < bound ? var_type[a[k]] : 0;
                if (t != b->type && !(t < bound && t && elem[t] == b->type)) continue;
                // Only a direct, constant-member access proves which member is
                // used. Whole loads, pointer copies/calls/phis and an access to
                // the array's struct alone retain the complete block.
                if (!((op == OP_ACCESS_CHAIN || op == OP_IN_BOUNDS_ACCESS_CHAIN) && k == 2)) b->used |= 3;
            }
        }
        i += l;
    }
    VkCtx *c = vk_ctx();
    uint32_t limit = c ? c->props.limits.maxCombinedClipAndCullDistances : 8;
    for (uint32_t j = 0; j < nblocks; j++) {
        CullFix *b = &blocks[j];
        if (!b->members || b->members > 8) continue;
        if (entry_points == 1 && (geometry_entries == 1 || (geometry_chain && vertex_entries == 1))) {
            uint32_t tail = b->members;
            if (b->cull == tail - 1 && !(b->used & 1)) tail--;
            if (tail && b->clip == tail - 1 && !(b->used & 2)) tail--;
            // Keeping a nonempty block avoids needing to delete its variables
            // and entry-point interface references as well.
            if (tail) b->drop = b->members - tail;
            cull_ok |= b->drop;
            continue;
        }
        if (b->cull != b->members - 1 || b->clip >= b->members || (b->used & 1)) continue;
        uint32_t clip_t = b->member_types[b->clip], cull_t = b->member_types[b->cull];
        uint32_t clip_n = clip_t < bound ? len[clip_t] : 0, cull_n = cull_t < bound ? len[cull_t] : 0;
        b->drop = clip_n && cull_n && (uint64_t)clip_n + cull_n > limit && clip_n <= limit;
        cull_ok |= b->drop;
    }
    int strip = interlock_declared && !interlock_used;
    if (!strip && !cull_ok) goto out;
    o = malloc(n * 4);
    if (!o) goto out;
    uint64_t on = 5;
    memcpy(o, w, 5 * 4);
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, l = w[i] >> 16;
        const uint32_t *a = w + i + 1;
        int drop = 0;
        if (strip && op == OP_CAPABILITY && (a[0] == CAP_PIXEL_INTERLOCK || a[0] == CAP_SAMPLE_INTERLOCK || a[0] == CAP_RATE_INTERLOCK)) drop = 1;
        if (strip && op == OP_EXTENSION && l >= 2 && !strncmp((const char *)a, "SPV_EXT_fragment_shader_interlock", (l - 1) * 4)) drop = 1;
        uint32_t shorten = 0;
        for (uint32_t j = 0; j < nblocks; j++) if (blocks[j].drop) {
            if ((op == OP_MEMBER_DECORATE || op == OP_MEMBER_NAME) && l >= 3 &&
                a[0] == blocks[j].type && a[1] >= blocks[j].members - blocks[j].drop) drop = 1;
            if (op == OP_TYPE_STRUCT && a[0] == blocks[j].type) shorten = blocks[j].drop;
        }
        if (!drop) {
            memcpy(o + on, w + i, l * 4);
            if (shorten) o[on] = (l - shorten) << 16 | op;
            on += l - shorten;
        }
        i += l;
    }
    *out_n = on;
out:
    free(val), free(elem), free(len), free(pointee), free(var_type), free(constant), free(blocks);
    return o;
}

static uint32_t *spirv_fixup(const uint32_t *w, uint64_t n, uint64_t *out_n) {
    return spirv_fixup_linked(w, n, out_n, 0);
}

// True when the module declares a GroupNonUniform* or SubgroupBallot-style capability (lane masks assume 32 lanes).
static int uses_subgroup_ops(const uint32_t *w, uint64_t words) {
    for (uint64_t i = 5; i < words;) {
        uint32_t op = w[i] & 0xffff, n = w[i] >> 16;
        if (!n || i + n > words) return 0;
        if (op != 17) return 0;   // OpCapability; capabilities come first, stop at the first other instruction
        uint32_t cap = w[i + 1];
        if ((cap >= 61 && cap <= 68) || cap == 4423 || cap == 4431) return 1;   // NonUniform*, SubgroupBallotKHR, SubgroupVoteKHR
        i += n;
    }
    return 0;
}

// Chain {size 32} into a stage create info when the stage uses subgroup ops and the driver allows the pin.
static void pin_subgroup32(VkPipelineShaderStageCreateInfo *st, VkPipelineShaderStageRequiredSubgroupSizeCreateInfo *rq,
                           const uint32_t *w, uint64_t words) {
    if (!(P.sg32_stages & st->stage) || !w || !uses_subgroup_ops(w, words)) return;
    *rq = (VkPipelineShaderStageRequiredSubgroupSizeCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, .requiredSubgroupSize = vk_caps.sg_size};
    st->pNext = rq;
}

static int spirv_is_mesh(const uint32_t *w, uint64_t n);
static uint64_t spv_fnv(const uint32_t *w, uint64_t words) {
    uint64_t h = 0xcbf29ce484222325ull;
    const uint8_t *b = (const uint8_t *)w;
    for (uint64_t i = 0; i < words * 4; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

// LOCAL TEST (Ocarina): TOTK_SPV_OVERRIDE=<dir>: a shader whose hash has <dir>/sh_<hash>.spv uses that SPIR-V instead.
static uint32_t *spv_override(const uint32_t *w, uint64_t words, uint64_t *out_words) {
    static const char *dir;
    static int init;
    if (!init) dir = getenv("TOTK_SPV_OVERRIDE"), init = 1;
    if (!dir || !*dir) return NULL;
    char path[600];
    snprintf(path, sizeof path, "%s/sh_%016llx.spv", dir, (unsigned long long)spv_fnv(w, words));
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *o = malloc((size_t)n);
    if (!o || fread(o, 1, (size_t)n, f) != (size_t)n) rt_fatal("spv override: cannot read %s", path);
    fclose(f);
    *out_words = (uint64_t)n / 4;
    rt_log("vk_pipeline: LOCAL TEST: shader override %s", path);
    return o;
}

// LOCAL TEST (Ocarina): FMin, FMax and FClamp become NMin, NMax and NClamp, whose NaN operand gives the other one, as on the
// console: a material's min(0 * (1 / 0), 1) came out NaN on NVIDIA and left the world black. A copy of the module, or NULL
// when nothing changes. ponytail: not compute shaders, so their big upscaler stays in the driver's cache; add them if a
// compute pass shows NaN.
static uint32_t *spirv_nan_minmax(const uint32_t *w, uint64_t n) {
    uint32_t glsl = 0;
    uint32_t *o = NULL;
    for (uint64_t i = 5; i < n && w[i] >> 16;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (i + len > n) break;
        if (op == 15 && w[i + 1] == 5) {   // OpEntryPoint GLCompute
            free(o);
            return NULL;
        }
        if (op == 11 && len >= 3 && !memcmp(&w[i + 2], "GLSL.std.450", 12)) glsl = w[i + 1];
        if (op == 12 && len >= 5 && glsl && w[i + 3] == glsl) {
            uint32_t to = w[i + 4] == 37 ? 79 : w[i + 4] == 40 ? 80 : w[i + 4] == 43 ? 81 : 0;
            if (to) {
                if (!o) {
                    if (!(o = malloc((size_t)n * 4))) rt_fatal("vk_pipeline: out of memory");
                    memcpy(o, w, (size_t)n * 4);
                }
                o[i + 4] = to;
            }
        }
        i += len;
    }
    return o;
}

static const uint32_t *fs_spirv_over(const GfxShader *fs, uint64_t *words) {   // the override is kept: a few shaders
    const uint32_t *w = gfx_shader_spirv(fs, words);
    uint64_t n = 0;
    uint32_t *o = w ? spv_override(w, *words, &n) : NULL;
    if (o) *words = n;
    return o ? o : w;
}

static VkShaderModule make_module_linked(const uint32_t *w, uint64_t words, int geometry_chain) {
    uint64_t over_n = 0;
    uint32_t *over = spv_override(w, words, &over_n);
    if (over) w = over, words = over_n;
    uint32_t *minmax = spirv_nan_minmax(w, words);
    if (minmax) w = minmax;
    uint64_t fixed_n = 0;
    uint32_t *fixed = spirv_is_mesh(w, words) ? NULL : geometry_chain ? spirv_fixup_linked(w, words, &fixed_n, 1) : spirv_fixup(w, words, &fixed_n);
    if (fixed) w = fixed, words = fixed_n;
    VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = (size_t)words * 4, .pCode = w};
    VkShaderModule m = VK_NULL_HANDLE;
    VkResult r = vkCreateShaderModule(P.device, &ci, NULL, &m);
    free(fixed);
    free(minmax);
    free(over);
    return r == VK_SUCCESS ? m : VK_NULL_HANDLE;
}

static VkShaderModule make_module(const uint32_t *w, uint64_t words) {
    return make_module_linked(w, words, 0);
}

// Experimental GS accepts a deliberately bounded interface: unpacked 32-bit
// scalar/vectors, the GS outer vertex array, and numeric built-in blocks. Parse
// the same fixup used at upload, rather than infer widths/counts from Locations.
// Vulkan Shader Interfaces: the outer GS array is disregarded for matching and
// component limits include built-ins. Device checks use VkPhysicalDeviceLimits.
typedef struct {
    uint32_t op, x, y, type, storage, location, builtin;
    uint8_t component, listed, block, flat, invariant, interpolation, unsupported;
} GeometryId;
typedef struct {
    uint32_t mask, components;
    GfxShaderIo io[32];
    uint8_t invariant[32];
    uint64_t builtin_mask, builtin[64];
    uint64_t standalone_builtins;
    uint8_t block_members, block_invariant, block_builtin[16];
    uint8_t system_inputs; // bit 0: subgroup lane ID, bit 1: subgroup less-than mask; neither is VS-linked
} GeometryInterface;

// These standalone GS inputs are supplied by the implementation, not the VS.
// Match the exact U32 forms emitted by Ryujinx IoMap, without putting large
// BuiltIn IDs into the per-vertex 64-bit mask or consuming vertex components.
static int geometry_system_input(const GeometryId *id, uint32_t bound, const GeometryId *var,
                                 uint32_t type, uint8_t *seen) {
    uint32_t builtin = var->builtin - 1;
    uint8_t bit = builtin == 41 ? 1 : builtin == 4420 ? 2 : 0;
    if (!bit || (*seen & bit) || var->location || var->component || var->invariant ||
        var->interpolation || var->unsupported || var->block || !type || type >= bound) return 0;
    if (bit == 2) {
        if (id[type].op != 23 || id[type].y != 4) return 0;
        type = id[type].x;
        if (!type || type >= bound) return 0;
    }
    if (id[type].op != 21 || id[type].x != 32 || id[type].y) return 0;
    *seen |= bit;
    return 1;
}

// Reject known subgroup extension families without rejecting unrelated shader
// extensions. The literal must terminate within its instruction, with zero pad.
static int geometry_subgroup_extension(const uint32_t *words, uint32_t count) {
    if (!count) return 0;
    const char *name = (const char *)words;
    size_t bytes = (size_t)count * 4;
    const char *end = memchr(name, 0, bytes);
    if (!end || end == name) return 0;
    for (const char *p = end + 1; p < name + bytes; p++) if (*p) return 0;
    static const char *tokens[] = {"subgroup", "sub_group", "group_non_uniform", "shader_ballot"};
    for (unsigned t = 0; t < sizeof tokens / sizeof *tokens; t++) {
        size_t length = strlen(tokens[t]);
        for (const char *p = name; (size_t)(end - p) >= length; p++) {
            size_t i;
            for (i = 0; i < length; i++) {
                unsigned char c = (unsigned char)p[i];
                if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
                if (c != (unsigned char)tokens[t][i]) break;
            }
            if (i == length) return 0;
        }
    }
    return 1;
}

// Capability support applies even without an instruction; stage support is a
// separate requirement for actual subgroup-scoped group operations (Vulkan
// SPIR-V appendix, VUID-RuntimeSpirv-None-06343). The required-size pin is not
// evidence for either. Admit only the group families currently emitted here.
static int geometry_subgroups(const GfxShader *s, uint8_t system_inputs) {
    uint64_t words = 0;
    const uint32_t *w = gfx_shader_spirv(s, &words);
    if (!w || words < 5 || !w[3] || w[3] > (1u << 16)) return 0;
    uint32_t bound = w[3], declared = 0, operations = 0;
    GeometryId *id = calloc(bound, sizeof *id);
    int ok = 0;
    if (!id) return 0;
    for (uint64_t i = 5; i < words;) {
        uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (!n || n > words - i) goto done;
        const uint32_t *a = w + i + 1;
        if (op == 17) {
            if (n != 2) goto done;
            if (a[0] >= 61 && a[0] <= 68) declared |= 1u << (a[0] - 61);
            else if (a[0] == 58 || a[0] == 4423 || a[0] == 4431 || a[0] == 5297 ||
                     (a[0] >= 5568 && a[0] <= 5570) || a[0] == 5579 ||
                     (a[0] >= 5696 && a[0] <= 5698)) goto done;
        } else if (op == 10) {
            if (!geometry_subgroup_extension(a, n - 1)) goto done;
        } else if (op == 21 && n == 4 && a[0] < bound) {
            id[a[0]].op = op; id[a[0]].x = a[1]; id[a[0]].y = a[2];
        } else if (op == 43 && n == 4 && a[1] < bound) {
            id[a[1]].op = op; id[a[1]].type = a[0]; id[a[1]].x = a[2];
        } else if (op >= 333 && op <= 366) {
            if (n < 5 || op == 333 || op > 348 || !a[2] || a[2] >= bound) goto done;
            uint32_t expected = (op <= 336 || op == 338 || op == 339 || op == 340 || op == 343 || op == 344) ? 5 : 6;
            if (n != expected) goto done;
            const GeometryId *scope = &id[a[2]];
            if (scope->op != 43 || scope->x != 3 || !scope->type || scope->type >= bound ||
                id[scope->type].op != 21 || id[scope->type].x != 32 || id[scope->type].y) goto done;
            operations |= op <= 336 ? VK_SUBGROUP_FEATURE_VOTE_BIT :
                op <= 344 ? VK_SUBGROUP_FEATURE_BALLOT_BIT : op <= 346 ? VK_SUBGROUP_FEATURE_SHUFFLE_BIT :
                VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
        } else if ((op >= 259 && op <= 271) || op == 5296 || op == 224 ||
                   (op >= 4421 && op <= 4422) || (op >= 4428 && op <= 4430) || op == 4432 ||
                   (op >= 5000 && op <= 5007) || (op >= 5571 && op <= 5578) ||
                   (op >= 5580 && op <= 5581) || (op >= 5713 && op <= 5816)) goto done;
        i += n;
    }
    // The specialized GroupNonUniform capabilities imply GroupNonUniform;
    // Ryujinx does not repeat that base declaration in generated modules.
    if (declared) declared |= VK_SUBGROUP_FEATURE_BASIC_BIT;
    uint32_t required = (system_inputs & 1 ? VK_SUBGROUP_FEATURE_BASIC_BIT : 0) |
                        (system_inputs & 2 ? VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT : 0);
    if ((required | operations) & ~declared) goto done;
    // The selected system-input subset also requires reported GS support.
    // This is a conservative admission policy, not a claim that a builtin load
    // itself is a group operation covered by VUID 06343.
    if ((declared & ~P.subgroup_operations) ||
        ((system_inputs || operations) && !(P.subgroup_stages & VK_SHADER_STAGE_GEOMETRY_BIT))) goto done;
    ok = 1;
done:
    free(id);
    return ok;
}

static int geometry_numeric(const GeometryId *id, uint32_t bound, uint32_t type, int array,
                            uint32_t *components, uint64_t *signature, GfxShaderIo *io) {
    if (!type || type >= bound) return 0;
    const GeometryId *t = &id[type];
    if (array && t->op == 28) {
        if (!t->y || t->y >= bound || id[t->y].op != 43 || !id[t->y].type || id[t->y].type >= bound ||
            id[id[t->y].type].op != 21 || id[id[t->y].type].x != 32 || !id[t->y].x || id[t->y].x > 4096) return 0;
        uint32_t count;
        uint64_t element;
        GfxShaderIo ignored;
        if (!geometry_numeric(id, bound, t->x, 0, &count, &element, &ignored)) return 0;
        *components = count * id[t->y].x;
        *signature = element | (uint64_t)id[t->y].x << 32 | UINT64_C(1) << 63;
        return 1;
    }
    uint32_t count = 1;
    if (t->op == 23) {
        count = t->y;
        if (count < 2 || count > 4 || !t->x || t->x >= bound) return 0;
        t = &id[t->x];
    }
    if (t->op == 21 && t->y > 1) return 0;
    uint32_t kind = t->op == 22 ? 1 : t->op == 21 ? (t->y ? 2 : 3) : t->op == 20 ? 4 : 0;
    if (!kind || (kind != 4 && t->x != 32) || (kind == 4 && count != 1)) return 0;
    *components = count;
    *signature = kind | (uint64_t)(kind == 4 ? 1 : 32) << 8 | (uint64_t)count << 16;
    *io = (GfxShaderIo){(uint8_t)(kind - 1), (uint8_t)count, 0, (uint8_t)(kind == 4 ? 1 : 32)};
    return 1;
}

static int geometry_member(const uint32_t *w, uint64_t words, uint32_t type, uint32_t member,
                           uint32_t *builtin, uint32_t *invariant) {
    int found = 0;
    *invariant = 0;
    for (uint64_t i = 5; i < words;) {
        uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (!n || n > words - i) return 0;
        if (op == 72 && n >= 4 && w[i + 1] == type && w[i + 2] == member) {
            if (w[i + 3] == 11) {
                if (n != 5 || found++ || w[i + 4] >= 64) return 0;
                *builtin = w[i + 4];
            } else if (w[i + 3] == 18 && n == 4) *invariant = 1;
            else if (w[i + 3] != 0 || n != 4) return 0;
        }
        i += n;
    }
    return found == 1;
}

static int geometry_interface(const GfxShader *s, uint32_t storage, uint32_t vertices, GeometryInterface *out) {
    memset(out, 0, sizeof *out);
    uint64_t words = 0, fixed_words = 0;
    const uint32_t *w = gfx_shader_spirv(s, &words);
    if (!w || words < 5 || words > (1u << 20) || !w[3] || w[3] > (1u << 16)) return 0;
    int stage = gfx_shader_stage(s);
    uint32_t *fixed = spirv_fixup_linked(w, words, &fixed_words, stage == GFX_STAGE_VS);
    if (fixed) w = fixed, words = fixed_words;
    uint32_t bound = w[3], entries = 0;
    GeometryId *id = calloc(bound, sizeof *id);
    int ok = 0;
    if (!id) goto done;
    for (uint64_t i = 5; i < words;) {
        uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (!n || n > words - i) goto done;
        const uint32_t *a = w + i + 1;
        if (op == 15) {
            if (n < 4 || entries++ || a[0] != (stage == GFX_STAGE_VS ? 0u : stage == GFX_STAGE_GS ? 3u : 4u)) goto done;
            const char *name = (const char *)(a + 2), *end = memchr(name, 0, (n - 3) * 4);
            if (!end || end - name != 4 || memcmp(name, "main", 4)) goto done;
            uint32_t start = 2 + ((uint32_t)(end - name) + 4) / 4;
            for (uint32_t j = start; j < n - 1; j++) {
                if (!a[j] || a[j] >= bound || id[a[j]].listed) goto done;
                id[a[j]].listed = 1;
            }
        } else if (op == 17 && n == 2) {
            if ((a[0] == 24 && !P.f_gs_point_size) || (a[0] == 32 && !P.f_clip) || (a[0] == 33 && !P.f_cull) ||
                a[0] == 53 || a[0] == 54) goto done;       // transform feedback / geometry streams not supported here
        } else if (op == 220 || op == 221) goto done;    // stream emission needs a different contract
        else if (op == 73 || op == 74 || op == 75 || op == 332) goto done; // indirect decorations are not modeled
        else if (op == 71 && n >= 3 && a[0] < bound) {
            GeometryId *d = &id[a[0]];
            if (a[1] == 30) { if (n != 4 || a[2] >= 32 || d->location) goto done; d->location = a[2] + 1; }
            else if (a[1] == 11) { if (n != 4 || (a[2] >= 64 && a[2] != 4420) || d->builtin) goto done; d->builtin = a[2] + 1; }
            else if (a[1] == 31) d->component = 1;
            else if (a[1] == 2) d->block = 1;
            else if (a[1] == 14) d->flat = d->interpolation = 1;
            else if (a[1] == 13 || a[1] == 16 || a[1] == 17) d->interpolation = 1;
            else if (a[1] == 18) d->invariant = 1;
            else if (a[1] == 15 || a[1] == 5271 || a[1] == 5285) goto done;
            else if (a[1] != 0) d->unsupported = 1;
        } else if ((op == 20 || op == 21 || op == 22 || op == 23 || op == 28 || op == 30 || op == 32) && n >= 2 && a[0] < bound) {
            GeometryId *t = &id[a[0]];
            t->op = op;
            if (op == 30) t->x = (uint32_t)i + 2, t->y = n - 2;
            else if (op == 32) { if (n != 4) goto done; t->x = a[2]; t->y = a[1]; }
            else { if (n >= 3) t->x = a[1]; if (n >= 4) t->y = a[2]; }
        } else if (op == 43 && n == 4 && a[1] < bound) id[a[1]].op = op, id[a[1]].x = a[2], id[a[1]].type = a[0];
        else if (op == 59 && n >= 4 && a[1] < bound) id[a[1]].type = a[0], id[a[1]].storage = a[2];
        i += n;
    }
    if (entries != 1) goto done;
    for (uint32_t v = 1; v < bound; v++) {
        GeometryId *var = &id[v];
        if (!var->listed) continue;
        if (!var->type || var->type >= bound || id[var->type].op != 32 || id[var->type].y != var->storage) goto done;
        if (var->storage != storage) continue;
        uint32_t type = id[var->type].x;
        if (!type || type >= bound || var->component || var->unsupported ||
            (var->builtin && var->interpolation)) goto done;
        if (var->builtin == 42 || var->builtin == 4421) {
            if (stage != GFX_STAGE_GS || storage != 1 ||
                !geometry_system_input(id, bound, var, type, &out->system_inputs)) goto done;
            continue;
        }
        if (stage == GFX_STAGE_GS && storage == 1 && id[type].op == 28) {
            uint32_t length = id[type].y;
            if (!length || length >= bound || id[length].op != 43 || id[length].x != vertices ||
                !id[length].type || id[length].type >= bound || id[id[length].type].op != 21 ||
                id[id[length].type].x != 32) goto done;
            type = id[type].x;
            if (!type || type >= bound) goto done;
        } else if (stage == GFX_STAGE_GS && storage == 1 && var->builtin != 8 && var->builtin != 9) goto done;
        if (id[type].op == 30) {
            if (!id[type].block || id[type].unsupported || id[type].invariant || id[type].interpolation ||
                var->interpolation || var->location || var->builtin ||
                !id[type].y || id[type].y > 16 || out->block_members) goto done;
            out->block_members = (uint8_t)id[type].y;
            out->block_invariant = var->invariant;
            for (uint32_t member = 0; member < id[type].y; member++) {
                uint32_t builtin = 0, invariant = 0, components;
                uint64_t signature;
                GfxShaderIo ignored;
                if (!geometry_member(w, words, type, member, &builtin, &invariant) ||
                    !geometry_numeric(id, bound, w[id[type].x + member], 1, &components, &signature, &ignored) ||
                    (out->builtin_mask & (UINT64_C(1) << builtin))) goto done;
                out->builtin_mask |= UINT64_C(1) << builtin;
                out->block_builtin[member] = (uint8_t)builtin;
                out->builtin[builtin] = signature | (uint64_t)invariant << 62;
                out->components += components;
            }
        } else {
            uint32_t components;
            uint64_t signature;
            GfxShaderIo io;
            if (!geometry_numeric(id, bound, type, var->builtin != 0, &components, &signature, &io)) goto done;
            if (var->location && !var->builtin) {
                uint32_t location = var->location - 1;
                if (io.scalar_bits != 32 || io.type > GFX_TYPE_UINT || (out->mask >> location & 1) ||
                    (stage == GFX_STAGE_FS && io.type != GFX_TYPE_FLOAT && !var->flat)) goto done;
                out->mask |= 1u << location;
                out->io[location] = io;
                out->invariant[location] = var->invariant;
            } else if (var->builtin && !var->location) {
                uint32_t builtin = var->builtin - 1;
                if (out->builtin_mask & (UINT64_C(1) << builtin)) goto done;
                out->builtin_mask |= UINT64_C(1) << builtin;
                out->standalone_builtins |= UINT64_C(1) << builtin;
                out->builtin[builtin] = signature | (uint64_t)var->invariant << 62;
            } else goto done;
            out->components += components;
        }
    }
    GfxShaderIo reflected[32];
    uint32_t mask = storage == 1 ? gfx_shader_inputs(s, reflected) : gfx_shader_outputs(s, reflected);
    if (mask != out->mask) goto done;
    for (unsigned i = 0; i < 32; i++) if (mask >> i & 1)
        if (reflected[i].packed || reflected[i].scalar_bits != 32 || reflected[i].type != out->io[i].type ||
            reflected[i].comps != out->io[i].comps) goto done;
    ok = 1;
done:
    free(id), free(fixed);
    return ok;
}

static int geometry_link(const GeometryInterface *producer, const GeometryInterface *consumer) {
    if (consumer->mask & ~producer->mask) return 0;
    for (unsigned i = 0; i < 32; i++) if (consumer->mask >> i & 1)
        if (memcmp(&producer->io[i], &consumer->io[i], sizeof producer->io[i]) ||
            producer->invariant[i] != consumer->invariant[i]) return 0;
    return 1;
}

static int geometry_locations(uint32_t mask, uint32_t limit) {
    for (unsigned i = 0; i < 32; i++) if ((mask >> i & 1) && i >= limit / 4) return 0;
    return 1;
}

static int geometry_builtin_link(const GeometryInterface *producer, const GeometryInterface *consumer) {
    // Built-in blocks match as structures, not as an unordered subset of their
    // BuiltIn IDs (VUID-RuntimeSpirv-OpVariable-08746). GS outer vertex arrays have
    // already been removed by geometry_interface. Unsupported blocks fail there.
    if (consumer->block_members) {
        if (producer->block_members != consumer->block_members ||
            producer->block_invariant != consumer->block_invariant) return 0;
        for (unsigned i = 0; i < consumer->block_members; i++) {
            unsigned builtin = consumer->block_builtin[i];
            if (producer->block_builtin[i] != builtin || producer->builtin[builtin] != consumer->builtin[builtin]) return 0;
        }
    }
    const uint64_t per_vertex = UINT64_C(1) | UINT64_C(1) << 1 | UINT64_C(1) << 3 | UINT64_C(1) << 4;
    return !(consumer->standalone_builtins & per_vertex & ~producer->standalone_builtins);
}

static int geometry_per_vertex(const GeometryInterface *io, const VkPhysicalDeviceLimits *limits) {
    uint32_t distances = 0;
    const uint64_t scalar = 1 | UINT64_C(32) << 8 | UINT64_C(1) << 16;
    for (unsigned i = 0; i < 5; i++) if (io->builtin_mask >> i & 1) {
        uint64_t signature = io->builtin[i] & ~(UINT64_C(1) << 62); // Invariant does not change the type
        if (i == 0) { if (signature != (1 | UINT64_C(32) << 8 | UINT64_C(4) << 16)) return 0; }
        else if (i == 1) { if (signature != scalar) return 0; }
        else if (i == 3 || i == 4) {
            uint32_t n = (uint32_t)(signature >> 32) & 0x3fffffff;
            if (!(signature >> 63) || (uint32_t)signature != scalar || !n ||
                n > (i == 3 ? limits->maxClipDistances : limits->maxCullDistances)) return 0;
            distances += n;
        } else return 0;
    }
    return distances <= limits->maxCombinedClipAndCullDistances;
}

static const char *geometry_validate(const VkPipeState *s, const VkPhysicalDeviceLimits *limits) {
    static const uint8_t input_vertices[] = {0, 1, 2, 4, 3, 6};
    if (!s->vs || !s->gs || gfx_shader_stage(s->vs) != GFX_STAGE_VS ||
        (s->fs && gfx_shader_stage(s->fs) != GFX_STAGE_FS)) return "geometry shader stage mismatch";
    const GfxShaderGeometry *g = gfx_shader_geometry(s->gs);
    if (!P.f_geometry || !g ||
        (g->version != GFX_GEOMETRY_VERSION && g->version != GFX_GEOMETRY_PASSTHROUGH_VERSION) || g->passthrough ||
        !vk_geometry_input(s->prim) || vk_geometry_input(s->prim) != g->input ||
        g->input_vertices != input_vertices[g->input] || g->output < GFX_GEOMETRY_OUTPUT_POINTS ||
        g->output > GFX_GEOMETRY_OUTPUT_TRIANGLE_STRIP) return "geometry input/stage contract differs from draw";
    if (g->version == GFX_GEOMETRY_PASSTHROUGH_VERSION &&
        (!s->fs || g->input != GFX_GEOMETRY_TRIANGLES || g->output != GFX_GEOMETRY_OUTPUT_TRIANGLE_STRIP ||
         g->invocations != 1 || g->max_output_vertices != 3)) return "unsupported portable geometry contract";
    // VUID-VkGraphicsPipelineCreateInfo-shaderTessellationAndGeometryPointSize-08776
    // requires PointSize on every emitted point. A declaration is not proof of
    // those writes. Polygon-mode points can also consume last-stage PointSize
    // depending on polygonModePointSize; that property is not contracted here.
    if (g->output == GFX_GEOMETRY_OUTPUT_POINTS || s->poly.mode == NVN_POLYGON_MODE_POINT)
        return "geometry point-size emission/rasterization is not implemented";
    if (!g->invocations || g->invocations > limits->maxGeometryShaderInvocations || !g->max_output_vertices ||
        g->max_output_vertices > limits->maxGeometryOutputVertices) return "geometry invocation/output-vertex device limit";
    const GfxShader *shaders[] = {s->vs, s->gs};
    for (unsigned j = 0; j < 2; j++) for (int kind = GFX_RES_SSBO; kind < GFX_RES_KINDS; kind++) {
        const GfxShaderRes *res;
        int count = gfx_shader_resources(shaders[j], kind, &res);
        for (int i = 0; i < count; i++)
            if (!P.f_vertex_stores && ((kind == GFX_RES_SSBO && (res[i].flags & 1)) ||
                                      (kind == GFX_RES_IMG && (res[i].flags & 4)))) return "pre-raster stores require vertexPipelineStoresAndAtomics";
    }
    GeometryInterface vo, gi, go, fi;
    if (!geometry_interface(s->vs, 3, 0, &vo)) return "unsupported vertex output interface or missing scalar-width metadata";
    if (!geometry_interface(s->gs, 1, g->input_vertices, &gi)) return "unsupported geometry input interface or missing scalar-width metadata";
    if (!geometry_interface(s->gs, 3, 0, &go)) return "unsupported geometry output interface or missing scalar-width metadata";
    if (!geometry_subgroups(s->gs, gi.system_inputs)) return "unsupported geometry subgroup capability, operation or stage";
    uint64_t per_vertex = UINT64_C(1) | UINT64_C(1) << 1 | UINT64_C(1) << 3 | UINT64_C(1) << 4;
    uint64_t generated = (UINT64_C(1) << 7) | (UINT64_C(1) << 8); // PrimitiveId / InvocationId are not VS outputs
    if ((gi.builtin_mask & ~(per_vertex | generated)) || (go.builtin_mask & ~per_vertex))
        return "geometry built-in outside the supported per-vertex subset";
    const uint64_t scalar_int = 2 | UINT64_C(32) << 8 | UINT64_C(1) << 16;
    for (unsigned i = 7; i <= 8; i++)
        if ((gi.builtin_mask >> i & 1) && gi.builtin[i] != scalar_int) return "geometry invocation/primitive ID type mismatch";
    if (!geometry_per_vertex(&vo, limits) || !geometry_per_vertex(&gi, limits) || !geometry_per_vertex(&go, limits))
        return "geometry per-vertex built-in type/device limit";
    uint64_t distance_components = 0;
    for (unsigned i = 3; i <= 4; i++) {
        if (gi.builtin_mask >> i & 1) distance_components += gi.builtin[i] >> 32 & 0x3fffffff;
        if (go.builtin_mask >> i & 1) distance_components += go.builtin[i] >> 32 & 0x3fffffff;
    }
    if (distance_components > limits->maxCombinedClipAndCullDistances)
        return "geometry combined input/output distance device limit";
    if (!geometry_link(&vo, &gi)) return "vertex-to-geometry varying mismatch";
    if (!geometry_builtin_link(&vo, &gi)) return "vertex-to-geometry built-in block mismatch";
    uint64_t need = gi.builtin_mask & ~generated;
    if ((need & ~vo.builtin_mask) || !(go.builtin_mask & 1)) return "geometry per-vertex built-in mismatch";
    for (unsigned i = 0; i < 64; i++) if ((need >> i & 1) && gi.builtin[i] != vo.builtin[i]) return "geometry built-in type mismatch";
    if (gi.components > limits->maxGeometryInputComponents || go.components > limits->maxGeometryOutputComponents ||
        (uint64_t)go.components * g->max_output_vertices > limits->maxGeometryTotalOutputComponents ||
        vo.components > limits->maxVertexOutputComponents || !geometry_locations(gi.mask, limits->maxGeometryInputComponents) ||
        !geometry_locations(go.mask, limits->maxGeometryOutputComponents) || !geometry_locations(vo.mask, limits->maxVertexOutputComponents))
        return "geometry interface component device limit";
    if (s->fs) {
        if (!geometry_interface(s->fs, 1, 0, &fi)) return "unsupported fragment input interface or missing scalar-width metadata";
        // VUID-VkGraphicsPipelineCreateInfo-pStages-00740 requires the GS to
        // write PrimitiveId on every path when FS reads it. This subset admits
        // no such GS output and must not inherit the rasterizer-generated ID.
        if (fi.builtin_mask & (UINT64_C(1) << 7)) return "geometry-to-fragment PrimitiveId output is not implemented";
        if (!geometry_link(&go, &fi)) return "geometry-to-fragment varying mismatch";
        if (fi.components > limits->maxFragmentInputComponents || !geometry_locations(fi.mask, limits->maxFragmentInputComponents))
            return "fragment input component device limit";
    }
    return NULL;
}

// ---- descriptor layouts

typedef struct {
    uint32_t n;
    VkDescriptorSetLayoutBinding b[MAX_BINDINGS];
    VkPipeBinding pb[MAX_BINDINGS];
} Bindings;

static VkDescriptorType descriptor_type(int kind, const GfxShaderRes *r) {
    int buffer = (r->type & 0xff) == 2;
    switch (kind) {
    case GFX_RES_UBO: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case GFX_RES_SSBO: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case GFX_RES_TEX:
        if (buffer) return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
        if (r->separate) LOG_ONCE("vk_pipeline: a separate texture handle: bound as SAMPLED_IMAGE");
        return r->separate ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    default: return buffer ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
}

static int mesh_vs(const VkPipeState *s) { return s->vs && gfx_shader_is_mesh(s->vs); }   // LOCAL TEST (Ocarina)
// The first entry point of a module is MeshEXT: it takes no vertex stage fixups.
static int spirv_is_mesh(const uint32_t *w, uint64_t n) {
    for (uint64_t i = 5; i < n && w[i] >> 16; i += w[i] >> 16)
        if ((w[i] & 0xffff) == 15) return i + 1 < n && w[i + 1] == 5365;
    return 0;
}

static VkShaderStageFlags stage_bit(int stage) {
    switch (stage) {
    case GFX_STAGE_VS: return VK_SHADER_STAGE_VERTEX_BIT;
    case GFX_STAGE_GS: return VK_SHADER_STAGE_GEOMETRY_BIT;
    case GFX_STAGE_FS: return VK_SHADER_STAGE_FRAGMENT_BIT;
    case GFX_STAGE_CS: return VK_SHADER_STAGE_COMPUTE_BIT;
    default: return 0;
    }
}

// One past the highest binding a shader uses (1 with no resources: binding 0 is the support buffer).
static uint32_t binding_span(const GfxShader *s) {
    uint32_t span = 1;
    for (int k = 0; k < GFX_RES_KINDS; k++) {
        const GfxShaderRes *r;
        int n = gfx_shader_resources(s, k, &r);
        for (int i = 0; i < n; i++)
            if (r[i].binding + (uint32_t)(r[i].count ? r[i].count : 1) > span) span = r[i].binding + (r[i].count ? r[i].count : 1);
    }
    return span;
}

// Adds a stage's resources with bindings b >= 1 moved by off; binding 0 (support buffer) is shared.
static int add_stage(Bindings *B, const GfxShader *s, uint32_t off) {
    int stage = gfx_shader_stage(s);
    VkShaderStageFlags bit = gfx_shader_is_task(s) ? VK_SHADER_STAGE_TASK_BIT_EXT : gfx_shader_is_mesh(s) ? VK_SHADER_STAGE_MESH_BIT_EXT : stage_bit(stage);
    if (!bit) return 0;
    if (gfx_shader_is_task(s)) stage = GFX_STAGE_TCS;   // LOCAL TEST (Ocarina): the vertex slot's tables (vk_exec_draw.c nvn_stage_of)
    if (!B->n) {
        B->b[0] = (VkDescriptorSetLayoutBinding){0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, 0, NULL};
        B->pb[0] = (VkPipeBinding){0, (uint8_t)stage, VK_PIPE_SUPPORT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, 0, NULL};
        B->n = 1;
    }
    B->b[0].stageFlags |= bit;
    B->pb[0].stages |= bit;
    for (int k = 0; k < GFX_RES_KINDS; k++) {
        const GfxShaderRes *r;
        int n = gfx_shader_resources(s, k, &r);
        for (int i = 0; i < n; i++) {
            if (B->n == MAX_BINDINGS || !r[i].binding) return 0;
            VkDescriptorType t = descriptor_type(k, &r[i]);
            uint32_t binding = r[i].binding + off, count = r[i].count ? r[i].count : 1;
            if (binding > UINT16_MAX) return 0;           // VkPipeBinding must not truncate a stage's binding
            B->b[B->n] = (VkDescriptorSetLayoutBinding){binding, t, count, bit, NULL};
            B->pb[B->n] = (VkPipeBinding){(uint16_t)binding, (uint8_t)stage, (uint8_t)k, t, (uint16_t)count, bit, &r[i]};
            B->n++;
        }
    }
    return 1;
}

// The first binding a module's resource variables use that the binding list b[0..n) lacks (module bindings >= 1 moved
// by off), or -1. A reflection that misses a resource would otherwise hand the driver an invalid layout.
static int module_binding_missing(const GfxShader *s, const VkDescriptorSetLayoutBinding *b, uint32_t n, uint32_t off) {
    uint64_t words = 0;
    const uint32_t *w = gfx_shader_spirv(s, &words);
    uint32_t shared[16];   // LOCAL TEST (Ocarina): variables of set 2 are not in this layout's lists
    int nshared = 0;
    for (uint64_t i = 5; w && i < words;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > words) break;
        if (op == SPV_OP_DECORATE && len >= 4 && w[i + 2] == SPV_DEC_DESCRIPTOR_SET && w[i + 3] == 2 && nshared < 16) shared[nshared++] = w[i + 1];
        i += len;
    }
    for (uint64_t i = 5; w && i < words;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > words) break;
        int keep = 0;
        for (int k = 0; k < nshared; k++) keep |= shared[k] == w[i + 1];
        if (op == SPV_OP_DECORATE && len >= 4 && w[i + 2] == SPV_DEC_BINDING && !keep) {
            uint32_t want = w[i + 3] + (w[i + 3] >= 1 ? off : 0), j = 0;
            while (j < n && b[j].binding != want) j++;
            if (j == n) return (int)w[i + 3];
        }
        i += len;
    }
    return -1;
}

// Checks every stage against its layout; 0 with why filled for a missing binding.
static int layout_covers(const VkPipeState *s, const Layout *l, uint32_t fs_off, char *why, size_t cap) {
    uint32_t nv = l->set1 ? l->n0 : l->n;
    int m = module_binding_missing(s->vs, l->b, nv, 0);
    const char *stage = "vertex";
    if (m < 0 && s->ts) {
        m = module_binding_missing(s->ts, l->b, l->n, binding_span(s->vs) - 1);
        stage = "task";
    }
    if (m < 0 && s->gs) {
        if (l->set1) { snprintf(why, cap, "geometry requires a monolithic layout"); return 0; }
        m = module_binding_missing(s->gs, l->b, l->n, binding_span(s->vs) - 1);
        stage = "geometry";
    }
    if (m < 0 && s->fs) {
        m = l->set1 ? module_binding_missing(s->fs, l->b + l->n0, l->n - l->n0, 0) : module_binding_missing(s->fs, l->b, l->n, fs_off);
        stage = "fragment";
    }
    if (m >= 0) snprintf(why, cap, "the %s shader uses binding %d, which its reflection and layout lack", stage, m);
    return m < 0;
}

// The update template of a set whose bindings are b[0..n): one entry per binding, in order, over the packed array the
// replay fills from the set's first descriptor, a VkDescSlot per descriptor. The bind point follows the stages (compute
// layouts hold the compute stage only). VK_NULL_HANDLE when templates are off, for an empty set, for a push set
// without vkCmdPushDescriptorSetWithTemplateKHR or when creation fails: the replay then writes the set with write
// structs.
static VkDescriptorUpdateTemplate make_tpl(const VkDescriptorSetLayoutBinding *b, uint32_t n, VkDescriptorSetLayout sl,
                                           int push, VkPipelineLayout pl, uint32_t set) {
    if (!P.tpl || !n || (push && !P.tpl_push)) return VK_NULL_HANDLE;
    VkDescriptorUpdateTemplateEntry e[MAX_BINDINGS];
    size_t off = 0;
    VkShaderStageFlags stages = 0;
    for (uint32_t i = 0; i < n; i++) {
        e[i] = (VkDescriptorUpdateTemplateEntry){b[i].binding, 0, b[i].descriptorCount, b[i].descriptorType, off, sizeof(VkDescSlot)};
        off += sizeof(VkDescSlot) * b[i].descriptorCount;
        stages |= b[i].stageFlags;
    }
    VkDescriptorUpdateTemplateCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO, .descriptorUpdateEntryCount = n,
        .pDescriptorUpdateEntries = e,
        .templateType = push ? VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR : VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET,
        .descriptorSetLayout = sl,
        .pipelineBindPoint = stages & VK_SHADER_STAGE_COMPUTE_BIT ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS,
        .pipelineLayout = pl, .set = set};
    VkDescriptorUpdateTemplate t;
    if (P.tpl_create(P.device, &ci, NULL, &t) != VK_SUCCESS) {
        LOG_ONCE("vk_pipeline: descriptor update template creation failed: that set is written with write structs");
        return VK_NULL_HANDLE;
    }
    return t;
}

// Finds or creates the layout of a binding list (sorted here); NULL on failure. Caller holds P.lock.
// Sorts a binding list by binding and hashes it into *h; the descriptor total, or -1 when a binding repeats.
static int64_t bindings_sort(Bindings *B, uint64_t *h) {
    for (uint32_t i = 1; i < B->n; i++)                     // insertion sort by binding; the lists are short
        for (uint32_t j = i; j > 0 && B->b[j - 1].binding > B->b[j].binding; j--) {
            VkDescriptorSetLayoutBinding t = B->b[j]; B->b[j] = B->b[j - 1]; B->b[j - 1] = t;
            VkPipeBinding u = B->pb[j]; B->pb[j] = B->pb[j - 1]; B->pb[j - 1] = u;
        }
    int64_t total = 0;
    for (uint32_t i = 0; i < B->n; i++) {
        if (i && B->b[i].binding == B->b[i - 1].binding) return -1;
        uint32_t k[4] = {B->b[i].binding, B->b[i].descriptorType, B->b[i].descriptorCount, B->b[i].stageFlags};
        *h = fnv(*h, k, sizeof k);
        total += B->b[i].descriptorCount;
    }
    return total;
}

static Layout *layout_get(Bindings *B) {
    uint64_t h = FNV0;
    int64_t total = bindings_sort(B, &h);
    if (total < 0) return NULL;
    Layout **head = &layouts[h % LAYOUT_BUCKETS];
    for (Layout *l = *head; l; l = l->next)
        if (l->hash == h && l->n == B->n && !memcmp(l->b, B->b, B->n * sizeof *B->b)) return l;
    Layout *l = calloc(1, sizeof *l + B->n * sizeof *B->b);
    if (!l) return NULL;
    l->hash = h, l->n = B->n;
    memcpy(l->b, B->b, B->n * sizeof *B->b);
    l->push = total <= (int64_t)P.max_push;
    VkDescriptorSetLayoutCreateInfo si = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .flags = l->push ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0,
                                          .bindingCount = l->n, .pBindings = l->b};
    if (vkCreateDescriptorSetLayout(P.device, &si, NULL, &l->set) != VK_SUCCESS) {
        free(l);
        return NULL;
    }
    VkDescriptorSetLayout sets3[3] = {l->set, vk_bindless_empty(), vk_bindless_layout()};
    VkPipelineLayoutCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = sets3[2] ? 3 : 1,
                                     .pSetLayouts = sets3};
    if (vkCreatePipelineLayout(P.device, &pi, NULL, &l->layout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(P.device, l->set, NULL);
        free(l);
        return NULL;
    }
    l->tpl[0] = make_tpl(l->b, l->n, l->set, l->push, l->layout, 0);
    l->next = *head;
    *head = l;
    P.stats.layouts++;
    P.stats.push_layouts += (uint64_t)l->push;
    return l;
}

// Independent layouts (P.indep): the vertex stage's bindings in set 0, the fragment stage's in set 1 (empty without a
// fragment shader), each with its own support buffer at binding 0, the pipeline layout created with
// VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT so stage libraries can use it. Only the P.push_stage set is a
// push-descriptor set (a layout holds one), when it fits maxPushDescriptors; the other is allocated per draw from the
// command buffer's pools. On success *B holds both lists, set 0 first, VkPipeBinding.set filled in. Caller holds P.lock.
static Layout *layout_pair(Bindings *B, Bindings *F) {
    uint64_t h = fnv(FNV0, "indep", 5);
    int64_t tv = bindings_sort(B, &h);
    h = fnv(h, "|", 1);
    int64_t tf = bindings_sort(F, &h);
    if (tv < 0 || tf < 0 || B->n + F->n > MAX_BINDINGS) return NULL;
    uint32_t n0 = B->n, n = B->n + F->n;
    memcpy(B->b + n0, F->b, F->n * sizeof *F->b);
    memcpy(B->pb + n0, F->pb, F->n * sizeof *F->pb);
    for (uint32_t i = n0; i < n; i++) B->pb[i].set = 1;
    B->n = n;
    Layout **head = &layouts[h % LAYOUT_BUCKETS];
    for (Layout *l = *head; l; l = l->next)
        if (l->hash == h && l->set1 && l->n == n && l->n0 == n0 && !memcmp(l->b, B->b, n * sizeof *B->b)) return l;
    Layout *l = calloc(1, sizeof *l + n * sizeof *B->b);
    if (!l) return NULL;
    l->hash = h, l->n = n, l->n0 = n0;
    memcpy(l->b, B->b, n * sizeof *B->b);
    l->push = P.push_stage == GFX_STAGE_VS && tv <= (int64_t)P.max_push;
    l->push1 = P.push_stage == GFX_STAGE_FS && F->n && tf <= (int64_t)P.max_push;
    VkDescriptorSetLayoutCreateInfo s0 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .flags = l->push ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0,
                                          .bindingCount = n0, .pBindings = l->b};
    VkDescriptorSetLayoutCreateInfo s1 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .flags = l->push1 ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0,
                                          .bindingCount = n - n0, .pBindings = l->b + n0};
    if (vkCreateDescriptorSetLayout(P.device, &s0, NULL, &l->set) != VK_SUCCESS) {
        free(l);
        return NULL;
    }
    if (vkCreateDescriptorSetLayout(P.device, &s1, NULL, &l->set1) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(P.device, l->set, NULL);
        free(l);
        return NULL;
    }
    VkDescriptorSetLayout sets[3] = {l->set, l->set1, vk_bindless_layout()};
    VkPipelineLayoutCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .flags = VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT, .setLayoutCount = sets[2] ? 3 : 2,
                                     .pSetLayouts = sets};
    if (vkCreatePipelineLayout(P.device, &pi, NULL, &l->layout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(P.device, l->set, NULL);
        vkDestroyDescriptorSetLayout(P.device, l->set1, NULL);
        free(l);
        return NULL;
    }
    l->tpl[0] = make_tpl(l->b, n0, l->set, l->push, l->layout, 0);
    l->tpl[1] = make_tpl(l->b + n0, n - n0, l->set1, l->push1, l->layout, 1);
    l->next = *head;
    *head = l;
    P.stats.layouts++;
    P.stats.push_layouts += (uint64_t)(l->push | l->push1);
    return l;
}

// Vertex outputs with independent layouts: each vertex shader declares the union of the inputs of every fragment
// shader met with it (pipeline-keys.bin at the prewarm, then each pipeline built), so its module does not depend on the
// fragment shader of one pipeline. Unwritten ones read (0, 0, 0, 1) as before. TOTK_VS_OUTS=all declares all 32.
#define VS_OUTS_CAP 8192
static struct {
    PlatMutex lock;
    const GfxShader *vs[VS_OUTS_CAP];
    uint32_t mask[VS_OUTS_CAP];
    GfxShaderIo io[VS_OUTS_CAP][32];
    uint32_t n, full;
} VO = {.lock = PLAT_MUTEX_INIT};

// Adds fs's inputs (packed ones excepted) to vs's union; returns the union, types in io (the first type met wins).
static uint32_t vs_outs(const GfxShader *vs, const GfxShader *fs, GfxShaderIo io[32]) {
    GfxShaderIo fin[32];
    uint32_t need = fs ? gfx_shader_inputs(fs, fin) : 0, mask;
    for (int i = 0; i < 32; i++)
        if (need >> i & 1 && fin[i].packed) need &= ~(1u << i);
    plat_mutex_lock(&VO.lock);
    uint32_t j = (uint32_t)(((uintptr_t)vs >> 4) * 0x9e3779b1u) & (VS_OUTS_CAP - 1);
    while (VO.vs[j] && VO.vs[j] != vs) j = (j + 1) & (VS_OUTS_CAP - 1);
    if (!VO.vs[j] && VO.n + 1 >= VS_OUTS_CAP * 3 / 4) {   // full: this pair alone
        VO.full++;
        plat_mutex_unlock(&VO.lock);
        LOG_ONCE("vk_pipeline: vertex output union table full: outputs from the pair");
        memset(io, 0, 32 * sizeof *io);
        for (int i = 0; i < 32; i++)
            if (need >> i & 1) io[i] = fin[i];
        return need;
    }
    if (!VO.vs[j]) VO.vs[j] = vs, VO.n++;
    for (int i = 0; i < 32; i++)
        if (!(need >> i & 1)) continue;
        else if (!(VO.mask[j] >> i & 1)) VO.io[j][i] = fin[i], VO.mask[j] |= 1u << i;
        else if (VO.io[j][i].type == fin[i].type && VO.io[j][i].comps < fin[i].comps) VO.io[j][i].comps = fin[i].comps;
    mask = VO.mask[j];
    memcpy(io, VO.io[j], 32 * sizeof *io);
    plat_mutex_unlock(&VO.lock);
    return mask;
}

// ---- pipeline cache table

static Pipe *pipe_find(const void *key, size_t size, uint64_t h) {
    for (Pipe *p = pipes[h % PIPE_BUCKETS]; p; p = p->next)
        if (p->hash == h && p->key_size == size && !memcmp(p->key, key, size)) return p;
    return NULL;
}

// Stores a result (a failed one too, so it is not retried every draw). Caller holds P.lock.
static Pipe *pipe_add(const void *key, size_t size, uint64_t h, VkPipeline pipeline, const Layout *l, const Bindings *B,
                      VkPipelineBindPoint bp, uint32_t vs_inputs) {
    Pipe *p = calloc(1, sizeof *p + size);
    if (!p) return NULL;
    p->hash = h, p->key_size = size;
    memcpy(p->key, key, size);
    if (l && B) {
        p->bindings = malloc(B->n * sizeof *p->bindings);
        if (p->bindings) memcpy(p->bindings, B->pb, B->n * sizeof *p->bindings);
    }
    p->info = (VkPipeInfo){pipeline, l ? l->layout : VK_NULL_HANDLE, l ? l->set : VK_NULL_HANDLE, l ? l->push : 0, bp,
                           p->bindings ? B->n : 0, p->bindings, vs_inputs};
    p->info.set_layout1 = l ? l->set1 : VK_NULL_HANDLE;
    p->info.push1 = l ? l->push1 : 0;
    p->info.n_sets = l && l->set1 ? 2 : 1;
    if (l) p->info.tpl[0] = l->tpl[0], p->info.tpl[1] = l->tpl[1];
    p->lay = l;
    Pipe **head = &pipes[h % PIPE_BUCKETS];
    p->next = *head;
    *head = p;
    return p;
}

// ---- graphics pipelines

typedef struct {                       // normalized: zero where the state is dynamic or unused
    const GfxShader *vs, *fs, *gs, *ts;
    uint32_t color[8], depth, stencil; // VkFormat
    uint8_t n_color, samples, topo_class, minus_w;
    uint8_t clamp, poly_mode, a2c, logic_op;   // logic_op: VkLogicOp + 1 when enabled
    uint8_t blend_en, dsb, fbl, alpha;  // dsb: an enabled target blends with a SRC1 factor; fbl: VkPipeState.feedback;
                                       // alpha: NVN_ALPHA_FUNC_* of an alpha test (spirv_alpha_test), else 0
    uint32_t wmask;
    uint8_t blend[8][6];               // src, dst, op, src alpha, dst alpha, alpha op (VkBlendFactor / VkBlendOp)
    uint32_t attr_fmt[32], attr_off[32];
    uint8_t attr_bind[32];
    uint8_t rate[17];                  // per binding: 0 unused, 1 vertex, 2 instance
    uint8_t indep;                     // P.indep when built: the probe draws through both layouts
    uint8_t geometry_input, pad2;
    float alpha_ref;                   // 0 without an alpha test
} GKey;

static int blendable(VkFormat f) {
    return f != VK_FORMAT_UNDEFINED && (format_features(f, 1) & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT);
}

static int target_count(const VkPipeState *s) {
    int n = 0;
    for (int i = 0; i < 8; i++)
        if (s->color_fmt[i]) n = i + 1;
    return n;
}

// SRC1 blend factors read the fragment shader's second output (dual-source blending); the pipeline then needs a
// fragment module whose output 1 is Location 0, Index 1 (spirv_dual_source).
static int is_src1(unsigned f) { return f >= NVN_BLEND_FUNC_SRC1_COLOR && f <= NVN_BLEND_FUNC_ONE_MINUS_SRC1_ALPHA; }
// Runs per draw (gkey_fill, target_wmask): only the blend factors of blend-enabled targets are read until one is SRC1,
// which TotK never sets (0 of 876 pipelines in the field and fight saves).
static int uses_src1(const VkPipeState *s) {
    for (uint32_t m = s->color.blend_en & 0xffu; m; m &= m - 1) {
        int i = __builtin_ctz(m);
        const blend_st *b = &s->blend[i];
        if ((is_src1(b->src_rgb) || is_src1(b->dst_rgb) || is_src1(b->src_a) || is_src1(b->dst_a)) &&
            i < target_count(s) && blendable(target_format(s->color_fmt[i], NULL, NULL)))
            return 1;
    }
    return 0;
}

// Colour write mask of target i: NVN's channel mask, zero for a target outside the fragment shader's output map (SPH
// omap). Maxwell leaves such a target untouched (nxnative 04-graphics.md, "Targets the fragment shader does not
// write"); Vulkan writes undefined values, and on AMD they lit grass in front of Link as flat lime. Zeros are wrong
// too: TOTK's half-res transparency pass binds its linear depth (R32F) as target 7, samples it in the same draws and
// writes only targets 0-2, 4 and 5; zeroed, the depth that later passes rebuild from it was 0 in tiles, and the fog
// over distant geometry showed rectangles of clear sky.
static uint32_t target_wmask(const VkPipeState *s, int i) {
    uint32_t m = s->chmask.mask >> (4 * i) & 0xf;
    if (s->fs && !(gfx_shader_outputs(s->fs, NULL) >> i & 1)) m = 0;
    if (s->fs) m &= gfx_shader_output_comps(s->fs, i);   // the channels it does not write stay, as on Maxwell
    if (i == 1 && s->fs && P.f_dual_src && uses_src1(s)) m = 0;   // output 1 is the second blend source (spirv_dual_source)
    return m;
}

uint32_t vk_pipeline_target_wmask(const VkPipeState *s, int i) { return target_wmask(s, i); }

static void gkey_fill(const VkPipeState *s, GKey *k) {
    memset(k, 0, sizeof *k);
    k->vs = s->vs, k->fs = s->fs, k->gs = s->gs, k->ts = s->ts;
    k->indep = s->gs || mesh_vs(s) ? 0 : (uint8_t)P.indep;
    k->geometry_input = s->gs ? vk_geometry_input(s->prim) : 0;
    k->n_color = (uint8_t)target_count(s);
    for (int i = 0; i < k->n_color; i++) k->color[i] = target_format(s->color_fmt[i], NULL, NULL);
    int d, st;
    VkFormat df = target_format(s->depth_fmt, &d, &st);
    k->depth = d ? df : VK_FORMAT_UNDEFINED;
    k->stencil = st ? df : VK_FORMAT_UNDEFINED;
    k->samples = s->samples > 1 ? s->samples : 1;
    k->topo_class = (uint8_t)topology_class(vk_pipeline_topology(s->prim));
    k->minus_w = s->depth_minus_w != 0;
    k->dsb = s->fs && P.f_dual_src && uses_src1(s);
    k->fbl = s->fs ? s->feedback : 0;
    if (s->fs && s->color.alpha_func >= NVN_ALPHA_FUNC_NEVER && s->color.alpha_func < NVN_ALPHA_FUNC_ALWAYS)
        k->alpha = (uint8_t)s->color.alpha_func, k->alpha_ref = s->alpha_ref;
    if (!P.dyn_clamp) k->clamp = s->depth_clamp && P.f_depth_clamp;
    if (!P.dyn_poly) k->poly_mode = (uint8_t)polygon_mode(s->poly.mode);
    if (!P.dyn_a2c) k->a2c = s->ms.a2c != 0;
    if (P.f_logic_op && s->color.logic_op != NVN_LOGIC_OP_COPY) k->logic_op = (uint8_t)(s->color.logic_op + 1);
    for (int i = 0; i < k->n_color; i++) {
        int on = (s->color.blend_en >> i & 1) && blendable(k->color[i]);
        if (!P.dyn_blend_en && on) k->blend_en |= (uint8_t)(1u << i);
        if (!P.dyn_wmask) k->wmask |= target_wmask(s, i) << (4 * i);
        if (!P.dyn_blend_eq && on) {
            const blend_st *b = &s->blend[i];
            uint8_t *o = k->blend[i];
            o[0] = (uint8_t)blend_factor(b->src_rgb), o[1] = (uint8_t)blend_factor(b->dst_rgb), o[2] = (uint8_t)blend_op(b->eq_rgb);
            o[3] = (uint8_t)blend_factor(b->src_a), o[4] = (uint8_t)blend_factor(b->dst_a), o[5] = (uint8_t)blend_op(b->eq_a);
        }
    }
    if (!P.dyn_vi) {
        VtxInput v;
        vertex_input(s, &v);
        for (uint32_t i = 0; i < v.n_attr; i++) {
            uint32_t l = v.a[i].location;
            k->attr_fmt[l] = v.a[i].format, k->attr_off[l] = v.a[i].offset, k->attr_bind[l] = (uint8_t)v.a[i].binding;
        }
        for (uint32_t i = 0; i < v.n_bind; i++)
            k->rate[v.b[i].binding] = v.b[i].inputRate == VK_VERTEX_INPUT_RATE_INSTANCE ? 2 : 1;
    }
}

// Independent layouts: the outputs the vertex shader lacks from its union with s->fs's inputs added (vs_outs), types in
// fin. The vertex module (and its pre-rasterization library) depends on the vertex shader and this mask only.
static uint32_t vs_union_missing(const VkPipeState *s, GfxShaderIo fin[32]) {
    GfxShaderIo vout[32];
    uint32_t need = vs_outs(s->vs, s->fs, fin), have = gfx_shader_outputs(s->vs, vout), missing = 0;
    if (s->fs) {   // a location another fragment shader reads as another type: this pair's type (rare)
        GfxShaderIo fi[32];
        uint32_t fneed = gfx_shader_inputs(s->fs, fi);
        for (int i = 0; i < 32; i++)
            if (fneed >> i & 1 && !fi[i].packed &&
                (!(need >> i & 1) || fin[i].type != fi[i].type || fin[i].comps < fi[i].comps)) {
                if (need >> i & 1 && fin[i].type != fi[i].type)
                    LOG_ONCE("vk_pipeline: varying %d read as type %d and %d by fragment shaders of one vertex shader", i, fin[i].type, fi[i].type);
                fin[i] = fi[i], need |= 1u << i;
            }
    }
    if (P.vs_outs_all)
        for (int i = 0; i < 32; i++)
            if (!(need >> i & 1)) fin[i] = (GfxShaderIo){GFX_TYPE_FLOAT, 4, 0, 32}, need |= 1u << i;
    for (int i = 0; i < 32; i++)
        if (need >> i & 1 && !(have >> i & 1)) missing |= 1u << i;
    return missing;
}

#define PIPE_NOOPT_WORDS_AMD 0   // TOTK_PIPE_NOOPT_WORDS default on AMD: off. Unoptimized big pipelines lost the device on the RX 5500 XT (driver 26.10.1)

// The big compiles left for the next start and the extra compile workers below rely on finished pipelines being kept.
// Without VK_KHR_pipeline_binary (an older driver, TOTK_PIPE_BINARIES=0) a compile left for later would never be kept,
// so those cards keep the behaviour they had.
static int amd_keeps_binaries(void) {
    VkCtx *c = vk_ctx();
    return vk_caps.amd_sc && c && c->has_pipeline_binary;
}

// ---- compile memory gate
// AMD's Windows compiler takes gigabytes for Ocarina's big pipelines: six workers compiling at once took a 16 GB PC
// from 10 to 33 GB committed, left 44 MB of RAM free, the driver crashed and the PC rebooted (RX 5500 XT, driver
// 26.10.1). A pipeline or library compile starts only while the system keeps TOTK_PIPE_MIN_FREE_MB of RAM free
// (default: an eighth of the RAM, at least 1536 MB), unless no other compile is running, so one always goes on.
// TOTK_PIPE_MIN_FREE_MB=0 turns it off. Builds from a pipeline binary and fast links do not pass through it. Only on
// drivers with amd_keeps_binaries. A compile on the replay thread holding P.lock never waits here (tl_holds_plock): the
// picture would freeze for as long as the other compile runs. It still counts as running, so the others wait for it.
// Tested: docs/test-environment-rx5500xt.md (preparation: 1380 pipelines and 50 big compute shaders, free RAM never
// below 2.3 GB).
static struct {
    PlatMutex m;
    PlatCond cv;
    int running;                       // compiles between mg_enter and mg_leave
    int big_running;                   // of them, big ones (big_pipeline): one at a time in play
    uint64_t big_last_ns;              // when the last big one started
    int64_t min_mb;                    // -1 not read yet, 0 off
    uint32_t waited, logged;
} MG = {.m = PLAT_MUTEX_INIT, .cv = PLAT_COND_INIT, .min_mb = -1};

// Before the guest starts (the prewarm; TOTK_PREWARM_ONLY) big compiles run side by side: another starts once the last
// one had 45 s to grow and TOTK_PIPE_PREP_FREE_MB (default 7168) of RAM and of commit (RAM + page file) are still free.
// One takes 1-10 minutes and grows to about 4-5 GB with the game closed. With 5120 MB of RAM, 30 s and no commit check,
// the free RAM fell to 285 MB and the free commit to 299 MB (16 GB, a 7 GB page file that Windows grows on its own).
static float mg_free_mb(void);
static double mg_commit_free_mb(void) {
    double proc, used, limit;
    plat_commit_info(&proc, &used, &limit);
    return limit > 0 && used >= 0 ? limit - used : -1;
}
static int mg_big_parallel_ok(void) {
    static int64_t free_mb = -1;
    if (free_mb < 0) {
        const char *e = getenv("TOTK_PIPE_PREP_FREE_MB");
        free_mb = e && *e ? strtoll(e, NULL, 0) : 7168;
    }
    if (rt_guest_start_ns || plat_now_ns() - MG.big_last_ns < 45000000000ull) return 0;
    float f = mg_free_mb();
    double c = mg_commit_free_mb();
    return f >= 0 && f >= (float)free_mb && (c < 0 || c >= (double)free_mb);
}

static float mg_free_mb(void) {
    float used, total, proc;
    plat_mem_info(&used, &total, &proc);
    return total > 0 && used >= 0 ? total - used : -1;
}

static _Thread_local int tl_holds_plock;   // this thread compiles synchronously with P.lock held (pipe_miss, vk_pipeline_compute)

static void mg_enter(int big) {
    if (!amd_keeps_binaries()) return;
    plat_mutex_lock(&MG.m);
    if (MG.min_mb < 0) {
        const char *e = getenv("TOTK_PIPE_MIN_FREE_MB");
        float used, total, proc;
        plat_mem_info(&used, &total, &proc);
        int64_t def = total > 0 ? (int64_t)(total / 8) : 1536;
        MG.min_mb = e && *e ? strtoll(e, NULL, 0) : def < 1536 ? 1536 : def;
        if (MG.min_mb < 0) MG.min_mb = 0;
        if (MG.min_mb) rt_log("vk_pipeline: a compile waits while another runs and less than %lld MB of RAM is free",
                              (long long)MG.min_mb);
    }
    float f = 0;
    int waits = 0;
    uint64_t t0 = 0;
    double cf;
    // Commit too: with the page file at its current size Windows refuses allocations at the limit, and a driver
    // allocation that fails mid-compile crashed amdvlk64.dll.
    while (!tl_holds_plock &&
           ((big && MG.big_running > 0 && !mg_big_parallel_ok()) ||
            (MG.min_mb && MG.running > 0 && (f = mg_free_mb()) >= 0 && f < (float)MG.min_mb) ||
            (MG.min_mb && MG.running > 0 && (cf = mg_commit_free_mb()) >= 0 && cf < (double)MG.min_mb))) {
        if (!waits++) {
            t0 = plat_now_ns(), MG.waited++;
            if (MG.logged < 8)
                MG.logged++, rt_log("vk_pipeline: compile held: %.0f MB of RAM free, %d compiles running (%d big)%s",
                                    mg_free_mb(), MG.running, MG.big_running, big ? "; this one is big" : "");
        }
        plat_cond_wait_ms(&MG.cv, &MG.m, 250);
    }
    if (waits && MG.logged < 8) rt_log("vk_pipeline: compile released after %.1f s", (double)(plat_now_ns() - t0) / 1e9);
    MG.running++, MG.big_running += big != 0;
    if (big) MG.big_last_ns = plat_now_ns();
    plat_mutex_unlock(&MG.m);
}

static void mg_leave(int big) {
    if (!amd_keeps_binaries()) return;
    plat_mutex_lock(&MG.m);
    MG.running--, MG.big_running -= big != 0;
    plat_cond_broadcast(&MG.cv);
    plat_mutex_unlock(&MG.m);
}

// ---- big compiles left for the next start
// A graphics pipeline, or the vertex or fragment shader library of one, of TOTK_PIPE_BIG_WORDS SPIR-V words or more
// (default 40000 on AMD, 0 = off elsewhere) is big. Ocarina's ~65,000-word fragment shaders took AMD's Windows
// compiler 4-5 minutes and gigabytes each: with 16 GB the game alone holds about 10 GB, and one such compile in play took
// the free RAM from 2.2 GB to 160 MB in 7 s (RX 5500 XT, driver 26.10.1). So:
// - big compiles run one at a time (mg_enter);
// - TOTK_PREWARM_ONLY=1, which jogar\jogar-ocarina.ps1 runs (jogar\preparar-shaders.ps1) before it opens the game,
//   builds them, and the pipeline binary keeps the result;
// - while the guest runs one starts only with TOTK_PIPE_BIG_FREE_MB (default 10240) of RAM free. Otherwise it is left
//   for the next preparation: the pipeline's key goes to pipeline-keys.bin and its draws are skipped this run. The
//   game's own boot prewarm leaves them too, so a game started without the preparation keeps skipping them (logged once).
// TOTK_PIPE_BIG_WORDS=0 turns all of this off.
static _Thread_local int tl_deferred;   // the last create_graphics / lib_create on this thread left its compile
static struct {
    int64_t words, free_mb;            // -1 not read yet
    volatile int32_t deferred;
} BIG = {-1, -1, 0};

static int big_pipeline(uint64_t words) {
    if (BIG.words < 0) {   // read once; racing threads read the same values
        const char *w = getenv("TOTK_PIPE_BIG_WORDS"), *f = getenv("TOTK_PIPE_BIG_FREE_MB");
        int64_t fm = f && *f ? strtoll(f, NULL, 0) : 10240;
        BIG.free_mb = fm < 0 ? 0 : fm;
        // 40000: a 44,000-word pipeline compiled in play crashed amdvlk64.dll in a pipe worker with the free RAM at 433 MB
        int64_t bw = w && *w ? strtoll(w, NULL, 0) : amd_keeps_binaries() ? 40000 : 0;
        BIG.words = bw < 0 ? 0 : bw;
        if (BIG.words)
            rt_log("vk_pipeline: compiles of %lld SPIR-V words or more run one at a time; in play they wait for the next "
                   "start unless %lld MB of RAM is free", (long long)BIG.words, (long long)BIG.free_mb);
    }
    return BIG.words > 0 && words >= (uint64_t)BIG.words;
}

// 1 (and tl_deferred set) when a big compile must wait for the next start.
// Above TOTK_PREP_CS_MAX_WORDS (default 1,000,000) never in play, whatever the free RAM: the title screen's compute
// shaders of 1.4-7.2 million words take the driver minutes and over 30 GB; one started with 10.7 GB free and took the
// free RAM to 21 MB in a minute. The preparation leaves them out too, so their dispatches stay skipped.
static uint64_t prep_cs_max_words(void) {
    static uint64_t w = ~0ull;
    if (w == ~0ull) {
        const char *e = getenv("TOTK_PREP_CS_MAX_WORDS");
        w = e && *e ? strtoull(e, NULL, 0) : 1000000u;
    }
    return w;
}

// Only TOTK_PREWARM_ONLY (the preparation, the game closed) compiles big ones at once. The game's own boot prewarm
// leaves them too: started with 10 GB free before the guest, three ran on into play after TOTK_PREWARM_WAIT_MS.
static int big_defer(uint64_t words, const char *what) {
    if (!big_pipeline(words)) return 0;
    static int prep = -1;
    if (prep < 0) {
        const char *po = getenv("TOTK_PREWARM_ONLY");
        prep = po && !strcmp(po, "1");
    }
    if (prep) return 0;
    float f = mg_free_mb();
    if (rt_guest_start_ns && words <= prep_cs_max_words() && (f < 0 || f >= (float)BIG.free_mb)) return 0;
    tl_deferred = 1;
    {
        static volatile int32_t said;
        if (plat_inc32(&said) == 1)
            rt_log("vk_pipeline: big pipelines (%lld SPIR-V words or more) are built only by jogar\\preparar-shaders.ps1 "
                   "(TOTK_PREWARM_ONLY=1) or in play with %lld MB of RAM free. Started any other way, their draws and dispatches "
                   "are skipped on every run", (long long)BIG.words, (long long)BIG.free_mb);
    }
    if (plat_inc32(&BIG.deferred) <= 16)
        rt_log("vk_pipeline: %s of %llu words left for the next start (%.0f MB of RAM free, %lld wanted); its draws are "
               "skipped this run", what, (unsigned long long)words, f, (long long)BIG.free_mb);
    return 1;
}

// Compile threads the RAM allows on AMD: one per 8 GB (16 GB, 16315 MB usable -> 2), at least 1.
static int mg_compilers_by_ram(void) {
    float used, total, proc;
    plat_mem_info(&used, &total, &proc);
    int n = total > 0 ? (int)((total + 1024) / 8192) : 2;
    return n < 1 ? 1 : n;
}

// ---- pipeline binaries (VK_KHR_pipeline_binary, vk_device.c has_pipeline_binary)
// Built only with Vulkan headers that have the extension (1.3.296+); older ones (Ubuntu 24.04's 1.3.280) get the stubs
// after the section, and has_pipeline_binary stays 0.
#ifndef VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR
#define VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR 0x80000000ULL
#endif
#ifdef VK_KHR_pipeline_binary
// A graphics pipeline that took TOTK_PIPE_BINARY_MS (500; 0 on AMD) or more to compile is kept as the driver's own binaries in
// <exe dir>/pipeline-binaries/<global key>/<pipeline key>.bin, and later runs create it from them without compiling.
// AMD's Windows driver keeps 4 MiB of pipelines per application on disk and dropped Ocarina's big ones: three took it
// 290-301 s each at every start, and new areas met more (RX 5500 XT, driver 26.10.1). The global key changes with the
// driver, which starts a new folder; a binary the driver refuses falls back to a normal compile.
// Tested: docs/test-environment-rx5500xt.md.
#define PB_MAGIC 0x4e494250u   // "PBIN"
#define PB_MAX 16
static struct {
    int state;   // 0 not set up, 1 on, -1 off
    char dir[PLAT_PATH_MAX * 2];
    uint32_t save_ms;
    PFN_vkGetPipelineKeyKHR key;
    PFN_vkCreatePipelineBinariesKHR create;
    PFN_vkGetPipelineBinaryDataKHR data;
    PFN_vkReleaseCapturedPipelineDataKHR release;
    PFN_vkDestroyPipelineBinaryKHR destroy;
    volatile int32_t loaded, saved, refused;
} PB;
static PlatMutex pb_lock = PLAT_MUTEX_INIT;

static void pb_hex(const uint8_t *b, uint32_t n, char *out) {
    static const char d[] = "0123456789abcdef";
    for (uint32_t i = 0; i < n; i++) out[2 * i] = d[b[i] >> 4], out[2 * i + 1] = d[b[i] & 15];
    out[2 * n] = 0;
}

static int pb_on(void) {
    int st = __atomic_load_n(&PB.state, __ATOMIC_ACQUIRE);
    if (st) return st > 0;
    plat_mutex_lock(&pb_lock);
    if (!PB.state) {
        VkCtx *c = vk_ctx();
        int on = 0;
        if (c && amd_keeps_binaries() && P.cache_path[0]) {
            PB.key = (PFN_vkGetPipelineKeyKHR)vkGetDeviceProcAddr(c->device, "vkGetPipelineKeyKHR");
            PB.create = (PFN_vkCreatePipelineBinariesKHR)vkGetDeviceProcAddr(c->device, "vkCreatePipelineBinariesKHR");
            PB.data = (PFN_vkGetPipelineBinaryDataKHR)vkGetDeviceProcAddr(c->device, "vkGetPipelineBinaryDataKHR");
            PB.release = (PFN_vkReleaseCapturedPipelineDataKHR)vkGetDeviceProcAddr(c->device, "vkReleaseCapturedPipelineDataKHR");
            PB.destroy = (PFN_vkDestroyPipelineBinaryKHR)vkGetDeviceProcAddr(c->device, "vkDestroyPipelineBinaryKHR");
            VkPipelineBinaryKeyKHR gk = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
            if (PB.key && PB.create && PB.data && PB.release && PB.destroy && PB.key(c->device, NULL, &gk) == VK_SUCCESS &&
                gk.keySize && gk.keySize <= VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR) {
                char hex[2 * VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR + 1];
                pb_hex(gk.key, gk.keySize, hex);
                snprintf(PB.dir, sizeof PB.dir, "%s", P.cache_path);
                char *cut = plat_last_sep(PB.dir);
                if (cut) {
                    *cut = 0;
                    size_t l = strlen(PB.dir);
                    snprintf(PB.dir + l, sizeof PB.dir - l, "/pipeline-binaries/%.16s", hex);
                    plat_file_mkdir_p(PB.dir);
                    // On AMD every compiled pipeline is kept (0 ms): a binary loads in a few ms with no compiler
                    // memory, and its compile took gigabytes of RAM (compile memory gate); a few hundred KB each on disk.
                    const char *e = getenv("TOTK_PIPE_BINARY_MS");
                    PB.save_ms = e && *e ? (uint32_t)strtoul(e, NULL, 0) : 0u;
                    on = plat_dir_exists(PB.dir);
                }
            }
        }
        if (on) rt_log("vk_pipeline: graphics pipelines that compile in %u ms or more are kept as binaries in %s", PB.save_ms, PB.dir);
        else rt_log("vk_pipeline: pipeline binaries off");
        __atomic_store_n(&PB.state, on ? 1 : -1, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&pb_lock);
    return PB.state > 0;
}

// The file of the pipeline pci (a VkGraphicsPipelineCreateInfo or VkComputePipelineCreateInfo) would create; 0 when
// the driver gives no key.
static int pb_path(const void *pci, char *path, size_t cap) {
    VkPipelineCreateInfoKHR ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_INFO_KHR, .pNext = (void *)pci};
    VkPipelineBinaryKeyKHR k = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
    if (PB.key(P.device, &ci, &k) != VK_SUCCESS || !k.keySize || k.keySize > VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR) return 0;
    char hex[2 * VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR + 1];
    pb_hex(k.key, k.keySize, hex);
    return snprintf(path, cap, "%s/%s.bin", PB.dir, hex) < (int)cap;
}

// File: u32 magic, u32 count, then per binary u32 key size, 32 key bytes, u64 data size, data.
// A graphics pipeline (gi) or a compute one (cci, gi NULL).
static VkPipeline pb_load(VkGraphicsPipelineCreateInfo *gi, VkComputePipelineCreateInfo *cci, const char *path) {
    uint64_t size = 0;
    uint8_t *f = plat_file_read(path, &size);
    if (!f) return VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkPipelineBinaryKeyKHR keys[PB_MAX];
    VkPipelineBinaryDataKHR datas[PB_MAX];
    uint32_t magic = 0, count = 0, n = 0;
    uint64_t o = 8;
    if (size >= 8) memcpy(&magic, f, 4), memcpy(&count, f + 4, 4);
    if (magic != PB_MAGIC || !count || count > PB_MAX) goto out;
    for (; n < count; n++) {
        uint32_t ks;
        uint64_t ds;
        if (o + 4 + VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR + 8 > size) goto out;
        memcpy(&ks, f + o, 4), o += 4;
        if (!ks || ks > VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR) goto out;
        keys[n] = (VkPipelineBinaryKeyKHR){.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR, .keySize = ks};
        memcpy(keys[n].key, f + o, VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR), o += VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR;
        memcpy(&ds, f + o, 8), o += 8;
        if (!ds || ds > size - o) goto out;
        datas[n] = (VkPipelineBinaryDataKHR){.dataSize = (size_t)ds, .pData = f + o};
        o += ds;
    }
    {
        VkPipelineBinaryKeysAndDataKHR kd = {.binaryCount = n, .pPipelineBinaryKeys = keys, .pPipelineBinaryData = datas};
        VkPipelineBinaryCreateInfoKHR bci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR, .pKeysAndDataInfo = &kd};
        VkPipelineBinaryKHR bins[PB_MAX];
        VkPipelineBinaryHandlesInfoKHR hi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR,
                                             .pipelineBinaryCount = n, .pPipelineBinaries = bins};
        if (PB.create(P.device, &bci, NULL, &hi) == VK_SUCCESS) {
            const void **pn = gi ? &gi->pNext : &cci->pNext;
            VkPipelineBinaryInfoKHR bi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR, .pNext = *pn,
                                          .binaryCount = hi.pipelineBinaryCount, .pPipelineBinaries = bins};
            const void *chain = *pn;
            *pn = &bi;
            VkResult cr = gi ? vkCreateGraphicsPipelines(P.device, VK_NULL_HANDLE, 1, gi, NULL, &pipe)
                             : vkCreateComputePipelines(P.device, VK_NULL_HANDLE, 1, cci, NULL, &pipe);
            if (cr != VK_SUCCESS) pipe = VK_NULL_HANDLE;
            *pn = chain;
            for (uint32_t i = 0; i < hi.pipelineBinaryCount; i++) PB.destroy(P.device, bins[i], NULL);
        }
    }
out:
    free(f);
    return pipe;
}

static void pb_release(VkPipeline pipe) {
    VkReleaseCapturedPipelineDataInfoKHR ri = {.sType = VK_STRUCTURE_TYPE_RELEASE_CAPTURED_PIPELINE_DATA_INFO_KHR, .pipeline = pipe};
    PB.release(P.device, &ri, NULL);
}

// pipe was created with VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR: its binaries go to path, then the data is released.
static int pb_save(VkPipeline pipe, const char *path) {
    int ok = 0;
    VkPipelineBinaryCreateInfoKHR bci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR, .pipeline = pipe};
    VkPipelineBinaryHandlesInfoKHR hi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
    VkPipelineBinaryKHR bins[PB_MAX];
    uint32_t n = 0;
    if (PB.create(P.device, &bci, NULL, &hi) == VK_SUCCESS && hi.pipelineBinaryCount && hi.pipelineBinaryCount <= PB_MAX) {
        hi.pPipelineBinaries = bins;
        if (PB.create(P.device, &bci, NULL, &hi) == VK_SUCCESS) n = hi.pipelineBinaryCount;
    }
    uint8_t *buf = NULL;
    size_t len = 8, cap = 0;
    for (uint32_t i = 0; i < n; i++) {
        VkPipelineBinaryDataInfoKHR di = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR, .pipelineBinary = bins[i]};
        VkPipelineBinaryKeyKHR k = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
        size_t ds = 0;
        if (PB.data(P.device, &di, &k, &ds, NULL) != VK_SUCCESS || !ds) goto done;
        size_t need = len + 4 + VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR + 8 + ds;
        if (need > cap) {
            uint8_t *nb = realloc(buf, need * 2);
            if (!nb) goto done;
            buf = nb, cap = need * 2;
        }
        uint8_t *p = buf + len;
        uint64_t ds64 = ds;
        if (PB.data(P.device, &di, &k, &ds, p + 4 + VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR + 8) != VK_SUCCESS) goto done;
        memcpy(p, &k.keySize, 4);
        memcpy(p + 4, k.key, VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR);
        memcpy(p + 4 + VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR, &ds64, 8);
        len = need;
    }
    if (n && buf) {
        uint32_t magic = PB_MAGIC;
        memcpy(buf, &magic, 4), memcpy(buf + 4, &n, 4);
        ok = plat_file_write_atomic(path, buf, len);
    }
done:
    free(buf);
    for (uint32_t i = 0; i < n; i++) PB.destroy(P.device, bins[i], NULL);
    pb_release(pipe);
    return ok;
}
#else
static struct { uint32_t save_ms; volatile int32_t loaded, saved, refused; } PB;
static int pb_on(void) { return 0; }
static int pb_path(const void *pci, char *path, size_t cap) { (void)pci, (void)path, (void)cap; return 0; }
static VkPipeline pb_load(VkGraphicsPipelineCreateInfo *gi, VkComputePipelineCreateInfo *cci, const char *path) {
    (void)gi, (void)cci, (void)path;
    return VK_NULL_HANDLE;
}
static void pb_release(VkPipeline pipe) { (void)pipe; }
static int pb_save(VkPipeline pipe, const char *path) { (void)pipe, (void)path; return 0; }
#endif

// pipeline-crash.txt next to pipeline-keys.bin: the crash note of each graphics or compute pipeline the driver crashed in
// (platform last-chance filter), a line per crash. An RX 6750 XT crashed in amdvlk64.dll building a mesh pipeline
// early in play. Crashed once: built without optimization (a mesh pipeline: not built). Twice: not built, its draws
// are skipped. Deleting the file tries them all again.
static char *crashed;   // the file's text; written once at init (keys_open), read by the build threads
static void crashed_open(const char *keys_path) {
    static char path[1024];
    snprintf(path, sizeof path, "%s", keys_path);
    char *slash = strrchr(path, '\\'), *fslash = strrchr(path, '/');
    if (fslash > slash) slash = fslash;
    snprintf(slash ? slash + 1 : path, sizeof path - (size_t)(slash ? slash + 1 - path : 0), "pipeline-crash.txt");
    plat_crash_file(path);
    FILE *f = fopen(path, "rb");
    if (!f) return;
    char *t = calloc(1, 1 << 16);
    size_t n = t ? fread(t, 1, (1 << 16) - 1, f) : 0;
    fclose(f);
    if (!n) {
        free(t);
        return;
    }
    crashed = t;
    int lines = 0;
    for (size_t i = 0; i < n; i++) lines += t[i] == '\n';
    rt_log("vk_pipeline: %s: %d crash%s in the driver; those pipelines are built without optimization, or skipped after "
           "a second crash", path, lines, lines == 1 ? "" : "es");
}
static int crash_count(const char *note) {
    int n = 0;
    size_t len = strlen(note);
    for (const char *at = crashed; at && (at = strstr(at, note)); at += len)
        n += (at == crashed || at[-1] == '\n') && (at[len] == '\n' || at[len] == '\r');
    return n;
}

static uint64_t big_cs_words(void);
// LOCAL TEST (Ocarina, AMD): set by a pool worker around a cache probe of a large job: the build fails with
// VK_PIPELINE_COMPILE_REQUIRED instead of compiling when the pipeline is not in the cache (pipelineCreationCacheControl).
static _Thread_local int gfx_probe;
static VkPipeline create_graphics(const VkPipeState *s, const GKey *k, const Layout *l, uint32_t fs_off,
                                  VkPipelineCache cache) {
    if (gfx_shader_is_task(s->vs) || (s->ts && !mesh_vs(s))) {   // a task shader needs its mesh stage (PR #29, qiyuey)
        rt_log("vk_pipeline: task shader requires a mesh stage");
        return VK_NULL_HANDLE;
    }
    if (s->gs && (l->set1 || !k->geometry_input || k->geometry_input > GFX_GEOMETRY_TRIANGLES_ADJACENCY)) return VK_NULL_HANDLE;
    VkShaderModule vm = VK_NULL_HANDLE, fm = VK_NULL_HANDLE, gm = VK_NULL_HANDLE;
    uint64_t vw, fw;
    const uint32_t *vcode = gfx_shader_spirv(s->vs, &vw);
    uint32_t *vpatched = NULL, *fpatched = NULL;
    if (!s->gs && l->set1) {   // independent layouts: outputs from the vertex shader's union
        GfxShaderIo fin[32];
        uint32_t missing = vs_union_missing(s, fin);
        if (missing) {
            uint64_t n2 = 0;
            vpatched = spirv_add_outputs(vcode, vw, missing, fin, &n2);
            if (!vpatched) LOG_ONCE("vk_pipeline: cannot add missing vertex outputs (mask %#x)", missing);
            else vcode = vpatched, vw = n2;
        }
    }
    if (s->fs) {
        if (!s->gs) {
            GfxShaderIo fin[32], vout[32];
            uint32_t need = gfx_shader_inputs(s->fs, fin), have = gfx_shader_outputs(s->vs, vout), missing = 0;
            for (int i = 0; i < 32; i++) {
                if (!(need >> i & 1) || fin[i].packed) continue;
                if (!(have >> i & 1)) missing |= 1u << i;
                else if (vout[i].type != fin[i].type)
                    LOG_ONCE("vk_pipeline: varying %d is type %d in the vertex stage, %d in the fragment stage", i, vout[i].type, fin[i].type);
            }
            if (missing && !l->set1) {
                uint64_t n2 = 0;
                vpatched = spirv_add_outputs(vcode, vw, missing, fin, &n2);
                if (!vpatched) LOG_ONCE("vk_pipeline: cannot add missing vertex outputs (mask %#x)", missing);
                else vcode = vpatched, vw = n2;
            }
        }
        const uint32_t *fcode = fs_spirv_over(s->fs, &fw);
        if (k->alpha && fcode) {   // before spirv_dual_source, which adds a second output at Location 0
            uint64_t n2 = 0;
            uint32_t *a = spirv_alpha_test(fcode, fw, k->alpha, k->alpha_ref, &n2);
            if (a) fcode = fpatched = a, fw = n2;
            else LOG_ONCE("vk_pipeline: alpha test but the fragment shader has no float vec4 output at location 0");
        }
        if (k->dsb) {
            uint64_t n2 = 0;
            uint32_t *d = spirv_dual_source(fcode, fw, &n2);
            if (d) free(fpatched), fcode = fpatched = d, fw = n2;
            else LOG_ONCE("vk_pipeline: SRC1 blend factors but the fragment shader writes no colour output 1");
        }
        if (fcode && mesh_vs(s)) {   // LOCAL TEST (Ocarina): the mesh stage's per-primitive outputs
            uint64_t mwn = 0, n2 = 0;
            const uint32_t *mw = gfx_shader_spirv(s->vs, &mwn);
            uint32_t *pm = mw ? spirv_mark_prim_inputs(fcode, fw, spirv_prim_mask(mw, mwn), &n2) : NULL;
            if (pm) free(fpatched), fcode = fpatched = pm, fw = n2;
        }
        if (l->set1 || fs_off) {
            uint32_t *r = !fcode ? NULL : l->set1 ? spirv_set(fcode, fw, 1) : spirv_rebind(fcode, fw, fs_off);
            free(fpatched);
            fcode = fpatched = r;
        }
        // A bound colour target outside the shader's output map gets write mask 0 (target_wmask)
        if (fcode) fm = make_module(fcode, fw);
    }
    if (s->gs) {
        uint64_t gw = 0;
        const uint32_t *gcode = gfx_shader_spirv(s->gs, &gw);
        uint32_t off = binding_span(s->vs) - 1;
        uint32_t *patched = off && gcode ? spirv_rebind(gcode, gw, off) : NULL;
        if (gcode && (!off || patched)) gm = make_module(patched ? patched : gcode, gw);
        free(patched);
    }
    // Only a VS linked to GS uses the GS-chain dead-distance declaration policy.
    // Plain VS/FS modules retain their original fixup path and uploaded bytes.
    vm = make_module_linked(vcode, vw, s->gs != NULL);
    free(vpatched), free(fpatched);
    VkShaderModule tm = VK_NULL_HANDLE;   // LOCAL TEST (Ocarina): the task stage, bindings after the mesh stage's
    if (s->ts) {
        uint64_t tw = 0;
        const uint32_t *tcode = gfx_shader_spirv(s->ts, &tw);
        uint32_t off = binding_span(s->vs) - 1;
        uint32_t *patched = off && tcode ? spirv_rebind(tcode, tw, off) : NULL;
        if (tcode && (!off || patched)) tm = make_module(patched ? patched : tcode, tw);
        free(patched);
    }
    VkPipeline pipe = VK_NULL_HANDLE;
    if (!vm || (s->fs && !fm) || (s->gs && !gm) || (s->ts && !tm)) goto out;

    VkPipelineShaderStageCreateInfo stages[4];
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rq[4];
    const GfxShader *shaders[4] = {s->vs, s->gs, s->fs, s->ts};
    VkShaderModule modules[4] = {vm, gm, fm, tm};
    uint32_t stage_count = 0;
    for (int i = 0; i < 4; i++) if (shaders[i]) {
        VkPipelineShaderStageCreateInfo *st = &stages[stage_count];
        *st = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = gfx_shader_is_task(shaders[i]) ? VK_SHADER_STAGE_TASK_BIT_EXT : gfx_shader_is_mesh(shaders[i]) ? VK_SHADER_STAGE_MESH_BIT_EXT : stage_bit(gfx_shader_stage(shaders[i])),
            .module = modules[i], .pName = "main"};
        uint64_t sw = 0;
        const uint32_t *scode = gfx_shader_spirv(shaders[i], &sw);
        pin_subgroup32(st, &rq[stage_count], scode, sw);
        stage_count++;
    }
    {
        // Vulkan needs a vertex or a mesh stage (VUID-VkGraphicsPipelineCreateInfo-stage-02096). A vertex slot whose
        // shader resolved to another stage gave the driver a pipeline without one: NVIDIA rejects it, AMD's can fault.
        int pre_raster = 0;
        for (uint32_t i = 0; i < stage_count; i++)
            pre_raster |= stages[i].stage == VK_SHADER_STAGE_VERTEX_BIT || stages[i].stage == VK_SHADER_STAGE_MESH_BIT_EXT;
        if (!pre_raster) {
            LOG_ONCE("vk_pipeline: a graphics pipeline has no vertex or mesh stage (vertex slot is stage %#x); not built",
                     stage_count ? (unsigned)stages[0].stage : 0u);
            goto out;
        }
    }
    VkVertexInputAttributeDescription va[32];
    VkVertexInputBindingDescription vb[17];
    uint32_t na = 0, nb = 0;
    if (!P.dyn_vi) {
        VtxInput v;
        vertex_input(s, &v);
        for (uint32_t i = 0; i < v.n_attr; i++) va[na++] = (VkVertexInputAttributeDescription){v.a[i].location, v.a[i].binding, v.a[i].format, v.a[i].offset};
        for (uint32_t i = 0; i < v.n_bind; i++) vb[nb++] = (VkVertexInputBindingDescription){v.b[i].binding, v.b[i].stride, v.b[i].inputRate};
    }
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                                               .vertexBindingDescriptionCount = nb, .pVertexBindingDescriptions = vb,
                                               .vertexAttributeDescriptionCount = na, .pVertexAttributeDescriptions = va};
    static const VkPrimitiveTopology class_base[4] = {VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                                                      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_PATCH_LIST};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = class_base[k->topo_class]};
    if (s->gs) {
        static const VkPrimitiveTopology gs_base[] = {VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
            VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY};
        ia.topology = gs_base[k->geometry_input];
    }
    VkPipelineViewportDepthClipControlCreateInfoEXT clip = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
                                                            .negativeOneToOne = k->minus_w};
    VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .pNext = &clip};
    // NVN's default is the last vertex (SetProvokingVertexMode, nvn_ext.c); Vulkan's is the first.
    VkPipelineRasterizationProvokingVertexStateCreateInfoEXT pvs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
        .provokingVertexMode = VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .pNext = P.pv_last ? &pvs : NULL,
                                                 .depthClampEnable = k->clamp, .polygonMode = (VkPolygonMode)k->poly_mode,
                                                 .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)k->samples,
                                               .alphaToCoverageEnable = k->a2c};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                                                .maxDepthBounds = 1.0f};
    VkPipelineColorBlendAttachmentState cba[8];
    memset(cba, 0, sizeof cba);
    for (int i = 0; i < k->n_color; i++) {
        const uint8_t *b = k->blend[i];
        cba[i].blendEnable = k->blend_en >> i & 1;
        cba[i].srcColorBlendFactor = (VkBlendFactor)b[0], cba[i].dstColorBlendFactor = (VkBlendFactor)b[1];
        cba[i].colorBlendOp = (VkBlendOp)b[2], cba[i].srcAlphaBlendFactor = (VkBlendFactor)b[3];
        cba[i].dstAlphaBlendFactor = (VkBlendFactor)b[4], cba[i].alphaBlendOp = (VkBlendOp)b[5];
        cba[i].colorWriteMask = k->wmask >> (4 * i) & 0xf;
    }
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .logicOpEnable = k->logic_op != 0,
                                              .logicOp = k->logic_op ? (VkLogicOp)(k->logic_op - 1) : VK_LOGIC_OP_COPY,
                                              .attachmentCount = k->n_color, .pAttachments = cba};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = P.n_dyn, .pDynamicStates = P.dyn};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = k->n_color,
                                        .pColorAttachmentFormats = (const VkFormat *)k->color,
                                        .depthAttachmentFormat = (VkFormat)k->depth, .stencilAttachmentFormat = (VkFormat)k->stencil};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
                                       .stageCount = stage_count, .pStages = stages, .pVertexInputState = &vi,
                                       .pInputAssemblyState = &ia, .pViewportState = &vp, .pRasterizationState = &rs,
                                       .pMultisampleState = &ms, .pDepthStencilState = &ds, .pColorBlendState = &cb,
                                       .pDynamicState = &dy, .layout = l->layout};
    if (k->fbl & 1) pi.flags |= VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
    if (k->fbl & 2) pi.flags |= VK_PIPELINE_CREATE_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
    VkDynamicState mesh_dyn[sizeof P.dyn / sizeof *P.dyn];
    if (mesh_vs(s)) {   // no vertex input or input assembly state in a mesh pipeline, static or dynamic
        uint32_t n = 0;
        for (uint32_t i = 0; i < P.n_dyn; i++)
            if (P.dyn[i] != VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY && P.dyn[i] != VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE &&
                P.dyn[i] != VK_DYNAMIC_STATE_VERTEX_INPUT_EXT && P.dyn[i] != VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE)
                mesh_dyn[n++] = P.dyn[i];
        dy.dynamicStateCount = n, dy.pDynamicStates = mesh_dyn;
        pi.pVertexInputState = NULL, pi.pInputAssemblyState = NULL;
    }
    uint64_t p0 = perf_begin();
    static _Thread_local char note[200];   // named in the log if the driver crashes in this create (platform last-chance filter)
    uint64_t total_words = 0;
    {
        const GfxShader *st[4] = {s->ts, s->vs, s->gs, s->fs};
        int at = snprintf(note, sizeof note, "graphics pipeline, mesh %d task %d:", mesh_vs(s), s->ts != NULL);
        for (int i = 0; i < 4 && at < (int)sizeof note; i++) {
            uint64_t w = 0, h = 1469598103934665603ull;   // FNV-1a of the SPIR-V: the same shader across runs
            const uint32_t *c = st[i] ? gfx_shader_spirv(st[i], &w) : NULL;
            for (uint64_t j = 0; c && j < w; j++) h = (h ^ c[j]) * 1099511628211ull;
            if (c) total_words += w;
            if (c) at += snprintf(note + at, sizeof note - at, " %s %llu words %016llx", (const char *[]){"ts", "vs", "gs", "fs"}[i],
                                  (unsigned long long)w, (unsigned long long)h);
        }
    }
    // TOTK_PIPE_NOOPT_WORDS=<n> (0 = off): pipelines whose stages total n SPIR-V words or more are built without
    // optimization. AMD's Windows compiler took minutes for a few of Ocarina's pipelines and kept none of them in its
    // caches, so every boot and every first visit waited for them again (RX 5500 XT, driver 26.10.1).
    static uint64_t noopt_words = ~0ull;
    if (noopt_words == ~0ull) {
        const char *e = getenv("TOTK_PIPE_NOOPT_WORDS");
        noopt_words = e && *e ? strtoull(e, NULL, 0) : vk_caps.amd_sc ? PIPE_NOOPT_WORDS_AMD : 0;
        if (noopt_words) rt_log("vk_pipeline: graphics pipelines of %llu SPIR-V words or more are built without optimization",
                                (unsigned long long)noopt_words);
    }
    int noopt = noopt_words && total_words >= noopt_words;
    if (noopt) pi.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    int crashes = crash_count(note);
    // A mesh pipeline is skipped after one crash: AMD's driver (amdvlk64.dll+0xf3c1af on an RX 9070 XT, #153, #157)
    // crashed in the same mesh pipeline without optimization too, so the game only started on the third try.
    if (crashes >= (mesh_vs(s) ? 1 : 2)) {
        static volatile int32_t told;
        if (plat_inc32(&told) <= 8) rt_log("vk_pipeline: not built, the driver crashed in it before: %s", note);
        perf_end(PERF_PIPE, p0);
        goto out;
    }
    if (crashes) {
        pi.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
        rt_log("vk_pipeline: built without optimization, the driver crashed in it before: %s", note);
    }
    uint64_t gwords = 0;
    {   // LOCAL TEST (Ocarina): OOT_BIG_GFX_NOOPT=1 builds a guarded graphics pipeline (vertex + fragment of big_cs_words()
        // or more, OOT_BIG_GFX_BINDINGS bindings or more) without optimization. Off by default on every GPU: on an RX 580 the
        // unoptimized build was not faster (8-10 minutes each) and the result ran slower.
        static int opt = -1;
        if (opt < 0) opt = getenv("OOT_BIG_GFX_NOOPT") == NULL;
        const GfxShader *st[4] = {s->ts, s->vs, s->gs, s->fs};
        for (int i = 0; i < 4; i++) {
            uint64_t w = 0;
            if (st[i]) gfx_shader_spirv(st[i], &w);
            gwords += w;
        }
        static int min_b = -1;   // guarded programs have 51-73 bindings; ordinary ones far fewer
        if (min_b < 0) { const char *e = getenv("OOT_BIG_GFX_BINDINGS"); min_b = e ? atoi(e) : 48; }
        if (!opt && gwords >= (uint64_t)big_cs_words() && l->n >= (uint32_t)min_b) pi.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    }
    if (gfx_probe) pi.flags |= VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT;
    uint64_t g0 = plat_now_ns(), c0 = g0;
    plat_crash_note(note);
    // Pipeline binaries (above): a kept one is created without compiling; otherwise the compile captures its data so a
    // slow one can be kept. A probe of the cache does neither.
    char pbp[PLAT_PATH_MAX * 2 + 160];
    int pbk = !gfx_probe && !noopt && !crashes && pb_on() && pb_path(&pi, pbp, sizeof pbp), from_bin = 0;
    VkResult r = VK_ERROR_UNKNOWN;
    if (pbk && plat_file_exists(pbp)) {
        pipe = pb_load(&pi, NULL, pbp);
        if (pipe) r = VK_SUCCESS, from_bin = 1, plat_inc32(&PB.loaded);
        else if (plat_inc32(&PB.refused) <= 8) rt_log("vk_pipeline: pipeline binary refused, compiling: %s", pbp);
    }
    int big = big_pipeline(total_words) && !gfx_probe;
    if (!from_bin && !gfx_probe && big_defer(total_words, "graphics pipeline")) {
        plat_crash_note(NULL);
        perf_end(PERF_PIPE, p0);
        pipe = VK_NULL_HANDLE;
        goto out;
    }
    int fallback = 0;   // built without optimization after the optimized build failed: never kept (below)
    if (!from_bin) {
        VkPipelineCreateFlags2CreateInfoKHR f2 = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR,
                                                  .pNext = pi.pNext,
                                                  .flags = (VkPipelineCreateFlags2KHR)pi.flags | VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR};
        if (pbk) pi.pNext = &f2;
        mg_enter(big);
        if (big) {   // jogar\preparar-shaders.ps1 shows it
            c0 = plat_now_ns();   // the compile alone, not the wait for the gate
            rt_log("vk_pipeline: compiling a big graphics pipeline: %llu words", (unsigned long long)total_words);
        }
        VkPipelineCache build_cache = pbk ? VK_NULL_HANDLE : cache;   // Vulkan requires VK_NULL_HANDLE with the capture flag
        int shared = build_cache && build_cache == P.cache;
        if (shared) plat_mutex_lock_shared(&cache_access);
        r = vkCreateGraphicsPipelines(P.device, build_cache, 1, &pi, NULL, &pipe);
        // AMD's compiler gives up on large programs (-1, out of host memory, on an RX 6750 XT: the scene stayed black at
        // 1 fps). Built again without optimization, as NotRexed/Hookshot's RX_VK_NOOPT does; NVIDIA never gets here.
        if (r != VK_SUCCESS && r != VK_PIPELINE_COMPILE_REQUIRED && vk_caps.amd &&
            !(pi.flags & VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT)) {
            VkResult first = r;
            pi.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
            f2.flags |= VK_PIPELINE_CREATE_2_DISABLE_OPTIMIZATION_BIT_KHR;   // when chained, f2's flags replace pi.flags
            r = vkCreateGraphicsPipelines(P.device, build_cache, 1, &pi, NULL, &pipe);
            fallback = 1;
            rt_log("vk_pipeline: graphics pipeline built again without optimization: %s (%d, then %d): %s",
                   r == VK_SUCCESS ? "ok" : "failed too", (int)first, (int)r, note);
        }
        if (shared) plat_mutex_unlock_shared(&cache_access);
        mg_leave(big);
        if (pbk) pi.pNext = f2.pNext;
    }
    plat_crash_note(NULL);
    double c_s = (double)(plat_now_ns() - c0) / 1e9;
    if (pbk && !from_bin && r == VK_SUCCESS && pipe) {
        // pbp is the optimized build's name, and the slower fallback must not load from it on every later start.
        if (!fallback && c_s * 1000.0 >= PB.save_ms) {
            int ok = pb_save(pipe, pbp);
            if (ok) plat_inc32(&PB.saved);
            rt_log("vk_pipeline: pipeline binary %s after a %.1f s compile: %s", ok ? "kept" : "NOT kept", c_s, pbp);
        } else {
            pb_release(pipe);
        }
    }
    if (from_bin && __atomic_load_n(&PB.loaded, __ATOMIC_RELAXED) <= 8)
        rt_log("vk_pipeline: pipeline created from its binary in %.1f ms, %llu words", c_s * 1000.0, (unsigned long long)total_words);
    if (c_s >= 3.0) rt_log("vk_pipeline: slow graphics pipeline: %.1f s%s, %llu words; %s", c_s, noopt ? " (unoptimized)" : "",
                           (unsigned long long)total_words, note);
    perf_end(PERF_PIPE, p0);   // counts on the render thread only, so background builds stay out of it
    {
        double gs_ = (double)(plat_now_ns() - g0) / 1e9;
        if (gs_ > 2.0) {
            rt_log("vk_pipeline: graphics pipeline of %llu words built in %.1f s%s", (unsigned long long)gwords, gs_,
                   pi.flags & VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT ? " (no optimization)" : "");
        }
    }
    if (r != VK_SUCCESS) {
        if (r != VK_PIPELINE_COMPILE_REQUIRED) rt_log("vk_pipeline: vkCreateGraphicsPipelines failed (%d)", (int)r);
        pipe = VK_NULL_HANDLE;
    } else {
        perf_count(PERF_CNT_PIPE_BUILT);
    }
out:
    if (vm) vkDestroyShaderModule(P.device, vm, NULL);
    if (fm) vkDestroyShaderModule(P.device, fm, NULL);
    if (gm) vkDestroyShaderModule(P.device, gm, NULL);
    return pipe;
}

// ---- graphics pipeline library (vk_caps.gpl)
// A pipeline is linked from four libraries, each cached by the part of the pipeline key it depends on: vertex input
// (topology class, vertex attributes without dynamic vertex input), pre-rasterization (vertex shader, the outputs its
// union adds, static rasterization state), fragment shader (fragment shader, dual-source rewrite, samples) and fragment
// output (targets, blend, samples). With independent layouts each stage's set depends on its shader alone, so the
// pre-rasterization library uses {set 0, null} and the fragment shader library {null, set 1}. The fast link runs on the
// thread of the draw that needs it (no link-time optimisation); the optimised pipeline is the job pipe_miss queued
// (a monolithic build with the same modules), which replaces the fast one in the Pipe when done. Libraries and fast
// pipelines live until vk_pipeline_shutdown.
// Background build signalling, both with P.lock: work_cv wakes the pipeline workers, done_cv wakes threads waiting for a
// pipeline or a stage library to be published.
static PlatCond work_cv = PLAT_COND_INIT, done_cv = PLAT_COND_INIT;
#define LIB_BUCKETS 1024
enum { LIB_VI, LIB_PR, LIB_FS, LIB_FO };
typedef struct Lib { struct Lib *next; uint64_t hash; VkPipeline pipe; uint32_t size; uint8_t pre, used, building, pad;
                     struct LibJob *job; uint8_t key[]; } Lib;
// pre: built by the boot prebuild ("stage library prebuild" below); used: a draw's link found it; building: a thread
// builds it or job holds it queued (pipe not set yet); entries live until libs_destroy
static Lib *libs[LIB_BUCKETS];

typedef struct { uint32_t kind, topo_class, attr_fmt[32], attr_off[32]; uint8_t attr_bind[32], rate[17], pad[3]; } ViKey;
typedef struct { uint32_t kind, missing; const GfxShader *vs; uint8_t clamp, poly_mode, minus_w, pad; GfxShaderIo fin[32]; } PrKey;
typedef struct { uint32_t kind; uint8_t dsb, samples, a2c, alpha; const GfxShader *fs; float alpha_ref; uint32_t pad; } FsKey;
typedef struct { uint32_t kind; uint8_t n_color, samples, a2c, logic_op, blend_en, pad[3]; uint32_t color[8], depth, stencil, wmask;
                 uint8_t blend[8][6]; } FoKey;

static uint64_t qpc(void) { return plat_now_ns(); }
static double ms_since(uint64_t t0) { return (double)(qpc() - t0) / 1e6; }

static VkPipelineLayout half_layout(Layout *l, int h) {
    if (l->half[h]) return l->half[h];
    VkDescriptorSetLayout sets[3] = {h ? VK_NULL_HANDLE : l->set, h ? l->set1 : VK_NULL_HANDLE, vk_bindless_layout()};
    VkPipelineLayoutCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .flags = VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT, .setLayoutCount = sets[2] ? 3 : 2,
                                     .pSetLayouts = sets};
    if (vkCreatePipelineLayout(P.device, &pi, NULL, &l->half[h]) != VK_SUCCESS) l->half[h] = VK_NULL_HANDLE;
    return l->half[h];
}

// Builds one library from the pipeline's state; key is the library's key (PrKey: fin holds the added outputs).
static VkPipeline lib_create(int kind, const VkPipeState *s, const GKey *k, VkPipelineLayout lay, const void *key) {
    static const VkGraphicsPipelineLibraryFlagsEXT part[4] = {
        VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
        VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT};
    VkGraphicsPipelineLibraryCreateInfoEXT gl = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, .flags = part[kind]};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .pNext = &gl};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = P.n_dyn, .pDynamicStates = P.dyn};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
                                       .flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR, .pDynamicState = &dy};
    VkShaderModule m = VK_NULL_HANDLE;
    VkPipelineShaderStageCreateInfo st = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pName = "main"};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rq;
    VkVertexInputAttributeDescription va[32];
    VkVertexInputBindingDescription vb[17];
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    static const VkPrimitiveTopology class_base[4] = {VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                                                      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_PATCH_LIST};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = class_base[k->topo_class]};
    VkPipelineViewportDepthClipControlCreateInfoEXT clip = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
                                                            .negativeOneToOne = k->minus_w};
    VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .pNext = &clip};
    VkPipelineRasterizationProvokingVertexStateCreateInfoEXT pvs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
        .provokingVertexMode = VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .pNext = P.pv_last ? &pvs : NULL, .depthClampEnable = k->clamp,
                                                 .polygonMode = (VkPolygonMode)k->poly_mode, .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)k->samples,
                                               .alphaToCoverageEnable = k->a2c};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                                                .maxDepthBounds = 1.0f};
    VkPipelineColorBlendAttachmentState cba[8];
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    uint32_t *patched = NULL;
    VkPipeline pipe = VK_NULL_HANDLE;
    uint64_t stage_words = 0;   // the shader's own size, for big_pipeline
    if (kind == LIB_VI) {
        if (!P.dyn_vi) {
            VtxInput v;
            vertex_input(s, &v);
            for (uint32_t i = 0; i < v.n_attr; i++)
                va[i] = (VkVertexInputAttributeDescription){v.a[i].location, v.a[i].binding, v.a[i].format, v.a[i].offset};
            for (uint32_t i = 0; i < v.n_bind; i++)
                vb[i] = (VkVertexInputBindingDescription){v.b[i].binding, v.b[i].stride, v.b[i].inputRate};
            vi.vertexAttributeDescriptionCount = v.n_attr, vi.pVertexAttributeDescriptions = va;
            vi.vertexBindingDescriptionCount = v.n_bind, vi.pVertexBindingDescriptions = vb;
        }
        pi.pVertexInputState = &vi, pi.pInputAssemblyState = &ia;
    } else if (kind == LIB_PR) {
        const PrKey *pk = key;
        uint64_t vw, sw, n2 = 0;
        const uint32_t *vcode = gfx_shader_spirv(s->vs, &vw), *scode = vcode;
        sw = vw, stage_words = vw;
        if (pk->missing && (patched = spirv_add_outputs(vcode, vw, pk->missing, pk->fin, &n2))) vcode = patched, vw = n2;
        else if (pk->missing) LOG_ONCE("vk_pipeline: cannot add missing vertex outputs (mask %#x)", pk->missing);
        if (!(m = make_module(vcode, vw))) goto out;
        st.stage = VK_SHADER_STAGE_VERTEX_BIT, st.module = m;
        pin_subgroup32(&st, &rq, scode, sw);
        pi.stageCount = 1, pi.pStages = &st, pi.pViewportState = &vp, pi.pRasterizationState = &rs;
        pi.layout = lay;
    } else if (kind == LIB_FS) {
        if (s->fs) {
            uint64_t fw, sw, n2 = 0;
            const uint32_t *fcode = fs_spirv_over(s->fs, &fw), *scode = fcode;
            sw = fw, stage_words = fw;
            uint32_t *d = NULL, *at = NULL;
            if (k->alpha && fcode) {   // before spirv_dual_source, which adds a second output at Location 0
                if ((at = spirv_alpha_test(fcode, fw, k->alpha, k->alpha_ref, &n2))) fcode = at, fw = n2;
                else LOG_ONCE("vk_pipeline: alpha test but the fragment shader has no float vec4 output at location 0");
            }
            if (k->dsb && fcode) {
                if ((d = spirv_dual_source(fcode, fw, &n2))) fcode = d, fw = n2;
                else LOG_ONCE("vk_pipeline: SRC1 blend factors but the fragment shader writes no colour output 1");
            }
            patched = fcode ? spirv_set(fcode, fw, 1) : NULL;
            free(d), free(at);
            if (!patched || !(m = make_module(patched, fw))) goto out;
            st.stage = VK_SHADER_STAGE_FRAGMENT_BIT, st.module = m;
            pin_subgroup32(&st, &rq, scode, sw);
            pi.stageCount = 1, pi.pStages = &st;
        }
        pi.pMultisampleState = &ms, pi.pDepthStencilState = &ds;
        pi.layout = lay;
    } else {
        memset(cba, 0, sizeof cba);
        for (int i = 0; i < k->n_color; i++) {
            const uint8_t *b = k->blend[i];
            cba[i].blendEnable = k->blend_en >> i & 1;
            cba[i].srcColorBlendFactor = (VkBlendFactor)b[0], cba[i].dstColorBlendFactor = (VkBlendFactor)b[1];
            cba[i].colorBlendOp = (VkBlendOp)b[2], cba[i].srcAlphaBlendFactor = (VkBlendFactor)b[3];
            cba[i].dstAlphaBlendFactor = (VkBlendFactor)b[4], cba[i].alphaBlendOp = (VkBlendOp)b[5];
            cba[i].colorWriteMask = k->wmask >> (4 * i) & 0xf;
        }
        cb.logicOpEnable = k->logic_op != 0, cb.logicOp = k->logic_op ? (VkLogicOp)(k->logic_op - 1) : VK_LOGIC_OP_COPY;
        cb.attachmentCount = k->n_color, cb.pAttachments = cba;
        ri.colorAttachmentCount = k->n_color, ri.pColorAttachmentFormats = (const VkFormat *)k->color;
        ri.depthAttachmentFormat = (VkFormat)k->depth, ri.stencilAttachmentFormat = (VkFormat)k->stencil;
        pi.pColorBlendState = &cb, pi.pMultisampleState = &ms;
    }
    int big = big_pipeline(stage_words);
    if (big && big_defer(stage_words, kind == LIB_FS ? "fragment shader library" : "vertex shader library")) goto out;
    mg_enter(big);
    plat_mutex_lock_shared(&cache_access);
    if (vkCreateGraphicsPipelines(P.device, P.cache, 1, &pi, NULL, &pipe) != VK_SUCCESS) pipe = VK_NULL_HANDLE;
    plat_mutex_unlock_shared(&cache_access);
    mg_leave(big);
out:
    if (m) vkDestroyShaderModule(P.device, m, NULL);
    free(patched);
    return pipe;
}

// The cached library for key. Caller holds P.lock. build 0: lookup only (VK_NULL_HANDLE and *miss when absent);
// build 1: a missing one is built with P.lock released (a failed build is cached too).
// The library entry for key, NULL when absent. Caller holds P.lock.
static Lib *lib_find(uint64_t h, const void *key, uint32_t size) {
    for (Lib *b = libs[h % LIB_BUCKETS]; b; b = b->next)
        if (b->hash == h && b->size == size && !memcmp(b->key, key, size)) return b;
    return NULL;
}

// Enters a library built with P.lock released; another thread's entry made meanwhile wins and pipe is destroyed.
// Returns the entered handle. Caller holds P.lock.
static VkPipeline lib_put(uint64_t h, const void *key, uint32_t size, VkPipeline pipe) {
    Lib *b = lib_find(h, key, size);
    if (b && b->building) {   // another thread's build publishes the entry
        if (pipe) vkDestroyPipeline(P.device, pipe, NULL);
        return VK_NULL_HANDLE;
    }
    if (b) {
        if (pipe) vkDestroyPipeline(P.device, pipe, NULL);
        return b->pipe;
    }
    if (!(b = calloc(1, sizeof *b + size))) return pipe;
    b->hash = h, b->size = size, b->pipe = pipe;
    memcpy(b->key, key, size);
    b->next = libs[h % LIB_BUCKETS];
    libs[h % LIB_BUCKETS] = b;
    return pipe;
}

// Shader libraries a waiting draw needs (lib_wait): queued here ahead of every pipeline job, so an idle worker builds
// each at once and the vertex and fragment libraries of one draw build in parallel. A job no worker took is built by
// the first thread that needs its library (lib_get, lib_wait). Under P.lock.
typedef struct LibJob { struct LibJob *next; Lib *b; int kind; Layout *l; VkPipeState s; GKey k; } LibJob;
static struct {
    LibJob *head, *tail;
    int idle;                          // workers asleep on work_cv
    int running;                       // jobs workers are building
    int on;                            // TOTK_PIPE_WAIT: 1 workers (default), 0 the drawing thread builds them (draw);
                                       // set by vk_pipeline_async_enable, before the replay runs
    uint64_t waits, queued, by_worker, stolen, timeouts;
} U;

// A building entry for key, NULL when out of memory. Caller holds P.lock and found none.
static Lib *lib_new(uint64_t h, const void *key, uint32_t size) {
    Lib *b = calloc(1, sizeof *b + size);
    if (!b) return NULL;
    b->hash = h, b->size = size, b->building = 1;
    memcpy(b->key, key, size);
    b->next = libs[h % LIB_BUCKETS];
    libs[h % LIB_BUCKETS] = b;
    return b;
}

// Builds b (building, not queued) with P.lock released and publishes it; a failed build is entered as VK_NULL_HANDLE.
static VkPipeline lib_build(Lib *b, int kind, const VkPipeState *s, const GKey *k, Layout *l) {
    VkPipelineLayout lay = kind == LIB_PR ? half_layout(l, 0) : kind == LIB_FS ? half_layout(l, 1) : VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    if (lay || kind == LIB_VI || kind == LIB_FO) {
        plat_mutex_unlock(&P.lock);
        uint64_t t0 = qpc();
        pipe = lib_create(kind, s, k, lay, b->key);
        double ms = ms_since(t0);
        plat_mutex_lock(&P.lock);
        P.g.lib_n[kind]++, P.g.lib_ms[kind] += ms;
        if (ms > P.g.lib_max[kind]) P.g.lib_max[kind] = ms;
    }
    if (!pipe && ++P.g.lib_fail <= 8) rt_log("vk_pipeline: gpl: stage library %d not built", kind);
    b->pipe = pipe, b->building = 0;
    plat_cond_broadcast(&done_cv);
    return pipe;
}

// Takes b's queued job out of the queue and builds it on this thread. Caller holds P.lock.
static void lib_steal(Lib *b) {
    LibJob *j = b->job, **pp = &U.head, *prev = NULL;
    while (*pp && *pp != j) prev = *pp, pp = &(*pp)->next;
    if (*pp) *pp = j->next;
    if (U.tail == j) U.tail = prev;
    b->job = NULL;
    lib_build(b, j->kind, &j->s, &j->k, j->l);
    free(j);
}

static VkPipeline lib_get(int kind, const VkPipeState *s, const GKey *k, Layout *l, const void *key, uint32_t size,
                          int build, int *miss) {
    uint64_t h = key_hash(key, size);
    Lib *have = lib_find(h, key, size);
    while (have && have->building) {   // being built (wait) or queued (build it here)
        if (!build) {
            *miss = 1;
            return VK_NULL_HANDLE;
        }
        if (have->job) U.stolen++, lib_steal(have);
        else plat_cond_wait(&done_cv, &P.lock);
    }
    if (have) {
        if (have->pre && !have->used) have->used = 1, P.g.pre_used[kind]++;
        return have->pipe;
    }
    VkPipelineLayout lay = kind == LIB_PR ? half_layout(l, 0) : kind == LIB_FS ? half_layout(l, 1) : VK_NULL_HANDLE;
    if (!build || ((kind == LIB_PR || kind == LIB_FS) && !lay)) {
        *miss = 1;
        return VK_NULL_HANDLE;
    }
    Lib *b = lib_new(h, key, size);   // entered first, so other threads wait for this build instead of repeating it
    if (b) return lib_build(b, kind, s, k, l);
    plat_mutex_unlock(&P.lock);
    uint64_t t0 = qpc();
    VkPipeline pipe = lib_create(kind, s, k, lay, key);
    double ms = ms_since(t0);
    plat_mutex_lock(&P.lock);
    if ((have = lib_find(h, key, size)) && !have->building) {   // another thread built it meanwhile
        if (pipe) vkDestroyPipeline(P.device, pipe, NULL);
        return have->pipe;
    }
    P.g.lib_n[kind]++, P.g.lib_ms[kind] += ms;
    if (ms > P.g.lib_max[kind]) P.g.lib_max[kind] = ms;
    if (!pipe && ++P.g.lib_fail <= 8) rt_log("vk_pipeline: gpl: stage library %d not built", kind);
    return lib_put(h, key, size, pipe);
}

static void libs_destroy(void) {
    for (int i = 0; i < LIB_BUCKETS; i++) {
        for (Lib *b = libs[i], *n; b; b = n) {
            n = b->next;
            if (b->pipe) vkDestroyPipeline(P.device, b->pipe, NULL);
            free(b);
        }
        libs[i] = NULL;
    }
}

// The four library keys of pipeline s (key k).
static void lib_keys(const VkPipeState *s, const GKey *k, ViKey *v, PrKey *pk, FsKey *fk, FoKey *ok) {
    memset(v, 0, sizeof *v);
    v->kind = LIB_VI, v->topo_class = k->topo_class;
    memcpy(v->attr_fmt, k->attr_fmt, sizeof v->attr_fmt), memcpy(v->attr_off, k->attr_off, sizeof v->attr_off);
    memcpy(v->attr_bind, k->attr_bind, sizeof v->attr_bind), memcpy(v->rate, k->rate, sizeof v->rate);
    memset(pk, 0, sizeof *pk);
    GfxShaderIo fin[32];
    pk->kind = LIB_PR, pk->vs = s->vs, pk->clamp = k->clamp, pk->poly_mode = k->poly_mode, pk->minus_w = k->minus_w;
    pk->missing = vs_union_missing(s, fin);
    for (int i = 0; i < 32; i++)
        if (pk->missing >> i & 1) pk->fin[i] = fin[i];
    memset(fk, 0, sizeof *fk);
    fk->kind = LIB_FS, fk->fs = s->fs, fk->dsb = k->dsb, fk->samples = k->samples, fk->a2c = k->a2c;
    fk->alpha = k->alpha, fk->alpha_ref = k->alpha_ref;
    memset(ok, 0, sizeof *ok);
    ok->kind = LIB_FO, ok->n_color = k->n_color, ok->samples = k->samples, ok->a2c = k->a2c, ok->logic_op = k->logic_op;
    ok->blend_en = k->blend_en, ok->depth = k->depth, ok->stencil = k->stencil, ok->wmask = k->wmask;
    memcpy(ok->color, k->color, sizeof ok->color), memcpy(ok->blend, k->blend, sizeof ok->blend);
}

// A fast-linked pipeline for s from its stage libraries. Caller holds P.lock.
// build 0 (drawing thread): links only when the shader libraries exist (vertex input and fragment output libraries,
// which hold no shader, are built here); *final 0 = not yet, a worker builds them. build 1 (worker): builds the
// missing libraries and links with P.lock released. VK_NULL_HANDLE with *final 1 when GPL cannot express the pipeline
// (shared layout, feedback loop) or a build failed.
static VkPipeline gpl_link(const VkPipeState *s, const GKey *k, const Layout *lc, int build, int *final) {
    Layout *l = (Layout *)lc;
    *final = 1;
    if (s->gs || mesh_vs(s) || !l->set1 || k->fbl) return VK_NULL_HANDLE;
    uint64_t t0 = qpc();
    ViKey vk_;
    PrKey pk;
    FsKey fk;
    FoKey ok;
    lib_keys(s, k, &vk_, &pk, &fk, &ok);
    int miss = 0;
    VkPipeline lib[4];
    lib[1] = lib_get(LIB_PR, s, k, l, &pk, sizeof pk, build, &miss);   // shader libraries first: no work when absent
    lib[2] = miss ? VK_NULL_HANDLE : lib_get(LIB_FS, s, k, l, &fk, sizeof fk, build, &miss);
    if (miss) {
        if (!build) *final = 0, P.g.defer++;
        return VK_NULL_HANDLE;
    }
    lib[0] = lib_get(LIB_VI, s, k, l, &vk_, sizeof vk_, 1, &miss);
    lib[3] = lib_get(LIB_FO, s, k, l, &ok, sizeof ok, 1, &miss);
    VkPipeline pipe = VK_NULL_HANDLE;
    if (lib[0] && lib[1] && lib[2] && lib[3]) {
        VkPipelineLibraryCreateInfoKHR li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, .libraryCount = 4,
                                             .pLibraries = lib};
        VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &li,
                                           .layout = l->layout};
        if (build) plat_mutex_unlock(&P.lock);
        uint64_t t1 = qpc();
        if (vkCreateGraphicsPipelines(P.device, VK_NULL_HANDLE, 1, &pi, NULL, &pipe) != VK_SUCCESS) pipe = VK_NULL_HANDLE;
        double ms = ms_since(t1);   // the link alone: library builds are counted apart
        if (build) plat_mutex_lock(&P.lock);
        P.g.link_ms += ms;
        if (ms > P.g.link_max) P.g.link_max = ms;
    }
    if (pipe) P.g.links++;
    else if (!P.g.link_fail++) rt_log("vk_pipeline: gpl: fast link failed; such draws wait or skip as without GPL");
    double total = ms_since(t0);
    if (!build && total > 5.0 && P.g.slow++ < 16)
        rt_log("vk_pipeline: gpl: fast link on the drawing thread took %.1f ms", total);
    return pipe;
}

// ---- background graphics pipeline builds (vk_pipeline_async_enable)
// A miss in vk_pipeline_graphics_async enters the Pipe as building and queues a job; the replay skips draws with it
// until a worker publishes the handle. The job holds copies of the state and the key. What they point to outlives it:
// shaders (gfx_shader_get results) live for the process (gfx_shader_drop_memory is the shader probe's, which builds no
// pipelines), layouts until vk_pipeline_shutdown, which drains the queue first. Workers call vkCreateGraphicsPipelines
// without P.lock: P.cache is created without VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT, so the driver
// synchronises it, vkGetPipelineCacheData in vk_pipeline_save included. Compute pipelines stay synchronous: a skipped
// dispatch can change what the game computes.
//
// Two queues: draws' misses (A.head) and the boot prebuild (A.low, see "pipeline keys" below). Workers take the low
// queue only when the other is empty and run its jobs at PLAT_PRIO_BELOW_NORMAL, so guest threads win. A draw
// that finds a prebuild job still queued moves it to the draw queue; one that finds it running raises its worker.
typedef struct Job {
    struct Job *next, *prev;           // prev: low queue only
    Pipe *p;
    const Layout *l;
    uint32_t fs_off;
    uint8_t low;                       // in the low queue, or running at below-normal priority
    uint8_t prebuild, demanded;        // from the key file; a draw asked for it before it was ready
    uint8_t large;                     // keep in the low queue so ordinary draws retain a worker
    uint8_t probed, slot;              // large: cache probe done (or off); holds one of the OOT_BIG_GFX_WORKERS slots
    uint8_t big;                       // AMD: big_cs_words() or more, set once when queued (head_pick)
    int worker;                        // running on A.thread[worker]; -1 queued
    VkPipeState s;
    GKey k;
} Job;

#define MAX_WORKERS 8

static struct {
    int on;                            // set once by vk_pipeline_async_enable, before the replay runs
    int threads;
    PlatThread *thread[MAX_WORKERS];
    uint8_t prio_low[MAX_WORKERS];     // worker runs below normal (under P.lock)
    Job *head, *tail;                  // draws' misses
    Job *low, *low_tail;               // prebuild
    uint32_t inflight;                 // queued or running, both queues
    int low_running;                   // pool workers running a job taken from the low queue
    PlatCond idle;           // with P.lock: no job queued or running
} A = {.idle = PLAT_COND_INIT};

// ---- pipeline keys (pipeline-keys.bin next to the pipeline cache)
// Every graphics pipeline built in a game run appends a KeyRec: the VkPipeState it was built from with the shader
// pointers zeroed and the shaders named by their gfx_shader input ids instead, which do not depend on the translator,
// so the record rebuilds the pipeline without guest memory. At the next boot vk_pipeline_prewarm (rt_boot, before the
// guest runs; TOTK_PIPE_PREBUILD=0 skips it) loads each record's shaders, translating the ones not cached under the
// current translator from their stored inputs (gfx_shader_prewarm; a record whose shader has neither is dropped),
// enters each Pipe as building under the key a draw would compute, queues it to the low queue and builds the queue on
// every core. Records are deduplicated by a hash of their bytes; they are collected under P.lock by the worker that
// built the pipeline and appended by the present thread (vk_pipeline_autosave) or at exit (vk_pipeline_save). A file
// with another magic, version or record size is ignored and rewritten. Bump KEYS_VERSION when VkPipeState or the
// record changes meaning without changing size.
#define KEYS_VERSION 4                 // 3: shaders named by input id (gfx_shader.c); 4: the task stage too
typedef struct { char magic[8]; uint32_t version, rec_size; } KeysHeader;
typedef struct {
    uint8_t vs[32], fs[32];            // gfx_shader_input_id of each stage; fs all zero with has_fs 0
    uint32_t has_fs, pad;
    // LOCAL TEST (Ocarina): a mesh pipeline's task stage (vs holds the mesh stage). Mesh pipelines were left out, and the
    // Deku Tree scene still built them on a first start after the shader preparation (8.4 s).
    uint8_t ts[32];
    uint32_t has_ts, pad2;
    // Frozen two-stage prefix. Adding the experimental GS pointer must not
    // change existing version-3 records or reinterpret their bytes at boot.
    _Alignas(VkPipeState) uint8_t state[offsetof(VkPipeState, gs)];
} KeyRec;
static struct {
    char path[PLAT_PATH_MAX * 2];
    int header_ok;                     // the file exists with our header: append
    uint64_t *seen;                    // open addressing set of record hashes (0 empty)
    uint32_t seen_cap, seen_n;
    KeyRec *pend;                      // records not written yet
    uint32_t pend_n, pend_cap;
    uint8_t *file;                     // the file as read at boot, until the prewarm is done with it
    size_t file_n;                     // records in it
    PlatMutex io;                        // one writer
    // prebuild progress (under P.lock)
    int loading;                       // the prewarm still queues records
    int warm;                          // vk_pipeline_prewarm owns the file (prebuild_start leaves it)
    uint32_t left;                     // prebuild jobs queued or running
    uint32_t built, failed, promoted;
    double t0;                         // now_ms() when the prebuild started
} K = {.io = PLAT_MUTEX_INIT};

static uint64_t rec_hash(const KeyRec *r) {
    uint64_t h = fnv(FNV0, r, sizeof *r);
    return h ? h : 1;
}

// 1 if h was new. Caller holds P.lock.
static int seen_add(uint64_t h) {
    if ((K.seen_n + 1) * 2 > K.seen_cap) {
        uint32_t cap = K.seen_cap ? K.seen_cap * 2 : 4096;
        uint64_t *t = calloc(cap, sizeof *t);
        if (!t) return 1;
        for (uint32_t i = 0; i < K.seen_cap; i++) {
            if (!K.seen[i]) continue;
            uint32_t j = (uint32_t)K.seen[i] & (cap - 1);
            while (t[j]) j = (j + 1) & (cap - 1);
            t[j] = K.seen[i];
        }
        free(K.seen);
        K.seen = t, K.seen_cap = cap;
    }
    uint32_t j = (uint32_t)h & (K.seen_cap - 1);
    while (K.seen[j]) {
        if (K.seen[j] == h) return 0;
        j = (j + 1) & (K.seen_cap - 1);
    }
    K.seen[j] = h;
    K.seen_n++;
    return 1;
}

static void rec_fill(KeyRec *r, const VkPipeState *s) {
    memset(r, 0, sizeof *r);
    memcpy(r->vs, gfx_shader_input_id(s->vs), 32);
    if (s->fs) memcpy(r->fs, gfx_shader_input_id(s->fs), 32), r->has_fs = 1;
    if (s->ts) memcpy(r->ts, gfx_shader_input_id(s->ts), 32), r->has_ts = 1;
    VkPipeState legacy = *s;
    legacy.vs = legacy.fs = legacy.gs = legacy.ts = NULL;
    legacy.prim_restart = 0;   // dynamic state: the same pipeline either way
    memcpy(r->state, &legacy, sizeof r->state);
}

// A pipeline was built for a draw: queue its record for the file. Caller holds P.lock.
static void keys_note(const VkPipeState *s) {
    if (s->gs || !A.on || !K.path[0]) return;             // GS prewarm records need a future explicit format
    KeyRec r;
    rec_fill(&r, s);
    if (!seen_add(rec_hash(&r))) return;
    if (K.pend_n == K.pend_cap) {
        uint32_t cap = K.pend_cap ? K.pend_cap * 2 : 64;
        KeyRec *t = realloc(K.pend, cap * sizeof *t);
        if (!t) return;
        K.pend = t, K.pend_cap = cap;
    }
    K.pend[K.pend_n++] = r;
}

// Appends the pending records (present thread or exit; never the render thread).
static void keys_flush(void) {
    if (!K.path[0]) return;
    plat_mutex_lock(&K.io);
    plat_mutex_lock(&P.lock);
    KeyRec *r = K.pend;
    uint32_t n = K.pend_n;
    K.pend = NULL, K.pend_n = K.pend_cap = 0;
    plat_mutex_unlock(&P.lock);
    if (n) {
        FILE *f = fopen(K.path, K.header_ok ? "ab" : "wb");
        int ok = f != NULL;
        if (ok && !K.header_ok) {
            KeysHeader h = {{'T', 'O', 'T', 'K', 'P', 'K', 'E', 'Y'}, KEYS_VERSION, sizeof(KeyRec)};
            ok = fwrite(&h, sizeof h, 1, f) == 1;
        }
        if (ok) ok = fwrite(r, sizeof *r, n, f) == n;
        if (ok) ok = plat_file_sync(f);   // appended records on the disk before the next append
        if (f) ok = !fclose(f) && ok;
        if (ok) K.header_ok = 1;
        else rt_log("vk_pipeline: cannot write %s", K.path);
    }
    free(r);
    plat_mutex_unlock(&K.io);
}

static void crashed_open(const char *keys_path);
// Reads the key file into K.file and its record hashes into the set. Caller holds P.lock (ensure_init).
static void keys_open(void) {
    if (!A.on || !P.cache_path[0]) return;
    snprintf(K.path, sizeof K.path, "%s", P.cache_path);
    char *slash = strrchr(K.path, '\\'), *fslash = strrchr(K.path, '/');
    if (fslash > slash) slash = fslash;
    snprintf(slash ? slash + 1 : K.path, sizeof K.path - (size_t)(slash ? slash + 1 - K.path : 0), "pipeline-keys.bin");
    gfx_shader_record_at(K.path);   // shader-answers.bin next to it (gfx_shader.c)
    crashed_open(K.path);
    FILE *f = fopen(K.path, "rb");
    if (!f) return;
    KeysHeader h;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len >= (long)sizeof h && fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, "TOTKPKEY", 8) &&
        h.version == KEYS_VERSION && h.rec_size == sizeof(KeyRec)) {
        K.header_ok = 1;
        size_t n = ((size_t)len - sizeof h) / sizeof(KeyRec);
        if (n && (K.file = malloc(n * sizeof(KeyRec))) && fread(K.file, sizeof(KeyRec), n, f) == n) K.file_n = n;
        else free(K.file), K.file = NULL;
        for (size_t i = 0; i < K.file_n; i++) seen_add(rec_hash((const KeyRec *)K.file + i));
        if (len != (long)(sizeof h + n * sizeof(KeyRec)))   // a torn append: rewrite the file from what was read
            K.header_ok = 0;
    } else {
        rt_log("vk_pipeline: %s has another format or version; it is rewritten", K.path);
    }
    fclose(f);
    if (!K.header_ok && K.file_n) {   // keep the whole records of a torn file
        KeyRec *t = malloc(K.file_n * sizeof *t);
        if (t) memcpy(t, K.file, K.file_n * sizeof *t), K.pend = t, K.pend_n = K.pend_cap = (uint32_t)K.file_n;
    }
}

static double now_ms(void) { return (double)plat_now_ns() / 1e6; }

// Caller holds P.lock.
static void prebuild_report(void) {
    rt_log("vk_pipeline: prebuild done: %u built, %u failed, %u asked for by a draw first; %.0f ms after the device was "
           "ready", K.built, K.failed, K.promoted, now_ms() - K.t0);
}

// LOCAL TEST (Ocarina, AMD): a big (guarded) graphics pipeline compiles for ~10 minutes on the AMD driver. Only worker
// 0 takes those, and every worker takes small jobs first: with both workers on big ones, the ordinary pipelines of
// the scene (milliseconds each) waited behind them and the Kokiri forest stayed black. OOT_BIG_GFX_WORKERS=n lets
// workers 0..n-1 take big jobs (default 1). Other GPUs take the queue in order.
// The draw-queue job worker `me` should run next (small first, then big if allowed), unlinked when take; P.lock held.
static Job *head_pick(int me, int take) {
    static int big_workers = -1;
    if (big_workers < 0) {
        const char *e = getenv("OOT_BIG_GFX_WORKERS");
        big_workers = e ? atoi(e) : 1;
    }
    Job *pick = NULL, *pick_prev = NULL, *big = NULL, *big_prev = NULL;
    if (!vk_caps.amd) pick = A.head;
    else
        for (Job *j = A.head, *prev = NULL; j; prev = j, j = j->next) {
            if (!j->big) { pick = j, pick_prev = prev; break; }
            if (!big) big = j, big_prev = prev;
        }
    if (!pick && big && me < big_workers) pick = big, pick_prev = big_prev;
    if (pick && take) {
        if (pick_prev) pick_prev->next = pick->next;
        else A.head = pick->next;
        if (A.tail == pick) A.tail = pick_prev;
    }
    return pick;
}

// LOCAL TEST (Ocarina, AMD): upstream keeps large AMD jobs in the low queue, where two workers could still run two of
// them at once (10+ GB of driver memory each on a 16 GB machine). At most OOT_BIG_GFX_WORKERS large jobs run at once;
// a worker skips past queued large jobs to the next ordinary one. P.lock held.
static int large_running;
static int prewarm_big_bg;   // set while the prewarm queues its large records for the pool (vk_pipeline_prewarm)
static int probe_on = 1;   // OOT_PIPE_PROBE=0: large jobs compile in turn without a cache probe
static Job *low_pick(int take) {
    static int big_workers = -1;
    if (big_workers < 0) {
        const char *e = getenv("OOT_BIG_GFX_WORKERS");
        big_workers = e ? atoi(e) : 1;
    }
    // A large job not yet probed may run without a slot: only its cache probe, which takes milliseconds when the
    // pipeline is in pipeline-cache.bin, so cached ones no longer wait 15-20 minutes behind one that compiles.
    Job *j = A.low;
    while (j && j->large && j->probed && large_running >= big_workers) j = j->next;
    if (j && j->large && j->probed && j->prebuild && !j->demanded)   // a slot goes to a draw's job first
        for (Job *q = j->next; q; q = q->next)
            if (q->large && (!q->prebuild || q->demanded)) { j = q; break; }
    if (j && take) {
        if (j->prev) j->prev->next = j->next;
        else A.low = j->next;
        if (j->next) j->next->prev = j->prev;
        else A.low_tail = j->prev;
        j->slot = j->large && large_running < big_workers;
        if (j->slot) large_running++;
    }
    return j;
}

static void pipe_worker(void *arg) {
    int me = (int)(intptr_t)arg;
    plat_mutex_lock(&P.lock);
    for (;;) {
        // With 3 or more workers one never takes low-queue work, so a waiting draw's library starts at once
        // (not with TOTK_PIPE_WAIT=draw).
        while (!U.head && !head_pick(me, 0) && !(low_pick(0) && (A.threads < 3 || !U.on || A.low_running < A.threads - 1))) {
            U.idle++;
            plat_cond_wait(&work_cv, &P.lock);
            U.idle--;
        }
        if (U.head) {   // a library a draw waits for
            LibJob *lj = U.head;
            if (!(U.head = lj->next)) U.tail = NULL;
            lj->b->job = NULL;
            if (A.prio_low[me]) A.prio_low[me] = 0, plat_thread_priority(PLAT_PRIO_NORMAL);
            U.by_worker++, U.running++;
            lib_build(lj->b, lj->kind, &lj->s, &lj->k, lj->l);   // releases P.lock while building
            U.running--;
            free(lj);
            continue;
        }
        Job *j = head_pick(me, 1);
        int lowrun = !j;
        if (!j) {
            j = low_pick(1);
            A.low_running++;
        }
        j->next = j->prev = NULL;
        j->worker = me;
        if (j->low != A.prio_low[me]) {   // under P.lock, so a draw raising this thread (promote) is not undone
            A.prio_low[me] = j->low;
            plat_thread_priority(j->low ? PLAT_PRIO_BELOW_NORMAL : PLAT_PRIO_NORMAL);
        }
        // Step 1 for a job a draw waits for: its stage libraries and the fast link. Step 2, the optimised build, goes to
        // the head of the low queue, so a burst of new pipelines gets every fast link before any optimised build.
        if (P.gpl && !j->s.gs && (!j->prebuild || j->demanded) && !j->p->info.pipeline && !j->p->fast_tried && j->p->lay) {
            int final;
            VkPipeline f = gpl_link(&j->s, &j->k, j->p->lay, 1, &final);   // releases P.lock while building
            j->p->fast_tried = 1;
            if (f && !j->p->info.pipeline) {
                j->p->fast = j->p->info.pipeline = f, P.g.wlinks++;
                plat_cond_broadcast(&done_cv);
                j->low = 1, j->worker = -1, P.g.requeued++;
                j->prev = NULL, j->next = A.low;
                if (A.low) A.low->prev = j;
                else A.low_tail = j;
                A.low = j;
                A.low_running -= lowrun;
                continue;
            } else if (f) {
                vkDestroyPipeline(P.device, f, NULL);
            }
        }
        plat_mutex_unlock(&P.lock);
        VkPipeline pipe = VK_NULL_HANDLE;
        int deferred = 0;
        if (j->large && !j->probed) {   // LOCAL TEST (Ocarina, AMD): in the cache? (see low_pick)
            double t0 = now_ms();
            gfx_probe = 1;
            pipe = create_graphics(&j->s, &j->k, j->l, j->fs_off, P.cache);
            gfx_probe = 0;
            double dt = now_ms() - t0;
            rt_log("vk_pipeline: large pipeline %s in %.0f ms%s", pipe ? "found in the cache" : "not in the cache", dt,
                   pipe || j->slot ? "" : "; waits for a slot");
            if (dt > 10000 && probe_on) {
                probe_on = 0;
                rt_log("vk_pipeline: the driver compiled during a cache probe: probes off");
            }
            if (!pipe && !j->slot) {   // back to the head of the low queue, for the next free slot
                plat_mutex_lock(&P.lock);
                j->probed = 1, j->worker = -1;
                j->prev = NULL, j->next = A.low;
                if (A.low) A.low->prev = j;
                else A.low_tail = j;
                A.low = j;
                A.low_running -= lowrun;
                continue;
            }
        }
        if (!pipe) {
            // As with the big compute worker, do not hold the shared driver
            // cache during a large compilation.
            VkPipelineCache own = j->large ? cache_private() : VK_NULL_HANDLE;
            tl_deferred = 0;
            pipe = create_graphics(&j->s, &j->k, j->l, j->fs_off, own ? own : P.cache);
            deferred = !pipe && tl_deferred;
            if (own) {
                if (pipe) cache_publish(own);
                else vkDestroyPipelineCache(P.device, own, NULL);
            }
        }
        plat_mutex_lock(&P.lock);
        if (j->slot) large_running--;
        if (pipe || !j->p->fast) j->p->info.pipeline = pipe;   // a failed optimised build keeps the fast-linked one
        if (pipe && j->p->fast) P.g.swaps++;
        j->p->building = 0;
        j->p->job = NULL;
        if (pipe) {
            P.stats.graphics++, P.stats.async_built++;
            if (!j->prebuild && !j->large) keys_note(&j->s);
        } else if (deferred) {
            keys_note(&j->s);   // the next start's prewarm builds it (big compiles)
        } else {
            P.stats.failures++;
            rt_log("vk_pipeline: graphics pipeline not built: pipeline creation failed");
        }
        if (j->prebuild) {
            if (pipe) K.built++;
            else K.failed++;
            K.promoted += j->demanded;
            if (!--K.left && !K.loading) prebuild_report();
        }
        A.low_running -= lowrun;
        if (!--A.inflight) plat_cond_broadcast(&A.idle);
        plat_cond_broadcast(&done_cv);
        if (pipe && (!j->prebuild || j->demanded)) perf_count(PERF_CNT_PIPE_ASYNC);   // a draw waited for it
        free(j);
    }
    return;
}

// Starts the pool on first use: a quarter of the logical processors, 2 to 4; on AMD half of them, 2 to MAX_WORKERS
// (its Windows compiler takes seconds to minutes per pipeline, and the game leaves most cores idle), but no more
// than one per 8 GB of RAM: that compiler takes gigabytes per big pipeline, and six at once used up a 16 GB PC
// (see the compile memory gate). TOTK_PIPE_WORKERS=<n> sets it. Caller holds P.lock.
static int workers_start(void) {
    if (A.threads) return 1;
    uint32_t cpus = plat_cpu_usable();
    int cap = amd_keeps_binaries() ? MAX_WORKERS : 4;   // the room for 8 is for that driver only: elsewhere the limit stays 4
    int n = cpus / 4 < 2 ? 2 : cpus / 4 > (uint32_t)cap ? cap : (int)(cpus / 4);
    if (amd_keeps_binaries()) {
        n = cpus / 2 < 2 ? 2 : cpus / 2 > (uint32_t)cap ? cap : (int)(cpus / 2);
        if (n > mg_compilers_by_ram()) n = mg_compilers_by_ram();
    }
    const char *we = getenv("TOTK_PIPE_WORKERS");
    if (we && atoi(we) > 0) n = atoi(we) > cap ? cap : atoi(we);
    {   // LOCAL TEST (Ocarina, AMD): OOT_PIPE_WORKERS overrides the count. On AMD the big jobs are serialized on worker 0
        // (head_pick), so the default count stays.
        const char *e = getenv("OOT_PIPE_WORKERS");
        if (e && *e) n = atoi(e);
        n = n < 1 ? 1 : n > cap ? cap : n;
    }
    for (int i = 0; i < n; i++) {
        PlatThread *t = plat_thread_new(pipe_worker, (void *)(intptr_t)A.threads, "pipe worker");
        if (!t) break;
        A.thread[A.threads++] = t;   // kept: promote() raises a worker running a prebuild job
    }
    if (A.threads) rt_log("vk_pipeline: %d background graphics pipeline workers", A.threads);
    else rt_log("vk_pipeline: cannot start pipeline workers; building synchronously");
    return A.threads != 0;
}

static void workers_drain(void) {
    plat_mutex_lock(&P.lock);
    K.loading = 0;
    for (Job *j = A.low, *n; j; j = n) {   // queued prebuild jobs are dropped, not built
        n = j->next;
        j->p->building = 0, j->p->job = NULL;
        A.inflight--;
        if (j->prebuild) K.left--;
        free(j);
    }
    A.low = A.low_tail = NULL;
    for (LibJob *j = U.head, *n; j; j = n) {   // queued library jobs too: their entries read as failed builds
        n = j->next;
        j->b->job = NULL, j->b->building = 0;
        free(j);
    }
    U.head = U.tail = NULL;
    while (A.inflight) plat_cond_wait(&A.idle, &P.lock);
    while (U.running) plat_cond_wait(&done_cv, &P.lock);
    plat_mutex_unlock(&P.lock);
}

void vk_pipeline_async_enable(void) {
    U.on = strcmp(cfg_value("TOTK_PIPE_WAIT"), "draw") != 0;
    const char *e = cfg_get("TOTK_PIPE_SYNC");
    if (e && *e && strcmp(e, "0")) return;
    A.on = 1;
}

int vk_pipeline_async_on(void) { return A.on; }

// A draw needs a pipeline a prebuild job owns: the job moves to the draw queue, or its worker to normal priority.
// Caller holds P.lock.
static void promote(Pipe *p) {
    Job *j = p->job;
    if (!j || !j->low || j->large) return;
    j->low = 0, j->demanded = 1;
    if (j->worker < 0) {
        if (j->prev) j->prev->next = j->next;
        else A.low = j->next;
        if (j->next) j->next->prev = j->prev;
        else A.low_tail = j->prev;
        j->next = j->prev = NULL;
        if (A.tail) A.tail->next = j;
        else A.head = j;
        A.tail = j;
        plat_cond_broadcast(&work_cv);
    } else if (A.prio_low[j->worker]) {
        A.prio_low[j->worker] = 0;
        plat_thread_set_priority(A.thread[j->worker], PLAT_PRIO_NORMAL);
    }
}

static uint64_t graphics_shader_words(const GfxShader *shader) {
    uint64_t words = 0;
    if (shader) gfx_shader_spirv(shader, &words);
    return words;
}

// Enters a missing pipeline: built here (mode 0), or queued for a draw (1) or for the prebuild (2). Failures are
// entered too. *queued: 1 when a job owns it. Caller holds P.lock.
static Pipe *pipe_miss(const VkPipeState *s, const GKey *k, uint64_t h, int mode, int *queued) {
    *queued = 0;
    const char *why = NULL;
    Bindings B = {0};
    Layout *l = NULL;
    uint32_t span = binding_span(s->vs), fs_off = span - 1;
    if (s->gs) fs_off += binding_span(s->gs) - 1;
    if (s->ts) fs_off += binding_span(s->ts) - 1;
    if (gfx_shader_stage(s->vs) != GFX_STAGE_VS || (s->fs && gfx_shader_stage(s->fs) != GFX_STAGE_FS) ||
        (s->gs && gfx_shader_stage(s->gs) != GFX_STAGE_GS)) why = "stage mismatch";
    else if (k->topo_class == 3) why = "patches need tessellation stages, which are not translated";
    if (!why && s->gs) why = geometry_validate(s, &vk_ctx()->props.limits);
    if (!why && k->indep) {
        Bindings F = {0};
        if (!add_stage(&B, s->vs, 0) || (s->fs && !add_stage(&F, s->fs, 0))) why = "too many bindings";
        else if (!(l = layout_pair(&B, &F))) why = "descriptor layout creation failed";
        fs_off = 0;
    } else if (!why && (!add_stage(&B, s->vs, 0) || (s->gs && !add_stage(&B, s->gs, span - 1)) || (s->ts && !add_stage(&B, s->ts, span - 1)) ||
               (s->fs && !add_stage(&B, s->fs, fs_off)))) why = "too many bindings";
    else if (!why && !l && !(l = layout_get(&B))) why = "descriptor layout creation failed";
    char lack[128];
    if (!why && !layout_covers(s, l, fs_off, lack, sizeof lack)) why = lack, l = NULL;
    VkPipeline pipe = VK_NULL_HANDLE;
    Job *j = NULL;
    if (!why && mode && workers_start() && (j = calloc(1, sizeof *j))) {
        uint64_t words = graphics_shader_words(s->vs) + graphics_shader_words(s->fs) + graphics_shader_words(s->gs) +
                         graphics_shader_words(s->ts);
        int large = (mode == 1 || (mode == 2 && prewarm_big_bg)) && vk_caps.amd && !P.gpl && U.on && A.threads >= 3 &&
                    words >= 60000;
        *j = (Job){.l = l, .fs_off = fs_off, .low = mode == 2 || large, .prebuild = mode == 2,
                   .large = large, .probed = !probe_on || (getenv("OOT_PIPE_PROBE") && !strcmp(getenv("OOT_PIPE_PROBE"), "0")),
                   .big = vk_caps.amd && words >= big_cs_words(), .worker = -1, .s = *s, .k = *k};
    } else if (!why) {
        tl_deferred = 0;
        tl_holds_plock = 1;
        if (!(pipe = create_graphics(s, k, l, fs_off, P.cache)) && !tl_deferred) why = "pipeline creation failed";
        tl_holds_plock = 0;
    }
    if (why) {
        P.stats.failures++;
        rt_log("vk_pipeline: graphics pipeline not built: %s", why);
    } else if (!j) {
        if (pipe) P.stats.graphics++;
        if (mode != 2) keys_note(s);   // a big compile left for the next start (tl_deferred) is recorded too
    }
    Pipe *p = pipe_add(k, sizeof *k, h, pipe, l, l ? &B : NULL, VK_PIPELINE_BIND_POINT_GRAPHICS, gfx_shader_inputs(s->vs, NULL));
    if (!j || !p) {
        free(j);
        return p;
    }
    j->p = p;
    p->building = 1, p->job = j;
    // LOCAL TEST (Ocarina, AMD): a large job takes 6-20 minutes and may still be queued when the game closes; its record
    // goes to pipeline-keys.bin now, so the next boot queues it in the background (prewarm) even if it never finished.
    if (j->large && !j->prebuild) keys_note(s);
    if (j->low) {
        j->prev = A.low_tail;
        if (A.low_tail) A.low_tail->next = j;
        else A.low = j;
        A.low_tail = j;
        if (j->prebuild) K.left++;
    } else {
        if (A.tail) A.tail->next = j;
        else A.head = j;
        A.tail = j;
    }
    A.inflight++;
    plat_cond_broadcast(&work_cv);
    *queued = 1;
    return p;
}

// A job still in the draw queue goes to the head of the low queue: its pipeline has (or is about to have) a fast-linked
// one, so the optimised build waits behind other draws' misses. Caller holds P.lock.
static void job_lower(Job *j) {
    if (!j || j->worker >= 0 || j->low) return;
    if (A.head == j) {
        if (!(A.head = j->next)) A.tail = NULL;
    } else {
        Job *q = A.head;
        while (q && q->next != j) q = q->next;
        if (!q) return;
        if (!(q->next = j->next)) A.tail = q;
    }
    j->low = 1, P.g.requeued++;
    j->prev = NULL, j->next = A.low;
    if (A.low) A.low->prev = j;
    else A.low_tail = j;
    A.low = j;
}

// A draw waits for pipeline s: its missing pre-rasterization and fragment shader libraries go to the head of the
// worker queues (U), so they build in parallel on idle workers, and this thread sleeps until both are published or
// end (now_ms) passes. A job no worker took within 1 ms (all busy) is built here. Caller holds P.lock.
static void lib_wait(const VkPipeState *s, const GKey *k, Layout *l, double end) {
    ViKey vk_;
    PrKey pk;
    FsKey fk;
    FoKey ok;
    lib_keys(s, k, &vk_, &pk, &fk, &ok);
    U.waits++;
    const void *key[2] = {&pk, &fk};
    uint32_t size[2] = {sizeof pk, sizeof fk};
    Lib *b[2];
    int queued = 0;
    for (int i = 0; i < 2; i++) {
        uint64_t h = key_hash(key[i], size[i]);
        if ((b[i] = lib_find(h, key[i], size[i]))) continue;
        if (!half_layout(l, i)) return;   // gpl_link reports it
        LibJob *j = calloc(1, sizeof *j);
        if (!j || !(b[i] = lib_new(h, key[i], size[i]))) {
            free(j);
            return;
        }
        *j = (LibJob){.b = b[i], .kind = i ? LIB_FS : LIB_PR, .l = l, .s = *s, .k = *k};
        b[i]->job = j;
        if (U.tail) U.tail->next = j;
        else U.head = j;
        U.tail = j;
        U.queued++, queued++;
    }
    if (queued) plat_cond_broadcast(&work_cv);
    double steal_at = now_ms() + 1.0;
    for (;;) {
        if (!b[0]->building && !b[1]->building) return;
        double now = now_ms();
        if (now >= end) {
            U.timeouts++;
            return;
        }
        Lib *q = b[0]->job ? b[0] : b[1]->job ? b[1] : NULL;
        if (q && (now >= steal_at || !U.idle)) {
            U.stolen++;
            lib_steal(q);   // releases P.lock while building
            continue;
        }
        double until = q && steal_at < end ? steal_at : end;
        plat_cond_wait_ms(&done_cv, &P.lock, (uint32_t)(until - now) + 1);
    }
}

static VkPipeline graphics(const VkPipeState *s, const VkPipeInfo **info, int async, int *pending, uint32_t wait_ms) {
    if (info) *info = NULL;
    if (pending) *pending = 0;
    if (!s->vs || !ensure_init()) return VK_NULL_HANDLE;
    GKey k;
    gkey_fill(s, &k);
    uint64_t h = key_hash(&k, sizeof k);
    plat_mutex_lock(&P.lock);
    Pipe *p = pipe_find(&k, sizeof k, h);
    int queued;
    if (p) {
        if (!p->building) P.stats.hits++;
    } else {
        p = pipe_miss(s, &k, h, async, &queued);
    }
    // A draw that would wait (wait_ms) gets its missing shader libraries built at once, the vertex and fragment ones in
    // parallel on idle workers (lib_wait; about 0.12 ms per library from the driver's disk cache, 5-180 ms cold), and
    // links them here: its own job may queue behind optimised builds of earlier pipelines. The job then only does the
    // optimised build, from the low queue. Other draws leave the libraries to the job and are skipped meanwhile.
    // TOTK_PIPE_WAIT=draw builds them on this thread one after the other instead, as before; TOTK_GPL_DRAW_BUILD=0
    // leaves them to the job.
    static int draw_build = -1;
    if (draw_build < 0) {
        const char *e = cfg_get("TOTK_GPL_DRAW_BUILD");
        draw_build = !(e && !strcmp(e, "0"));
        rt_log("vk_pipeline: a draw that waits for its pipeline %s", !draw_build ? "leaves its libraries to its job"
               : U.on ? "has its libraries built in parallel on the workers, ahead of other jobs"
                      : "builds its libraries itself (TOTK_PIPE_WAIT=draw)");
    }
    // TOTK_PIPE_WAIT_ONCE (default 1 with amd_keeps_binaries, 0 elsewhere): once a draw's wait for a pipeline ran out, later draws with it skip at once until
    // it is built. AMD's Windows compiler took minutes for some of Ocarina's pipelines; hundreds of one-shot draws at
    // the first area waited a second each for the same few pipelines and were skipped anyway, freezing the picture for
    // minutes (RX 5500 XT, driver 26.10.1).
    static int wait_once = -1;
    if (wait_once < 0) {
        const char *e = getenv("TOTK_PIPE_WAIT_ONCE");
        wait_once = e && *e ? strcmp(e, "0") != 0 : amd_keeps_binaries();
    }
    if (wait_ms && wait_once && p && p->waited_out && p->building && !p->info.pipeline) wait_ms = 0;
    double end = wait_ms ? now_ms() + wait_ms : 0;
    // An ordinary full-screen wait is at most 50 ms. If a one-shot wait
    // times out, later frames can keep drawing with the available pipelines
    // while the same slow AMD driver job continues in the background.
    if (p && p->building && async && vk_caps.amd && !P.gpl && wait_ms > 50) {
        if (!p->one_shot_deadline_ms) p->one_shot_deadline_ms = end;
        else if (end > p->one_shot_deadline_ms) end = p->one_shot_deadline_ms;
    }
    if (p && p->building && P.gpl && !s->gs && !mesh_vs(s) && !p->info.pipeline && !p->fast_tried && p->lay) {
        int final = 0, build = wait_ms != 0 && draw_build;
        uint64_t p0 = perf_begin(), t0 = qpc();
        if (build && U.on) {
            job_lower(p->job);   // its worker would only wait for the same libraries; it links them if this draw cannot
            lib_wait(s, &k, (Layout *)p->lay, end);   // releases P.lock while waiting
            build = 0;                                // links here when both libraries exist (gpl_link build 0)
        } else if (build) {
            p->fast_tried = 1;   // the job, if a worker takes it meanwhile, goes straight to the optimised build
        }
        VkPipeline f = p->info.pipeline ? VK_NULL_HANDLE : gpl_link(s, &k, p->lay, build, &final);   // build 1 releases P.lock
        perf_end(PERF_PIPE, p0);
        if (wait_ms && draw_build) {
            double ms = ms_since(t0);
            P.g.dlinks++, P.g.dlink_ms += ms;
            if (ms > P.g.dlink_max) P.g.dlink_max = ms;
            if (ms > 5.0 && P.g.slow++ < 16)
                rt_log("vk_pipeline: gpl: a waiting draw %s its libraries in %.1f ms", U.on ? "got" : "built", ms);
        }
        if (final) p->fast_tried = 1;
        if (f && !p->info.pipeline) {
            p->fast = p->info.pipeline = f;
            job_lower(p->job);
            plat_cond_broadcast(&done_cv);
        } else if (f) {
            vkDestroyPipeline(P.device, f, NULL);   // a worker published one while the lock was free
        }
    }
    if (p && p->building && !p->info.pipeline) {
        promote(p);
        uint64_t p0 = wait_ms ? perf_begin() : 0;   // the wait counts as the render thread's pipeline time
        double now;
        while (p->building && !p->info.pipeline && wait_ms && (now = now_ms()) < end)
            plat_cond_wait_ms(&done_cv, &P.lock, (uint32_t)(end - now) + 1);
        if (wait_ms && p->building && !p->info.pipeline) p->waited_out = 1;
        if (wait_ms) perf_end(PERF_PIPE, p0);
        if (wait_ms && !p->building) perf_count(PERF_CNT_PIPE_WAIT);
    }
    int wait = p && p->building && !p->info.pipeline;
    if (wait) P.stats.pending_skips++;
    VkPipeline pipe = p ? p->info.pipeline : VK_NULL_HANDLE;
    plat_mutex_unlock(&P.lock);
    if (wait) {
        perf_count(PERF_CNT_PIPE_SKIP);
        if (pending) *pending = 1;
        return VK_NULL_HANDLE;
    }
    if (!p) return VK_NULL_HANDLE;
    if (info) *info = &p->info;
    return pipe;
}

VkPipeline vk_pipeline_graphics(const VkPipeState *s, const VkPipeInfo **info) {
    return graphics(s, info, 0, NULL, 0);
}

VkPipeline vk_pipeline_graphics_async(const VkPipeState *s, const VkPipeInfo **info, int *pending, uint32_t wait_ms) {
    return graphics(s, info, A.on, pending, wait_ms);
}

// ---- boot prewarm from the key file (comment at "pipeline keys")

static int id_cmp(const void *x, const void *y) { return memcmp(x, y, 32); }

// Builds queued prebuild jobs until the low queue is empty, beside the pipe_worker pool (same bookkeeping).
static void warm_builder(void *arg) {
    (void)arg;
    plat_mutex_lock(&P.lock);
    while (A.low) {
        Job *j = A.low;
        if ((A.low = j->next)) A.low->prev = NULL;
        else A.low_tail = NULL;
        j->next = j->prev = NULL;
        j->low = 0;                    // not on a pool thread: promote() has nothing to raise
        plat_mutex_unlock(&P.lock);
        tl_deferred = 0;
        VkPipeline pipe = create_graphics(&j->s, &j->k, j->l, j->fs_off, P.cache);
        int deferred = !pipe && tl_deferred;   // the guest started meanwhile (TOTK_PREWARM_WAIT_MS ran out)
        plat_mutex_lock(&P.lock);
        if (pipe || !j->p->fast) j->p->info.pipeline = pipe;   // a failed optimised build keeps the fast-linked one
        if (pipe && j->p->fast) P.g.swaps++;
        j->p->building = 0;
        j->p->job = NULL;
        if (pipe) {
            P.stats.graphics++, P.stats.async_built++, K.built++;
        } else if (deferred) {
            K.failed++;   // its record stays in the key file for the next start
        } else {
            P.stats.failures++, K.failed++;
            rt_log("vk_pipeline: graphics pipeline not built: pipeline creation failed");
        }
        K.promoted += j->demanded;
        K.left--;
        if (!--A.inflight) plat_cond_broadcast(&A.idle);
        plat_cond_broadcast(&done_cv);
        free(j);
    }
    plat_mutex_unlock(&P.lock);
    return;
}

static void lb_start(const uint32_t pr_hist[16], const uint32_t fs_hist[4]);

void vk_pipeline_prewarm(void) {
    const char *gpu = rt_config()->gpu;
    uint32_t pr_hist[16] = {0}, fs_hist[4] = {0};   // library state of the recorded pipelines (lb_start)
    if (!A.on || (gpu && !strcmp(gpu, "null"))) return;
    if (!cfg_bool("TOTK_PIPE_PREBUILD")) {   // prebuild_start logs the off case
        if (ensure_init()) lb_start(pr_hist, fs_hist);
        return;
    }
    double t0 = now_ms();
    K.warm = 1;
    if (!ensure_init() || !K.file_n) {
        rt_log("vk_pipeline: prewarm: no pipeline keys (%s)", K.path[0] ? K.path : "<none>");
        if (P.ready) lb_start(pr_hist, fs_hist);
        return;
    }
    {   // jogar-ocarina.ps1 runs TOTK_PREWARM_ONLY on every AMD card: without binaries to keep, it would compile everything
        // at every start for nothing. After ensure_init, which drops the records unless K.warm is set.
        const char *only = getenv("TOTK_PREWARM_ONLY");
        if (only && !strcmp(only, "1") && !amd_keeps_binaries()) {
            rt_log("vk_pipeline: prewarm only: this driver keeps no pipeline binaries, nothing to prepare");
            return;
        }
    }
    const KeyRec *r = (const KeyRec *)K.file;
    size_t n = K.file_n, m = 0;
    uint8_t (*ids)[32] = malloc(3 * n * 32);
    const GfxShader **sh = calloc(3 * n, sizeof *sh);
    if (!ids || !sh) {
        free(ids), free(sh);
        return;
    }
    for (size_t i = 0; i < n; i++) {
        memcpy(ids[m++], r[i].vs, 32);
        if (r[i].has_fs) memcpy(ids[m++], r[i].fs, 32);
        if (r[i].has_ts) memcpy(ids[m++], r[i].ts, 32);
    }
    qsort(ids, m, 32, id_cmp);
    size_t u = 0;
    for (size_t i = 0; i < m; i++)
        if (!u || memcmp(ids[u - 1], ids[i], 32)) memcpy(ids[u++], ids[i], 32);
    int cpus = (int)plat_cpu_usable();
    const char *pe = cfg_get("TOTK_PREWARM_XLATE");   // no fixed default: half the CPUs, at most 16, when unset
    int procs = pe && *pe ? atoi(pe) : cpus / 2 > 16 ? 16 : cpus / 2;
    gfx_shader_prewarm((const uint8_t(*)[32])ids, u, sh, cpus, procs < 1 ? 1 : procs);
    double t1 = now_ms();

    if (P.indep) {   // vertex output unions first: workers may take a job before the queue is complete
        uint64_t pair = 0, uni = 0, pairs = 0;
        for (size_t i = 0; i < n; i++) {
            const uint8_t (*v)[32] = bsearch(r[i].vs, ids, u, 32, id_cmp);
            const uint8_t (*f)[32] = r[i].has_fs ? bsearch(r[i].fs, ids, u, 32, id_cmp) : NULL;
            if (!v || !f || !sh[v - ids] || !sh[f - ids]) continue;
            GfxShaderIo io[32];
            vs_outs(sh[v - ids], sh[f - ids], io);
            pair += (uint64_t)__builtin_popcount(gfx_shader_inputs(sh[f - ids], NULL)), pairs++;
        }
        for (uint32_t j = 0; j < VS_OUTS_CAP; j++) uni += VO.vs[j] ? (uint64_t)__builtin_popcount(VO.mask[j]) : 0;
        rt_log("vk_pipeline: vertex output unions: %u vertex shaders, %.2f locations each; %llu pairs read %.2f each",
               VO.n, VO.n ? (double)uni / VO.n : 0.0, (unsigned long long)pairs, pairs ? (double)pair / pairs : 0.0);
    }
    uint32_t queued = 0, dropped = 0, known = 0, deferred = 0;
    struct DeferRec { size_t i; const GfxShader *vs, *fs, *ts; } *dr = calloc(n ? n : 1, sizeof *dr);
    plat_mutex_lock(&P.lock);
    K.t0 = t1;
    K.loading = 1;
    for (size_t i = 0; i < n; i++) {
        const uint8_t (*v)[32] = bsearch(r[i].vs, ids, u, 32, id_cmp);
        const uint8_t (*f)[32] = r[i].has_fs ? bsearch(r[i].fs, ids, u, 32, id_cmp) : NULL;
        const uint8_t (*tq)[32] = r[i].has_ts ? bsearch(r[i].ts, ids, u, 32, id_cmp) : NULL;
        const GfxShader *vs = v ? sh[v - ids] : NULL, *fs = f ? sh[f - ids] : NULL, *ts = tq ? sh[tq - ids] : NULL;
        if (!vs || (r[i].has_fs && !fs) || (r[i].has_ts && !ts)) {
            dropped++;
            continue;
        }
        {   // LOCAL TEST (Ocarina, AMD): a big (guarded) pipeline is left to play: built here on every core at once they
            // held the boot for 15 minutes and ran the driver out of memory. OOT_PREWARM_BIG=1 builds them here.
            // RDNA 1/2: the prepared binaries and big_defer handle these.
            static int big = -1;
            if (big < 0) big = getenv("OOT_PREWARM_BIG") != NULL;
            if (!big && A.on && vk_caps.amd && !amd_keeps_binaries() && graphics_shader_words(vs) + graphics_shader_words(fs) + graphics_shader_words(ts) >= big_cs_words()) {
                if (dr) dr[deferred] = (struct DeferRec){i, vs, fs, ts};
                deferred++;
                continue;
            }
        }
        VkPipeState s = {0};
        memcpy(&s, r[i].state, sizeof r[i].state);
        s.vs = vs, s.fs = fs, s.ts = ts;
        GKey k;
        gkey_fill(&s, &k);
        pr_hist[(k.clamp & 1) | (k.poly_mode & 3) << 1 | (k.minus_w & 1) << 3]++;
        fs_hist[k.samples >= 8 ? 3 : k.samples >= 4 ? 2 : k.samples >= 2 ? 1 : 0]++;
        uint64_t h = key_hash(&k, sizeof k);
        int q = 0;
        if (pipe_find(&k, sizeof k, h)) known++;
        else pipe_miss(&s, &k, h, 2, &q), queued += (uint32_t)q;
    }
    K.loading = 0;
    plat_mutex_unlock(&P.lock);
    free(ids), free(sh);

    PlatThread *t[64];
    int nt = 0;
    // On AMD two per 8 GB of RAM: a record without a pipeline binary (a new driver) compiles, and that takes gigabytes
    // (compile memory gate). TOTK_PREWARM_ONLY (the game closed) uses every core: the gate keeps the RAM.
    int wt = cpus;
    const char *po = getenv("TOTK_PREWARM_ONLY");
    if (amd_keeps_binaries() && !(po && !strcmp(po, "1")) && wt > 2 * mg_compilers_by_ram()) wt = 2 * mg_compilers_by_ram();
    for (int i = 0; i < wt && i < 64 && queued; i++)
        if ((t[nt] = plat_thread_new(warm_builder, NULL, "pipe prewarm"))) nt++;
    if (!nt) warm_builder(NULL);
    for (int i = 0; i < nt; i++)
        while (!plat_thread_join_ns(t[i], 2000000000u))
            rt_log("vk_pipeline: prewarm: %u of %u pipelines, %.0f ms", K.built + K.failed, queued, now_ms() - t1);
    plat_mutex_lock(&P.lock);
    while (K.left) plat_cond_wait(&done_cv, &P.lock);   // jobs a pool worker took
    rt_log("vk_pipeline: prewarm: %zu records in %s: %u pipelines built, %u failed, %u already built, %u deferred, %u dropped "
           "(shader neither cached nor stored); shaders %.0f ms, pipelines %.0f ms on %d threads, total %.0f ms",
           n, K.path, K.built, K.failed, known, deferred, dropped, t1 - t0, now_ms() - t1, nt, now_ms() - t0);
    // LOCAL TEST (Ocarina, AMD): the deferred (big) records go to the pool's low queue now that the boot no longer
    // waits: large ones one at a time after a cache probe (pipe_worker), so a scene's big pipelines are usually ready
    // before a draw asks for them instead of all queuing behind the first that compiles. OOT_PREWARM_BG=0: left to play.
    const char *bge = getenv("OOT_PREWARM_BG");
    if (dr && deferred && !(bge && !strcmp(bge, "0"))) {
        uint32_t bg = 0, before = K.left;
        prewarm_big_bg = 1, K.loading = 1;
        for (uint32_t d = 0; d < deferred; d++) {
            VkPipeState s = {0};
            memcpy(&s, r[dr[d].i].state, sizeof r[dr[d].i].state);
            s.vs = dr[d].vs, s.fs = dr[d].fs, s.ts = dr[d].ts;
            GKey k;
            gkey_fill(&s, &k);
            uint64_t h = key_hash(&k, sizeof k);
            int q = 0;
            if (!pipe_find(&k, sizeof k, h)) pipe_miss(&s, &k, h, 2, &q), bg += (uint32_t)q;
        }
        prewarm_big_bg = 0, K.loading = 0;
        K.built = K.failed = K.promoted = 0;   // the next "prebuild done" line counts these alone
        rt_log("vk_pipeline: prewarm: %u big pipelines queued in the background (%u were waiting before)", bg, before);
    }
    free(dr);
    free(K.file);
    K.file = NULL;
    plat_mutex_unlock(&P.lock);
    lb_start(pr_hist, fs_hist);
}

// ensure_init's hook: without a prewarm (TOTK_PIPE_PREBUILD=0, or the device came up first elsewhere) the records
// are not built. Caller holds no lock.
static void prebuild_start(void) {
    plat_mutex_lock(&P.lock);
    if (K.file && !K.warm) {
        rt_log("vk_pipeline: prebuild off%s; %zu records not built",
               !cfg_bool("TOTK_PIPE_PREBUILD") ? " (TOTK_PIPE_PREBUILD=0)" : " (no boot prewarm)", K.file_n);
        free(K.file);
        K.file = NULL;
    }
    plat_mutex_unlock(&P.lock);
}

// A compute pipeline for cs in layout l; VK_NULL_HANDLE on failure. No lock needed (P.cache is driver-synchronised).
// LOCAL TEST (Ocarina): SPIR-V words from which a compute shader builds in the background, unoptimized: shaders of 65,000 and 172,000 words froze
// the replay for a minute or more. OOT_BIG_CS_WORDS sets it, 0 turns it off.
static uint64_t big_cs_words(void) {
    static uint64_t w = ~0ull;
    if (w == ~0ull) {
        const char *e = getenv("OOT_BIG_CS_WORDS");
        w = e ? strtoull(e, NULL, 0) : (1u << 15);
        if (!w) w = ~0ull - 1;
    }
    return w;
}

static _Thread_local VkPipelineCache cs_build_cache;   // the big compute worker's own cache, see big_cs_worker
// LOCAL TEST (Ocarina): --bench-cs hands compute_create another SPIR-V for the same interface (vk_pipeline_bench_cs).
static _Thread_local const uint32_t *cs_code_override;
static _Thread_local uint64_t cs_words_override;
static VkPipeline compute_create(const GfxShader *cs, const Layout *l) {
    uint64_t words;
    const uint32_t *code = gfx_shader_spirv(cs, &words);
    if (cs_code_override) code = cs_code_override, words = cs_words_override;
    VkShaderModule m = make_module(code, words);
    if (!m) return VK_NULL_HANDLE;
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rqc;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = l->layout};
    pin_subgroup32(&ci.stage, &rqc, code, words);
    {   // LOCAL TEST (Ocarina): a big guarded shader takes the driver minutes and over 30 GB: built without optimization, as Lichii #273 does for
        // two shaders (they run slower). OOT_BIG_CS_OPT=1 keeps the optimizer.
        // LOCAL TEST (Ocarina, AMD): on AMD's driver the unoptimized build is the slow path: on an RX 580 the 166958-word forest
        // shader takes 29 s and 1.1 GB optimized, minutes and 10+ GB without, and the unoptimized ones cost ~130 ms of GPU per
        // frame. AMD keeps the optimizer (and retries without it if the driver refuses, below); OOT_BIG_CS_NOOPT=1 turns it off there.
        static int opt = -1;
        if (opt < 0) opt = vk_caps.amd ? getenv("OOT_BIG_CS_NOOPT") == NULL : getenv("OOT_BIG_CS_OPT") != NULL;
        if (!opt && words >= big_cs_words()) ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    }
    {   // LOCAL TEST (Ocarina): shaders that compare a lane with their thread index, or count lanes, need every subgroup full; NVIDIA
        // otherwise splits invocations differently from run to run, which made results differ from frame to frame.
        uint32_t ls[3];
        gfx_shader_local_size(cs, ls);
        if (ci.stage.pNext && ls[0] && ls[0] % vk_caps.sg_size == 0) ci.stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    }
    // A compute pipeline the driver crashed in (pipeline-crash.txt, as for graphics; NotRexed/Hookshot #758 saw an
    // access violation in AMD's driver while compute pipelines were prepared at start) is built without optimization
    // after one crash and skipped after a second, so a start that crashed does not crash again in the same place.
    static _Thread_local char note[96];
    uint64_t h = 1469598103934665603ull;   // FNV-1a of the SPIR-V, as for graphics
    for (uint64_t j = 0; j < words; j++) h = (h ^ code[j]) * 1099511628211ull;
    snprintf(note, sizeof note, "compute pipeline: cs %llu words %016llx", (unsigned long long)words, (unsigned long long)h);
    int crashes = crash_count(note);
    if (crashes >= 2) {
        static volatile int32_t told;
        if (plat_inc32(&told) <= 8) rt_log("vk_pipeline: not built, the driver crashed in it before: %s", note);
        vkDestroyShaderModule(P.device, m, NULL);
        return VK_NULL_HANDLE;
    }
    if (crashes) {
        ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
        rt_log("vk_pipeline: built without optimization, the driver crashed in it before: %s", note);
    }
    VkPipeline pipe = VK_NULL_HANDLE;
    plat_crash_note(note);   // also over pb_load: a driver crash loading a binary must not repeat at every start
    // Pipeline binaries (see graphics pipelines): neither the driver's disk cache nor ours kept the big compute shaders on
    // AMD, so each run compiled them again, for minutes and gigabytes (130,000-166,000 words at the first area, RX 5500 XT,
    // driver 26.10.1). A kept one is created in milliseconds. One the driver crashed in takes the normal path (crash_count).
    char pbp[PLAT_PATH_MAX * 2 + 160];
    int pbk = !crashes && pb_on() && pb_path(&ci, pbp, sizeof pbp);
    // The driver's key need not cover the flag: a big shader built with the optimizer must not load a binary kept from a
    // build without it, which ran slower. Those stay on the disk, unused.
    if (pbk && !(ci.flags & VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT) && words >= big_cs_words()) {
        size_t pn = strlen(pbp);
        if (pn > 4 && pn + 4 < sizeof pbp) strcpy(pbp + pn - 4, ".opt.bin");
    }
    if (pbk && plat_file_exists(pbp)) {
        if ((pipe = pb_load(NULL, &ci, pbp))) plat_inc32(&PB.loaded);
        else if (plat_inc32(&PB.refused) <= 8) rt_log("vk_pipeline: pipeline binary refused, compiling: %s", pbp);
    }
    // Background-built ones (big_cs_words, 32,768 words) count as big for the gate: one at a time. A 79,000-word one
    // took 3 GB in 20 s at the preparation, and twelve threads started several at once.
    int big = big_pipeline(words) || (amd_keeps_binaries() && words >= big_cs_words());
    if (!pipe && big_defer(words, "compute shader")) {   // big compiles left for the next start: its dispatches stay skipped
        plat_crash_note(NULL);
        vkDestroyShaderModule(P.device, m, NULL);
        return VK_NULL_HANDLE;
    }
    if (!pipe) {
        VkPipelineCreateFlags2CreateInfoKHR f2 = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR,
                                                  .pNext = ci.pNext,
                                                  .flags = (VkPipelineCreateFlags2KHR)ci.flags | VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR};
        if (pbk) ci.pNext = &f2;
        mg_enter(big);
        uint64_t c0 = plat_now_ns();
        if (big) rt_log("vk_pipeline: compiling a big compute shader: %llu words", (unsigned long long)words);
        // A capturing compile takes no pipeline cache (Vulkan requires VK_NULL_HANDLE with the capture flag).
        VkPipelineCache build_cache = pbk ? VK_NULL_HANDLE : cs_build_cache ? cs_build_cache : P.cache;
        int shared = build_cache && build_cache == P.cache;
        if (shared) plat_mutex_lock_shared(&cache_access);
        if (vkCreateComputePipelines(P.device, build_cache, 1, &ci, NULL, &pipe) != VK_SUCCESS) pipe = VK_NULL_HANDLE;
        int fallback = 0;   // built without optimization after the optimized build failed: never kept (below)
        if (!pipe && vk_caps.amd && !(ci.flags & VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT)) {   // as for graphics, above
            ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
            f2.flags |= VK_PIPELINE_CREATE_2_DISABLE_OPTIMIZATION_BIT_KHR;
            if (vkCreateComputePipelines(P.device, build_cache, 1, &ci, NULL, &pipe) != VK_SUCCESS) pipe = VK_NULL_HANDLE;
            fallback = 1;
            rt_log("vk_pipeline: compute pipeline built again without optimization: %s", pipe ? "ok" : "failed too");
        }
        if (shared) plat_mutex_unlock_shared(&cache_access);
        mg_leave(big);
        if (pbk) ci.pNext = f2.pNext;
        double c_s = (double)(plat_now_ns() - c0) / 1e9;
        if (pbk && pipe) {
            // pbp is the optimized build's name: the slower fallback must not load from it on every later start. Below
            // TOTK_PIPE_BINARY_MS (default 0 here) a pipeline compiles again next start: a capturing compile is not in the cache.
            if (!fallback && c_s * 1000.0 >= PB.save_ms) {
                int ok = pb_save(pipe, pbp);
                if (ok) plat_inc32(&PB.saved);
                if (!ok || c_s >= 1.0)
                    rt_log("vk_pipeline: compute pipeline binary %s after a %.1f s compile, %llu words", ok ? "kept" : "NOT kept",
                           c_s, (unsigned long long)words);
            } else {
                pb_release(pipe);
            }
        }
    }
    plat_crash_note(NULL);
    vkDestroyShaderModule(P.device, m, NULL);
    return pipe;
}

int vk_pipeline_keeps_binaries(void) { return P.ready ? amd_keeps_binaries() : -1; }

// The layout of a compute shader in *B; NULL with *why on failure. Caller holds P.lock.
static Layout *compute_layout(const GfxShader *cs, Bindings *B, const char **why) {
    Layout *l = NULL;
    if (gfx_shader_stage(cs) != GFX_STAGE_CS) *why = "not a compute shader";
    else if (!add_stage(B, cs, 0)) *why = "too many bindings";
    else if (!(l = layout_get(B))) *why = "descriptor layout creation failed";
    return l;
}

// ---- stage library prebuild
// Two places build what a draw would otherwise build while the game waits:
// - totk.exe --prepare-shaders (lb_warm_prepared), after the translation, on every core: for each prepared variant
//   (shader-prepared.bin) the pre-rasterization library of a vertex shader, the fragment-shader library of a fragment
//   shader and the pipeline of a compute shader, each built from a private shader copy and destroyed at once. What
//   stays is the driver's own disk cache: on AMD a library built again in a later process takes 0.1 ms instead of
//   55 ms (vertex) / 28 ms (fragment). Our pipeline-cache.bin is not written there:
//   it would grow by about 10 KB per library (~240 MB) and be read into memory at every boot.
// - boot (lb_worker), after the pipeline-key prewarm, on cpus/4 threads at PLAT_PRIO_IDLE (the short P.lock
//   sections and shader loads at normal priority, so a preempted thread never holds the lock the replay needs):
//   compute pipelines of every recorded and prepared compute shader, kept (vk_pipeline_compute builds compute
//   synchronously on the replay thread; 163 prepared), and the libraries of the first TOTK_LIB_PREBUILD_KEEP variants
//   play recorded (priority 0), kept in libs[] with their shaders. KEEP defaults to 0: a kept library costs 140-460 KB
//   of process memory (10,000 kept took 9 GB in run 1) to save a rebuild the driver cache makes 0.1 ms.
// Library state: the vertex output union known now (vs_union_peek; the prewarm fills it from pipeline-keys.bin, the
// preparation has none), the clamp, polygon mode and depth mode most recorded pipelines use (defaults without a key
// file: clamp 0, fill, -w..w), single-sampled, no dual source; the state dynamic on this GPU is 0 in every key.
// A draw that meets such a variant in a new pipeline then gets its libraries from libs[] or the driver cache and
// fast-links (gpl_link). TOTK_LIB_PREBUILD=0 turns the boot part off, =N stops it after N variants;
// TOTK_LIB_PREBUILD_THREADS sets its thread count.
#define LB_MAX_THREADS 16

static struct {
    GfxWarmEnt *e;
    size_t n;
    volatile int64_t next;
    volatile int32_t stop, running;
    int threads;
    PlatThread *th[LB_MAX_THREADS];
    uint8_t clamp, poly, minus_w, samples;             // library state (GKey fields)
    double t0, ms[3];                                  // ms: build time per kind (vertex, fragment, compute)
    // warmed: driver cache only; nolib: vertex and fragment variants skipped with the graphics pipeline library off
    uint64_t built[3], have[3], fail[3], warmed[3], noshader, nolib, done, cs_draw, big_skipped;
    double warm_ms[3];
    volatile int32_t kept;                                // libraries entered in libs[] so far
    int32_t keep_n;
    int idle;                                          // build at PLAT_PRIO_IDLE (boot), normal (preparation)
    PlatMutex stat;
} LB = {.stat = PLAT_MUTEX_INIT};

// LOCAL TEST (Ocarina): compute shaders with guarded candidates reach 17-30 MB of SPIR-V; the driver takes a minute or more and over 30 GB
// for one, and it ran on the replay thread under P.lock: the game froze at the first broken pot. A big shader is built here first, by
// big_cs_workers() threads, and its dispatches are skipped until the pipeline is in the live table.
// Each worker takes the smallest queued shader, and at most one of them builds a huge one (big_cs_huge_words()) at a time, so memory
// stays bounded and a huge build never holds the others. One worker taking the queue in order let a 346,302-word Kokiri shader
// (c968ad44272dcbb4, 504 s on NVIDIA 621.01) hold every big shader met after it: the exposure (435a0138b9013040) and volumetric fog
// (ee7764a6b1ef459c, 27ee9883e2a20323) passes and the falling leaves were skipped for the whole of a first session after an update
// (washed-out picture, leaves still in the air, #167).
// ponytail: a linear table of 8192 shaders and polling workers; a dispatch skipped this way loses its effect for those frames.
// TOTK_ASYNC_CS_WORDS (default: the big-shader size; opt-in lower, e.g. 1024 on AMD): compute shaders of at least this many words
// go through the same background build, so the replay thread never compiles one under P.lock. AMD's Windows compiler took
// seconds to minutes for medium-sized shaders that NVIDIA builds in milliseconds, and the game froze for minutes at the
// first area (RX 5500 XT, driver 26.10.1). Only big shaders (big_cs_words) are built without optimization. Medium shaders
// get ASYNC_CS_MEDIUM workers of their own, so they do not queue behind a giant one.
enum { BIG_CS_N = 8192, BIG_CS_MAX_THREADS = 4, ASYNC_CS_MEDIUM = 2 };
enum { BIG_CS_QUEUED, BIG_CS_BUILDING, BIG_CS_DONE };
static struct { const GfxShader *cs; uint64_t words; volatile int state; uint8_t big; } big_cs[BIG_CS_N];
static volatile int big_cs_n;
static int big_cs_thread;
static PlatMutex big_cs_lock = PLAT_MUTEX_INIT;   // big_cs[].state from BIG_CS_QUEUED, big_cs_huge_busy
static int big_cs_huge_busy;

// OOT_BIG_CS_THREADS: workers building big compute shaders (at most BIG_CS_MAX_THREADS). Default 2 with 24 GB of RAM or more,
// else 1: the 346,302-word shader alone took 5.9 GB of RAM for 504 s (RTX 4090, driver 621.01), and the README's minimum is 16 GB.
static int big_cs_workers(void) {
    static int n;
    if (!n) {
        const char *e = getenv("OOT_BIG_CS_THREADS");
        float used = 0, total = 0, proc = 0;
        if (!e) plat_mem_info(&used, &total, &proc);
        int v = e ? atoi(e) : total >= 24.0f * 1024 ? 2 : 1;
        n = v < 1 ? 1 : v > BIG_CS_MAX_THREADS ? BIG_CS_MAX_THREADS : v;
    }
    return n;
}

// OOT_BIG_CS_HUGE: SPIR-V words from which a big compute shader counts as huge: one worker at a time builds those (default 262144;
// Kokiri meets 346-393 K word shaders, the rest are 33-260 K).
static uint64_t big_cs_huge_words(void) {
    static uint64_t w;
    if (!w) {
        const char *e = getenv("OOT_BIG_CS_HUGE");
        w = e && strtoull(e, NULL, 0) ? strtoull(e, NULL, 0) : 262144;
    }
    return w;
}

// The queued shader a worker builds next: the smallest, skipping huge ones while another worker builds one, and big ones for
// a medium-only worker. -1: none.
static int big_cs_take(int medium_only) {
    int best = -1, n = __atomic_load_n(&big_cs_n, __ATOMIC_ACQUIRE);
    plat_mutex_lock(&big_cs_lock);
    for (int i = 0; i < n; i++) {
        if (__atomic_load_n(&big_cs[i].state, __ATOMIC_RELAXED) != BIG_CS_QUEUED) continue;
        if (medium_only && big_cs[i].big) continue;
        if (big_cs_huge_busy && big_cs[i].words >= big_cs_huge_words()) continue;
        if (best < 0 || big_cs[i].words < big_cs[best].words) best = i;
    }
    if (best >= 0) {
        __atomic_store_n(&big_cs[best].state, BIG_CS_BUILDING, __ATOMIC_RELAXED);
        if (big_cs[best].words >= big_cs_huge_words()) big_cs_huge_busy = 1;
    }
    plat_mutex_unlock(&big_cs_lock);
    return best;
}

// A worker is done with shader i (built, failed or left for the next start): another may take a huge one.
static void big_cs_release(int i) {
    plat_mutex_lock(&big_cs_lock);
    if (big_cs[i].words >= big_cs_huge_words()) big_cs_huge_busy = 0;
    plat_mutex_unlock(&big_cs_lock);
}

// LOCAL TEST (Ocarina): a copy of the shared cache, for --bench-cs with OOT_BENCH_SEED (play seeds its private caches with
// cache_private). Falls back to an empty cache.
static VkPipelineCache cache_copy(void) {
    VkPipelineCacheCreateInfo cci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VkPipelineCache c = VK_NULL_HANDLE;
    size_t size = 0;
    void *data = NULL;
    // Exclusive, as vk_pipeline_save takes it: vkGetPipelineCacheData while other threads build into the same cache crashed
    // Intel's driver (#90).
    plat_mutex_lock(&cache_access);
    if (P.cache && vkGetPipelineCacheData(P.device, P.cache, &size, NULL) == VK_SUCCESS && size && (data = malloc(size)) &&
        vkGetPipelineCacheData(P.device, P.cache, &size, data) == VK_SUCCESS)
        cci.initialDataSize = size, cci.pInitialData = data;
    plat_mutex_unlock(&cache_access);
    if (vkCreatePipelineCache(P.device, &cci, NULL, &c) != VK_SUCCESS) {
        cci.initialDataSize = 0, cci.pInitialData = NULL;
        if (vkCreatePipelineCache(P.device, &cci, NULL, &c) != VK_SUCCESS) c = VK_NULL_HANDLE;
    }
    free(data);
    return c;
}

static uint64_t async_cs_words(void) {
    static uint64_t w = ~0ull;
    if (w == ~0ull) {
        const char *e = getenv("TOTK_ASYNC_CS_WORDS");
        w = e && *e ? strtoull(e, NULL, 0) : big_cs_words();   // opt-in: a skipped dispatch can leave stale indirect arguments
        if (!w || w > big_cs_words()) w = big_cs_words();
        rt_log("vk_pipeline: compute shaders of %llu words or more are built in the background (TOTK_ASYNC_CS_WORDS)",
               (unsigned long long)w);
    }
    return w;
}

static void big_cs_worker(void *arg) {
    int medium_only = (int)(intptr_t)arg;   // 1: only shaders below big_cs_words
    for (;;) {
        int next = big_cs_take(medium_only);
        if (next < 0) {
            plat_sleep_ms(5);
            continue;
        }
        const GfxShader *cs = big_cs[next].cs;
        const char *why = NULL;
        Bindings B = {0};
        plat_mutex_lock(&P.lock);
        Layout *l = compute_layout(cs, &B, &why);
        plat_mutex_unlock(&P.lock);
        uint64_t t0 = plat_now_ns();
        if (rt_trace & RT_TRACE_PERF) rt_log("vk_pipeline: big compute shader %d: building", next);
        // A cache of its own: a compile of minutes inside the shared cache held every other pipeline on the replay thread
        // behind it, and the game froze for over a minute. The result is merged into the shared one when it is done.
        VkPipelineCache own = cache_private();   // upstream v0.2.4: seeded from pipeline-cache.bin
        cs_build_cache = own;
        tl_deferred = 0;
        VkPipeline pipe = why ? VK_NULL_HANDLE : compute_create(cs, l);
        cs_build_cache = VK_NULL_HANDLE;
        if (!pipe && tl_deferred) {   // left for the next start (big compiles): not done, so its dispatches stay skipped
            if (own) vkDestroyPipelineCache(P.device, own, NULL);
            big_cs_release(next);     // stays BIG_CS_BUILDING: no worker takes it again this run
            continue;
        }
        // Kept, as lb_compute keeps its pipelines: built again on the replay thread, the driver's cache missed and the game froze for
        // the whole compile a second time (62 s after big compute shader 31, a 21-minute run).
        int built = pipe != VK_NULL_HANDLE;
        if (pipe) {
            struct { const GfxShader *cs; uint32_t compute; } k = {cs, 1};
            uint64_t h = key_hash(&k, sizeof k);
            plat_mutex_lock(&P.lock);
            if (!pipe_find(&k, sizeof k, h)) {
                pipe_add(&k, sizeof k, h, pipe, l, &B, VK_PIPELINE_BIND_POINT_COMPUTE, 0);
                P.stats.compute++;
                pipe = VK_NULL_HANDLE;
            }
            plat_mutex_unlock(&P.lock);
            if (pipe) vkDestroyPipeline(P.device, pipe, NULL);
        }
        double secs = (double)(plat_now_ns() - t0) / 1e9;
        if (big_cs[next].big || secs >= 1.0 || !built)
            rt_log("vk_pipeline: %s compute shader %d (%llu words) built in %.1f s%s", big_cs[next].big ? "big" : "background", next,
                   (unsigned long long)big_cs[next].words, secs, built ? "" : " (failed)");
        big_cs_release(next);
        __atomic_store_n(&big_cs[next].state, BIG_CS_DONE, __ATOMIC_RELEASE);
        if (own) {
            if (built) cache_publish(own);
            else vkDestroyPipelineCache(P.device, own, NULL);
        }
    }
}

// LOCAL TEST (Ocarina): ocarina.exe --bench-cs builds one recorded compute shader the way big_cs_worker does, with no game,
// and prints the time and the process's peak commit, to compare ways of making the big guarded shaders cheaper to build.
// Without OOT_BENCH_WORDS it lists the recorded compute shaders of big_cs_words() or more. OOT_BENCH_WORDS=<n> picks the
// shader of that many SPIR-V words; OOT_BENCH_SPV=<file> builds that SPIR-V in its place (same interface, e.g. from
// spirv-opt); OOT_BENCH_SALT=1 changes the generator word so no driver cache can answer; OOT_BIG_CS_OPT=1 keeps the
// driver's optimizer, as in play.
int vk_pipeline_bench_cs(void) {
    vk_pipeline_async_enable();   // as the game does: keys_open then records the shaders met in play
    if (!ensure_init()) {
        fprintf(stderr, "bench-cs: no Vulkan device\n");
        return 1;
    }
    const char *we = getenv("OOT_BENCH_WORDS"), *spv = getenv("OOT_BENCH_SPV");
    uint64_t want = we ? strtoull(we, NULL, 0) : 0;
    GfxWarmEnt *all = NULL;
    size_t n = gfx_shader_warm_list(&all);
    const GfxShader *pick = NULL;
    for (size_t i = 0; i < n; i++) {
        if (all[i].stage != GFX_STAGE_CS) continue;
        const GfxShader *sh = gfx_shader_load_input(all[i].iid, 0);
        uint64_t w = 0;
        if (!sh || !gfx_shader_spirv(sh, &w)) continue;
        if (!want && w >= big_cs_words()) printf("bench-cs: recorded big compute shader of %llu words\n", (unsigned long long)w);
        if (want && w == want) pick = sh;
    }
    free(all);
    if (!want) return 0;
    if (!pick) {
        fprintf(stderr, "bench-cs: no recorded compute shader of %llu words\n", (unsigned long long)want);
        return 1;
    }
    uint32_t *code = NULL;
    uint64_t words = 0;
    if (spv) {
        uint64_t size = 0;
        code = (uint32_t *)plat_file_read(spv, &size);
        words = size / 4;
        if (!code || words < 5 || code[0] != 0x07230203u) {
            fprintf(stderr, "bench-cs: %s is not SPIR-V\n", spv);
            return 1;
        }
    } else {
        const uint32_t *c = gfx_shader_spirv(pick, &words);
        code = malloc(words * 4);
        if (!code) return 1;
        memcpy(code, c, words * 4);
    }
    if (getenv("OOT_BENCH_SALT")) code[2] ^= (uint32_t)plat_now_ns() | 1u;   // generator word: the driver's cache key changes
    Bindings B = {0};
    const char *why = NULL;
    plat_mutex_lock(&P.lock);
    Layout *l = compute_layout(pick, &B, &why);
    plat_mutex_unlock(&P.lock);
    if (!l) {
        fprintf(stderr, "bench-cs: no layout: %s\n", why ? why : "?");
        return 1;
    }
    VkPipelineCacheCreateInfo cci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VkPipelineCache own = VK_NULL_HANDLE;
    if (getenv("OOT_BENCH_SEED")) own = cache_copy();   // as big_cs_worker: pipeline-cache.bin may answer
    else vkCreatePipelineCache(P.device, &cci, NULL, &own);   // empty: our pipeline-cache.bin cannot answer
    float used0, total, proc0;
    plat_mem_info(&used0, &total, &proc0);
    cs_build_cache = own, cs_code_override = code, cs_words_override = words;
    uint64_t t0 = plat_now_ns();
    VkPipeline pipe = compute_create(pick, l);
    double s = (double)(plat_now_ns() - t0) / 1e9;
    cs_build_cache = VK_NULL_HANDLE, cs_code_override = NULL;
    float used1, proc1;
    plat_mem_info(&used1, &total, &proc1);
    printf("bench-cs: %llu words%s%s: %s in %.1f s; system memory in use %.0f MB before, %.0f MB after (of %.0f MB)\n",
           (unsigned long long)words, spv ? " from " : "", spv ? spv : "", pipe ? "built" : "FAILED", s, (double)used0,
           (double)used1, (double)total);
    if (pipe) vkDestroyPipeline(P.device, pipe, NULL);
    if (own) vkDestroyPipelineCache(P.device, own, NULL);
    free(code);
    return pipe ? 0 : 1;
}

// LOCAL TEST (Ocarina): big-compute.txt next to pipeline-cache.bin, a line "<input id> <origin>" for each compute shader of
// big_cs_words() or more the replay met, before DKB_SKIP_CS. A stored input does not keep its origin (nvn2_origin knows this
// process's lowered code only), so --warm-big-cs reads it here. The list also leaves out what play no longer meets: the shader
// cache keeps the inputs of older lowerings, and on an RX 580 one of those (4.3 M words, HMMA inlined) took 22 minutes and
// the next crashed the driver.
static void big_met_path(char *out, size_t cap) {
    snprintf(out, cap, "%s", P.cache_path);
    char *slash = strrchr(out, '\\'), *fslash = strrchr(out, '/');
    if (fslash > slash) slash = fslash;
    snprintf(slash ? slash + 1 : out, cap - (size_t)(slash ? slash + 1 - out : 0), "big-compute.txt");
}

void vk_pipeline_big_cs_met(const GfxShader *cs, uint64_t origin) {   // the replay thread
    static const GfxShader *met[256];
    static int nmet, loaded;
    static char *listed;   // the file as it was at the first call
    uint64_t words = 0;
    if (!cs || !gfx_shader_spirv(cs, &words) || words < big_cs_words()) return;
    for (int i = 0; i < nmet; i++)
        if (met[i] == cs) return;
    if (nmet == (int)(sizeof met / sizeof *met) || !ensure_init() || !P.cache_path[0]) return;
    met[nmet++] = cs;
    char path[sizeof P.cache_path], line[64 + 1 + 16 + 2];
    big_met_path(path, sizeof path);
    if (!loaded) {
        loaded = 1;
        FILE *f = fopen(path, "rb");
        size_t got = f && (listed = calloc(1, 1 << 16)) ? fread(listed, 1, (1 << 16) - 1, f) : 0;
        (void)got;
        if (f) fclose(f);
    }
    const uint8_t *id = gfx_shader_input_id(cs);
    for (int i = 0; i < 32; i++) snprintf(line + 2 * i, 3, "%02x", id[i]);
    if (listed && strstr(listed, line)) return;
    snprintf(line + 64, sizeof line - 64, " %016llx\n", (unsigned long long)origin);
    FILE *f = fopen(path, "ab");
    if (f) fputs(line, f), fclose(f);
}

// LOCAL TEST (Ocarina): ocarina.exe --warm-big-cs builds, with no game, the big compute shaders play met (big-compute.txt)
// into pipeline-cache.bin, one at a time, saving after each, so play never waits for them (big_cs_worker finds them through
// cache_private). On an RX 580 two of the forest's take about 4 minutes and 13-15 GB each the first time; afterwards well under a second.
// The shaders whose dispatches DKB_SKIP_CS skips are left out, matched by origin as the replay does.
int vk_pipeline_warm_big_cs(void) {
    vk_pipeline_async_enable();
    if (!ensure_init()) {
        fprintf(stderr, "warm-big-cs: no Vulkan device\n");
        return 1;
    }
    const char *se = getenv("DKB_SKIP_CS");   // main.c's game_defaults sets the default before the mode runs
    if (!se) se = "";
    uint64_t skip[32];
    int nskip = 0;
    for (char *end; *se && nskip < 32; se = *end ? end + 1 : end) skip[nskip++] = strtoull(se, &end, 16);
    char path[sizeof P.cache_path], line[256];
    big_met_path(path, sizeof path);
    FILE *f = P.cache_path[0] ? fopen(path, "rb") : NULL;
    if (!f) {
        printf("warm-big-cs: %s is missing: play once, until the scenes you want, then run this again\n", path);
        return 1;
    }
    const GfxShader **todo = NULL;
    uint64_t *tw = NULL;
    size_t m = 0, cap = 0, left_out = 0, absent = 0;
    while (fgets(line, sizeof line, f)) {
        uint8_t iid[32];
        unsigned v;
        int ok = strlen(line) >= 81 && line[64] == ' ';
        for (int i = 0; ok && i < 32; i++) ok = sscanf(line + 2 * i, "%2x", &v) == 1, iid[i] = (uint8_t)v;
        if (!ok) continue;
        uint64_t origin = strtoull(line + 65, NULL, 16);
        int skipped = 0;
        for (int k = 0; k < nskip; k++) skipped |= skip[k] == origin;
        if (skipped) { left_out++; continue; }
        const GfxShader *sh = gfx_shader_load_input(iid, 0);
        uint64_t w = 0;
        if (!sh || !gfx_shader_spirv(sh, &w)) { absent++; continue; }
        int drop = w < big_cs_words();
        for (size_t j = 0; j < m; j++) drop |= todo[j] == sh;
        if (drop) continue;
        if (m == cap) {
            cap = cap ? cap * 2 : 64;
            const GfxShader **nt = realloc(todo, cap * sizeof *todo);
            uint64_t *nw = nt ? realloc(tw, cap * sizeof *tw) : NULL;
            if (nt) todo = nt;
            if (nw) tw = nw;
            if (!nt || !nw) break;
        }
        todo[m] = sh, tw[m] = w, m++;
    }
    fclose(f);
    for (size_t i = 1; i < m; i++)   // smallest first: the quick ones are in the cache even if the run is stopped
        for (size_t j = i; j > 0 && tw[j] < tw[j - 1]; j--) {
            uint64_t t = tw[j]; tw[j] = tw[j - 1], tw[j - 1] = t;
            const GfxShader *s = todo[j]; todo[j] = todo[j - 1], todo[j - 1] = s;
        }
    printf("warm-big-cs: %zu big compute shaders to build, one at a time (closing this window stops it; what is done stays); %zu "
           "left out by DKB_SKIP_CS, %zu not translated by this translator yet (play does it)\n", m, left_out, absent);
    fflush(stdout);
    uint64_t t_all = plat_now_ns();
    int failed = 0;
    // OOT_WARM_BUDGET_S: no new build starts after that many seconds (the installer sets it; an AMD card took 3 hours for
    // the list, #188). What is left is built in play as before; the list is smallest first, so the quick ones are done.
    const char *be = getenv("OOT_WARM_BUDGET_S");
    double budget = be ? atof(be) : 0;
    for (size_t i = 0; i < m; i++) {
        double spent = (double)(plat_now_ns() - t_all) / 1e9;
        if (budget > 0 && spent > budget) {
            printf("warm-big-cs: time budget of %.0f s reached after %zu of %zu: the %zu left are built in play\n", budget, i, m,
                   m - i);
            break;
        }
        // "[i/m]": the installer's progress bar reads it (Instalar-Ocarina.ps1 Atualizar)
        printf("warm-big-cs: [%zu/%zu] building a compute shader of %llu words\n", i + 1, m, (unsigned long long)tw[i]);
        fflush(stdout);
        Bindings B = {0};
        const char *why = NULL;
        plat_mutex_lock(&P.lock);
        Layout *l = compute_layout(todo[i], &B, &why);
        plat_mutex_unlock(&P.lock);
        uint64_t t0 = plat_now_ns();
        VkPipeline pipe = l ? compute_create(todo[i], l) : VK_NULL_HANDLE;   // straight into P.cache: nothing else runs here
        double s = (double)(plat_now_ns() - t0) / 1e9;
        if (pipe) vkDestroyPipeline(P.device, pipe, NULL);
        else failed++;
        if (pipe && s > 1.0) vk_pipeline_save();   // a long one is kept at once
        printf("warm-big-cs: %zu/%zu: %llu words %s in %.1f s\n", i + 1, m, (unsigned long long)tw[i], pipe ? "built" : "FAILED", s);
        fflush(stdout);
    }
    vk_pipeline_save();
    printf("warm-big-cs: done in %.0f s, %d failed\n", (double)(plat_now_ns() - t_all) / 1e9, failed);
    free(todo), free(tw);
    return failed ? 1 : 0;
}

int vk_pipeline_compute_ready(const GfxShader *cs) {
    uint64_t words = 0;
    if (!cs || !ensure_init()) return 1;
    gfx_shader_spirv(cs, &words);
    if (words < async_cs_words()) return 1;
    int n = __atomic_load_n(&big_cs_n, __ATOMIC_RELAXED);   // only the replay thread adds
    for (int i = 0; i < n; i++)
        if (big_cs[i].cs == cs) return __atomic_load_n(&big_cs[i].state, __ATOMIC_ACQUIRE) == BIG_CS_DONE;
    if (n == BIG_CS_N) return 1;
    if (!big_cs_thread) {
        int started = 0;
        for (int i = 0; i < big_cs_workers(); i++)
            started += plat_thread_spawn(big_cs_worker, (void *)(intptr_t)0, "big compute", 0) != 0;
        big_cs_thread = started ? started : -1;
        rt_log("vk_pipeline: %d big compute workers (OOT_BIG_CS_THREADS), smallest first, one huge (%llu+ words, OOT_BIG_CS_HUGE) at a time",
               started, (unsigned long long)big_cs_huge_words());
        if (started && async_cs_words() < big_cs_words())
            for (int i = 0; i < ASYNC_CS_MEDIUM; i++) plat_thread_spawn(big_cs_worker, (void *)(intptr_t)1, "compute builder", 0);
    }
    if (big_cs_thread < 0) return 1;
    big_cs[n].cs = cs;
    big_cs[n].words = words;
    big_cs[n].big = words >= big_cs_words();
    __atomic_store_n(&big_cs[n].state, BIG_CS_QUEUED, __ATOMIC_RELAXED);
    __atomic_store_n(&big_cs_n, n + 1, __ATOMIC_RELEASE);
    rt_log("vk_pipeline: big compute shader %d queued (%llu words): its dispatches are skipped until it is built", n,
           (unsigned long long)words);
    return 0;
}

// TOTK_PREWARM_ONLY (boot.c, after the prewarm): every compute shader play recorded (shader-answers.bin) or prepared
// that builds in the background (big_cs_words or more), one at a time with the RAM the game would hold, each kept as a
// pipeline binary; in play they are then created in milliseconds instead of compiled for minutes. Shaders over
// TOTK_PREP_CS_MAX_WORDS (default 1,000,000) are left out: the guarded ones reach millions of words and over 30 GB.
static struct {
    const GfxShader **todo;
    size_t n;
    volatile int64_t next;
    volatile int32_t done, failed;
    double t0;
} PC;

static void prep_cs_worker(void *arg) {
    (void)arg;
    for (;;) {
        size_t i = (size_t)plat_inc64(&PC.next) - 1;
        if (i >= PC.n) return;
        Bindings B = {0};
        const char *why = NULL;
        plat_mutex_lock(&P.lock);
        Layout *l = compute_layout(PC.todo[i], &B, &why);
        plat_mutex_unlock(&P.lock);
        VkPipeline p = why ? VK_NULL_HANDLE : compute_create(PC.todo[i], l);
        if (p) vkDestroyPipeline(P.device, p, NULL);
        else plat_inc32(&PC.failed);
        plat_inc32(&PC.done);
    }
}

void vk_pipeline_prep_compute(void) {
    if (!ensure_init() || !pb_on()) return;
    GfxWarmEnt *all = NULL;
    size_t total = gfx_shader_warm_list(&all), n = 0;
    uint64_t maxw = prep_cs_max_words();
    const GfxShader **todo = total ? malloc(total * sizeof *todo) : NULL;
    for (size_t i = 0; todo && i < total; i++) {
        if (all[i].stage != GFX_STAGE_CS) continue;
        const GfxShader *sh = gfx_shader_load_input(all[i].iid, 0);
        uint64_t w = 0;
        if (sh) gfx_shader_spirv(sh, &w);
        if (sh && w >= big_cs_words() && w <= maxw) todo[n++] = sh;
    }
    free(all);
    PC.todo = todo, PC.n = n, PC.next = 0, PC.done = 0, PC.failed = 0, PC.t0 = now_ms();
    uint32_t loaded0 = (uint32_t)PB.loaded;
    // Every core: the compile memory gate starts the big ones side by side only while the RAM allows.
    int cpus = (int)plat_cpu_usable(), k = 0;
    PlatThread *t[64];
    for (int i = 0; i < cpus && i < 64 && (size_t)i < n; i++)
        if ((t[k] = plat_thread_new(prep_cs_worker, NULL, "prep compute"))) k++;
    if (!k) prep_cs_worker(NULL);
    for (int i = 0; i < k; i++)
        while (!plat_thread_join_ns(t[i], 2000000000u))
            rt_log("vk_pipeline: prewarm: %d of %zu big compute shaders, %.0f ms", (int)PC.done, n, now_ms() - PC.t0);
    rt_log("vk_pipeline: prewarm: %zu big compute shaders ready (%u from their binary, %d failed) in %.0f ms on %d threads", n,
           (uint32_t)PB.loaded - loaded0, (int)PC.failed, now_ms() - PC.t0, k ? k : 1);
    free(todo);
}

VkPipeline vk_pipeline_compute(const GfxShader *cs, const VkPipeInfo **info) {
    if (info) *info = NULL;
    if (!cs || !ensure_init()) return VK_NULL_HANDLE;
    struct { const GfxShader *cs; uint32_t compute; } k = {cs, 1};
    uint64_t h = key_hash(&k, sizeof k);
    plat_mutex_lock(&P.lock);
    Pipe *p = pipe_find(&k, sizeof k, h);
    if (p) {
        P.stats.hits++;
    } else {
        const char *why = NULL;
        Bindings B = {0};
        Layout *l = compute_layout(cs, &B, &why);
        VkPipeline pipe = VK_NULL_HANDLE;
        if (!why) {
            // The driver compile runs without P.lock, as lb_compute does: holding it stalled every other pipeline lookup
            // (graphics draws on the replay thread, the prebuild workers) for the whole compile, a visible hitch.
            plat_mutex_unlock(&P.lock);
            uint64_t p0 = perf_begin();
            tl_holds_plock = 1;
            pipe = compute_create(cs, l);
            tl_holds_plock = 0;
            perf_end(PERF_PIPE, p0);
            plat_mutex_lock(&P.lock);
            Pipe *other = pipe_find(&k, sizeof k, h);   // a prebuild worker finished the same pipeline meanwhile
            if (other) {
                if (pipe) vkDestroyPipeline(P.device, pipe, NULL);
                p = other;
                P.stats.hits++;
            } else if (!pipe) why = "pipeline creation failed";
        }
        if (!p) {
            if (why) {
                P.stats.failures++;
                rt_log("vk_pipeline: compute pipeline not built: %s", why);
            } else {
                P.stats.compute++;
                LB.cs_draw++;
            }
            p = pipe_add(&k, sizeof k, h, pipe, l, l ? &B : NULL, VK_PIPELINE_BIND_POINT_COMPUTE, 0);
        }
    }
    plat_mutex_unlock(&P.lock);
    if (!p) return VK_NULL_HANDLE;
    if (info) *info = &p->info;
    return p->info.pipeline;
}

static void lb_lock(void) {
    plat_thread_priority(PLAT_PRIO_NORMAL);
    plat_mutex_lock(&P.lock);
}

static void lb_unlock(void) {
    plat_mutex_unlock(&P.lock);
    if (LB.idle) plat_thread_priority(PLAT_PRIO_IDLE);
}

static void lb_count(int kind, int what, double ms) {   // what: 0 built, 1 already there, 2 failed, 3 cache only
    plat_mutex_lock(&LB.stat);
    if (what == 3) LB.warmed[kind]++, LB.warm_ms[kind] += ms;
    else (what == 0 ? LB.built : what == 1 ? LB.have : LB.fail)[kind]++, LB.ms[kind] += ms;
    plat_mutex_unlock(&LB.stat);
}

// vs_union_missing for vs alone, without entering vs into the union table (it holds VS_OUTS_CAP shaders).
static uint32_t vs_union_peek(const GfxShader *vs, GfxShaderIo fin[32]) {
    GfxShaderIo vout[32];
    uint32_t need = 0, have = gfx_shader_outputs(vs, vout), missing = 0;
    memset(fin, 0, 32 * sizeof *fin);
    plat_mutex_lock_shared(&VO.lock);
    uint32_t j = (uint32_t)(((uintptr_t)vs >> 4) * 0x9e3779b1u) & (VS_OUTS_CAP - 1);
    while (VO.vs[j] && VO.vs[j] != vs) j = (j + 1) & (VS_OUTS_CAP - 1);
    if (VO.vs[j]) need = VO.mask[j], memcpy(fin, VO.io[j], 32 * sizeof *fin);
    plat_mutex_unlock_shared(&VO.lock);
    if (P.vs_outs_all)
        for (int i = 0; i < 32; i++)
            if (!(need >> i & 1)) fin[i] = (GfxShaderIo){GFX_TYPE_FLOAT, 4, 0, 32}, need |= 1u << i;
    for (int i = 0; i < 32; i++)
        if (need >> i & 1 && !(have >> i & 1)) missing |= 1u << i;
    return missing;
}

static void lb_compute(const GfxShader *cs) {
    // LOCAL TEST (Ocarina): a big shader is left to big_cs_worker (one at a time, its own cache, only once a dispatch needs it).
    // Here it was built on every prebuild thread at once in the shared cache, including the skipped movie network (up to 7 million
    // words): on AMD (LLPC) those compiles ran for many minutes, the graphics pipelines waited behind them and the scene after the
    // intro stayed black (ocarina-recomp #12, #20, #27).
    uint64_t words = 0;
    gfx_shader_spirv(cs, &words);
    if (words >= big_cs_words()) return;
    struct { const GfxShader *cs; uint32_t compute; } k = {cs, 1};
    uint64_t h = key_hash(&k, sizeof k);
    Bindings B = {0};
    const char *why = NULL;
    lb_lock();
    Layout *l = pipe_find(&k, sizeof k, h) ? NULL : compute_layout(cs, &B, &why);
    int have = !l && !why;
    lb_unlock();
    if (!l) {
        lb_count(2, have ? 1 : 2, 0);
        return;
    }
    double t0 = now_ms();
    VkPipeline pipe = compute_create(cs, l);
    double ms = now_ms() - t0;
    lb_lock();
    if (pipe && !pipe_find(&k, sizeof k, h)) {
        pipe_add(&k, sizeof k, h, pipe, l, &B, VK_PIPELINE_BIND_POINT_COMPUTE, 0);
        P.stats.compute++;
    } else if (pipe) {
        vkDestroyPipeline(P.device, pipe, NULL);
        have = 1;
    }
    lb_unlock();
    lb_count(2, pipe ? have : 2, ms);
}

// The pre-rasterization (vertex) or fragment-shader library of one shader with the LB state.
// keep 0: build into the driver's pipeline cache only (sh is a private copy; the library is destroyed).
static void lb_library(const GfxShader *sh, int vertex, int keep) {
    int kind = vertex ? LIB_PR : LIB_FS, stat = vertex ? 0 : 1;
    VkPipeState s;
    vk_pipe_state_defaults(&s);
    GKey k;
    memset(&k, 0, sizeof k);
    k.clamp = LB.clamp, k.poly_mode = LB.poly, k.minus_w = LB.minus_w, k.samples = LB.samples;
    PrKey pk;
    FsKey fk;
    memset(&pk, 0, sizeof pk), memset(&fk, 0, sizeof fk);
    const void *key;
    uint32_t size;
    if (vertex) {
        GfxShaderIo fin[32];
        s.vs = sh;
        pk.kind = LIB_PR, pk.vs = sh, pk.clamp = k.clamp, pk.poly_mode = k.poly_mode, pk.minus_w = k.minus_w;
        pk.missing = vs_union_peek(sh, fin);
        for (int i = 0; i < 32; i++)
            if (pk.missing >> i & 1) pk.fin[i] = fin[i];
        key = &pk, size = sizeof pk;
    } else {
        s.fs = sh;
        fk.kind = LIB_FS, fk.fs = sh, fk.samples = k.samples;
        key = &fk, size = sizeof fk;
    }
    uint64_t h = key_hash(key, size);
    Bindings B = {0}, F = {0};
    VkPipelineLayout lay = VK_NULL_HANDLE;
    lb_lock();
    int have = keep && lib_find(h, key, size) != NULL;
    if (!have && add_stage(vertex ? &B : &F, sh, 0)) {
        Layout *l = layout_pair(&B, &F);
        if (l) lay = half_layout(l, vertex ? 0 : 1);
    }
    lb_unlock();
    if (have || !lay) {
        lb_count(stat, have ? 1 : 2, 0);
        return;
    }
    double t0 = now_ms();
    VkPipeline pipe = lib_create(kind, &s, &k, lay, key);
    double ms = now_ms() - t0;
    if (!keep) {
        if (pipe) vkDestroyPipeline(P.device, pipe, NULL);
        lb_count(stat, pipe ? 3 : 2, ms);
        return;
    }
    lb_lock();
    have = lib_find(h, key, size) != NULL;
    VkPipeline got = lib_put(h, key, size, pipe);
    if (!have && pipe) {
        Lib *b = lib_find(h, key, size);
        if (b) b->pre = 1;
    }
    lb_unlock();
    lb_count(stat, !pipe ? 2 : have || got != pipe ? 1 : 0, ms);
}

// A vertex or fragment variant with the graphics pipeline library off: no stage library to build ahead.
static void lb_skip_lib(void) {
    plat_mutex_lock(&LB.stat);
    LB.nolib++;
    plat_mutex_unlock(&LB.stat);
}

static void lb_report(const char *how) {
    float mem_used, mem_total, mem_proc;
    plat_mem_info(&mem_used, &mem_total, &mem_proc);
    plat_mutex_lock_shared(&LB.stat);
    rt_log("vk_pipeline: stage library prebuild %s: %llu of %zu variants in %.0f ms; kept: vertex %llu built (avg %.1f ms), "
           "%llu there; fragment %llu built (avg %.1f ms), %llu there; compute %llu built (avg %.1f ms), %llu there; driver "
           "cache only: vertex %llu (avg %.1f ms), fragment %llu (avg %.1f ms), compute %llu (avg %.1f ms); %llu failed, %llu big compute left to play, %llu not in the shader cache, "
           "%llu vertex/fragment skipped (graphics pipeline library off); process working set %.0f MB", how, (unsigned long long)LB.done, LB.n,
           now_ms() - LB.t0, (unsigned long long)LB.built[0], LB.built[0] ? LB.ms[0] / LB.built[0] : 0.0,
           (unsigned long long)LB.have[0], (unsigned long long)LB.built[1], LB.built[1] ? LB.ms[1] / LB.built[1] : 0.0,
           (unsigned long long)LB.have[1], (unsigned long long)LB.built[2], LB.built[2] ? LB.ms[2] / LB.built[2] : 0.0,
           (unsigned long long)LB.have[2], (unsigned long long)LB.warmed[0], LB.warmed[0] ? LB.warm_ms[0] / LB.warmed[0] : 0.0,
           (unsigned long long)LB.warmed[1], LB.warmed[1] ? LB.warm_ms[1] / LB.warmed[1] : 0.0,
           (unsigned long long)LB.warmed[2], LB.warmed[2] ? LB.warm_ms[2] / LB.warmed[2] : 0.0,
           (unsigned long long)(LB.fail[0] + LB.fail[1] + LB.fail[2]), (unsigned long long)LB.big_skipped,
           (unsigned long long)LB.noshader, (unsigned long long)LB.nolib, (double)mem_proc);
    plat_mutex_unlock_shared(&LB.stat);
}

// Boot: compute pipelines and kept libraries, all entered for the process.
static void lb_worker(void *arg) {
    (void)arg;
    plat_thread_priority(PLAT_PRIO_IDLE);
    for (;;) {
        size_t i = (size_t)plat_inc64(&LB.next) - 1;
        if (i >= LB.n || LB.stop) break;
        plat_thread_priority(PLAT_PRIO_NORMAL);   // gfx_shader's cache lock, disk reads
        const GfxShader *sh = gfx_shader_load_input(LB.e[i].iid, 0);
        plat_thread_priority(PLAT_PRIO_IDLE);
        if (!sh) {
            plat_mutex_lock(&LB.stat);
            LB.noshader++;
            plat_mutex_unlock(&LB.stat);
        } else if (gfx_shader_stage(sh) == GFX_STAGE_CS) {
            lb_compute(sh);
        } else if (gfx_shader_stage(sh) == GFX_STAGE_VS || gfx_shader_stage(sh) == GFX_STAGE_FS) {
            if (P.gpl) lb_library(sh, gfx_shader_stage(sh) == GFX_STAGE_VS, 1);
            else lb_skip_lib();
        }
        plat_mutex_lock(&LB.stat);
        LB.done++;
        plat_mutex_unlock(&LB.stat);
    }
    if (!plat_dec32(&LB.running)) lb_report(LB.stop ? "stopped" : "done");
    return;
}

// The library state from the key file's records (prewarm); defaults without one.
static void lb_state(const uint32_t pr_hist[16], const uint32_t fs_hist[4]) {
    // state index: clamp | poly (0 fill, 1 line, 2 point -> VkPolygonMode) << 1 | minus_w << 3 (pr); samples log2 (fs)
    uint32_t best = 0, bf = 0;
    for (uint32_t i = 1; i < 16; i++) best = pr_hist[i] > pr_hist[best] ? i : best;
    for (uint32_t i = 1; i < 4; i++) bf = fs_hist[i] > fs_hist[bf] ? i : bf;
    if (!pr_hist[best]) best = 8;   // no key file: TotK clips -w..w (all 2706 records of a recorded key file)
    LB.clamp = (uint8_t)(best & 1), LB.poly = (uint8_t)(best >> 1 & 3), LB.minus_w = (uint8_t)(best >> 3 & 1);
    LB.samples = (uint8_t)(1u << bf);
}

// Boot: compute shaders of the warm list (recorded and prepared), then the first KEEP recorded graphics variants.
static void lb_start(const uint32_t pr_hist[16], const uint32_t fs_hist[4]) {
    const char *e = cfg_value("TOTK_LIB_PREBUILD"), *ke = cfg_get("TOTK_LIB_PREBUILD_KEEP");
    const char *te = cfg_get("TOTK_LIB_PREBUILD_THREADS");   // explicit overrides keep their existing meaning
    if (!A.on || !cfg_bool("TOTK_LIB_PREBUILD")) return;
    GfxWarmEnt *all = NULL;
    size_t total = gfx_shader_warm_list(&all), n = 0, n_cs = 0;
    LB.keep_n = ke && *ke ? atoi(ke) : 0;
    LB.e = malloc((total + 1) * sizeof *LB.e);
    for (size_t i = 0; LB.e && i < total; i++)
        if (all[i].stage == GFX_STAGE_CS) LB.e[n++] = all[i];
    n_cs = n;
    for (size_t i = 0; LB.e && i < total && (int32_t)(n - n_cs) < LB.keep_n; i++)
        if (all[i].stage != GFX_STAGE_CS && all[i].prio == 0) LB.e[n++] = all[i];
    free(all);
    if (atoi(e) > 1 && (size_t)atoi(e) < n) n = (size_t)atoi(e);   // a count above 1 caps the list; 1 is plain on
    if (!n) {
        free(LB.e), LB.e = NULL;
        return;
    }
    LB.n = n;
    lb_state(pr_hist, fs_hist);
    LB.idle = 1;
    int cpus = (int)plat_cpu_usable();
    // Speculative NVIDIA builds can slow foreground replay despite idle worker
    // priority. Keep prebuild enabled, but serialize its default worker pool.
    const VkCtx *ctx = vk_ctx();
    int default_threads = ctx && ctx->props.vendorID == 0x10de ? 1 : cpus / 4;
    int t = te && *te ? atoi(te) : default_threads;
    t = t < 1 ? 1 : t > LB_MAX_THREADS ? LB_MAX_THREADS : t;
    if ((size_t)t > n) t = (int)n;
    LB.t0 = now_ms();
    LB.running = t;
    int started = 0;
    for (int i = 0; i < t; i++)
        if ((LB.th[started] = plat_thread_new(lb_worker, NULL, "lib prebuild"))) started++;
    plat_add32(&LB.running, started - t);
    LB.threads = started;
    rt_log("vk_pipeline: stage library prebuild: %zu compute pipelines and %zu recorded variants' libraries "
           "(TOTK_LIB_PREBUILD_KEEP %d) on %d idle-priority threads; library state clamp %u, polygon mode %u, depth "
           "-w..w %u, samples %u", n_cs < n ? n_cs : n, n > n_cs ? n - n_cs : 0, LB.keep_n, started, LB.clamp, LB.poly,
           LB.minus_w, LB.samples);
}

static void lb_stop(void) {
    plat_xchg32(&LB.stop, 1);
    for (int i = 0; i < LB.threads; i++)
        if (LB.th[i]) plat_thread_join(LB.th[i]), LB.th[i] = NULL;
    LB.threads = 0;
}

// Preparation: one prepared variant at a time into the driver's disk cache (private shader copy, nothing kept).
static void lb_warm_worker(void *arg) {
    (void)arg;
    for (;;) {
        size_t i = (size_t)plat_inc64(&LB.next) - 1;
        if (i >= LB.n) return;
        int owned = 0;
        const GfxShader *sh = gfx_shader_load_private(LB.e[i].iid, &owned);
        int st = sh ? gfx_shader_stage(sh) : -1;
        uint64_t words = 0;
        if (!sh) {
            plat_mutex_lock(&LB.stat);
            LB.noshader++;
            plat_mutex_unlock(&LB.stat);
        } else if (st == GFX_STAGE_CS && gfx_shader_spirv(sh, &words) && words >= big_cs_words()) {
            // As lb_compute: a big shader is left to big_cs_worker in play. Here it held the preparation for many minutes
            // (some are the movie network the game skips, up to 7 million words). Suggested by robk3 on issue #15.
            plat_mutex_lock(&LB.stat);
            LB.big_skipped++;
            plat_mutex_unlock(&LB.stat);
        } else if (st == GFX_STAGE_CS) {
            Bindings B = {0};
            const char *why = NULL;
            lb_lock();
            Layout *l = compute_layout(sh, &B, &why);
            lb_unlock();
            double t0 = now_ms();
            VkPipeline pipe = l ? compute_create(sh, l) : VK_NULL_HANDLE;
            if (pipe) vkDestroyPipeline(P.device, pipe, NULL);
            lb_count(2, pipe ? 3 : 2, now_ms() - t0);
        } else if (st == GFX_STAGE_VS || st == GFX_STAGE_FS) {
            if (P.gpl) lb_library(sh, st == GFX_STAGE_VS, 0);
            else lb_skip_lib();
        }
        if (owned) gfx_shader_release_private(sh);
        plat_mutex_lock(&LB.stat);
        uint64_t d = ++LB.done;
        plat_mutex_unlock(&LB.stat);
        if (d % 4000 == 0) lb_report("preparation progress");
    }
}

// Preparation: every variant of <dir>/shader-prepared.bin on every core at normal priority.
static void lb_warm_prepared(const char *dir) {
    char path[PLAT_PATH_MAX * 2 + 32];
    snprintf(path, sizeof path, "%s/shader-prepared.bin", dir);
    uint64_t size = 0;
    uint8_t *f = plat_file_read(path, &size);
    size_t n = f && size >= 16 ? (size_t)(size - 16) / 36 : 0;
    LB.e = n ? calloc(n, sizeof *LB.e) : NULL;
    for (size_t i = 0; LB.e && i < n; i++)
        memcpy(LB.e[i].iid, f + 16 + 36 * i + 4, 32), LB.e[i].stage = f[16 + 36 * i], LB.e[i].prio = f[16 + 36 * i + 1];
    free(f);
    if (!LB.e) return;
    LB.n = n, LB.idle = 0, LB.t0 = now_ms();
    static const uint32_t no_pr[16], no_fs[4];
    lb_state(no_pr, no_fs);
    int cpus = (int)plat_cpu_usable(), k = 0;
    PlatThread *t[64];
    for (int i = 0; i < cpus && i < 64; i++)
        if ((t[k] = plat_thread_new(lb_warm_worker, NULL, "lib warm"))) k++;
    if (!k) lb_warm_worker(NULL);
    for (int i = 0; i < k; i++) plat_thread_join(t[i]);
    lb_report("preparation done");
    // with gpl off only compute variants are built: say so, rather than count the skipped ones as built
    if (LB.nolib)
        printf("prepare-shaders: %zu variants on %d threads in %.0f ms, %llu vertex/fragment variants skipped (graphics "
               "pipeline library off, they build at their first draw)\n", n, k, now_ms() - LB.t0, (unsigned long long)LB.nolib);
    else
        printf("prepare-shaders: %zu variants built into the driver cache on %d threads in %.0f ms\n", n, k, now_ms() - LB.t0);
    free(LB.e), LB.e = NULL, LB.n = 0, LB.next = 0, LB.done = 0;
}

// totk.exe --prepare-shaders (tools/prepare_shaders.py): the device sets the translator's host answers (they are in every
// cache key), the seed next to totk.exe is translated on every core, then every prepared variant's stage library or
// compute pipeline is built once into the driver's disk cache (lb_warm_prepared). No game runs.
int vk_pipeline_prepare_shaders(void) {
    if (!ensure_init()) {
        fprintf(stderr, "prepare-shaders: no Vulkan device\n");
        return 1;
    }
    char dir[PLAT_PATH_MAX * 2];
    snprintf(dir, sizeof dir, "%s", P.cache_path);
    char *cut = plat_last_sep(dir);
    if (!cut) return 1;
    *cut = 0;
    int cpus = (int)plat_cpu_usable();
    const char *pe = cfg_get("TOTK_PREWARM_XLATE");   // no fixed default: half the CPUs, at most 16, when unset
    int procs = pe && *pe ? atoi(pe) : cpus / 2 > 16 ? 16 : cpus / 2;
    int rc = gfx_shader_prepare_seed(dir, cpus, procs < 1 ? 1 : procs);
    if (!rc) lb_warm_prepared(dir);
    return rc;
}

// totk.exe --warm-shaders: every variant of shader-prepared.bin built into the driver's disk cache again from the shader
// cache, without translating the seed. The AMD cache is keyed to the exe path (a relink keeps it: 29,587 variants in
// 1.8 s) and holds a limited amount: preparing another build dir evicted this one's libraries, and a library a draw
// needed then took 40-220 ms in play instead of 0.2 ms. Seconds when the libraries are still there.
int vk_pipeline_warm_shaders(void) {
    if (!ensure_init()) {
        fprintf(stderr, "warm-shaders: no Vulkan device\n");
        return 1;
    }
    char dir[PLAT_PATH_MAX * 2];
    snprintf(dir, sizeof dir, "%s", P.cache_path);
    char *cut = plat_last_sep(dir);
    if (!cut) return 1;
    *cut = 0;
    char prp[PLAT_PATH_MAX * 2 + 32];
    snprintf(prp, sizeof prp, "%s/shader-prepared.bin", dir);
    if (!plat_file_exists(prp)) {
        fprintf(stderr, "warm-shaders: no %s (tools/prepare_shaders.py writes it)\n", prp);
        return 1;
    }
    lb_warm_prepared(dir);
    return 0;
}

// ---- dynamic state

uint32_t vk_pipeline_set_dynamic(VkCommandBuffer cb, const VkPipeState *s) {
    if (!ensure_init()) return VK_PIPE_ALL_BINDINGS;
    const poly_st *pl = &s->poly;
    const ds_st *d = &s->ds;
    // The front face maps unchanged: TOTK sets the upper-left window origin, and the y-flipped viewport (y_flip) puts
    // NDC +y at row 0 as NVN does, so the framebuffer winding Vulkan tests is the one NVN tests (MacNativeGames passes
    // it to Metal unchanged; LAS Recomp exec.c mirrors it only for the lower-left origin).
    int front_cw = pl->front == NVN_FRONT_FACE_CW;
    static int no_cull = -1;   // LOCAL TEST (Ocarina): OOT_NO_CULL=1 draws both faces of everything
    if (no_cull < 0) no_cull = getenv("OOT_NO_CULL") != NULL;
    vkCmdSetCullMode(cb, no_cull ? VK_CULL_MODE_NONE : (VkCullModeFlags)(pl->cull & 3));   // NVN_FACE_* has VkCullModeFlagBits' values
    vkCmdSetFrontFace(cb, front_cw ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE);
    vkCmdSetPrimitiveTopology(cb, vk_pipeline_topology(s->prim));
    vkCmdSetDepthTestEnable(cb, d->dt);
    vkCmdSetDepthWriteEnable(cb, d->dw);
    vkCmdSetDepthCompareOp(cb, compare_op(d->df));
    vkCmdSetStencilTestEnable(cb, d->st);
    vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_BIT, stencil_op(d->fail_f), stencil_op(d->zpass_f), stencil_op(d->zfail_f),
                      compare_op(d->sf_f));
    vkCmdSetStencilOp(cb, VK_STENCIL_FACE_BACK_BIT, stencil_op(d->fail_b), stencil_op(d->zpass_b), stencil_op(d->zfail_b),
                      compare_op(d->sf_b));
    // NVN_POLYGON_OFFSET_ENABLE_* bit of the polygon mode in use: point 1, line 2, fill 4
    unsigned mode = pl->mode <= NVN_POLYGON_MODE_FILL ? pl->mode : NVN_POLYGON_MODE_FILL;
    vkCmdSetDepthBiasEnable(cb, (pl->offset_en >> mode & 1) != 0);
    vkCmdSetRasterizerDiscardEnable(cb, s->raster_discard != 0);
    vkCmdSetPrimitiveRestartEnable(cb, s->prim_restart != 0);
    // The vertex bindings the draw reads: those of its vertex input, set here or made into the pipeline from the same
    // state by the same vertex_input.
    uint32_t streams = VK_PIPE_ALL_BINDINGS;
    if (s->vs) {
        VtxInput v;
        vertex_input(s, &v);
        if (P.dyn_vi) rec_vertex_input(P.set_vi, cb, v.n_bind, v.b, v.n_attr, v.a);
        streams = 0;
        for (uint32_t i = 0; i < v.n_bind; i++) streams |= 1u << v.b[i].binding;
    }
    if (P.dyn_clamp) rec_bool(P.set_clamp, cb, s->depth_clamp != 0);
    if (P.dyn_poly) rec_polygon_mode(P.set_poly, cb, polygon_mode(pl->mode));
    if (P.dyn_a2c) rec_bool((PFN_vkCmdSetDepthClampEnableEXT)P.set_a2c, cb, s->ms.a2c != 0);   // the same signature
    int n = target_count(s);
    if (!n) return streams;
    VkBool32 en[8];
    VkColorBlendEquationEXT eq[8];
    VkColorComponentFlags wm[8];
    for (int i = 0; i < n; i++) {
        const blend_st *b = &s->blend[i];
        en[i] = (s->color.blend_en >> i & 1) && blendable(target_format(s->color_fmt[i], NULL, NULL));
        eq[i] = (VkColorBlendEquationEXT){blend_factor(b->src_rgb), blend_factor(b->dst_rgb), blend_op(b->eq_rgb),
                                          blend_factor(b->src_a), blend_factor(b->dst_a), blend_op(b->eq_a)};
        wm[i] = target_wmask(s, i);
    }
    if (P.dyn_blend_en) rec_blend_enable(P.set_blend_en, cb, 0, (uint32_t)n, en);
    if (P.dyn_blend_eq) rec_blend_equation(P.set_blend_eq, cb, 0, (uint32_t)n, eq);
    if (P.dyn_wmask) rec_write_mask(P.set_wmask, cb, 0, (uint32_t)n, wm);
    return streams;
}

// ---- statistics, persistence, teardown

void vk_pipeline_stats(VkPipeStats *s) {
    plat_mutex_lock_shared(&P.lock);
    *s = P.stats;
    plat_mutex_unlock_shared(&P.lock);
}

static PlatMutex save_lock = PLAT_MUTEX_INIT;   // one cache save at a time; shutdown takes it to destroy the cache

void vk_pipeline_save(void) {
    if (P.ready) keys_flush();
    if (P.ready && P.gpl) {
        rt_log("vk_pipeline: gpl: %llu fast links (%llu by workers; avg %.2f ms, max %.2f ms, %llu failed), %llu draws "
               "lookups found shader libraries missing (left to a worker), %llu optimised pipelines swapped in (%llu queued "
               "low after their fast link), %llu drawing-thread links over 5 ms; %llu waiting draws built their libraries "
               "(avg %.2f ms, max %.2f ms)",
               (unsigned long long)P.g.links, (unsigned long long)P.g.wlinks, P.g.links ? P.g.link_ms / P.g.links : 0.0,
               P.g.link_max, (unsigned long long)P.g.link_fail, (unsigned long long)P.g.defer,
               (unsigned long long)P.g.swaps, (unsigned long long)P.g.requeued, (unsigned long long)P.g.slow,
               (unsigned long long)P.g.dlinks, P.g.dlinks ? P.g.dlink_ms / P.g.dlinks : 0.0, P.g.dlink_max);
        if (U.waits || U.queued)
            rt_log("vk_pipeline: gpl: library jobs for waiting draws: %llu queued, %llu built by a worker, %llu by the "
                   "thread that needed them (no worker free), %llu waits ended at their limit",
                   (unsigned long long)U.queued, (unsigned long long)U.by_worker, (unsigned long long)U.stolen,
                   (unsigned long long)U.timeouts);
        static const char *const kn[4] = {"vertex input", "pre-rasterization", "fragment shader", "fragment output"};
        for (int i = 0; i < 4; i++)
            rt_log("vk_pipeline: gpl: %s libraries %llu (avg %.2f ms, max %.2f ms)", kn[i], (unsigned long long)P.g.lib_n[i],
                   P.g.lib_n[i] ? P.g.lib_ms[i] / P.g.lib_n[i] : 0.0, P.g.lib_max[i]);
        if (LB.n)
            rt_log("vk_pipeline: gpl: prebuilt libraries a draw used: pre-rasterization %llu, fragment shader %llu; "
                   "compute pipelines built on the replay thread %llu", (unsigned long long)P.g.pre_used[LIB_PR],
                   (unsigned long long)P.g.pre_used[LIB_FS], (unsigned long long)LB.cs_draw);
    }
    gfx_shader_record_flush(1);
    if (!P.ready || !P.cache || !P.cache_path[0]) return;
    // Not under P.lock: the driver synchronises P.cache, and holding P.lock through the copy and the file write (about
    // 60 ms on the 1.4.3 fight) stopped every pipeline lookup of the replay, a freeze at each 20 s autosave.
    plat_mutex_lock(&save_lock);
    if (!P.cache) {   // vk_pipeline_shutdown destroyed it
        plat_mutex_unlock(&save_lock);
        return;
    }
    size_t size = 0;
    void *data = NULL;
    cache_retry_deferred();
    // Exclusive: Intel's driver (Arc 140V, 101.8991) reads a null pointer in vkGetPipelineCacheData (igvk64.dll+0x653291,
    // the "pipe autosave" thread) while another thread builds a pipeline into the same cache. trylock, so a waiting
    // save never holds back the replay's builds behind a compile of seconds.
    while (!plat_mutex_trylock(&cache_access)) plat_sleep_ms(2);
    plat_mutex_lock(&deferred_access);
    VkPipelineCache snapshot = P.cache;
    // A driver compile may still be using the shared cache. Persist the
    // union via the deferred cache without waiting for an exclusive lock.
    if (deferred_cache) {
        if (vkMergePipelineCaches(P.device, deferred_cache, 1, &P.cache) == VK_SUCCESS) snapshot = deferred_cache;
        else rt_log("vk_pipeline: could not include deferred pipelines in cache save");
    }
    int got = vkGetPipelineCacheData(P.device, snapshot, &size, NULL) == VK_SUCCESS && size && (data = malloc(size)) &&
              vkGetPipelineCacheData(P.device, snapshot, &size, data) == VK_SUCCESS;
    plat_mutex_unlock(&deferred_access);
    plat_mutex_unlock(&cache_access);
    if (got) {
        if (!plat_file_write_atomic(P.cache_path, data, size)) rt_log("vk_pipeline: cannot write %s", P.cache_path);
    }
    free(data);
    plat_mutex_unlock(&save_lock);
}

// The game never shuts the device down (boot.c and the window end the process), so the cache is saved while it
// runs: at most every 20 s, and only after new pipelines were built. New pipeline keys are appended every 2 s.
static void autosave_once(void) {
    static uint64_t last_n, last_updates, last_ms, last_keys;
    if (!P.ready) return;
    cache_retry_deferred();
    uint64_t now = plat_now_ms();
    if (*(volatile uint32_t *)&K.pend_n && now - last_keys >= 2000) last_keys = now, keys_flush();
    gfx_shader_record_flush(0);
    if (now - last_ms < 20000) return;
    VkPipeStats s;
    vk_pipeline_stats(&s);
    uint64_t updates = __atomic_load_n(&cache_updates, __ATOMIC_RELAXED);
    if (s.graphics + s.compute == last_n && updates == last_updates) return;
    last_n = s.graphics + s.compute, last_ms = now;
    last_updates = updates;
    vk_pipeline_save();
}

// On its own thread: the save (vkGetPipelineCacheData and the file write, about 60 ms) and the key appends (a flush to
// the disk each) ran on the present thread, and the render thread waited for it to take the next frame.
static void autosave_main(void *arg) {
    (void)arg;
    plat_thread_priority(PLAT_PRIO_BELOW_NORMAL);
    for (;;) {
        plat_sleep_ms(500);
        autosave_once();
    }
}

void vk_pipeline_autosave(void) {
    static int32_t started;
    if (!P.ready || plat_xchg32(&started, 1)) return;
    if (!plat_thread_spawn(autosave_main, NULL, "pipe autosave", 0))
        rt_log("vk_pipeline: cannot start the autosave thread; the cache is saved at exit only");
}

void vk_pipeline_shutdown(void) {
    if (!P.ready) return;
    lb_stop();
    workers_drain();   // jobs point at layouts and Pipe entries freed below
    vk_pipeline_save();
    plat_mutex_lock(&P.lock);
    vkDeviceWaitIdle(P.device);
    for (int i = 0; i < PIPE_BUCKETS; i++) {
        for (Pipe *p = pipes[i], *n; p; p = n) {
            n = p->next;
            if (p->info.pipeline) vkDestroyPipeline(P.device, p->info.pipeline, NULL);
            if (p->fast && p->fast != p->info.pipeline) vkDestroyPipeline(P.device, p->fast, NULL);
            free(p->bindings), free(p);
        }
        pipes[i] = NULL;
    }
    libs_destroy();
    for (int i = 0; i < LAYOUT_BUCKETS; i++) {
        for (Layout *l = layouts[i], *n; l; l = n) {
            n = l->next;
            for (int k = 0; k < 2; k++)
                if (l->tpl[k]) P.tpl_destroy(P.device, l->tpl[k], NULL);
            vkDestroyPipelineLayout(P.device, l->layout, NULL);
            for (int h = 0; h < 2; h++)
                if (l->half[h]) vkDestroyPipelineLayout(P.device, l->half[h], NULL);
            vkDestroyDescriptorSetLayout(P.device, l->set, NULL);
            if (l->set1) vkDestroyDescriptorSetLayout(P.device, l->set1, NULL);
            free(l);
        }
        layouts[i] = NULL;
    }
    plat_mutex_lock(&save_lock);   // not while the autosave thread copies it
    plat_mutex_lock(&cache_access);
    plat_mutex_lock(&deferred_access);
    if (deferred_cache) vkDestroyPipelineCache(P.device, deferred_cache, NULL);
    deferred_cache = VK_NULL_HANDLE;
    deferred_count = 0;
    plat_mutex_unlock(&deferred_access);
    if (P.cache) vkDestroyPipelineCache(P.device, P.cache, NULL);
    P.cache = VK_NULL_HANDLE;
    plat_mutex_unlock(&cache_access);
    plat_mutex_unlock(&save_lock);
    memset(&P.stats, 0, sizeof P.stats);
    plat_xchg32(&P.ready, 0);
    plat_mutex_unlock(&P.lock);
}

// ---- self-test: census shaders (probe_census_dir, tools/shader_census.py) through gfx_shader_get into pipelines.
// Graphics pairs are a fragment shader and a vertex shader of the same source file, spread over the census; compute
// shaders are spread the same way. Every pipeline is built twice (the second call must hit the cache) and the
// dynamic state of the first graphics pipeline is recorded into a throwaway command buffer.

typedef struct { const char *key, *c1, *src; size_t c1_len, src_len; int stage; uint32_t local[3]; } CensusEntry;

static const char *st_field(const char *from, const char *to, const char *name, size_t *len) {
    char pat[32];
    snprintf(pat, sizeof pat, "\"%s\": \"", name);
    const char *p = strstr(from, pat);
    if (!p || p >= to) return NULL;
    p += strlen(pat);
    const char *e = memchr(p, '"', (size_t)(to - p));
    if (!e) return NULL;
    *len = (size_t)(e - p);
    return p;
}

static int census_cmp(const void *x, const void *y) {
    const CensusEntry *a = x, *b = y;
    size_t n = a->src_len < b->src_len ? a->src_len : b->src_len;
    int c = memcmp(a->src, b->src, n);
    if (c) return c;
    if (a->src_len != b->src_len) return a->src_len < b->src_len ? -1 : 1;
    return a->stage - b->stage;
}

static char *census_read(const char *root, size_t *size) {
    char path[PLAT_PATH_MAX * 2 + 64];
    snprintf(path, sizeof path, "%s/jobs.json", root);
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) free(b), b = NULL;
    fclose(f);
    if (b) b[n] = 0, *size = (size_t)n;
    return b;
}

static int census_parse(const char *jobs, CensusEntry *out, int max) {
    int n = 0;
    for (const char *p = jobs; n < max && (p = strstr(p, "\"job\": {")); p += 8) {
        const char *end = strstr(p, "\"origin\": ");
        if (!end) break;
        size_t kl, sl, cl, rl;
        const char *k = st_field(p, end, "key", &kl), *s = st_field(p, end, "stage", &sl), *c = st_field(p, end, "c1", &cl),
                   *r = st_field(p, end, "source", &rl);
        if (!k || !s || !c || !r || kl != 64) continue;
        CensusEntry *e = &out[n++];
        e->key = k, e->c1 = c, e->c1_len = cl / 2, e->src = r, e->src_len = rl;
        e->stage = !strncmp(s, "vs", sl) ? GFX_STAGE_VS : !strncmp(s, "ps", sl) ? GFX_STAGE_FS : GFX_STAGE_CS;
        e->local[0] = e->local[1] = e->local[2] = 1;
        const char *l = strstr(p, "\"local\": [");
        if (l && l < end) {
            l += 10;
            for (int i = 0; i < 3; i++) {
                while (*l == ' ' || *l == '\n' || *l == '\r' || *l == ',') l++;
                e->local[i] = (uint32_t)strtoul(l, (char **)&l, 10);
            }
        }
    }
    return n;
}

static int hexv(char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; }

// The program the game would hand NVN for a census entry, translated with default answers (as gfx_shader_probe.c).
static const GfxShader *census_shader(const char *root, const CensusEntry *e) {
    char path[PLAT_PATH_MAX * 2 + 128];
    static uint8_t c1[8192];
    uint8_t control[0x880] = {0};
    if (e->c1_len > sizeof c1) return NULL;
    for (size_t b = 0; b < e->c1_len; b++) c1[b] = (uint8_t)(hexv(e->c1[b * 2]) << 4 | hexv(e->c1[b * 2 + 1]));
    snprintf(path, sizeof path, "%s/code/%.64s.code", root, e->key);
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long body = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t skip = e->stage == GFX_STAGE_CS ? 0 : 0x30, total = skip + (size_t)body;
    uint8_t *code = calloc(1, total);
    int ok = code && fread(code + skip, 1, (size_t)body, f) == (size_t)body;
    fclose(f);
    if (!ok) return free(code), NULL;
    uint32_t c1_off = (uint32_t)(total - e->c1_len), length = (uint32_t)total;
    memcpy(control + 0x700, &c1_off, 4), memcpy(control + 0x704, &length, 4);
    GfxShaderAnswers a = {0};
    if (e->stage == GFX_STAGE_CS) for (int k = 0; k < 3; k++) a.local[k] = (uint16_t)e->local[k];
    const GfxShader *s = gfx_shader_get(e->stage, control, code, total, c1, &a);
    free(code);
    return s;
}

// A state for a census pair: one target per fragment output (RGBA8 of the output's type), D32S8 depth with the test on,
// alpha blending on target 0, one RGBA32 attribute per vertex input, interleaved in stream 0.
static void selftest_state(VkPipeState *s, const GfxShader *vs, const GfxShader *fs) {
    vk_pipe_state_defaults(s);
    s->vs = vs, s->fs = fs;
    GfxShaderIo io[32];
    uint32_t outs = gfx_shader_outputs(fs, io);
    for (int i = 0; i < 8; i++)
        if (outs >> i & 1)
            s->color_fmt[i] = io[i].type == GFX_TYPE_UINT ? NVN_FORMAT_RGBA8UI : io[i].type == GFX_TYPE_SINT ? NVN_FORMAT_RGBA8I
                                                                                                            : NVN_FORMAT_RGBA8;
    s->depth_fmt = NVN_FORMAT_DEPTH32F_STENCIL8;
    s->ds.dt = 1;
    s->color.blend_en = 1;
    s->blend[0].src_rgb = NVN_BLEND_FUNC_SRC_ALPHA, s->blend[0].dst_rgb = NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA;
    s->poly.cull = NVN_FACE_BACK;
    uint32_t ins = gfx_shader_inputs(vs, io), off = 0;
    for (int l = 0; l < 16; l++)
        if (ins >> l & 1) {
            va_st a = {io[l].type == GFX_TYPE_UINT ? NVN_FORMAT_RGBA32UI : io[l].type == GFX_TYPE_SINT ? NVN_FORMAT_RGBA32I
                                                                                                      : NVN_FORMAT_RGBA32F,
                       off, 0};
            memcpy(&s->vattr[l], &a, sizeof a);
            off += 16;
        }
    s->vstream[0].stride = off ? off : 16;
}

// Records the dynamic state of a pipeline into a command buffer that is never submitted (validation checks the calls).
static int selftest_record(const VkPipeState *s, VkPipeline pipe) {
    VkCtx *c = vk_ctx();
    VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = c->gfx_family};
    VkCommandPool pool;
    if (vkCreateCommandPool(c->device, &pi, NULL, &pool) != VK_SUCCESS) return 0;
    VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer cb;
    int ok = vkAllocateCommandBuffers(c->device, &ai, &cb) == VK_SUCCESS;
    if (ok) {
        VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                       .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        vkBeginCommandBuffer(cb, &bi);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        vk_pipeline_set_dynamic(cb, s);
        ok = vkEndCommandBuffer(cb) == VK_SUCCESS;
    }
    vkDestroyCommandPool(c->device, pool, NULL);
    return ok;
}

int vk_pipeline_selftest(void) {
    enum { MAXN = 20000, WANT_PAIRS = 64, WANT_CS = 16 };
    char root[PLAT_PATH_MAX * 2];
    if (!vk_init() || !ensure_init()) { printf("vk_pipeline selftest: FAILED: no Vulkan device\n"); return 1; }
    if (!probe_census_dir(root, sizeof root)) {   // the census folder, build/<version>/shaders
        printf("vk_pipeline selftest: FAILED: no build/v%d/shaders/jobs.json above the exe\n", rt_game_version);
        return 1;
    }
    size_t size;
    char *jobs = census_read(root, &size);
    CensusEntry *e = malloc(sizeof *e * MAXN);
    int n = jobs && e ? census_parse(jobs, e, MAXN) : 0;
    if (!n) { printf("vk_pipeline selftest: FAILED: no census entries\n"); free(jobs), free(e); return 1; }
    qsort(e, (size_t)n, sizeof *e, census_cmp);
    long err0 = vk_validation_errors_total();
    VkPipeStats st0;
    vk_pipeline_stats(&st0);

    // candidate pairs: each fragment shader of a source file with one of the file's vertex shaders (round robin)
    int (*pairs)[2] = malloc(sizeof *pairs * (size_t)n), np = 0, *cs = malloc(sizeof *cs * (size_t)n), nc = 0;
    for (int i = 0; i < n;) {
        int j = i, v0 = -1, nv = 0, nf = 0;
        for (; j < n && e[j].src_len == e[i].src_len && !memcmp(e[j].src, e[i].src, e[i].src_len); j++) {
            if (e[j].stage == GFX_STAGE_VS && v0 < 0) v0 = j;   // sorted by stage within a file: vertex shaders first
            nv += e[j].stage == GFX_STAGE_VS;
            if (e[j].stage == GFX_STAGE_CS) cs[nc++] = j;
        }
        for (int k = i; k < j && nv; k++)
            if (e[k].stage == GFX_STAGE_FS) pairs[np][0] = v0 + nf++ % nv, pairs[np][1] = k, np++;
        i = j;
    }
    int tried = 0, built = 0, cached = 1, shader_fail = 0, ctried = 0, cbuilt = 0, recorded = 0, varying_fix = 0;
    for (int t = 0; t < WANT_PAIRS && t < np; t++) {
        const int *pr = pairs[(long long)t * np / (np < WANT_PAIRS ? np : WANT_PAIRS)];
        tried++;
        const GfxShader *vs = census_shader(root, &e[pr[0]]), *fs = census_shader(root, &e[pr[1]]);
        if (!vs || !fs) { shader_fail++; continue; }
        if (gfx_shader_inputs(fs, NULL) & ~gfx_shader_outputs(vs, NULL)) varying_fix++;
        VkPipeState s;
        selftest_state(&s, vs, fs);
        GfxShaderAnswers fa = {0};
        vk_pipeline_answers(&s, NULL, &fa);
        const VkPipeInfo *info;
        VkPipeline p = vk_pipeline_graphics(&s, &info);
        if (!p) continue;
        built++;
        const VkPipeInfo *again;
        if (vk_pipeline_graphics(&s, &again) != p || again != info) cached = 0;
        if (!recorded) recorded = selftest_record(&s, p);
    }
    for (int t = 0; t < WANT_CS && t < nc; t++) {
        const CensusEntry *c = &e[cs[(long long)t * nc / (nc < WANT_CS ? nc : WANT_CS)]];
        ctried++;
        const GfxShader *s = census_shader(root, c);
        if (!s) { shader_fail++; continue; }
        const VkPipeInfo *info;
        VkPipeline p = vk_pipeline_compute(s, &info);
        if (!p) continue;
        cbuilt++;
        const VkPipeInfo *again;
        if (vk_pipeline_compute(s, &again) != p || again != info) cached = 0;
    }
    vk_pipeline_save();
    VkPipeStats st;
    vk_pipeline_stats(&st);
    long errors = vk_validation_errors_total() - err0;
    printf("vk_pipeline selftest: graphics %d/%d census pairs (%d needed vertex outputs added), compute %d/%d, "
           "%d shaders failed to translate; layouts %llu (%llu push); cache hits %llu; dynamic state recorded %d; "
           "validation errors %ld\n",
           built, tried, varying_fix, cbuilt, ctried, shader_fail, (unsigned long long)(st.layouts - st0.layouts),
           (unsigned long long)(st.push_layouts - st0.push_layouts), (unsigned long long)(st.hits - st0.hits), recorded,
           errors);
    free(pairs), free(cs), free(e), free(jobs);
    int ok = built > 0 && cbuilt > 0 && cached && recorded && errors == 0;
    printf("vk_pipeline selftest: %s\n", ok ? "passed" : "FAILED");
    return ok ? 0 : 1;
}
