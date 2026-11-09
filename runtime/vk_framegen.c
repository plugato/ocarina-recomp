// Frame interpolation on the present thread (vk_backend.c present_main): TOTK_FRAMEGEN=n (2..4, [display] framegen)
// shows n frames per game frame, n - 1 of them made from the game's last two and shown before it, evenly over the time
// since the last one. 1 (the default) shows the game's frames alone. Ported from NotRexed/Hookshot gpu/vk/vk_framegen.cpp
// (GPL-3.0, as this repository); its shaders are runtime/shaders/fg_*.comp:
//   1. the game's frame (its crop) is copied into an RGBA16F image through a view of its own format, as vk_fsr.c reads
//      it, and reduced to quarter-size luma. Ocarina's window texture is UNORM holding display values, the values the
//      thresholds of fg_synth.comp were tuned on (compared after a square root, as if linear, made white flashes);
//   2. block matching: a vector per 16x16 pixels within +-32 pixels, then a 3x3 median of them;
//   3. each made frame at time t blends the last frame moved back along t and the new one forward along 1 - t, and
//      where the two frames agree, or no vector explains the change, the plain blend or the nearer frame.
// Frames more than 100 ms apart (a loading hitch) are not interpolated. The made frames go to the window through
// vk_present_image, as any RGBA16F image (the FSR and colour paths apply to them as to the game's).
//
// Cost on the GPU per game frame: a copy, the luma, the block matching on the quarter-size luma and n - 1 synthesis
// passes over the crop. The made frames of a set are read by their presents; the next frame writes the other set.
#include "vk.h"

#include <math.h>
#include <string.h>

#include "cfg.h"
#include "fg_camflow_spv.h"
#include "fg_copy_spv.h"
#include "fg_flow_spv.h"
#include "fg_luma_spv.h"
#include "fg_median_spv.h"
#include "fg_synth_spv.h"
#include "host.h"
#include "platform/platform.h"
#include "vk_rec.h"

#define MAXMULT 4
#define FG_FMT VK_FORMAT_R16G16B16A16_SFLOAT

typedef struct { VkImage image; VmaAllocation alloc; VkImageView view; } Img;
typedef struct { VkDescriptorSetLayout dsl; VkPipelineLayout layout; VkPipeline pipe; } Pass;

static struct {
    int tried, ok;
    PFN_vkCmdPushDescriptorSetKHR push;
    Pass copy, luma, flow, median, synth, camflow;
    OotMotionFrame prev_cam;   // the camera of the last frame interpolated from
    int have_cam;
    VkSampler nearest, linear;
    VkCommandPool pool;
    VkCommandBuffer cb[2];
    VkFence fence[2];
    Img frame[2], luma_img[2], flowv[2], made[2][MAXMULT - 1];
    uint32_t w, h;
    int mult, cur, have_prev, set;
    uint64_t last_ns;
    VkImageView src_view[2];
    VkBuffer rb;   // OOT_FG_LOG: a readback of the vectors
    VmaAllocation rb_alloc;
    void *rb_map;
    int rb_pending, rb_n;
    VkFormat src_fmt;
} G;

int vk_framegen_mult(void) {
    int m = cfg_int("TOTK_FRAMEGEN");
    return m < 1 ? 1 : m > MAXMULT ? MAXMULT : m;
}

static int make_pass(VkCtx *c, Pass *p, const uint32_t *code, size_t size, const VkDescriptorType *types, int n,
                     uint32_t push, const VkSpecializationInfo *spec) {
    VkDescriptorSetLayoutBinding b[5];   // camflow binds 5: with 4 the fifth wrote past the array (stack check abort)
    if (n > (int)(sizeof b / sizeof *b)) return 0;
    for (int i = 0; i < n; i++)
        b[i] = (VkDescriptorSetLayoutBinding){.binding = (uint32_t)i, .descriptorType = types[i], .descriptorCount = 1,
                                              .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = (uint32_t)n, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, push ? push : 4};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &p->dsl, .pushConstantRangeCount = push ? 1 : 0,
                                     .pPushConstantRanges = &pr};
    if (vkCreateDescriptorSetLayout(c->device, &dci, NULL, &p->dsl) != VK_SUCCESS ||
        vkCreatePipelineLayout(c->device, &li, NULL, &p->layout) != VK_SUCCESS)
        return 0;
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code};
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS) return 0;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main",
                                                .pSpecializationInfo = spec},
                                      .layout = p->layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &p->pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    return r == VK_SUCCESS;
}

static int init(VkCtx *c) {
    G.tried = 1;
    G.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    if (!G.push) {
        rt_log("vk: frame interpolation off: no VK_KHR_push_descriptor");
        return 0;
    }
    const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    const int radius = 8;   // quarter texels: +-32 pixels
    const VkSpecializationMapEntry me = {0, 0, 4};
    const VkSpecializationInfo spec = {1, &me, 4, &radius};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0.25f};
    VkSamplerCreateInfo sl = si;
    sl.magFilter = sl.minFilter = VK_FILTER_LINEAR;
    VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                   .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = c->gfx_family};
    if (!make_pass(c, &G.copy, fg_copy_spv, sizeof fg_copy_spv, (VkDescriptorType[]){S, I}, 2, 16, NULL) ||
        !make_pass(c, &G.luma, fg_luma_spv, sizeof fg_luma_spv, (VkDescriptorType[]){S, I}, 2, 0, NULL) ||
        !make_pass(c, &G.flow, fg_flow_spv, sizeof fg_flow_spv, (VkDescriptorType[]){S, S, I}, 3, 0, &spec) ||
        !make_pass(c, &G.median, fg_median_spv, sizeof fg_median_spv, (VkDescriptorType[]){S, I}, 2, 0, NULL) ||
        !make_pass(c, &G.synth, fg_synth_spv, sizeof fg_synth_spv, (VkDescriptorType[]){S, S, S, I}, 4, 4, NULL) ||
        !make_pass(c, &G.camflow, fg_camflow_spv, sizeof fg_camflow_spv, (VkDescriptorType[]){S, S, S, S, I}, 5, 80, NULL) ||
        vkCreateSampler(c->device, &si, NULL, &G.nearest) != VK_SUCCESS ||
        vkCreateSampler(c->device, &sl, NULL, &G.linear) != VK_SUCCESS ||
        vkCreateCommandPool(c->device, &pci, NULL, &G.pool) != VK_SUCCESS) {
        rt_log("vk: frame interpolation off: its pipelines could not be created");
        return 0;
    }
    VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = G.pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 2};
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT};
    if (vkAllocateCommandBuffers(c->device, &ai, G.cb) != VK_SUCCESS || vkCreateFence(c->device, &fi, NULL, &G.fence[0]) ||
        vkCreateFence(c->device, &fi, NULL, &G.fence[1]))
        return 0;
    rt_log("vk: frame interpolation ready (made from the game's last two frames)");
    return G.ok = 1;
}

static void drop(VkCtx *c, Img *i) {
    if (i->view) vkDestroyImageView(c->device, i->view, NULL);
    if (i->image) vmaDestroyImage(c->vma, i->image, i->alloc);
    memset(i, 0, sizeof *i);
}

static int make(VkCtx *c, Img *i, uint32_t w, uint32_t h, VkFormat f) {
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ci, &ai, &i->image, &i->alloc, NULL) != VK_SUCCESS) return i->image = VK_NULL_HANDLE, 0;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = i->image,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = f,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    return vkCreateImageView(c->device, &vi, NULL, &i->view) == VK_SUCCESS;
}

static void free_images(VkCtx *c) {
    for (int k = 0; k < 2; k++) {
        drop(c, &G.frame[k]), drop(c, &G.luma_img[k]), drop(c, &G.flowv[k]);
        for (int m = 0; m < MAXMULT - 1; m++) drop(c, &G.made[k][m]);
    }
    G.w = G.h = 0;
    G.have_prev = 0;
}

static void barrier_all(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &d);
}

static void to_general(VkCommandBuffer cb, VkImage img) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                               .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &d);
}

// Images for frames of w x h: two of each per-frame image (this frame's and the last), two sets of made frames.
static int ensure_images(VkCtx *c, VkCommandBuffer cb, uint32_t w, uint32_t h, int mult) {
    if (G.frame[0].image && G.w == w && G.h == h && G.mult == mult) return 1;
    vk_queue_lock();   // earlier presents may still read the old images
    vkDeviceWaitIdle(c->device);
    vk_queue_unlock();
    free_images(c);
    uint32_t qw = (w + 3) / 4, qh = (h + 3) / 4, cw = (qw + 3) / 4, ch = (qh + 3) / 4;
    int ok = 1;
    for (int k = 0; k < 2 && ok; k++) {
        ok = make(c, &G.frame[k], w, h, FG_FMT) && make(c, &G.luma_img[k], qw, qh, VK_FORMAT_R16_SFLOAT) &&
             make(c, &G.flowv[k], cw, ch, VK_FORMAT_R16G16B16A16_SFLOAT);
        for (int m = 0; m < mult - 1 && ok; m++) ok = make(c, &G.made[k][m], w, h, FG_FMT);
    }
    if (!ok) {
        free_images(c);
        rt_log("vk: frame interpolation: its %ux%u images could not be made; off", w, h);
        return G.ok = 0;
    }
    for (int k = 0; k < 2; k++) {
        to_general(cb, G.frame[k].image), to_general(cb, G.luma_img[k].image), to_general(cb, G.flowv[k].image);
        for (int m = 0; m < mult - 1; m++) to_general(cb, G.made[k][m].image);
    }
    G.w = w, G.h = h, G.mult = mult;
    rt_log("vk: frame interpolation: %dx at %ux%u (source format %d)", mult, w, h, (int)G.src_fmt);
    return 1;
}

static void pass(VkCommandBuffer cb, const Pass *p, const VkImageView *views, const VkSampler *samplers, int n, uint32_t w,
                 uint32_t h, const void *push, uint32_t push_bytes) {
    VkDescriptorImageInfo ii[5];
    VkWriteDescriptorSet ws[5];
    for (int i = 0; i < n; i++) {
        ii[i] = (VkDescriptorImageInfo){samplers[i], views[i], VK_IMAGE_LAYOUT_GENERAL};
        ws[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = (uint32_t)i,
                                       .descriptorCount = 1,
                                       .descriptorType = samplers[i] ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                                     : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       .pImageInfo = &ii[i]};
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
    rec_push(G.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, (uint32_t)n, ws);
    if (push_bytes) vkCmdPushConstants(cb, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    barrier_all(cb);
}

int vk_framegen_make(VkImage src, VkFormat fmt, int sw, int sh, const int crop[4], const OotMotionFrame *mo, VkImage out[],
                     int *ow, int *oh) {
    int mult = vk_framegen_mult();
    if (mult < 2 || !src || !fmt) return G.have_prev = 0, 0;
    VkCtx *c = vk_ctx();
    if (!G.tried && !init(c)) return 0;
    if (!G.ok) return 0;
    G.src_fmt = fmt;
    int x0 = 0, y0 = 0, w = sw, h = sh;
    if (crop && crop[2] > 0 && crop[3] > 0) x0 = crop[0], y0 = crop[1], w = crop[2], h = crop[3];
    if (x0 < 0 || y0 < 0 || x0 + w > sw || y0 + h > sh || w <= 0 || h <= 0) return 0;
    uint64_t now = plat_now_ns();
    int gap = G.last_ns && now - G.last_ns > 100000000ull;   // a loading hitch: nothing to blend
    G.last_ns = now;
    int s = G.set;
    G.set ^= 1;
    vkWaitForFences(c->device, 1, &G.fence[s], VK_TRUE, 5000000000ull);   // this command buffer's last use is done
    if (G.rb_pending == s + 1) {   // OOT_FG_LOG: that readback is complete
        G.rb_pending = 0;
        vmaInvalidateAllocation(c->vma, G.rb_alloc, 0, VK_WHOLE_SIZE);
        const uint16_t *hv = G.rb_map;
        int camn = 0;
        double sc = 0, mag = 0;
        for (int i = 0; i < G.rb_n; i++) {
            uint16_t w4 = hv[4 * i + 3];
            if (w4 == 0x4000) {   // 2.0 in half
                camn++;
                uint16_t z = hv[4 * i + 2];
                int e = z >> 10 & 31, mm = z & 1023;
                sc += e ? ldexp((double)(mm | 1024), e - 25) : ldexp((double)mm, -24);
                for (int q = 0; q < 2; q++) {
                    uint16_t x = hv[4 * i + q];
                    int ex = x >> 10 & 31, mx = x & 1023;
                    double f = ex ? ldexp((double)(mx | 1024), ex - 25) : ldexp((double)mx, -24);
                    mag += f;
                }
            }
        }
        rt_log("vk: frame interpolation: %d of %d blocks took the camera's vector (mean cost %.4f, mean |v| %.1f px)", camn,
               G.rb_n, camn ? sc / camn : 0.0, camn ? mag / camn : 0.0);
    }
    VkCommandBuffer cb = G.cb[s];
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cb, &bi);
    if (!ensure_images(c, cb, (uint32_t)w, (uint32_t)h, mult)) {
        vkEndCommandBuffer(cb);
        return 0;
    }
    if (G.src_view[s]) vkDestroyImageView(c->device, G.src_view[s], NULL);
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = src, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                .format = fmt, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &G.src_view[s]) != VK_SUCCESS) {
        G.src_view[s] = VK_NULL_HANDLE;
        vkEndCommandBuffer(cb);
        return 0;
    }
    barrier_all(cb);   // the game's writes to the source, and the last presents' reads of the images written here
    uint32_t uw = (uint32_t)w, uh = (uint32_t)h, qw = (uw + 3) / 4, qh = (uh + 3) / 4, cw = (qw + 3) / 4, ch = (qh + 3) / 4;
    int cur = G.cur, prev = cur ^ 1, made = 0;
    const int32_t pc[4] = {x0, y0, w, h};
    pass(cb, &G.copy, (VkImageView[]){G.src_view[s], G.frame[cur].view}, (VkSampler[]){G.nearest, VK_NULL_HANDLE}, 2, uw,
         uh, pc, sizeof pc);
    pass(cb, &G.luma, (VkImageView[]){G.frame[cur].view, G.luma_img[cur].view}, (VkSampler[]){G.nearest, VK_NULL_HANDLE},
         2, qw, qh, NULL, 0);
    if (G.have_prev && !gap) {
        pass(cb, &G.flow, (VkImageView[]){G.luma_img[prev].view, G.luma_img[cur].view, G.flowv[0].view},
             (VkSampler[]){G.nearest, G.nearest, VK_NULL_HANDLE}, 3, cw, ch, NULL, 0);
        pass(cb, &G.median, (VkImageView[]){G.flowv[0].view, G.flowv[1].view}, (VkSampler[]){G.nearest, VK_NULL_HANDLE}, 2,
             cw, ch, NULL, 0);
        int fl = 1;   // the vectors the synthesis reads: the median's, or the camera's checked against them
        static int cam_on = -1;
        if (cam_on < 0) { const char *e = getenv("OOT_FG_CAMERA"); cam_on = !(e && *e == '0'); }
        if (cam_on && mo && G.have_cam && mo->depth) {
            struct { float m[12], proj[4], size[4]; } pk;
            oot_motion_delta(G.prev_cam.view, mo->view, pk.m);
            pk.proj[0] = mo->p00, pk.proj[1] = mo->p11, pk.proj[2] = mo->lin_y, pk.proj[3] = mo->lin_x;
            pk.size[0] = (float)uw, pk.size[1] = (float)uh, pk.size[2] = mo->near_, pk.size[3] = 0;
            pass(cb, &G.camflow, (VkImageView[]){G.luma_img[prev].view, G.luma_img[cur].view, G.flowv[1].view, mo->depth,
                                                 G.flowv[0].view},
                 (VkSampler[]){G.nearest, G.nearest, G.nearest, G.linear, VK_NULL_HANDLE}, 5, cw, ch, &pk, sizeof pk);
            fl = 0;
            static int log_on = -1;
            static uint32_t nframes;
            if (log_on < 0) log_on = getenv("OOT_FG_LOG") != NULL;
            if (log_on && ++nframes % 120 == 0 && !G.rb_pending) {
                if (!G.rb) {
                    VkBufferCreateInfo bi2 = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 8ull * 512 * 512,
                                              .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
                    VmaAllocationCreateInfo ai2 = {.usage = VMA_MEMORY_USAGE_AUTO,
                                                   .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
                    VmaAllocationInfo inf;
                    if (vmaCreateBuffer(c->vma, &bi2, &ai2, &G.rb, &G.rb_alloc, &inf) == VK_SUCCESS) G.rb_map = inf.pMappedData;
                }
                if (G.rb_map && cw * ch <= 512 * 512) {
                    VkBufferImageCopy rg = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {cw, ch, 1}};
                    vkCmdCopyImageToBuffer(cb, G.flowv[0].image, VK_IMAGE_LAYOUT_GENERAL, G.rb, 1, &rg);
                    barrier_all(cb);
                    G.rb_pending = s + 1, G.rb_n = (int)(cw * ch);
                }
            }
        }
        for (int k = 1; k < mult; k++) {
            float t = (float)k / (float)mult;
            pass(cb, &G.synth, (VkImageView[]){G.frame[prev].view, G.frame[cur].view, G.flowv[fl].view, G.made[s][k - 1].view},
                 (VkSampler[]){G.linear, G.linear, G.nearest, VK_NULL_HANDLE}, 4, uw, uh, &t, 4);
            out[made++] = G.made[s][k - 1].image;
        }
    }
    vkEndCommandBuffer(cb);
    vkResetFences(c->device, 1, &G.fence[s]);
    VkCommandBufferSubmitInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = 1, .pCommandBufferInfos = &cbi};
    vk_queue_lock();
    VkResult r = vkQueueSubmit2(c->gfx_queue, 1, &si, G.fence[s]);
    vk_queue_unlock();
    if (r != VK_SUCCESS) {
        rt_log("vk: frame interpolation: submit failed (%d); off", (int)r);
        G.ok = 0;
        return 0;
    }
    G.have_prev = 1;
    G.cur = prev;
    if (mo) G.prev_cam = *mo, G.have_cam = 1;
    else G.have_cam = 0;
    *ow = w, *oh = h;
    return made;
}

VkFormat vk_framegen_format(void) { return FG_FMT; }

void vk_framegen_release(void) {
    VkCtx *c = vk_ctx();
    if (!c || !G.tried) return;
    free_images(c);
    for (int k = 0; k < 2; k++)
        if (G.src_view[k]) vkDestroyImageView(c->device, G.src_view[k], NULL);
    Pass *ps[] = {&G.copy, &G.luma, &G.flow, &G.median, &G.synth, &G.camflow};
    for (int i = 0; i < 6; i++) {
        if (ps[i]->pipe) vkDestroyPipeline(c->device, ps[i]->pipe, NULL);
        if (ps[i]->layout) vkDestroyPipelineLayout(c->device, ps[i]->layout, NULL);
        if (ps[i]->dsl) vkDestroyDescriptorSetLayout(c->device, ps[i]->dsl, NULL);
    }
    for (int k = 0; k < 2; k++)
        if (G.fence[k]) vkDestroyFence(c->device, G.fence[k], NULL);
    if (G.pool) vkDestroyCommandPool(c->device, G.pool, NULL);
    if (G.nearest) vkDestroySampler(c->device, G.nearest, NULL);
    if (G.linear) vkDestroySampler(c->device, G.linear, NULL);
    memset(&G, 0, sizeof G);
}
