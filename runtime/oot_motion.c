// LOCAL TEST (Ocarina 1.1.1): the camera and the scene depth of each frame, for the frame interpolation's camera
// vectors (vk_framegen.c, fg_camflow.comp). The game's sun-visibility probe (vertex nvn2 e0da18529f3a86d0, a 1x1
// point pass after the scene) reads both: its c3 holds the main camera's view (rows 3-5: x, y, z) and projection
// (row 6 .x = P00, row 7 .y = P11, rows 8-9), the depth linearisation (row 10: near / (d * c10.y + c10.x), row 11 .x =
// near), and its texture 0 is the scene's D32F depth. When that pass ends, the depth is copied into an R32F image of
// our own (the game's UI draws over the depth's memory later in the frame); the present takes the newest copy with
// its camera. A ring of three: the present thread is at most a frame behind the replay.
#include "vk.h"

#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "fg_depth_spv.h"
#include "host.h"
#include "nvn_api.h"
#include "vk_exec.h"
#include "vk_rec.h"

#define RING 3
#define NVN_STAGE_VS 0   // as vk_exec_draw.c: the ubo, tex arrays of VkExec are indexed by NVN stage

static struct {
    int tried, ok;
    PFN_vkCmdPushDescriptorSetKHR push;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkSampler nearest;
    VkImage img[RING];
    VmaAllocation alloc[RING];
    VkImageView view[RING];
    uint32_t w, h;
    OotMotionFrame cam[RING];
    int wr, pending;
    const VkTexImage *pend_depth;
    OotMotionFrame pend;
    volatile int latest;   // ring slot of the newest copy, -1 none; taken by the present
    int written;           // ring slot of the newest copy, for the scene's temporal pass (replay thread)
    uint64_t seq;          // copies made
} M = {.latest = -1, .written = -1};

// LOCAL TEST: scene_upscaler = taa or fsr2 (vk_exec_draw.c) needs the camera and depth too.
int oot_motion_scene_taa(void) {
    const char *v = cfg_value("TOTK_SCENE_UPSCALER");   // taa and fsr2 (vk_exec_draw.c) take the camera and depth
    return !strcmp(v, "taa") || !strcmp(v, "fsr2");
}

static int init(VkCtx *c) {
    M.tried = 1;
    M.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    VkDescriptorSetLayoutBinding b[2] = {
        {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 2, .pBindings = b};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &M.dsl};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof fg_depth_spv,
                                   .pCode = fg_depth_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (!M.push || vkCreateDescriptorSetLayout(c->device, &dci, NULL, &M.dsl) != VK_SUCCESS ||
        vkCreatePipelineLayout(c->device, &li, NULL, &M.layout) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &M.nearest) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS)
        return 0;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = M.layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &M.pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) return 0;
    rt_log("vk: frame interpolation: camera vectors from the scene depth ready");
    return M.ok = 1;
}

static int ensure(VkCtx *c, VkCommandBuffer cb, uint32_t w, uint32_t h) {
    if (M.img[0] && M.w == w && M.h == h) return 1;
    for (int i = 0; i < RING; i++) {   // ponytail: a size change mid-play leaves the old images to the deferred free
        if (M.view[i]) vkDestroyImageView(c->device, M.view[i], NULL);
        if (M.img[i]) vmaDestroyImage(c->vma, M.img[i], M.alloc[i]);
        M.view[i] = VK_NULL_HANDLE, M.img[i] = VK_NULL_HANDLE;
    }
    M.latest = -1;
    for (int i = 0; i < RING; i++) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                .format = VK_FORMAT_R32_SFLOAT, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ci, &ai, &M.img[i], &M.alloc[i], NULL) != VK_SUCCESS) return M.img[i] = VK_NULL_HANDLE, 0;
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = M.img[i],
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R32_SFLOAT,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(c->device, &vi, NULL, &M.view[i]) != VK_SUCCESS) return 0;
        VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                   .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                   .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                   .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                   .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = M.img[i], .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
        vkCmdPipelineBarrier2(cb, &d);
    }
    M.w = w, M.h = h;
    return 1;
}

// prepare_draw of the probe's draw (vk_exec_draw.c): its camera from the uniform snapshot, its depth texture.
void oot_motion_note(VkExec *ex) {
    if (vk_framegen_mult() < 2 && !oot_motion_scene_taa()) return;
    const p_bind_buf *b = &ex->ubo[NVN_STAGE_VS][0];   // c3
    VkDescriptorBufferInfo bi;
    const float *c3;
    if (!b->size || b->size < 12 * 16 || !vk_usnap_find(b->addr, 12 * 16, 16, &bi) || !(c3 = (const float *)vk_usnap_map(&bi)))
        return;
    uint64_t h = ex->tex[NVN_STAGE_VS][0];
    view_t v;
    int hv = 0;
    const VkTexImage *t = h ? vk_texture_by_id(&ex->tex_pool, NVN_HANDLE_TEX(h), &v, &hv) : NULL;
    if (!t || !(t->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) return;
    OotMotionFrame *f = &M.pend;
    memcpy(f->view, c3 + 3 * 4, 12 * sizeof(float));   // rows 3, 4, 5
    f->p00 = c3[6 * 4 + 0], f->p11 = c3[7 * 4 + 1];
    f->lin_x = c3[10 * 4 + 0], f->lin_y = c3[10 * 4 + 1], f->near_ = c3[11 * 4 + 0];
    M.pend_depth = t;
    M.pending = 1;
}

// vk_exec_pass_end, outside the probe's pass: the depth copy.
void oot_motion_pass_end(VkExec *ex) {
    if (!M.pending) return;
    M.pending = 0;
    VkCtx *c = vk_ctx();
    if (!M.tried) init(c);
    const VkTexImage *t = M.pend_depth;
    if (!M.ok || !t || !ensure(c, ex->cb, t->width, t->height)) return;
    VkImageView dv = vk_exec_target_view(t, t->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1, 0);
    if (!dv) return;
    int s = M.wr;
    M.wr = (M.wr + 1) % RING;
    vk_exec_barrier(ex);
    VkDescriptorImageInfo ii[2] = {{M.nearest, dv, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, M.view[s], VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet w[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &ii[1]}};
    vkCmdBindPipeline(ex->cb, VK_PIPELINE_BIND_POINT_COMPUTE, M.pipe);
    rec_push(M.push, ex->cb, VK_PIPELINE_BIND_POINT_COMPUTE, M.layout, 0, 2, w);
    vkCmdDispatch(ex->cb, (M.w + 7) / 8, (M.h + 7) / 8, 1);
    ex->recorded++;
    vk_exec_forget_compute();
    vk_exec_barrier(ex);
    M.cam[s] = M.pend;
    M.cam[s].depth = M.view[s];
    M.cam[s].w = M.w, M.cam[s].h = M.h;
    M.written = s, M.seq++;
    __atomic_store_n(&M.latest, s, __ATOMIC_RELEASE);
}

// The scene's temporal pass, on the replay thread: this frame's copy (seq: which one; 0 when none yet).
int oot_motion_peek(OotMotionFrame *out, uint64_t *seq) {
    if (M.written < 0) return 0;
    *out = M.cam[M.written];
    *seq = M.seq;
    return 1;
}

// vk_be_present: the newest copy and its camera, once (0: none since the last present).
int oot_motion_take(OotMotionFrame *out) {
    int s = __atomic_exchange_n(&M.latest, -1, __ATOMIC_ACQ_REL);
    if (s < 0) return 0;
    *out = M.cam[s];
    return 1;
}

// The rows of last view * inverse(this view) (views are rigid: inverse = [R^T | -R^T t]).
void oot_motion_delta(const float *last, const float *cur, float m[12]) {
    float ri[3][3], ti[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) ri[i][j] = cur[j * 4 + i];
    for (int i = 0; i < 3; i++) ti[i] = -(ri[i][0] * cur[3] + ri[i][1] * cur[7] + ri[i][2] * cur[11]);
    for (int r = 0; r < 3; r++) {
        for (int j = 0; j < 3; j++) m[r * 4 + j] = last[r * 4] * ri[0][j] + last[r * 4 + 1] * ri[1][j] + last[r * 4 + 2] * ri[2][j];
        m[r * 4 + 3] = last[r * 4] * ti[0] + last[r * 4 + 1] * ti[1] + last[r * 4 + 2] * ti[2] + last[r * 4 + 3];
    }
}

