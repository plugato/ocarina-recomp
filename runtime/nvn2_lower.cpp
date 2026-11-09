// NVN 2 program stage -> the NVN 1 shape the rest of the host takes (gfx_shader.c, vk_exec_draw.c): Eglantine's
// lowering (runtime/third_party/eglantine, GPL-3.0-or-later) rewrites the Ampere code as Maxwell, laid out as TotK's
// translator reads it, and a synthetic control block carries the fields the host reads.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shader_recompiler/frontend/ampere/program_capture.h"

extern "C" void rt_log(const char *fmt, ...);
extern "C" uint64_t gfx_shader_nvn2_hash(const uint8_t *control, const uint8_t *code);

namespace {
constexpr size_t kControl = 0x880;   // gfx_shader.c CONTROL_SIZE
constexpr size_t kC0Layout = 0x860;  // "DKC1" + compute parameter layout (gfx_shader.c NVN2_C0_LAYOUT)
constexpr uint64_t kMaxwellNop = 0x50b0000000070f00ull;
constexpr uint64_t kMaxwellExit = 0xe30000000007000full;

struct Lowered { std::vector<uint8_t> control; std::vector<uint64_t> code; uint64_t origin; };
std::mutex lock;
// Keyed by content, not by address: a game frees program memory and loads other shaders at the same place (a scene
// change), and an address key handed those the old lowering. The key is gfx_shader_nvn2_hash (the dump name).
// ponytail: kept for the process (a few thousand small stages); free with the program if memory ever matters
std::map<uint64_t, Lowered *> cache;
// Lowered code buffer -> its key, for nvn2_origin: the draw path asks it on every draw (vk_exec_draw.c
// light_quad_fault_skip), and a scan of the cache there cost most of the render thread in heavy scenes.
std::unordered_map<const uint8_t *, uint64_t> origins;

void put32(std::vector<uint8_t> &c, size_t at, uint32_t v) { std::memcpy(c.data() + at, &v, 4); }

Lowered *lower(const uint8_t *control, const uint8_t *code) {
    uint32_t code_len, w0;
    std::memcpy(&code_len, control + 0x38, 4), std::memcpy(&w0, code, 4);
    const bool gfx = ((w0 & 31) == 1 || (w0 & 31) == 2) && ((w0 >> 5) & 31) == 4;   // SPH v4
    const uint32_t hdr = gfx ? 0x80 : 0;
    if (code_len <= hdr) {
        rt_log("nvn2: stage with %#x bytes of code (header %#x, word 0 %#x): nothing to lower", code_len, hdr, w0);
        return nullptr;
    }

    std::vector<Shader::Ampere::Instruction> native;
    Shader::Ampere::ProgramCaptureState capture;
    for (uint32_t at = hdr; at + 16 <= code_len; at += 16) {
        Shader::Ampere::Instruction insn;
        std::memcpy(insn.words.data(), code + at, 16);
        native.push_back(insn);
        const auto r = capture.Observe(insn, static_cast<uint32_t>(native.size() - 1));
        if (r != Shader::Ampere::ProgramCaptureState::Result::Continue) break;
    }
    // LOCAL TEST (Ocarina): Lichii's lowering takes the stage, as its shader_environment.cpp passes it (SPH shader type: bits
    // 10-13; 2, 3 tessellation, 4 geometry). Its old call with "full" turned into geometry_stage = true for every stage.
    const uint32_t sph_type = gfx ? (w0 >> 10) & 15 : 0;
    Shader::Ampere::LoweringOptions lower_options;
    lower_options.geometry = sph_type == 4;
    // LOCAL TEST (Ocarina): each HMMA becomes one pseudo instruction (0x5d31/0x5d33) that the translator (patch 0036)
    // turns into a call to a shared helper, instead of ~300 inline Maxwell instructions (the NDSR TAAU had 1.39 M SPIR-V
    // words). OOT_HMMA_COMPACT=0 brings back the inline expansion.
    static const bool compact_hmma = [] { const char *e = std::getenv("OOT_HMMA_COMPACT"); return !(e && *e == '0'); }();
    lower_options.tensor = compact_hmma;
    lower_options.native_code_address = reinterpret_cast<uint64_t>(code) + hdr;
    // LOCAL TEST (Ocarina): a stage in a tessellation slot that writes the stage buffer is a mesh program after a task.
    bool writes_stage_buffer = false;
    for (const auto &insn : native) writes_stage_buffer |= insn.Bits(0, 12) == 0x927;
    lower_options.tessellation_stage = writes_stage_buffer ? Shader::Ampere::TessellationStage::None
                                     : sph_type == 2 ? Shader::Ampere::TessellationStage::Control
                                     : sph_type == 3 ? Shader::Ampere::TessellationStage::Evaluation
                                                     : Shader::Ampere::TessellationStage::None;
    const auto lowered = Shader::Ampere::LowerToMaxwell(native, lower_options);
    if (!lowered) {
        rt_log("nvn2: Ampere lowering failed for %016llx at +%#x: %.*s", (unsigned long long)gfx_shader_nvn2_hash(control, code), lowered.error_offset, (int)lowered.error.size(),
               lowered.error.data());
        if (lowered.error_offset / 16 < native.size()) {   // LOCAL TEST (Ocarina): the instruction it stopped at
            const auto &w = native[lowered.error_offset / 16].words;
            rt_log("nvn2:   words %08x %08x %08x %08x (%zu instructions, gfx %d, w0 %#x)", w[0], w[1], w[2], w[3], native.size(), (int)gfx, w0);
        }
        return nullptr;
    }

    auto *out = new Lowered;
    auto &m = out->code;
    // The lowering places instruction 0 at 0x10 modulo 32 with a scheduling word at every 32-byte boundary. Ryujinx
    // decodes from the code start (graphics: SPH+0x50, compute: 0) and takes every 32-byte boundary counted from there
    // as a scheduling word, so the code starts with a scheduling word and a NOP and instruction 0 lands 0x10 in. For
    // graphics, the 0x30 bytes the host skips and the first 0x50 bytes of the SPH come first. (Instruction 0 straight
    // at SPH+0x50 was decoded as a scheduling word and shifted every group: one real instruction in four was lost.)
    if (gfx) {
        m.assign(6, 0);
        m.resize(16);
        std::memcpy(m.data() + 6, code, 0x50);
        // DKB_NATIVE_GS (vk_exec_draw.c, native_gs_control): the portable geometry contract reads a kind-1 version-3
        // header. The fields it reads sit at the same offsets in the native header, so a lowered geometry stage is
        // relabelled.
        if (((w0 >> 10) & 15) == 4) {
            const uint32_t relabelled = (w0 & ~uint32_t{0x3ff}) | 1u | 3u << 5;
            std::memcpy(m.data() + 6, &relabelled, 4);
        }
        if (((w0 >> 10) & 15) == 3 && writes_stage_buffer) {   // LOCAL TEST (Ocarina): translated as a vertex stage
            const uint32_t relabelled = (w0 & ~(uint32_t{15} << 10)) | 1u << 10;
            std::memcpy(m.data() + 6, &relabelled, 4);
        }
    }
    m.push_back(0), m.push_back(kMaxwellNop);
    size_t pos = 0x10;   // from the code start
    // A full scheduling group of EXITs closes the program: Ryujinx's decoder runs on until a block ends, and a lowered
    // program whose last native instruction is a conditional exit or a self-branch would read past the buffer.
    // The compiler's end marker, an unconditional branch to itself after the last EXIT, becomes an EXIT too: Ryujinx's
    // structuring can join it to the exit path, and the empty endless loop it then emits hangs the GPU (device lost).
    for (size_t i = 0; i < lowered.instructions.size() + 4; i++) {
        if (pos % 32 == 0) m.push_back(0), pos += 8;
        uint64_t insn = i < lowered.instructions.size() ? lowered.instructions[i] : kMaxwellExit;
        if ((insn >> 52) == 0xe24 && ((insn >> 20) & 0xffffff) == 0xfffff8 && ((insn >> 16) & 15) == 7) insn = kMaxwellExit;
        m.push_back(insn), pos += 8;
    }

    // Inline constants (c[0x1]) follow the code in the NVN 2 buffer, from control +0x40 to +0x44; they go after the
    // Maxwell code, where control +0x700 points as in NVN 1. A BRX jump table in them holds native instruction offsets
    // (index * 16, as the lowering requires); the lowered BRX jumps to Maxwell addresses, so those are rewritten.
    uint32_t c1_at, c1_end;
    std::memcpy(&c1_at, control + 0x40, 4), std::memcpy(&c1_end, control + 0x44, 4);
    const uint32_t c1_off = static_cast<uint32_t>(m.size() * 8);
    if (c1_end > c1_at && c1_end - c1_at <= 0x10000) {
        std::vector<uint32_t> c1((c1_end - c1_at) / 4);
        std::memcpy(c1.data(), code + c1_at, c1.size() * 4);
        bool brx = false;
        for (const auto &insn : native) brx |= insn.Bits(0, 12) == 0x949 || insn.Bits(0, 12) == 0x958;
        // ponytail: any 16-aligned word inside the program counts as a table entry; an integer constant that happens to
        // be one would be rewritten too. Track the LDC base of each BRX if that ever bites.
        if (brx)
            for (uint32_t &w : c1)
                if (w % 16 == 0 && w / 16 < lowered.source_instruction_offsets.size()) {
                    const uint32_t i = lowered.source_instruction_offsets[w / 16];
                    // The lowering's address_of (0x50-based). Instruction i sits at address_of(i) - 0x40 from the code
                    // start, and the lowered BRX's base works out to -0x40, so the value lands on instruction i
                    // (Ryujinx FindBrxTargets: target = BRX base + table value).
                    w = 0x50 + i * 8 + ((i + 1) / 3) * 8;
                }
        c1.resize((c1.size() + 1) & ~size_t{1});
        m.resize(m.size() + c1.size() / 2);
        std::memcpy(m.data() + c1_off / 8, c1.data(), c1.size() * 4);
    }

    out->control.assign(kControl, 0);
    const uint32_t len = static_cast<uint32_t>(m.size() * 8);
    const uint32_t type = (w0 >> 10) & 15;   // SPH shader type -> NVN stage
    const uint32_t stage = !gfx ? 5 : type == 5 ? 1 : type == 4 ? 2 : type == 2 ? 3 : type == 3 ? 4 : 0;
    put32(out->control, 0x700, c1_off);
    put32(out->control, 0x704, len);
    put32(out->control, 0x714, stage);
    // LOCAL TEST (Ocarina): a mesh program without a task sits in the vertex slot and writes the stage buffer. NVN 2 control
    // +0x58 low half: threads; +0x60: max primitives, max vertices. A task program has +0x60 zero and is not handled yet.
    uint32_t mesh_threads, mesh_sizes;
    std::memcpy(&mesh_threads, control + 0x58, 4), std::memcpy(&mesh_sizes, control + 0x60, 4);
    bool stage_writes = false;
    for (const auto &insn : native) stage_writes |= insn.Bits(0, 12) == 0x927;
    // Payload: control +0x58 bits 22-30, in lines of 32 words (Nemyu's SET_MESH_INIT_SHADER). Vertices low half of +0x60:
    // 0x7e0040 is the usual 64 vertices, 126 primitives.
    const uint32_t payload_lines = (mesh_threads >> 22) & 0x1ff;
    const uint16_t payload_words = static_cast<uint16_t>((payload_lines ? payload_lines : 1) * 32);
    if (gfx && type == 1 && stage_writes && !mesh_sizes) {
        const uint16_t task[4] = {static_cast<uint16_t>(mesh_threads), 0, 0, payload_words};
        std::memcpy(out->control.data() + kC0Layout, "DKT1", 4);
        std::memcpy(out->control.data() + kC0Layout + 4, task, sizeof task);
    }
    if (gfx && (type == 1 || type == 3) && stage_writes && mesh_sizes) {
        const uint16_t mesh[4] = {static_cast<uint16_t>(mesh_threads), static_cast<uint16_t>(mesh_sizes), static_cast<uint16_t>(mesh_sizes >> 16),
                                  static_cast<uint16_t>(type == 3 ? payload_words : 0)};
        std::memcpy(out->control.data() + kC0Layout, "DKM1", 4);
        std::memcpy(out->control.data() + kC0Layout + 4, mesh, sizeof mesh);
        {   // The per-primitive output map: the 128-byte program header after the code (code_len bytes in), the same layout as an SPH.
            // Its system byte (0x35) and 16 generic nibble bytes (0x36) go after the mesh fields; all zero without per-primitive outputs.
            uint32_t cl = 0;
            std::memcpy(&cl, control + 0x38, 4);
            uint32_t pw0 = 0;
            std::memcpy(&pw0, code + cl, 4);
            if (pw0) {
                std::memcpy(out->control.data() + kC0Layout + 12, code + cl + 0x35, 17);
            }
        }
    }
    if (!gfx) {
        std::memcpy(out->control.data() + 0x748, control + 0x58, 12);   // local size (NVN 2 control +0x58)
        // The compute driver bank is packed per program from these counts (gfx_shader.c, "NVN 2 compute c0"):
        // images +0xb6, uniform buffers +0xb1, storage buffers +0xb2, textures +0xb3.
        // DKC1 adds the eight root-table word counts. The marker and counts enter the code/cache hash, so an older
        // DKC0 translation cannot be reused with the new parameter-block layout.
        const uint8_t lay[8] = {'D', 'K', 'C', '1', control[0xb6], control[0xb1], control[0xb2], control[0xb3]};
        std::memcpy(out->control.data() + kC0Layout, lay, sizeof lay);
        // Shared memory bytes (NVN 2 control +0x64; Bananza's upscaling network uses 0x2a88): the translator sizes the
        // workgroup's shared array from it, and with 0 every LDS read back nothing.
        std::memcpy(out->control.data() + kC0Layout + 8, control + 0x64, 4);
        std::memcpy(out->control.data() + kC0Layout + 12, control + 0xa9, 8);
    }
    return out;
}
}  // namespace

// 1 and the replacement control and code for an NVN 2 stage, 0 when it does not lower.
extern "C" int nvn2_lower(const uint8_t *control, const uint8_t *code, const uint8_t **out_control,
                          const uint8_t **out_code) {
    const uint64_t key = gfx_shader_nvn2_hash(control, code);
    if (!key) {
        uint32_t w[6];
        std::memcpy(w, control + 0x30, sizeof w);
        rt_log("nvn2: stage with a control block of unexpected sizes (+0x30 %#x +0x38 %#x +0x44 %#x, code %p): not "
               "lowered", w[0], w[2], w[5], (const void *)code);
        return 0;
    }
    std::lock_guard<std::mutex> g(lock);
    auto [it, fresh] = cache.try_emplace(key, nullptr);
    if (fresh && (it->second = lower(control, code))) {
        it->second->origin = key;
        origins.emplace(reinterpret_cast<const uint8_t *>(it->second->code.data()), key);
    }
    if (!it->second) return 0;
    *out_control = it->second->control.data();
    *out_code = reinterpret_cast<const uint8_t *>(it->second->code.data());
    return 1;
}

// The dump name (gfx_shader_nvn2_hash) of the NVN 2 stage a lowered code buffer came from, or 0. Lowered buffers live
// for the process, so their addresses never name another stage.
// LOCAL TEST (Ocarina): a per-thread direct-mapped cache in front of the locked map: the replay asks several times per
// draw (3.2% of the replay thread). A buffer's answer never changes once it is in origins; a miss (0) is not cached,
// since the buffer may be lowered later.
extern "C" uint64_t nvn2_origin(const uint8_t *code) {
    static thread_local struct { const uint8_t *code; uint64_t origin; } memo[512];
    auto &m = memo[(reinterpret_cast<uintptr_t>(code) >> 4) * 0x9E3779B97F4A7C15ull >> 55];
    if (code && m.code == code) return m.origin;
    std::lock_guard<std::mutex> g(lock);
    auto it = origins.find(code);
    if (it == origins.end()) return 0;
    m.code = code, m.origin = it->second;
    return it->second;
}
