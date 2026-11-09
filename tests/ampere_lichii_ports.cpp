// Lichii nx2 lowering tests (GPL-3.0-or-later), ported without Catch2: #588 (TXD/ATOMG shadows,
// 33acb0e1a0, @vouion), #279 (uniform IMAD, c149054446) and #398 (TLD4.B from a second descriptor
// buffer, 54dedf97b6, MateusAugustoBrito). No game or GPU.
#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
using Shader::Ampere::Instruction; using Shader::Ampere::LowerToMaxwell; using Shader::Ampere::NativeRegisterCopy;
static int fails = 0;
#define CHECK(x) do { if (!(x)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #x); ++fails; } } while (0)
#define REQUIRE(x) do { if (!(x)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #x); ++fails; return; } } while (0)
constexpr Instruction Exit{{0x794d, 0, 0x03800000, 0}};
constexpr Instruction U{{0x0000fc02, 0, 0, 0}};
constexpr Instruction CapturedPerTapGather{{0x040a7b63, 0x33800408, 0x041e2f08, 0x100fc200}};
u64 Copy(u64 p, u32 t, u32 s) { return 0x5c98000000000000ULL | p | t | (15ULL << 39) | (u64{s} << 20); }
u64 MoveImmediate(u64 p, u32 t, u32 v) { return 0x0100000000000000ULL | p | t | (15ULL << 12) | (u64{v} << 20); }
std::array<Instruction, 4> PerTapProgram() { return {{{{0x00087802, 0xff00ff00, 0x00000f00, 0x000fe400}}, {{0x00097802, 0x01000001, 0x00000f00, 0x000fe400}}, CapturedPerTapGather, Exit}}; }

void atomg() {
    const Instruction a{{0xff0479a8, 0x000000f0, 0x081ee144, 0x000fe200}};
    const auto l = LowerToMaxwell(std::array{U, a, Exit}); if (!l) std::printf("  %s\n", std::string(l.error).c_str()); REQUIRE(l);
    auto b = l.instructions.begin() + l.source_instruction_offsets[1], e = l.instructions.begin() + l.source_instruction_offsets[2];
    auto n = std::find_if(b, e, [](u64 w) { return (w >> 56) == 0xed; }); REQUIRE(n != e);
    u32 d = (*n >> 20) & 0xff; CHECK(d < 216); CHECK(std::find(b, n, NativeRegisterCopy(0, d, 1, 240)) != n);
}
void txd(u32 w0, u32 w1, bool layer) {
    const Instruction g{{w0, w1, 0x001e03ff, 0x000fe200}};
    const auto l = LowerToMaxwell(std::array{U, g, Exit}); if (!l) std::printf("  %s\n", std::string(l.error).c_str()); REQUIRE(l);
    auto b = l.instructions.begin() + l.source_instruction_offsets[1], e = l.instructions.begin() + l.source_instruction_offsets[2];
    auto n = std::find_if(b, e, [](u64 w) { return (w >> 56) == 0xde; }); REQUIRE(n != e);
    u32 gr = (*n >> 20) & 0xff; CHECK(gr < 216);
    if (layer) { CHECK(((*n >> 8) & 0xff) == 210); CHECK((gr + 4 <= 210 || gr >= 213)); }
    else for (u32 i = 0; i < 4; ++i) CHECK(std::find(b, n, NativeRegisterCopy(0, gr + i, 1, 240 + i)) != n);
}
void imad() {
    for (const Instruction m : {Instruction{{0x1de57c24, 0x0000000a, 0x0f8e02ff, 0x000fe200}}, Instruction{{0x31e17c24, 0x0000000a, 0x0f8e02ff, 0x000fe200}}}) {
        const auto p = Shader::Ampere::PrepareRegisterPressure(m, 216); REQUIRE(p.valid);
        CHECK(p.instruction.Bits(16, 8) == 212 + m.Bits(16, 8) % 4);
        REQUIRE(p.prefix.size() == 8); REQUIRE(p.suffix.size() == 8);
        const auto l = LowerToMaxwell(std::array{U, m, Exit}); if (!l) std::printf("  %s\n", std::string(l.error).c_str()); REQUIRE(l);
        CHECK(std::equal(p.prefix.begin(), p.prefix.end(), l.instructions.begin()));
    }
}
void tld4b() {
    const Instruction depth{{0x180c7b63, 0x30800e1c, 0x001e4f0e, 0x000f4200}};
    const u64 pred = 7ULL << 16;
    const u64 handle = 0x4c98000000070000ULL | (15ULL << 39) | (4ULL << 20) | (14ULL << 34);
    auto t = [&](u32 meta, u32 ot) -> u64 { return 0xdec0000000000000ULL | pred | 250 | (250ULL << 8) | (u64{meta} << 20) | (2ULL << 28) | (15ULL << 31) | (u64{ot} << 36) | (1ULL << 49) | (7ULL << 51); };
    const std::array<u64, 4> res{Copy(pred, 10, 250), Copy(pred, 11, 251), Copy(pred, 8, 252), Copy(pred, 9, 253)};
    auto check = [&](const auto& l, u32 i, const auto& ex) {
        if (!l) std::printf("  %s\n", std::string(l.error).c_str()); REQUIRE(l); REQUIRE(l.required_texture_buffer == 2);
        auto b = l.instructions.begin() + l.source_instruction_offsets[i];
        CHECK(std::equal(ex.begin(), ex.end(), b)); CHECK(std::equal(res.begin(), res.end(), b + ex.size()));
    };
    const Instruction plain{{0x040a7b63, 0x33800408, 0x041e0f08, 0x100fc200}};
    check(LowerToMaxwell(std::array{depth, plain, Exit}), 1, std::array<u64, 4>{Copy(pred, 250, 4), Copy(pred, 251, 8), handle | 246, t(246, 0)});
    auto ao = plain; ao.words[2] |= 1U << 12;
    check(LowerToMaxwell(std::array{depth, ao, Exit}), 1, std::array<u64, 5>{Copy(pred, 250, 4), Copy(pred, 251, 5), handle | 246, Copy(pred, 247, 8), t(246, 1)});
    const auto pt = PerTapProgram();
    check(LowerToMaxwell(std::array{depth, pt[0], pt[1], pt[2], Exit}), 3, std::array<u64, 6>{MoveImmediate(7ULL << 16, 248, 0xff00ff00), MoveImmediate(7ULL << 16, 249, 0x01000001), Copy(pred, 250, 4), Copy(pred, 251, 5), handle | 247, t(247, 2)});
    const auto same = LowerToMaxwell(std::array{plain, plain, Exit}); REQUIRE(same);
    CHECK((same.instructions[same.source_instruction_offsets[1] + 2] >> 56) == 0xc8);
}
int main() {
    std::puts("atomg"); atomg(); std::puts("txd"); txd(0x08047b6c, 0x200000f0, false); std::puts("txd layer"); txd(0xd2047b6c, 0xa00000f0, true);
    std::puts("imad"); imad(); std::puts("tld4b"); tld4b();
    std::printf("%s (%d failures)\n", fails ? "FAILED" : "OK", fails); return fails != 0;
}
