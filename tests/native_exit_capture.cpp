// Original native instruction authors; exercises real capture and lowering, no game/GPU.
#include <cstdio>
#include <vector>
#include "shader_recompiler/frontend/ampere/program_capture.h"

using namespace Shader::Ampere;
static unsigned checks, failures;
#define CHECK(x) do { checks++; if (!(x)) { failures++; std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); } } while (0)

static void bits(Instruction& i,unsigned at,unsigned n,u32 value) {
    for(unsigned b=0;b<n;b++) {
        const unsigned word=(at+b)/32,bit=(at+b)%32;
        i.words[word]=(i.words[word]&~(1u<<bit))|(((value>>b)&1u)<<bit);
    }
}
static Instruction op(u32 opcode,u32 predicate=7) {
    Instruction i{};bits(i,0,12,opcode);bits(i,12,4,predicate);return i;
}
static Instruction exit_op(u32 predicate=7,u32 secondary=7) {
    auto i=op(0x94d,predicate);bits(i,87,4,secondary);return i;
}
static Instruction continuation() {
    auto i=op(0x802);bits(i,16,8,12);bits(i,32,32,0x13579bdf);bits(i,72,4,15);return i;
}

static void predicate_matrix() {
    for(unsigned normal=0;normal<16;normal++) for(unsigned secondary=0;secondary<16;secondary++) {
        // PT/PT terminates at the first EXIT. Every other guard may fall through
        // (or is !PT and never exits), so the authored MOV and final EXIT remain.
        const std::vector<Instruction> code{op(0x918),exit_op(normal,secondary),continuation(),exit_op()};
        const size_t expected=(normal==7 && secondary==7)?2:4;
        auto full=LowerToMaxwell(code,"full");CHECK(bool(full));
        if(full) CHECK(full.source_size==expected*sizeof(Instruction));
        ProgramCaptureState capture;
        size_t count=0;
        for(size_t index=0;index<code.size();index++) {
            const auto status=capture.Observe(code[index],static_cast<u32>(index));
            CHECK(status!=ProgramCaptureState::Result::InvalidTarget);
            count=index+1;
            if(status==ProgramCaptureState::Result::Complete)break;
        }
        CHECK(count==expected);
        auto captured=LowerToMaxwell(std::span(code.data(),count),"full");
        CHECK(bool(captured));
        if(captured && full) {
            CHECK(captured.instructions==full.instructions);
            CHECK(captured.source_instruction_offsets==full.source_instruction_offsets);
            CHECK(captured.source_size==full.source_size);
        }
    }
}

static void pending_call_boundary() {
    ProgramCaptureState capture;
    auto call=op(0x944);bits(call,16,8,20);bits(call,32,32,32); // target source index3
    CHECK(capture.Observe(call,0)==ProgramCaptureState::Result::Continue);
    CHECK(capture.Observe(exit_op(7,0),1)==ProgramCaptureState::Result::Continue);
    // A conditional main EXIT must not set main_exited: this RET can return to
    // the still-live main path. Only the later unconditional EXIT arms the end.
    CHECK(capture.Observe(op(0x950),3)==ProgramCaptureState::Result::Continue);
    CHECK(capture.Observe(exit_op(),4)==ProgramCaptureState::Result::Continue);
    CHECK(capture.Observe(op(0x950),5)==ProgramCaptureState::Result::Complete);
}

static void uniform_branch_boundary() {
    ProgramCaptureState capture;
    auto branch=op(0x947);bits(branch,32,32,33); // relative32 plus the .U bit
    CHECK(capture.Observe(branch,0)==ProgramCaptureState::Result::Continue);
    CHECK(capture.Observe(exit_op(),1)==ProgramCaptureState::Result::Continue);
    CHECK(capture.Observe(exit_op(),3)==ProgramCaptureState::Result::Complete);
}

// A per-tap gather (TLD4.PTP) whose packed offsets in R16, R17 are not immediates (an S2R here)
// lowers anyway: the offsets go to R248, R249 by register move and the translator splits the gather.
static void per_tap_runtime_offsets() {
    auto s2r=op(0x919);bits(s2r,16,8,16);
    auto mov=op(0x202);bits(mov,16,8,17);bits(mov,32,8,16);bits(mov,72,4,15);
    const std::vector<Instruction> code{s2r,mov,Instruction{{0x06187b63U,0x30007c10,0x001e2f0eU,0}},exit_op()};
    auto lowered=LowerToMaxwell(code);
    CHECK(bool(lowered));
    unsigned moves=0;
    for(const u64 insn:lowered.instructions)
        for(u64 i=0;i<2;i++) moves+=insn==(0x5c98000000000000ULL|(7ULL<<16)|(248+i)|(15ULL<<39)|((16+i)<<20));
    CHECK(moves==2);
}

int main() {
    predicate_matrix();pending_call_boundary();uniform_branch_boundary();per_tap_runtime_offsets();
    std::printf("native EXIT capture: %u checks, %u failures\n",checks,failures);
    return failures?1:0;
}
