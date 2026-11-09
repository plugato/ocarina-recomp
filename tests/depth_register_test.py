"""Translate a Maxwell pixel shader that writes depth after a partly written colour target, and check which register
becomes gl_FragDepth (tools/xlate/patches/0038-depth-register-after-whole-targets.patch).

The colours are read four registers per enabled render target (component c of the n-th such target in R(4n + c)),
then the sample mask, then the depth. With OmapTargets 0x10f (RT0 RGBA, RT2 R) the depth is R9; the translator used
to count enabled components and read R6. The fixture sets R6 = 0.25 and R9 = 0.75, so the stored depth must be the
constant 0.75. Original instructions, no game data or GPU.

Usage: python tests/depth_register_test.py --translator build/xlate/totk_xlate.exe [--spirv-val PATH]
       (a managed totk_xlate.dll runs through dotnet)
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PT = 7 << 16


def movi(dst, value):   # MOV32I Rd, value
    return 0x0100000000000000 | (value << 20) | (15 << 12) | PT | dst


def shader():
    header = [0] * 20
    header[0] = 5 << 10   # fragment stage
    header[18] = 0x10f    # RT0 RGBA, RT2 R
    header[19] = 2        # writes depth
    f = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]
    ops = [movi(r, 0) for r in (0, 1, 2, 3, 4)] + [movi(6, f(0.25)), movi(9, f(0.75)), 0xE30000000007000F]   # EXIT
    while len(ops) % 3:
        ops.append(0)
    words = []
    for i in range(0, len(ops), 3):
        words += [0, *ops[i:i + 3]]   # Maxwell scheduling word, then three instructions
    return struct.pack('<20I', *header) + struct.pack('<%dQ' % len(words), *words) + bytes(256)


def stored_depth(path):
    data = path.read_bytes()
    w = struct.unpack('<%dI' % (len(data) // 4), data)
    consts, alias, frag_depth, stores, i = {}, {}, set(), [], 5
    while i < len(w):
        n, op = w[i] >> 16, w[i] & 0xffff
        a = w[i + 1:i + n]
        if op == 71 and len(a) >= 3 and a[1] == 11 and a[2] == 22:   # OpDecorate BuiltIn FragDepth
            frag_depth.add(a[0])
        elif op == 43:                   # OpConstant
            consts[a[1]] = a[2]
        elif op == 124:                  # OpBitcast keeps the bits
            alias[a[1]] = a[2]
        elif op == 62:                   # OpStore
            stores.append((a[0], a[1]))
        i += n
    values = []
    for ptr, value in stores:
        if ptr in frag_depth:
            while value in alias:
                value = alias[value]
            values.append(consts.get(value))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--translator', required=True, type=Path)
    parser.add_argument('--spirv-val', type=Path)
    args = parser.parse_args()
    out = ROOT / 'build/test/depth-register'
    out.mkdir(parents=True, exist_ok=True)
    code = out / 'depth_register.bin'
    code.write_bytes(shader())
    jobs = out / 'jobs.json'
    jobs.write_text(json.dumps([{'key': 'depth_register', 'stage': 'ps', 'code': str(code.resolve()), 'debug': False}]))
    exe = args.translator.resolve()
    run = subprocess.run((['dotnet', str(exe)] if exe.suffix == '.dll' else [str(exe)]) + [str(jobs.resolve()), str(out.resolve())],
                         capture_output=True, encoding='utf-8')
    if run.returncode:
        raise AssertionError('translator failed: ' + run.stderr.encode('ascii', 'backslashreplace').decode())
    spirv = out / 'depth_register.spv'
    if args.spirv_val:
        subprocess.run([str(args.spirv_val.resolve()), '--target-env', 'vulkan1.2', str(spirv)], check=True)
    got = stored_depth(spirv)
    want = struct.unpack('<I', struct.pack('<f', 0.75))[0]
    ok = got == [want]
    print('depth register: stored %s, want [0x%08x] (R9 = 0.75; R6 = 0.25 is the old register)' %
          (['0x%08x' % v if v is not None else None for v in got], want), 'ok' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
