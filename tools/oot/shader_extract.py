"""Extracts every NVN 2 shader stage from the player's own Ocarina of Time (Switch 2) RomFS, for shader precompilation at install.

    .venv\\Scripts\\python -I tools\\oot\\shader_extract.py --romfs-image romfs.bin --index index.json --out <dir>
    .venv\\Scripts\\python -I tools\\oot\\shader_extract.py --romfs-image romfs.bin --index index.json --check <dump dir>
    .venv\\Scripts\\python -I tools\\oot\\shader_extract.py --romfs-image romfs.bin --index index.json --pack <file>

--out writes each stage as <fnv>.ctl (control section) and <fnv>.bin (code), the names and bytes DKB_SHADER_DUMP gives
(runtime/gfx_shader.c gfx_shader_nvn2_hash). --pack writes them all to one file instead, the input of
ocarina.exe --lower-stages (gfx_shader.c gfx_shader_lower_stages): "OOTSTGPK", version 1, count (u32 each), then per
stage control size, code size (u32 each), control, code; one file, as Windows Defender once quarantined a loose .bin.
--check compares against such a dump and prints counts only. The output
is game data: it stays on the player's PC and never goes into the repository.

Containers (decompressed with zstd, dictionaries from Pack/ZsDic.pack.zs; SARC archives opened member by member):
BNSH (bfsha in Shader/ and Layout2Shader/, effects in Effect/*.vsetb.byml), agl's BAHS (Lib/, Shader/ sharcb, UI),
agl's NDSR (the upscaler) and nn::ndve's NDVE (movie upscaling).

The container parsers follow seedds/nxnative dm/gfx/tools/shader_split.py at commit
24fe1e5a8a6fb21eebe6b0c0ff6078b387d7da57 (bnsh_shaders, bahs_shaders, ndsr_shaders, ndve_shaders, sarc_members, zs_dict_id):
    Copyright (c) 2026 seedds, MIT License (see THIRD_PARTY_NOTICES.md).
"""
import argparse
import json
import os
import struct
import sys
import time

import zstandard

CTL_MAGIC = bytes.fromhex('34127698c9000000')   # NVN 2 control section: magic 0x98761234, version 0xc9
ZSTD_MAGIC = bytes.fromhex('28b52ffd')
# Large folders without shaders (textures, terrain, models, sound, bakes, physics, culling): skipping them takes the scan
# from 25 GB to 0.8 GB. A sampled scan of them found no shader container.
SKIP = {'Tex', 'TerrainArc', 'Model', 'Sound', 'Bake', 'Event', 'Phive', 'PreComputedCulling', 'Voice', 'Foliage'}


def is_ctl(d, c):
    return 0 <= c and c + 0x58 <= len(d) and d[c:c + 8] == CTL_MAGIC


def sizes(d, c):
    """Control and code size the runtime hashes (gfx_shader.c nvn2_sizes): +0x30 (0 -> 0xd8), code +0x38 up to c1 end +0x44"""
    ctl_size, n, c1_end = struct.unpack_from('<I', d, c + 0x30)[0], *struct.unpack_from('<I', d, c + 0x38), struct.unpack_from('<I', d, c + 0x44)[0]
    ctl_size = ctl_size or 0xd8
    if c1_end > n and c1_end - n <= 0x10000: n = c1_end
    return ctl_size, n


def fnv(*parts):
    h = 0xcbf29ce484222325
    for p in parts:
        for b in p: h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
    return h


def stage(d, c, k, out):
    """Adds the stage whose control section is at d[c:] and code at d[k:] when both lie inside d"""
    if not is_ctl(d, c): return
    ctl_size, n = sizes(d, c)
    if c + ctl_size > len(d) or k < 0 or k + n > len(d) or not n: return
    out.append((d[c:c + ctl_size], d[k:k + n]))


def bnsh(d, base, out):
    """BNSH: grsc at +0x60, variation count +0x7c, table +0x80 (0x40 bytes each, program at +0x10); a program's stage
    descriptors at +0x08.. (vs hs ds gs ps cs) and NVN 2's task and mesh at +0x40 / +0x48; a descriptor holds the
    control pointer +0x08 and code pointer +0x10. Pointers count from the BNSH start."""
    if base + 0x88 > len(d) or d[base + 0x60:base + 0x64] != b'grsc': return
    nvar, pvar = struct.unpack_from('<IQ', d, base + 0x7c)
    for v in range(nvar):
        at = base + pvar + 0x40 * v + 0x10
        if at + 8 > len(d): return
        prog = struct.unpack_from('<Q', d, at)[0]
        if not prog: continue
        for off in (8, 0x10, 0x18, 0x20, 0x28, 0x30, 0x40, 0x48):
            if base + prog + off + 8 > len(d): continue
            sc = struct.unpack_from('<Q', d, base + prog + off)[0]
            if not sc or base + sc + 0x18 > len(d): continue
            pctl, pcode = struct.unpack_from('<QQ', d, base + sc + 8)
            stage(d, base + pctl, base + pcode, out)


def bahs(d, out):
    """agl BAHS: +0x18 code region length, +0x1c its start; 16-byte entries {code offset, 0, control length, signed
    control offset from the entry}. A false entry inside a control section is dropped (shader_split.py)."""
    rsz, region = struct.unpack_from('<II', d, 0x18)
    found = []
    for a in range(0, min(region, len(d)) - 16, 4):
        off, z, csz, rel = struct.unpack_from('<IIIi', d, a)
        c = a + rel
        if z or off % 0x100 or not 0x48 <= csz < 0x10000 or c < 0 or c + csz > region or not is_ctl(d, c): continue
        if off + struct.unpack_from('<I', d, c + 0x44)[0] > rsz: continue
        found.append((a, c, region + off))
    spans = [(c, c + 0x48) for _, c, _ in found]
    for a, c, k in found:
        if not any(lo <= a < hi for lo, hi in spans): stage(d, c, k, out)


def ndsr(d, out):
    """agl NDSR: 16-byte records from +0x10 (or +0x14) to a zero one: control offset, control length << 4 | type, code
    offset, ...; data records (not a control section) are skipped 8 bytes at a time"""
    if len(d) < 0x20: return
    def hits(o): return sum(1 for i in range(min(16, (len(d) - o) // 16)) if is_ctl(d, struct.unpack_from('<I', d, o + 16 * i)[0]))
    o = max((0x10, 0x14), key=hits)
    while o + 16 <= len(d):
        c, cl, k, x = struct.unpack_from('<4I', d, o)
        if not (c or cl or k or x): break
        if not is_ctl(d, c): o += 8; continue
        o += 16
        stage(d, c, k, out)


def ndve(d, out):
    """nn::ndve: +0xc NDSR pack length, +0x14 model count, the NDSR pack after the 8-byte model records, then the
    networks: control sections, each with its code right behind its own length (+0x30)"""
    pack = 0x18 + 8 * struct.unpack_from('<I', d, 0x14)[0]
    nets = pack + struct.unpack_from('<I', d, 0xc)[0]
    ndsr(d[pack:nets], out)
    c = d.find(CTL_MAGIC, nets)
    while c >= 0:
        if not c & 3 and is_ctl(d, c): stage(d, c, c + struct.unpack_from('<I', d, c + 0x30)[0], out)
        c = d.find(CTL_MAGIC, c + 4)


def sarc_members(d):
    hl = struct.unpack_from('<H', d, 4)[0]
    doff = struct.unpack_from('<I', d, 0xc)[0]
    for i in range(struct.unpack_from('<H', d, hl + 6)[0]):
        _, _, s, e = struct.unpack_from('<IIII', d, hl + 12 + 16 * i)
        yield d[doff + s:doff + e]


def scan(d, out, depth=0):
    if d[:4] == ZSTD_MAGIC: d = unzs(d)
    if d[:4] == b'SARC' and depth < 8:
        for m in sarc_members(d): scan(m, out, depth + 1)
        return
    if d[:4] == b'BAHS': return bahs(d, out)
    if d[:4] == b'NDSR': return ndsr(d, out)
    if d[:4] == b'NDVE': return ndve(d, out)
    at = d.find(b'BNSH\0\0\0\0')
    while at >= 0:
        bnsh(d, at, out)
        at = d.find(b'BNSH\0\0\0\0', at + 8)


DICTS = {}
def unzs(d):
    """A zstd file, with the dictionary its frame header names (Pack/ZsDic.pack.zs)"""
    fhd = d[4]
    n, o = (0, 1, 2, 4)[fhd & 3], 6 - (fhd >> 5 & 1)
    did = int.from_bytes(d[o:o + n], 'little')
    if did and did not in DICTS: sys.exit(f'a frame needs zstd dictionary {did}, which Pack/ZsDic.pack.zs does not have')
    z = zstandard.ZstdDecompressor(dict_data=DICTS[did]) if did else zstandard.ZstdDecompressor()
    return z.stream_reader(d).read()


def extract(image, index):
    files = json.load(open(index, encoding='utf-8'))['files']
    uniq = set()   # FNV-1a runs in Python: hash each distinct stage once
    with open(image, 'rb') as rom:
        def read(f):
            rom.seek(f['offset'])
            return rom.read(f['size'])
        for f in files:
            if f['path'] == 'Pack/ZsDic.pack.zs':
                for m in sarc_members(unzs(read(f))):
                    if m[:4] == bytes.fromhex('37a430ec'): DICTS[struct.unpack_from('<I', m, 4)[0]] = zstandard.ZstdCompressionDict(m)
        for f in files:
            if f['path'].split('/')[0] in SKIP or f['path'] == 'Pack/ZsDic.pack.zs': continue
            d = read(f)
            if d[:4] != ZSTD_MAGIC and b'SARC' != d[:4] and CTL_MAGIC not in d: continue
            out = []
            scan(d, out)
            uniq.update(out)
    return {fnv(ctl, code): (ctl, code) for ctl, code in uniq}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--romfs-image', required=True)
    ap.add_argument('--index', required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--out')
    g.add_argument('--check', metavar='DUMPDIR')
    g.add_argument('--pack', metavar='FILE')
    a = ap.parse_args()
    t = time.time()
    stages = extract(a.romfs_image, a.index)
    print(f'{len(stages)} distinct stages in {time.time() - t:.0f} s, {sum(len(c) + len(k) for c, k in stages.values()) >> 20} MiB')
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        for h, (ctl, code) in stages.items():
            for ext, b in (('ctl', ctl), ('bin', code)):
                with open(os.path.join(a.out, f'{h:016x}.{ext}'), 'wb') as f: f.write(b)
    elif a.pack:
        with open(a.pack, 'wb') as f:
            f.write(b'OOTSTGPK' + struct.pack('<II', 1, len(stages)))
            for ctl, code in stages.values(): f.write(struct.pack('<II', len(ctl), len(code)) + ctl + code)
    else:
        dumped = {int(n[:-4], 16) for n in os.listdir(a.check) if n.endswith('.ctl')}
        print(f'dump {len(dumped)}, reproduced {len(dumped & stages.keys())}, missing {len(dumped - stages.keys())}')


if __name__ == '__main__':
    main()
