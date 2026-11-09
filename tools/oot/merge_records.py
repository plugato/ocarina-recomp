"""Merges record files of runtime/gfx_shader.c and runtime/vk_pipeline.c (shader-answers.bin "TOTKSANS",
pipeline-keys.bin "TOTKPKEY"): the shader preparation seed from several play recordings, and a seed's pipeline keys
into a player's own at install time.

    python -I tools/oot/merge_records.py OUT IN [IN ...] [--zero-ids]

Every input has a header {magic[8], u32 version, u32 record size}; inputs whose header differs from the first
readable one are skipped (another version is rewritten by the game anyway). A torn tail is cut off. Records keep the
order of the inputs, duplicates dropped. --zero-ids clears the input id of answer records (bytes 36..68): a seed record
with a zero id gets it from the translator that prepares it (gfx_shader.c), so recordings made under another
translator still match. OUT may be one of the inputs. Prints the record counts. Game-derived data: write it outside the
repository.
"""
import struct
import sys
from pathlib import Path

HEAD = struct.Struct('<8sII')
ID_LO, ID_HI = 36, 68   # answer record: stage u32, code hash[32], input id[32], answers, priority


def read(path):
    data = Path(path).read_bytes()
    if len(data) < HEAD.size:
        return None, []
    magic, version, size = HEAD.unpack_from(data)
    if not size:
        return None, []
    n = (len(data) - HEAD.size) // size
    return (magic, version, size), [data[HEAD.size + i * size:HEAD.size + (i + 1) * size] for i in range(n)]


def merge(out, inputs, zero_ids=False):
    head, seen, recs, counts = None, set(), [], []
    for p in inputs:
        if not Path(p).is_file():
            counts.append((p, 'missing'))
            continue
        h, rs = read(p)
        if h is None or (head and h != head):
            counts.append((p, 'skipped (other header)'))
            continue
        head = head or h
        if zero_ids and h[0] == b'TOTKSANS':
            rs = [r[:ID_LO] + bytes(ID_HI - ID_LO) + r[ID_HI:] for r in rs]
        new = 0
        for r in rs:
            if r not in seen:
                seen.add(r)
                recs.append(r)
                new += 1
        counts.append((p, f'{len(rs)} records, {new} new'))
    if head is None:
        sys.exit('merge_records: no readable input')
    Path(out).write_bytes(HEAD.pack(*head) + b''.join(recs))
    return len(recs), counts


def demo():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        h = HEAD.pack(b'TOTKSANS', 2, 70)
        rec = lambda b, iid: bytes([b]) * 36 + bytes([iid]) * 32 + bytes([b]) * 2
        Path(d, 'a').write_bytes(h + rec(1, 9) + rec(2, 9) + b'torn')
        Path(d, 'b').write_bytes(h + rec(2, 7) + rec(3, 7))
        Path(d, 'c').write_bytes(HEAD.pack(b'TOTKSANS', 1, 70) + rec(4, 0))
        n, counts = merge(Path(d, 'o'), [Path(d, 'a'), Path(d, 'b'), Path(d, 'c')], zero_ids=True)
        assert n == 3, n                                   # ids zeroed: rec 2 of a and b are the same record
        assert counts[2][1].startswith('skipped'), counts
        _, rs = read(Path(d, 'o'))
        assert [r[0] for r in rs] == [1, 2, 3] and all(r[ID_LO:ID_HI] == bytes(32) for r in rs)
    print('merge_records: ok')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--zero-ids']
    if args == ['--test']:
        demo()
    elif len(args) < 2:
        sys.exit(__doc__)
    else:
        n, counts = merge(args[0], args[1:], '--zero-ids' in sys.argv)
        for p, c in counts:
            print(f'merge_records: {p}: {c}')
        print(f'merge_records: {args[0]}: {n} records')
