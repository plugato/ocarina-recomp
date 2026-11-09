"""Builds Ocarina-Setup-<version>-windows.zip: everything a clean PC needs to build the game, with nothing to install.

    .venv\\Scripts\\python tools\\oot\\make_setup.py --out build\\setup [--no-zip]

Needs on this PC: build\\xlate (tools/build_xlate.py), the MSYS2 UCRT64 clang and windres for the two small helper programs,
git and network access on the first run. Downloads are cached in toolchain\\release-dl and checked against the sha256 in PINS.

The folder holds, with nothing derived from the game (tools/check_content.py runs over it):

    Instalar-Ocarina.exe   instalador/launcher.c, which opens the script below
    instalador/            Instalar-Ocarina.ps1, the installer
    projeto/               this repository at HEAD (git archive), without the translator's sources and the tests
    ferramentas/python/    python.org embeddable 3.14, numpy and lz4 (recomp/gen.py), zstandard (tools/oot/shader_extract.py)
    ferramentas/llvm/      llvm-mingw, x86_64 only: what clang and lld need to compile and link the game
    ferramentas/deps/      headers, libraries and DLLs of Vulkan, VMA, SDL3, Opus, libvpx, dav1d and zstd
    ferramentas/bin/       ninja.exe, unzbic.exe (kinnay/zbic, GPL-2.0), totk_xlate.exe and totk_xlate.id
"""
import argparse
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import zstandard

ROOT = Path(__file__).resolve().parent.parent.parent
DL = ROOT / 'toolchain' / 'release-dl'
WORK = ROOT / 'toolchain' / 'release-work'
MSYS2 = 'https://repo.msys2.org/mingw/ucrt64/'
SDL = '3.4.18'
LLVM = '20261020'
ZBIC = '11b08f2712264bbed731545085cbd9702096ceb7'
PINS = {
    'python-embed': ('https://www.python.org/ftp/python/3.14.9/python-3.14.9-embed-amd64.zip', '3c01d7cfc666f746a6f064f60fb2156ab0ec464d3d738ab796371a46a9bb24a2'),
    'llvm-mingw': (f'https://github.com/mstorsjo/llvm-mingw/releases/download/{LLVM}/llvm-mingw-{LLVM}-ucrt-x86_64.zip', 'fd5462b7f5d93b2ff981fcd3f9c7e427e47f3203c64ea91c27bcc444eebd74cd'),
    'sdl3-dll': (f'https://github.com/libsdl-org/SDL/releases/download/release-{SDL}/SDL3-{SDL}-win32-x64.zip', '75c2c0fc74e7d1206aaedc22893a35887332f4f85f5c8b74fddc10d177985f2c'),
    'sdl3-devel': (f'https://github.com/libsdl-org/SDL/releases/download/release-{SDL}/SDL3-devel-{SDL}-mingw.tar.gz', '049b9dd711e3fb3f8d84feb998dfec54a75e17cf539c4a80ff4d058d839771cf'),
    'ninja': ('https://github.com/ninja-build/ninja/releases/download/v1.13.2/ninja-win.zip', '07fc8261b42b20e71d1720b39068c2e14ffcee6396b76fb7a795fb460b78dc65'),
    'opus': (MSYS2 + 'mingw-w64-ucrt-x86_64-opus-1.6.1-2-any.pkg.tar.zst', 'bb114bcf26225f50f874497ebfd6b8224fff2a20bda3f8ee45cd07ce5f02657b'),
    'libvpx': (MSYS2 + 'mingw-w64-ucrt-x86_64-libvpx-1.17.0-1-any.pkg.tar.zst', '8ff6919dbb88ea26ce28d22fd81d6f0d6e200cf56a4dd81ffc2d33124fe98024'),
    'vulkan-headers': (MSYS2 + 'mingw-w64-ucrt-x86_64-vulkan-headers-1~1.4.363.0-1-any.pkg.tar.zst', 'be602b2657cace531316e78b1eda4087e8f7f84ebb877f08fba198a2db06406a'),
    'vulkan-loader': (MSYS2 + 'mingw-w64-ucrt-x86_64-vulkan-loader-1~1.4.363.0-1-any.pkg.tar.zst', '065076fe13d39ab97eea4740596426d5ae741b5044e48e1c9495d04835c92dbe'),
    'vma': (MSYS2 + 'mingw-w64-ucrt-x86_64-vulkan-memory-allocator-3.4.0-1-any.pkg.tar.zst', '44c81162337915afecda3185f66fd98268a54454580b2717080c9dfef57fe781'),
    'zstd': (MSYS2 + 'mingw-w64-ucrt-x86_64-zstd-1.5.7-2-any.pkg.tar.zst', 'dbdb8427280046a2b41697780aa4c52983b708082b0da4755951dc3bea96ca89'),
    'dav1d': (MSYS2 + 'mingw-w64-ucrt-x86_64-dav1d-1.5.4-1-any.pkg.tar.zst', 'ccf26c6f29fd36f5692d57fd5e0fe3becec492e0b875fd5e04f47a9582eb8b37'),
}
SDL3_DLL_SHA256 = '2b4b128d8d510cc2d442693cf848c050d700d43dc667afe9e5fd82b0570d667e'
LLVM_BIN = {'clang.exe', 'clang++.exe', 'clang-23.exe', 'ld.lld.exe', 'llvm-ar.exe', 'llvm-ranlib.exe', 'llvm-nm.exe', 'llvm-objdump.exe',
            'llvm-objcopy.exe', 'llvm-strip.exe', 'llvm-readobj.exe', 'llvm-dlltool.exe', 'llvm-rc.exe', 'llvm-cvtres.exe', 'llvm-windres.exe',
            'llvm-wrapper.exe', 'mingw32-common.cfg', 'x86_64-w64-windows-gnu.cfg', 'x86_64-pc-windows-gnu.cfg', 'x86_64-w64-mingw32-clang.exe',
            'x86_64-w64-mingw32-clang++.exe', 'libLLVM-23.dll', 'libclang-cpp.dll', 'libc++.dll', 'libunwind.dll', 'libwinpthread-1.dll', 'libffi-8.dll'}
LLVM_SKIP_DIRS = {'aarch64-w64-mingw32', 'armv7-w64-mingw32', 'i686-w64-mingw32', 'busybox', 'python', 'share'}


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def fetch(name):
    url, want = PINS[name]
    DL.mkdir(parents=True, exist_ok=True)
    path = DL / Path(url).name.replace('%7E', '~')
    if not path.is_file():
        print('download', url, flush=True)
        part = path.with_name(path.name + '.part')
        with urllib.request.urlopen(url) as r, open(part, 'wb') as f:
            shutil.copyfileobj(r, f)
        part.replace(path)
    got = sha256(path)
    if want is None:
        print(f'{name}: sha256 {got} (no pin yet: write it in PINS)')
    elif got != want:
        path.unlink()
        sys.exit(f'{path}: sha256 {got}, pinned {want}: deleted')
    return path


def unpack(name):
    dest = WORK / name
    stamp = dest / '.unpacked'
    if stamp.is_file():
        return dest
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    path = fetch(name)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
    elif path.suffix == '.zst':
        with open(path, 'rb') as f:
            data = zstandard.ZstdDecompressor().stream_reader(f).read()
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            t.extractall(dest, filter='data')
    else:
        with tarfile.open(path) as t:
            t.extractall(dest, filter='data')
    stamp.write_text('')
    return dest


def copy_tree(src, dst, skip=lambda rel: False):
    for p in sorted(src.rglob('*')):
        rel = p.relative_to(src)
        if p.is_file() and not skip(rel.as_posix()):
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / rel)


def run(cmd, **kw):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def build_python(tools):
    py = tools / 'python'
    with zipfile.ZipFile(fetch('python-embed')) as z:
        z.extractall(py)
    # the embeddable Python ignores the script's folder once a ._pth exists: ferramentas\python, so the project is two levels up
    (py / 'python314._pth').write_text('python314.zip\n.\nLib/site-packages\n../../tools\n../../recomp\n', encoding='utf-8')
    run([sys.executable, '-m', 'pip', 'install', '-q', '--disable-pip-version-check', '--target', py / 'Lib' / 'site-packages', '--python-version', '3.14',
         '--implementation', 'cp', '--abi', 'cp314', '--platform', 'win_amd64', '--only-binary', ':all:', 'numpy==2.5.4', 'lz4==4.4.5', 'zstandard==0.25.0'])
    shutil.rmtree(py / 'Lib' / 'site-packages' / 'bin', ignore_errors=True)
    # numpy's tests and pip's compiled files are not used, and their long paths can break unpacking the ZIP in a long folder
    for d in list((py / 'Lib' / 'site-packages').rglob('tests')) + list((py / 'Lib' / 'site-packages').rglob('__pycache__')):
        shutil.rmtree(d, ignore_errors=True)


def build_llvm(tools):
    src = unpack('llvm-mingw') / f'llvm-mingw-{LLVM}-ucrt-x86_64'

    def skip(rel):
        top = rel.split('/')[0]
        if top in LLVM_SKIP_DIRS:
            return True
        if top == 'bin':
            return rel[4:] not in LLVM_BIN
        if top == 'lib':
            return not (rel.startswith('lib/clang/') and '/include/' in rel or rel.endswith('/lib/windows/libclang_rt.builtins-x86_64.a'))
        return Path(rel).suffix.lower() in {'.idl', '.tlb'} or rel.startswith('include/ddk/')

    copy_tree(src, tools / 'llvm', skip)


def build_deps(tools):
    d = tools / 'deps'
    for sub in ('include', 'lib', 'bin'):
        (d / sub).mkdir(parents=True, exist_ok=True)
    sdl = unpack('sdl3-devel') / f'SDL3-{SDL}' / 'x86_64-w64-mingw32'
    copy_tree(sdl / 'include', d / 'include')
    shutil.copy2(sdl / 'lib' / 'libSDL3.dll.a', d / 'lib')
    for pkg, dirs, libs in (('opus', ['opus'], ['libopus.dll.a']), ('libvpx', ['vpx'], ['libvpx.a']), ('dav1d', ['dav1d'], ['libdav1d.a']),
                            ('vulkan-headers', ['vulkan', 'vk_video'], []), ('vulkan-loader', [], ['libvulkan-1.dll.a']), ('vma', ['vma'], [])):
        root = unpack(pkg) / 'ucrt64'
        for inc in dirs:
            copy_tree(root / 'include' / inc, d / 'include' / inc)
        for lib in libs:
            shutil.copy2(root / 'lib' / lib, d / 'lib')
    zstd = unpack('zstd') / 'ucrt64'
    for name in ('zstd.h', 'zstd_errors.h'):
        shutil.copy2(zstd / 'include' / name, d / 'include')
    shutil.copy2(zstd / 'lib' / 'libzstd.a', d / 'lib')
    dll = unpack('sdl3-dll') / 'SDL3.dll'
    if sha256(dll) != SDL3_DLL_SHA256:
        sys.exit(f'{dll}: not the pinned SDL3.dll')
    shutil.copy2(dll, d / 'bin' / 'SDL3.dll')
    shutil.copy2(unpack('opus') / 'ucrt64' / 'bin' / 'libopus-0.dll', d / 'bin' / 'libopus-0.dll')


def build_bin(tools, clang):
    b = tools / 'bin'
    b.mkdir(parents=True, exist_ok=True)
    shutil.copy2(unpack('ninja') / 'ninja.exe', b / 'ninja.exe')
    for f in ('totk_xlate.exe', 'totk_xlate.id'):
        shutil.copy2(ROOT / 'build' / 'xlate' / f, b / f)
    zb = WORK / 'zbic'
    if not (zb / '.git').is_dir():
        run(['git', 'clone', 'https://github.com/kinnay/zbic', zb])
    run(['git', '-C', zb, 'checkout', '-q', ZBIC])
    run([clang, '-O2', '-w', '-static', f'-I{zb / "src"}', ROOT / 'tools' / 'oot' / 'unzbic.c', '-o', b / 'unzbic.exe'])
    (tools / 'licenses').mkdir(exist_ok=True)
    shutil.copy2(zb / 'LICENSE', tools / 'licenses' / 'zbic-LICENSE.txt')
    (tools / 'licenses' / 'LEIA-ME.txt').write_text(f'unzbic.exe: tools/oot/unzbic.c with kinnay/zbic at {ZBIC}, source at https://github.com/kinnay/zbic\n'
                                                    'totk_xlate.exe: tools/vendor/ryujinx with tools/xlate/patches, source in this project\'s repository\n'
                                                    'The other tools come from their official releases. See projeto/THIRD_PARTY_NOTICES.md.\n', encoding='utf-8')


def build_project(out):
    dest = out / 'projeto'
    dest.mkdir(parents=True)
    tar = out / 'projeto.tar'
    run(['git', '-C', ROOT, 'archive', '--format=tar', '-o', tar, 'HEAD', '--', '.', ':!tests', ':!tools/vendor/ryujinx'])
    with tarfile.open(tar) as t:
        t.extractall(dest, filter='data')
    tar.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, default=ROOT / 'build' / 'setup')
    ap.add_argument('--no-zip', action='store_true')
    ap.add_argument('--seed', type=Path, help='folder with seed-answers.bin and pipeline-keys.bin recorded in play: the shader '
                    'preparation seed, copied into projeto/tools/data/oot-seed (out of the repository, as the TotK shader list)')
    args = ap.parse_args()
    version = (ROOT / 'VERSION').read_text().strip()
    folder = args.out / f'Ocarina-Setup-{version}'
    shutil.rmtree(folder, ignore_errors=True)
    tools = folder / 'ferramentas'
    tools.mkdir(parents=True)
    msys = Path(r'C:\msys64\ucrt64\bin')
    build_python(tools)
    build_llvm(tools)
    build_deps(tools)
    build_bin(tools, msys / 'clang.exe')
    build_project(folder)
    if args.seed:
        # Our recorded answers and pipeline keys: shader hashes, input ids and draw state, no game code. The installer's
        # P-Shaders step prepares the player's own shaders from them.
        seed = folder / 'projeto' / 'tools' / 'data' / 'oot-seed'
        seed.mkdir(parents=True)
        for f in ('seed-answers.bin', 'pipeline-keys.bin'):
            shutil.copy2(args.seed / f, seed / f)
        if (args.seed / 'big-compute.txt').is_file():   # the big compute shaders play met: P-BigShaders builds them
            shutil.copy2(args.seed / 'big-compute.txt', seed / 'big-compute.txt')
        print('shader seed:', ', '.join(f'{p.name} {p.stat().st_size // 1024} KiB' for p in seed.iterdir()))
    env = dict(os.environ, PATH=f'{msys};{os.environ["PATH"]}')
    run(['sh', ROOT / 'instalador' / 'build_exe.sh'], env=env)
    shutil.copy2(ROOT / 'instalador' / 'Instalar-Ocarina.exe', folder / 'Instalar-Ocarina.exe')
    (folder / 'instalador').mkdir()
    shutil.copy2(ROOT / 'instalador' / 'Instalar-Ocarina.ps1', folder / 'instalador' / 'Instalar-Ocarina.ps1')
    longest = max((p.relative_to(folder).as_posix() for p in folder.rglob('*')), key=len)
    print(f'longest path inside the folder: {len(longest)} characters: {longest}')
    size = sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
    print(f'{folder}: {size / (1 << 20):.0f} MiB')
    if not args.no_zip:
        z = args.out / f'Ocarina-Setup-{version}-windows.zip'
        print('zip', z, flush=True)
        with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as f:
            for p in sorted(folder.rglob('*')):
                if p.is_file():
                    f.write(p, p.relative_to(args.out))
        print(f'{z}: {z.stat().st_size / (1 << 20):.0f} MiB, sha256 {sha256(z)}')


if __name__ == '__main__':
    main()
