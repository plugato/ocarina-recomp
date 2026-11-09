"""Compile and run tests/ampere_lichii_ports.cpp against the vendored lowering headers; no game or GPU.

Usage: python tests/ampere_lichii_ports_test.py [--cxx clang++]
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default=os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++'))
    args = parser.parse_args()
    if not args.cxx:
        parser.error('pass --cxx with a C++20 compiler')
    with tempfile.TemporaryDirectory() as out:
        exe = Path(out) / 'ampere-lichii-ports.exe'
        command = [args.cxx, '-std=c++20', '-O1', '-I', str(ROOT / 'runtime/third_party/eglantine'),
                   str(ROOT / 'tests/ampere_lichii_ports.cpp'), '-o', str(exe)]
        if os.name == 'nt':
            command += ['-static-libstdc++', '-static-libgcc']
        if subprocess.run(command, timeout=300).returncode:
            return 1
        return subprocess.run([str(exe)], timeout=30).returncode


if __name__ == '__main__':
    raise SystemExit(main())
