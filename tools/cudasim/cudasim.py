#!/usr/bin/env python3
"""Translate a .cu file into plain C++ that runs on the CPU with cuda_sim.h.

usage: cudasim.py input.cu output.cpp

Rewrites
  kernel<<<grid, block[, smem[, stream]]>>>(args);
into
  __sim_launch("kernel", dim3(grid), dim3(block), smem, [&]{ kernel(args); });
and
  extern __shared__ T name[];   ->   T *name = (T *)__sim_dyn_smem;
Everything else (the cuda_sim.h header, -I paths) is handled by build flags.
"""
import re
import sys

CUDA_HEADERS = re.compile(r'^\s*#\s*include\s*[<"](cuda_runtime\.h|cuda\.h|cuda_fp16\.h|cuda_runtime_api\.h|device_launch_parameters\.h)[>"].*$', re.M)
LAUNCH = re.compile(r'([A-Za-z_][\w:]*(?:\s*<[^<>;]*>)?)\s*<<<')
EXTERN_SHARED = re.compile(r'extern\s+__shared__\s+(?:__align__\(\d+\)\s+)?([\w: ]+?)\s+(\w+)\s*\[\s*\]\s*;')


def split_top(s, sep=','):
    """Split on `sep` at nesting depth 0."""
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ''
        else:
            cur += ch
    out.append(cur)
    return [x.strip() for x in out]


def match_paren(s, i):
    """s[i] == '(' ; return index of the matching ')'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == '(':
            depth += 1
        elif s[j] == ')':
            depth -= 1
            if depth == 0:
                return j
    raise ValueError('unbalanced parentheses in kernel launch')


def translate(src):
    src = CUDA_HEADERS.sub('', src)
    src = EXTERN_SHARED.sub(lambda m: f'{m.group(1)} *{m.group(2)} = ({m.group(1)} *)__sim_dyn_smem;', src)
    out, pos = [], 0
    while True:
        m = LAUNCH.search(src, pos)
        if not m:
            out.append(src[pos:])
            break
        name = m.group(1)
        cfg_start = m.end()
        cfg_end = src.index('>>>', cfg_start)
        cfg = split_top(src[cfg_start:cfg_end])
        k = cfg_end + 3
        while src[k].isspace():
            k += 1
        assert src[k] == '(', f'expected ( after launch of {name}'
        close = match_paren(src, k)
        args = src[k + 1:close]
        grid, block = cfg[0], cfg[1]
        smem = cfg[2] if len(cfg) > 2 else '0'
        out.append(src[pos:m.start()])
        out.append(f'__sim_launch("{name}", dim3({grid}), dim3({block}), (size_t)({smem}), [&]{{ {name}({args}); }})')
        pos = close + 1
    return ''.join(out)


if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        src = f.read()
    with open(sys.argv[2], 'w') as f:
        f.write(f'#line 1 "{sys.argv[1]}"\n' + translate(src))
