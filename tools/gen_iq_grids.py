#!/usr/bin/env python3
"""Writes src/kernels/iq_grids_half.h: the i-quant grids of the GEMV kernels' ggml-common.h as half4 pairs in
constant memory, for mako's IQ2_XXS, IQ3_XXS and IQ2_S row kernels.
  gen_iq_grids.py <ggml-common.h> <out.h>
"""
import re
import sys


def table(src, name, n):
    m = re.search(r'GGML_TABLE_BEGIN\((uint64_t|uint32_t), ' + name + r', ' + str(n) + r'\)(.*?)GGML_TABLE_END\(\)',
                  src, re.S)
    if not m:
        raise SystemExit(f'{name} not found')
    vals = [int(v, 16) for v in re.findall(r'0x[0-9a-fA-F]+', m.group(2))]
    if len(vals) != n:
        raise SystemExit(f'{name}: {len(vals)} entries, expected {n}')
    return vals


def main():
    src = open(sys.argv[1]).read()
    out = ['// The i-quant grids as half4 pairs in constant memory, generated from the GEMV kernels\' ggml-common.h tables by',
           '// tools/gen_iq_grids.py: entry i of a 64-bit grid is name[2 * i] (bytes 0-3) and name[2 * i + 1] (bytes 4-7), of a',
           '// 32-bit grid name[i].', '#pragma once', '']
    for name, n, width in (('iq2s_grid', 1024, 8), ('iq2xxs_grid', 256, 8), ('iq3xxs_grid', 256, 4)):
        entries = []
        for v in table(src, name, n):
            bs = [(v >> (8 * b)) & 255 for b in range(width)]
            for q in range(0, width, 4):
                entries.append('{%d, %d, %d, %d}' % tuple(bs[q:q + 4]))
        out.append('constant half4 %s_h[%d] = {' % (name, len(entries)))
        for i in range(0, len(entries), 8):
            out.append('    ' + ', '.join(entries[i:i + 8]) + ',')
        out.append('};')
        out.append('')
    open(sys.argv[2], 'w').write('\n'.join(out))


if __name__ == '__main__':
    main()
