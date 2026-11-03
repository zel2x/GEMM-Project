#!/usr/bin/env python3
"""Create and validate offline SVG charts from the report's visible tables."""
import argparse
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
FIGURES = {
    'final-warm-cold': 'Warm and cold time/cuBLAS ratios for the final kernels',
    'progressive-m8': 'GPU execution times for M,N,K = 8,6144,4096',
    'progressive-m32': 'GPU execution times for M,N,K = 32,6144,4096',
    'progressive-square': 'GPU execution times for M,N,K = 1024,1024,1024',
}
SHAPES = {(8, 6144, 4096): 'progressive-m8',
          (32, 6144, 4096): 'progressive-m32',
          (1024, 1024, 1024): 'progressive-square'}


def chart_block(key):
    return f'\n![{FIGURES[key]}](figures/{key}.svg)\n'


def strip_charts(document):
    for key in FIGURES:
        document = document.replace(chart_block(key), '')
    return document


def report_tables(document):
    lines = document.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1]+len(line))
    result, heading, fenced, i = [], '', False, 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith('```'):
            fenced = not fenced
        if not fenced and line.startswith('#'):
            heading = line.strip()
        if (not fenced and line.startswith('|') and i+1 < len(lines)
                and re.fullmatch(r'\|[-:| ]+\|\s*', lines[i+1])):
            start = i
            def cells(value):
                return [cell.strip() for cell in value.strip().strip('|').split('|')]
            columns = cells(line)
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                row = cells(lines[i]); assert len(row) == len(columns)
                rows.append(row); i += 1
            if columns[0] == 'M,N,K':
                key, shape, kind = 'final-warm-cold', None, 'final'
            else:
                assert columns[:2] == ['Kernel', 'Executed M'], columns
                match = re.search(r'M,N,K\s*=\s*(\d+),(\d+),(\d+)', heading)
                assert match, heading
                shape = tuple(int(v) for v in match.groups())
                key, kind = SHAPES[shape], 'progressive'
            result.append(dict(id=key, kind=kind, heading=heading, shape=shape,
                               columns=columns, rows=rows, start=offsets[start], end=offsets[i],
                               header=''.join(lines[start:start+2]), raw=''.join(lines[start:i])))
        else:
            i += 1
    assert [t['id'] for t in result] == list(FIGURES), 'Expected the four report tables'
    return result


def add_chart_links(document):
    document = strip_charts(document)
    for table in reversed(report_tables(document)):
        pos = table['end']
        document = document[:pos]+chart_block(table['id'])+document[pos:]
    return document


def kernel_key(cell):
    return cell[1:-1] if cell.startswith('`') and cell.endswith('`') else cell


def render_report(template, specs, measurements):
    """Fill only the chosen tables, retaining their visible rows and order."""
    final = {tuple(row['shape']): row for row in measurements['final']}
    groups = {tuple(group['shape']): group for group in measurements['progressive']}
    for spec in specs:
        text = spec['header']
        if spec['kind'] == 'final':
            for shape in spec['keys']:
                row = final[tuple(shape)]
                text += (f"| {','.join(str(v) for v in shape)} | `{row['kernel']}` | "
                         f"{row['warm_us']:.3f} | {row['warm_cublas_us']:.3f} | "
                         f"{row['warm_us']/row['warm_cublas_us']:.3f} | {row['cold_us']:.3f} | "
                         f"{row['cold_cublas_us']:.3f} | {row['cold_us']/row['cold_cublas_us']:.3f} |\n")
        else:
            group = groups[tuple(spec['shape'])]
            by_key = {(row['kernel'], row['executed_m']): row for row in group['rows']}
            for kernel, executed_m in spec['keys']:
                row = by_key[kernel, executed_m]
                name = kernel if kernel.startswith('cuBLAS') else f'`{kernel}`'
                m = str(executed_m)+(' (padded)' if executed_m != spec['shape'][0] else '')
                def dims(value):
                    return value if isinstance(value, str) else '('+','.join(str(v) for v in value)+')'
                text += (f"| {name} | {m} | {dims(row['grid'])} | {dims(row['block'])} | "
                         f"{row['gpu_us']:.3f} | {row['gpu_us']/group['cublas_us']:.3f} |\n")
        token = '@@PERFORMANCE_TABLE:'+spec['id']+'@@\n'
        assert template.count(token) == 1
        template = template.replace(token, text)
    return add_chart_links(template)


def refresh_report(document, measurements):
    """Refresh values while keeping the document's current prose and row choices."""
    template = strip_charts(document)
    tables = report_tables(template)
    specs = []
    for table in tables:
        spec = {key: table[key] for key in ('id', 'kind', 'shape', 'header')}
        if table['kind'] == 'final':
            spec['keys'] = [list(map(int, row[0].split(','))) for row in table['rows']]
        else:
            spec['keys'] = [[kernel_key(row[0]), int(row[1].split()[0])] for row in table['rows']]
        specs.append(spec)
    for table in reversed(tables):
        token = '@@PERFORMANCE_TABLE:'+table['id']+'@@\n'
        template = template[:table['start']]+token+template[table['end']:]
    return render_report(template, specs, measurements)


def payload(table):
    return {key: table[key] for key in ('id', 'kind', 'heading', 'columns', 'rows')}


def _matplotlib():
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'build/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'gemm-performance-plots',
                         'axes.unicode_minus': False, 'figure.facecolor': 'white',
                         'savefig.facecolor': 'white', 'text.color': '#202936',
                         'axes.labelcolor': '#202936', 'xtick.color': '#425066',
                         'ytick.color': '#202936'})
    return plt


def _finish(fig, table, preview=None):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    # Catch clipped titles, tick labels and annotations before saving.
    texts = list(fig.texts)
    for ax in fig.axes:
        texts.extend([*ax.texts, ax.title, ax.xaxis.label, ax.yaxis.label])
        # LogLocator also allocates ticks outside the visible limits; those are
        # not drawn and must not be mistaken for clipped content.
        for axis, limits, locations in [(ax.xaxis, ax.get_xlim(), ax.get_xticks()),
                                       (ax.yaxis, ax.get_ylim(), ax.get_yticks())]:
            low, high = sorted(limits)
            for location, tick in zip(locations, axis.get_major_ticks()):
                if low <= location <= high:
                    texts.extend([tick.label1, tick.label2])
        legend = ax.get_legend()
        if legend is not None:
            texts.extend(legend.get_texts())
    for text in texts:
        if text.get_visible() and text.get_text().strip():
            box = text.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= bounds.x1+1 and box.y1 <= bounds.y1+1, (
                table['id'], 'Text outside figure', text.get_text(), tuple(box.bounds))
    if preview is not None:
        Path(preview).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=120)
    output = io.StringIO()
    fig.savefig(output, format='svg', metadata={'Date': None, 'Creator': 'GEMM performance plots'})
    svg = output.getvalue()
    svg = re.sub(r'<!DOCTYPE svg.*?>', '', svg, flags=re.S)
    start = svg.index('<svg ')
    svg = svg[:start]+svg[start:].replace('<svg ', f'<svg role="img" aria-labelledby="{table["id"]}-title {table["id"]}-desc" ', 1)
    end = svg.index('>', start)+1
    description = ('Warm and cold time ratios use their own cuBLAS controls; lower is better.'
                   if table['kind'] == 'final' else
                   'GPU execution times on a logarithmic axis. The dashed line is cuBLAS at requested M. Square markers identify padded M.')
    content = (f'\n<title id="{table["id"]}-title">{html.escape(FIGURES[table["id"]])}</title>'
               f'\n<desc id="{table["id"]}-desc">{html.escape(description)}</desc>'
               '\n<metadata id="gemm-plot-data">'+html.escape(json.dumps(payload(table), ensure_ascii=False, sort_keys=True))+'</metadata>\n')
    return svg[:end]+content+svg[end:]


def render_chart(table, preview=None):
    plt = _matplotlib()
    if table['kind'] == 'final':
        rows = table['rows']; n = len(rows)
        fig = plt.figure(figsize=(11, max(5.6, 1.0*n+0.9)))
        ax = fig.add_axes([.355, .19, .57, .64])
        labels = [r[0]+'\n'+r[1].replace('`', '') for r in rows]
        warm, cold = [float(r[4]) for r in rows], [float(r[7]) for r in rows]
        for i, (w, c) in enumerate(zip(warm, cold)):
            ax.barh(i-.18, w, height=.28, color='#ba8125', edgecolor='none', label='Warm' if i == 0 else None, zorder=3)
            ax.barh(i+.18, c, height=.28, color='#3268b2', edgecolor='none', label='Cold' if i == 0 else None, zorder=3)
            for y, v in [(i-.18, w), (i+.18, c)]:
                ax.text(v+.013, y, f'{v:.3f}×', va='center', fontsize=10.5,
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.2})
        ax.set_yticks(range(n), labels, fontfamily='DejaVu Sans Mono', fontsize=9.8)
        ax.set_ylim(n-.45, -.60)
        ax.set_xlim(0, max(warm+cold)*1.27)
        ax.set_xticks([0, .25, .5, .75, 1., 1.25])
        ax.axvline(1., color='#5d6876', linestyle=(0, (4, 3)), linewidth=1.25, zorder=2)
        ax.grid(axis='x', color='#e6e9ee', linewidth=.7, zorder=0)
        ax.set_xlabel('Time / cuBLAS — lower is better', labelpad=11)
        ax.tick_params(axis='y', length=0, pad=10)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.035), frameon=False, ncol=2, borderaxespad=0)
        fig.text(.035, .965, 'Final kernels versus cuBLAS', fontsize=15, weight='normal', va='top')
        fig.text(.035, .915, 'Dashed reference = cuBLAS (1.0×). Table row order is preserved.', fontsize=10.5, va='top')
        fig.text(.035, .035, 'Warm and cold use different timing methods; each ratio has its own cuBLAS control.', fontsize=10.)
    else:
        from matplotlib import ticker, transforms
        rows = table['rows']; n = len(rows)
        fig = plt.figure(figsize=(11, max(6.2, .47*n+1.75)))
        ax = fig.add_axes([.35, .145, .405, .70])
        values = [float(r[4]) for r in rows]
        control = next(float(r[4]) for r in rows if r[0].startswith('cuBLAS') and 'padded' not in r[1])
        labels = []
        for i, row in enumerate(rows):
            padded = 'padded' in row[1]
            name = row[0].replace('`', '')
            labels.append(name+('\nM='+row[1] if padded else ''))
            color = '#566271' if name.startswith('cuBLAS') else ('#168573' if name.startswith('gemm_mma_llm_kernel') else '#3268b2')
            ax.scatter(values[i], i, s=54, marker='s' if padded else 'o', color=color, zorder=4)
            ax.axhline(i, color='#edf0f4', linewidth=.65, zorder=0)
            ax.text(1.045, i, f'{values[i]:.3f} µs  ·  {float(row[5]):.3f}×',
                    transform=transforms.blended_transform_factory(ax.transAxes, ax.transData),
                    fontsize=10.2, va='center', clip_on=False)
        ax.set_xscale('log')
        ax.set_xlim(min(values)*.73, max(values)*1.28)
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, pos: f'{value:,.0f}'))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.grid(axis='x', color='#e6e9ee', linewidth=.7, zorder=0)
        ax.axvline(control, color='#5d6876', linestyle=(0, (4, 3)), linewidth=1.25, zorder=2)
        ax.set_yticks(range(n), labels, fontfamily='DejaVu Sans Mono', fontsize=9.8)
        ax.set_ylim(n-.45, -.55)
        ax.tick_params(axis='y', length=0, pad=10)
        ax.set_xlabel('GPU time (µs, log scale) — lower is better', labelpad=11, fontsize=10.2)
        fig.text(.035, .965, table['heading'].lstrip('# '), fontsize=15, weight='normal', va='top')
        subtitle = 'Dashed line: cuBLAS at requested M.'
        subtitle += ' Square markers: padded M.' if any('padded' in r[1] for r in rows) else ' All rows use the requested M.'
        fig.text(.035, .917, subtitle, fontsize=10.5, va='top')
        ax.text(1.045, 1.025, 'GPU time  ·  time/cuBLAS', transform=ax.transAxes, fontsize=10.2, color='#425066')
        fig.text(.035, .035, 'Kernel order matches the table; spacing on the time axis is logarithmic.', fontsize=10.)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#c4ccd5')
    try:
        return _finish(fig, table, preview)
    finally:
        plt.close(fig)


def write_plots(document, output_dir=None, preview_dir=None):
    folder = Path(output_dir) if output_dir is not None else ROOT/'docs/figures'
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for table in report_tables(document):
        preview = Path(preview_dir)/(table['id']+'.png') if preview_dir else None
        path = folder/(table['id']+'.svg')
        path.write_text(render_chart(table, preview))
        files.append(path)
    return files


def check_plots(document, folder=None):
    assert add_chart_links(document) == document, 'Missing, duplicated or misplaced report charts'
    folder = Path(folder) if folder is not None else ROOT/'docs/figures'
    for table in report_tables(document):
        path = folder/(table['id']+'.svg')
        content = path.read_text()
        root = ET.fromstring(content)
        meta = next(e for e in root if e.tag.endswith('metadata') and e.attrib.get('id') == 'gemm-plot-data')
        assert json.loads(meta.text) == payload(table), ('Chart/table mismatch', path)
        assert content == render_chart(table), ('Stale or changed figure', path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check links, values and generated SVG content')
    parser.add_argument('--output-dir', type=Path, help='Stage figures elsewhere without editing the document')
    parser.add_argument('--preview-dir', type=Path, help='Write temporary PNG previews for visual inspection')
    args = parser.parse_args()
    path = ROOT/'docs/PERFORMANCE.md'; original = path.read_text()
    if args.check:
        check_plots(original, args.output_dir)
        print('PLOTS CHECK PASS: all four charts match their tables')
    else:
        document = add_chart_links(original)
        write_plots(document, args.output_dir, args.preview_dir)
        if args.output_dir is None:
            assert path.read_text() == original, 'Document changed during chart generation'
            path.write_text(document)
        print('PLOTS GENERATED: four SVG charts')


if __name__ == '__main__':
    main()
