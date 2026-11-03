#!/usr/bin/env python3
"""Validate and profile all ten 1024^3 implementations without changing dispatch."""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT/'build'
SMI = '/usr/lib/wsl/lib/nvidia-smi'
METRICS = ['gpu__time_duration.sum', 'dram__bytes_op_read.sum',
           'dram__cycles_elapsed.avg.per_second', 'sm__cycles_elapsed.avg.per_second']
EXPECTED = ['gemm_base_kernel', 'gemm_smem_kernel', 'gemm_smem_reg_kernel',
            'gemm_smem_reg_half2_kernel', 'gemm_wmma_smem_kernel', 'gemm_wmma_gmem_kernel',
            'gemm_wmma_pipeline_kernel<3>', 'gemm_wmma_swizzle_kernel<3>',
            'cuBLAS (gemm_cublas)', 'gemm_mma_llm_kernel<64, 128, 3, 32>']


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2)+'\n')


def command(argv, log, timeout=900):
    argv = [str(v) for v in argv]
    print('+ '+shlex.join(argv), flush=True)
    with Path(log).open('w') as output:
        proc = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f'{log}:\n'+Path(log).read_text()[-6000:])


def source_hashes():
    paths = [ROOT/'src/gemm.cu', *sorted((ROOT/'src/lib').glob('*.cuh')),
             ROOT/'benchmarks/square_benchmark.cu']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def idle_gpu():
    active = subprocess.check_output([SMI, '--query-compute-apps=pid,process_name',
                                      '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('GPU is in use; wait for these jobs to finish: '+active)


def ncu_path(override=None):
    path = override or os.environ.get('NCU') or shutil.which('ncu')
    if not path:
        candidates = sorted(Path('/opt/nvidia/nsight-compute').glob('*/ncu'),
                            key=lambda p: tuple(int(n) for n in re.findall(r'\d+', p.parent.name)))
        path = candidates[-1] if candidates else None
    if path is None or not Path(path).is_file():
        raise RuntimeError('Set NCU or --ncu to an installed Nsight Compute executable')
    return str(path)


def canonical_kernel(text):
    m = re.search(r'\b(gemm_\w+)(?:<([^>]*)>)?\s*\(', text)
    if not m:
        raise ValueError('Missing custom kernel name: '+text)
    if m[2] is None:
        return m[1]
    args = [re.sub(r'\((?:int|bool)\)', '', v).strip() for v in m[2].split(',')]
    return m[1]+'<'+', '.join(args)+'>'


def scale(unit, kind):
    units = {'time': {'ns': .001, 'nsecond': .001, 'us': 1., 'usecond': 1.,
                      'ms': 1000., 'msecond': 1000., 's': 1.e6, 'second': 1.e6},
             'bytes': {'byte': 1., 'Kbyte': 1000., 'Mbyte': 1.e6, 'Gbyte': 1.e9},
             'hz': {'hz': 1., 'Hz': 1., 'Khz': 1000., 'Mhz': 1.e6, 'Ghz': 1.e9,
                    'cycle/second': 1., 'cycle/nsecond': 1.e9}}
    if unit not in units[kind]:
        raise ValueError(f'Unsupported {kind} unit: {unit}')
    return units[kind][unit]


def calculate_results(folder):
    """Read raw profiler data; sum all kernels in each NVTX-tagged GEMM call."""
    folder = Path(folder)
    variants = json.loads((folder/'variants.json').read_text())
    assert [v['kernel'] for v in variants] == EXPECTED
    accuracy = json.loads((folder/'correctness.json').read_text())
    assert accuracy['passed'] and len(accuracy['cases']) == 10
    assert accuracy['rtol'] == accuracy['atol'] == .002 and accuracy['nrms_limit'] == .001
    assert all(r['pass'] and r['canaries_intact'] and r['inputs_unchanged'] for r in accuracy['cases'])
    lines = (folder/'square.raw.csv').read_text().splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    rows = list(csv.DictReader(lines[first:])); units, rows = rows[0], rows[1:]
    calls = collections.defaultdict(list)
    kernels = []
    for row in rows:
        label = re.search(r'square-v(\d\d)-t(\d\d)', row['Kernel Name'])
        if not label:
            label = re.search(r'square-v(\d\d)-t(\d\d)',
                              row.get('thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg', ''))
        assert label, ('Missing trial/variant identity', row['Kernel Name'])
        v, trial = map(int, label.groups()); assert 0 <= v < 10 and 0 <= trial < 9
        if v != 8:
            assert canonical_kernel(row['Kernel Name']) == EXPECTED[v], row['Kernel Name']
            grid = [int(x) for x in re.findall(r'\d+', row['Grid Size'])]
            block = [int(x) for x in re.findall(r'\d+', row['Block Size'])]
            assert grid == variants[v]['grid'] and block == variants[v]['block']
        def number(metric, kind):
            return float(row[metric].replace(',', ''))*scale(units[metric], kind)
        item = dict(variant=v, trial=trial, kernel=row['Kernel Name'],
                    time_us=number(METRICS[0], 'time'), dram_read_bytes=number(METRICS[1], 'bytes'),
                    dram_hz=number(METRICS[2], 'hz'), sm_hz=number(METRICS[3], 'hz'))
        assert all(math.isfinite(item[k]) and item[k] > 0 for k in ('time_us', 'sm_hz', 'dram_hz'))
        calls[v, trial].append(item); kernels.append(item)
    assert set(calls) == {(v, t) for v in range(10) for t in range(9)}, ('Incomplete profile', len(calls))
    samples = []
    for (v, trial), xs in sorted(calls.items()):
        samples.append(dict(variant=v, trial=trial, kernel=EXPECTED[v],
                            time_us=sum(x['time_us'] for x in xs), kernels=len(xs),
                            dram_read_bytes=sum(x['dram_read_bytes'] for x in xs)))
    result = []
    for v, variant in enumerate(variants):
        values = [r['time_us'] for r in samples if r['variant'] == v]
        assert len(values) == 9
        result.append(dict(**variant, executed_m=1024, samples=9,
                           median_us=statistics.median(values), min_us=min(values), max_us=max(values)))
    control = result[8]['median_us']
    for row in result:
        row['time_cublas'] = row['median_us']/control
    return result, samples, kernels


def load_results(folder):
    result, samples, kernels = calculate_results(folder)
    saved = json.loads((Path(folder)/'summary.json').read_text())
    assert result == saved, 'Square summary disagrees with raw profiler data'
    with (Path(folder)/'samples.csv').open() as f:
        values = list(csv.DictReader(f))
    assert len(values) == len(samples) == 90
    for a, b in zip(values, samples):
        assert int(a['variant']) == b['variant'] and int(a['trial']) == b['trial']
        assert math.isclose(float(a['time_us']), b['time_us'], rel_tol=1e-12)
    return result


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def snapshot_gpu():
    return subprocess.check_output([SMI, '--query-gpu=name,uuid,driver_version,memory.total,memory.used,'
                                    'temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu',
                                    '--format=csv'], text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ncu', help='Nsight Compute executable; also accepts the NCU environment variable')
    parser.add_argument('--build-only', action='store_true', help='Compile without running CUDA work')
    parser.add_argument('--validate-only', action='store_true', help='Run correctness checks without profiling or updating tables')
    args = parser.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    binary = BUILD/'square_benchmark'
    before_sources = source_hashes()
    compile_cmd = [os.environ['NVCC'], '-O3', '-std=c++17', '-lineinfo', '-arch='+os.environ['CUDA_ARCH'],
                   str(ROOT/'benchmarks/square_benchmark.cu'), '-lcublas', '-ldl', '-o', str(binary)]
    if os.environ.get('NVTX_INCLUDE'):
        compile_cmd[1:1] = ['-I'+os.environ['NVTX_INCLUDE']]
    command(compile_cmd, BUILD/'square_benchmark.build.log', 300)
    assert source_hashes() == before_sources, 'Source changed during compilation'
    variants = json.loads(subprocess.check_output([str(binary), '--list'], text=True))
    assert [v['kernel'] for v in variants] == EXPECTED
    if args.build_only:
        print('SQUARE BUILD PASS', flush=True); return
    profiler = ncu_path(args.ncu)
    idle_gpu()
    (BUILD/'runs').mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='square-', dir=BUILD/'runs'))
    print('RESULTS '+str(out), flush=True)
    dump(out/'variants.json', variants)
    metadata = dict(shape=[1024, 1024, 1024], alpha=1, beta=0, dtype='FP16 A/B/C; FP32 accumulation',
                    seed=1729+3*1024, input_generator='LCG 1664525*x+1013904223; top 24 bits mapped to [-0.5,0.5]',
                    warmups_per_variant=10, trials_per_variant=9, variant_order='rotate by trial index',
                    cuda_graph=False, eviction_bytes=256*1024*1024,
                    eviction='128-byte-line LCG permutation, all 32 lanes load ld.global.ca.u32, all words read once',
                    replay_mode='application', cache_control='none', clock_control='base',
                    timing='sum gpu__time_duration.sum for every kernel inside each complete-call NVTX range',
                    compiler_command=compile_cmd, source_sha256=before_sources,
                    ncu_version=subprocess.check_output([profiler, '--version'], text=True),
                    compiler_version=subprocess.check_output([os.environ['NVCC'], '--version'], text=True),
                    host=subprocess.check_output(['uname', '-a'], text=True),
                    linked_libraries=subprocess.check_output(['ldd', str(binary)], text=True),
                    gpu_before=snapshot_gpu(), status='started')
    dump(out/'metadata.json', metadata)
    doc_path, latest_path = ROOT/'docs/PERFORMANCE.md', ROOT/'results/reference/manifest.json'
    old_doc, old_latest = doc_path.read_bytes(), latest_path.read_bytes()
    import project
    assert project.performance_document(project.saved_tables()).encode() == old_doc, 'Document has edits not yet incorporated into its generator'
    stopped = threading.Event()
    def telemetry():
        with (out/'telemetry.csv').open('w') as f:
            f.write('utc,temperature_C,power_W,sm_MHz,memory_MHz,utilization_percent,memory_MiB\n')
            while not stopped.is_set():
                r = subprocess.run([SMI, '--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,'
                                    'utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                                   text=True, capture_output=True, timeout=15)
                f.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+','+r.stdout.strip()+'\n'); f.flush()
                stopped.wait(1)
    monitor = threading.Thread(target=telemetry, daemon=True); monitor.start()
    try:
        command([binary, '--validate', out/'correctness.json'], out/'correctness.log', 180)
        assert json.loads((out/'correctness.json').read_text())['passed']
        assert source_hashes() == before_sources, 'Source changed during validation'
        if args.validate_only:
            metadata['status'] = 'validated'; print('SQUARE CORRECTNESS PASS: 10 implementations', flush=True); return
        idle_gpu()
        profile_cmd = [profiler, '--nvtx', '--nvtx-include', 'gemm-profile/',
                       '--replay-mode', 'application', '--app-replay-mode', 'strict',
                       '--cache-control', 'none', '--clock-control', 'base',
                       '--metrics', ','.join(METRICS), '--export', str(out/'square'), str(binary), '--profile']
        metadata['profile_command'] = profile_cmd; dump(out/'metadata.json', metadata)
        print('SQUARE CORRECTNESS PASS: 10 implementations; profiling 90 logical calls', flush=True)
        command(profile_cmd, out/'profile.log', 900)
        assert source_hashes() == before_sources, 'Source changed during profiling; results not accepted'
        export_cmd = [profiler, '--import', str(out/'square.ncu-rep'), '--page', 'raw', '--csv',
                      '--print-units', 'base', '--print-nvtx-rename', 'kernel']
        command(export_cmd, out/'square.raw.csv', 120)
        result, samples, kernels = calculate_results(out)
        dump(out/'summary.json', result); write_csv(out/'samples.csv', samples); write_csv(out/'kernels.csv', kernels)
        load_results(out)
        assert doc_path.read_bytes() == old_doc and latest_path.read_bytes() == old_latest, 'Document/latest pointer changed; keep results without overwriting edits'
        new_latest = json.loads(old_latest); new_latest['progressive_square'] = str(out.relative_to(ROOT))
        rendered = project.performance_document(project.saved_tables(), latest=new_latest)
        assert source_hashes() == before_sources
        # Finalize files before publishing a self-contained reference dataset.
        stopped.set(); monitor.join(timeout=20)
        metadata['status'] = 'complete'; metadata['gpu_after'] = snapshot_gpu()
        dump(out/'metadata.json', metadata)
        project.publish_reference('progressive_square', out)
        doc_path.with_suffix('.md.square-tmp').write_text(rendered)
        os.replace(doc_path.with_suffix('.md.square-tmp'), doc_path)
        print('SQUARE BENCHMARK PASS: 10 implementations x 9 trials', flush=True)
        for row in result:
            print(f"{row['kernel']}: {row['median_us']:.3f} us; time/cuBLAS={row['time_cublas']:.3f}", flush=True)
    finally:
        stopped.set(); monitor.join(timeout=20)
        if 'gpu_after' not in metadata:metadata['gpu_after'] = snapshot_gpu()
        dump(out/'metadata.json', metadata)


if __name__ == '__main__':
    main()
