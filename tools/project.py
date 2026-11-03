#!/usr/bin/env python3
"""Build, validate and measure the current GEMM implementation."""
import argparse, collections, csv, ctypes
import json, math, os, re, shutil, statistics, subprocess, sys, tempfile, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SAVED=ROOT/'results/reference'
BUILD=ROOT/'build'

def run(argv, log=None, cwd=None):
    argv=[str(x) for x in argv]
    print('+', ' '.join(argv),flush=True)
    if log:
        Path(log).parent.mkdir(parents=True,exist_ok=True)
        if Path(log).exists():
            i=1
            while Path(str(log)+f'.previous-{i}').exists():i+=1
            shutil.copy2(log,Path(str(log)+f'.previous-{i}'))
        with open(log,'w') as f: p=subprocess.run(argv,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        if p.returncode:raise RuntimeError(f'{Path(log)}: '+Path(log).read_text()[-7000:])
    else:subprocess.run(argv,cwd=cwd,check=True,stdin=subprocess.DEVNULL)

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n')

def reference_manifest(verify=False):
    """Read the stable index; optional verification checks every retained artifact."""
    index = json.loads((SAVED/'manifest.json').read_text())
    assert index['schema_version'] == 1
    for key, folder in [('current', 'warm'), ('progressive_square', 'square')]:
        assert (ROOT/index[key]).resolve() == (SAVED/folder).resolve()
    if verify:
        actual = {str(p.relative_to(SAVED)) for p in SAVED.rglob('*') if p.is_file() and p.name != '.DS_Store'}
        assert actual == set(index['files']) | {'manifest.json'}, 'Unexpected or missing reference files'
        for name, item in index['files'].items():
            path = SAVED/name
            assert path.resolve().is_relative_to(SAVED.resolve()), name
            assert path.stat().st_size == item['size'] and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], name
        assert {p.name for p in (ROOT/'results').iterdir() if p.name != '.DS_Store'} == {'reference'}
    return index


def public_text(text):
    """Remove account-specific paths and device identifiers from shared records."""
    text = re.sub(r"/mnt/[a-z]/(?:Users)/[^/\s\"']+", '$HOME', text)
    text = re.sub(r"/(?:home|Users)/[^/\s\"']+", '$HOME', text)
    text = re.sub(r"[A-Za-z]:\\Users\\[^\\\s\"']+", '$HOME', text)
    return re.sub(r'\bGPU-[A-Fa-f0-9-]{20,}', '[redacted-device]', text)


def public_metadata(value):
    if isinstance(value, dict):
        return {key: 'benchmark-host' if key == 'host' else public_metadata(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    return public_text(value) if isinstance(value, str) else value


def publish_reference(kind, source):
    """Publish a complete validated run; older evidence moves to ignored scratch."""
    targets = {
        'current': ('warm', ['samples.csv', 'summary.json', 'method.json', 'environment.json', 'launches.log']),
        'progressive_square': ('square', ['square.raw.csv', 'samples.csv', 'kernels.csv',
                                          'summary.json', 'variants.json', 'metadata.json', 'correctness.json', 'telemetry.csv']),
    }
    folder, names = targets[kind]
    source = Path(source).resolve()
    scratch = BUILD/'runs'; scratch.mkdir(parents=True, exist_ok=True)
    assert source.is_relative_to(scratch.resolve()), 'Publish completed scratch runs, not arbitrary folders'
    assert all((source/name).is_file() for name in names), 'Incomplete reference dataset'
    index_path = SAVED/'manifest.json'; old_index = index_path.read_bytes()
    index = reference_manifest(verify=True)
    candidate = dict(index); candidate[kind] = str(source.relative_to(ROOT))
    # Recompute all visible table values before replacing any retained evidence.
    performance_measurements(saved_tables(), latest=candidate)
    target = SAVED/folder
    stage = Path(tempfile.mkdtemp(prefix='publish-'+folder+'-', dir=scratch))
    backup = Path(tempfile.mkdtemp(prefix='previous-'+folder+'-', dir=scratch)); backup.rmdir()
    temporary_index = SAVED/'.manifest-publish-tmp.json'
    promoted = False
    try:
        for name in names:
            shutil.copy2(source/name, stage/name)
            if name in ('environment.json', 'metadata.json'):
                write(stage/name, public_metadata(json.loads((stage/name).read_text())))
            elif name == 'launches.log':
                (stage/name).write_text(public_text((stage/name).read_text()))
        assert index_path.read_bytes() == old_index, 'Reference index changed during publication'
        for name, item in index['files'].items():
            if name.startswith(folder+'/'):
                assert hashlib.sha256((SAVED/name).read_bytes()).hexdigest() == item['sha256'], 'Reference data changed during publication'
        updated = dict(index)
        updated['files'] = {name: item for name, item in index['files'].items() if not name.startswith(folder+'/')}
        for name in names:
            path = stage/name
            updated['files'][folder+'/'+name] = {'size': path.stat().st_size,
                                                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        target.rename(backup)
        stage.rename(target); promoted = True
        write(temporary_index, updated)
        os.replace(temporary_index, index_path)
        print('REFERENCE PUBLISHED', target, flush=True)
    except Exception:
        if promoted:
            shutil.rmtree(target)
        if backup.exists():
            backup.rename(target)
        raise
    finally:
        if stage.exists():shutil.rmtree(stage)
        if temporary_index.exists():temporary_index.unlink()
    return updated


def new_run(kind):
    (BUILD/'runs').mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix=kind+'-',dir=BUILD/'runs'))
    write(out/'environment.json',{'python':sys.executable,'cuda_home':os.environ['CUDA_HOME'],
          'compiler':subprocess.check_output([os.environ['NVCC'],'--version'],text=True),
          'gpu':subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=name,driver_version,memory.total,clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv,noheader'],text=True)})
    print('RESULTS',out,flush=True)
    return out

def compile_one(source,output,extra=(),opt='-O3'):
    output.parent.mkdir(parents=True,exist_ok=True)
    args=[os.environ['NVCC'],opt,'-std=c++17','-lineinfo','-arch='+os.environ['CUDA_ARCH'],str(source),*extra,'-o',str(output)]
    run(args,output.with_suffix('.build.log'))

def require_idle_gpu():
    result=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True)
    if result.stdout.strip():raise RuntimeError('GPU is in use; wait for other CUDA jobs to finish. Active processes: '+result.stdout.strip())

def build():
    compile_one(ROOT/'src/gemm.cu',BUILD/'libgemm.so',['--shared','-Xcompiler','-fPIC','-lcublas'])
    symbols=subprocess.check_output(['nm','-D','--defined-only',str(BUILD/'libgemm.so')],text=True)
    assert re.search(r'\bT gemm_launcher$',symbols,re.M) and not re.search(r'\bT solve$',symbols,re.M)
    print('BUILD PASS: current GEMM library',flush=True)

def require_build():
    if not (BUILD/'libgemm.so').is_file():raise RuntimeError('Run ./scripts/build.sh first')

def correctness():
    require_idle_gpu()
    require_build()
    sys.path.insert(0, str(ROOT/'tests'))
    from test_gemm import run as run_tests
    run_tests(BUILD/'libgemm.so', new_run('tests'))


def current_benchmark():
    require_idle_gpu()
    import torch
    require_build();out=new_run('current');torch.set_num_threads(4);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False
    lib=ctypes.CDLL(str(BUILD/'libgemm.so'))
    funcs={k:getattr(lib,s) for k,s in [('custom','gemm_launcher'),('cublas','gemm_cublas')]}
    for f in funcs.values():f.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_float]*2;f.restype=None
    shapes=[(1,6144,4096),(8,6144,4096),(32,6144,4096),(2048,6144,4096),(2048,4096,14336)]
    samples=[];summary=[]
    for m,n,k in shapes:
        a=torch.randn((m,k),device='cuda',dtype=torch.float16)*.125;b=torch.randn((k,n),device='cuda',dtype=torch.float16)*.125
        c=torch.empty((m,n),device='cuda',dtype=torch.float16);ref=(a.float()@b.float()).half()
        args=(a.data_ptr(),b.data_ptr(),c.data_ptr(),m,n,k,1.,0.)
        for name,f in funcs.items():
            c.fill_(float('nan'))
            # Capture the untimed correctness launch, not any measured call.
            if name=='custom':
                stdout=os.dup(1)
                try:
                    with (out/'launches.log').open('ab') as trace:
                        os.dup2(trace.fileno(),1);f(*args);torch.cuda.synchronize()
                finally:
                    os.dup2(stdout,1);os.close(stdout)
            else:f(*args);torch.cuda.synchronize()
            torch.testing.assert_close(c,ref,atol=.002,rtol=.002)
            for _ in range(20):f(*args)
        torch.cuda.synchronize()
        for trial in range(11):
            for name in (['custom','cublas'] if trial%2==0 else ['cublas','custom']):
                c.fill_(float('nan'));torch.cuda.synchronize()
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();funcs[name](*args);end.record();end.synchronize()
                samples.append(dict(M=m,N=n,K=k,variant=name,trial=trial,time_us=start.elapsed_time(end)*1000))
        med={name:statistics.median(x['time_us'] for x in samples if (x['M'],x['N'],x['K'],x['variant'])==(m,n,k,name)) for name in funcs}
        summary.append(dict(M=m,N=n,K=k,custom_us=med['custom'],cublas_us=med['cublas'],time_ratio=med['custom']/med['cublas']))
        del a,b,c,ref;torch.cuda.empty_cache()
    with (out/'samples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(samples[0]));w.writeheader();w.writerows(samples)
    write(out/'summary.json',summary)
    write(out/'method.json',dict(timing='CUDA events; one ordinary default-stream GEMM per sample',warmups=20,trials=11,cache='reused inputs; no explicit eviction',alpha=1,beta=0))
    publish_reference('current',out)
    update_tables()
    print('CURRENT BENCHMARK PASS',len(summary),'shapes',flush=True)

def csvrows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))

def ncu_rows(p):
    lines=Path(p).read_text().splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('"ID"'))
    rows=list(csv.DictReader(lines[i:]));return rows[1:],rows[0]

def saved_tables():
    index=reference_manifest()
    suites=set(index['cold_suites'])
    samples=csvrows(SAVED/'cold_matrix/samples.csv');groups=collections.defaultdict(list)
    for row in samples:
        if row['suite'] in suites:
            groups[row['suite'],row['variant']].append(float(row['time_us']))
    expected=json.loads((SAVED/'cold_matrix/summary.json').read_text())
    cold=[]
    for (case,variant),values in groups.items():
        value=statistics.median(values)
        assert len(values)==9 and math.isclose(value,expected[case][variant]['median_us'],rel_tol=1e-10)
        cold.append(dict(case=case,variant=variant,median_us=value,samples=len(values)))
    rows,units=ncu_rows(SAVED/'m32/all_setups.raw.csv');by=collections.defaultdict(float)
    for row in rows:
        label=re.search(r'(\d\d_[A-Za-z0-9_]+)',row['Kernel Name'])
        if not label:label=re.search(r'(\d\d_[A-Za-z0-9_]+)',row.get('thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg',''))
        assert label,row['Kernel Name']
        by[label.group(1)]+=float(row['gpu__time_duration.sum'].replace(',',''))*{'ns':.001,'us':1.,'ms':1000.}[units['gpu__time_duration.sum']]
    assert len(cold)==18 and len(by)==13
    return {'cold_matrix':cold,'m32':[{'variant':name,'gpu_us':value} for name,value in sorted(by.items())]}


def _performance_kernel(text):
    """Use current kernel names, including explicit non-default template arguments."""
    match = re.search(r'\b(gemm_\w+)(?:<([^>]*)>)?\s*\(', text)
    assert match, ('Missing kernel identity', text)
    name = match.group(1).replace('gemm_mma_swizzle_kernel', 'gemm_wmma_swizzle_kernel')
    if match.group(2) is None:
        return name
    args = [re.sub(r'\((?:int|bool)\)', '', value).strip()
            for value in match.group(2).split(',')]
    if name in ('gemm_gemv_kernel', 'gemm_mma_llm_kernel') and len(args) == 4 and args[-1] == '1':
        args.pop()  # Omit CacheGlobal=true or GroupRows=1, matching the launch spelling.
    return name + '<' + ', '.join(args) + '>'


def performance_measurements(tables, latest=None):
    """Recompute the displayed values from complete-call samples and kernel traces."""
    latest = reference_manifest() if latest is None else latest
    warm_dir = ROOT/latest['current']
    method = json.loads((warm_dir/'method.json').read_text())
    assert method['warmups'] == 20 and method['trials'] == 11
    assert method['alpha'] == 1 and method['beta'] == 0
    assert method['cache'] == 'reused inputs; no explicit eviction'
    warm = {(r['M'], r['N'], r['K']): r
            for r in json.loads((warm_dir/'summary.json').read_text())}
    samples = collections.defaultdict(list)
    for row in csvrows(warm_dir/'samples.csv'):
        key = tuple(int(row[d]) for d in ('M', 'N', 'K')) + (row['variant'],)
        samples[key].append(float(row['time_us']))

    cold = {(r['case'], r['variant']): r for r in tables['cold_matrix']}
    cold_kernels = collections.defaultdict(set)
    for row in csvrows(SAVED/'cold_matrix/kernels.csv'):
        if row['variant'] != 'cublas':
            cold_kernels[row['suite'], row['variant']].add(_performance_kernel(row['kernel']))
    cold_method = json.loads((SAVED/'cold_matrix/metadata.json').read_text())
    assert cold_method['warmups_per_variant'] == 10 and cold_method['samples_per_variant'] == 9
    assert cold_method['eviction_bytes'] == 256 * 1024 * 1024
    assert cold_method['replay_mode'] == 'application' and cold_method['ncu_cache_control'] == 'none'
    assert cold_method['clock_control'] == 'base'

    cases = [
        ((1, 6144, 4096), 'gemm_gemv_kernel<8, 16, 4>', 'opt_m1', 'optimized'),
        ((8, 6144, 4096), 'gemm_mma_llm_kernel<16, 128, 3>', 'qkv', 'optimized'),
        ((32, 6144, 4096), 'gemm_mma_llm_kernel<32, 128, 3>', 'opt_m32', 'optimized'),
        ((2048, 6144, 4096), 'gemm_mma_llm_kernel<64, 128, 3, 32>', 'opt_prefill', 'optimized'),
        ((2048, 4096, 14336), 'gemm_mma_llm_kernel<128, 64, 4, 4>', 'opt_down', 'optimized'),
    ]
    launches={}
    for line in (warm_dir/'launches.log').read_text().splitlines():
        match=re.match(r'Launching (.+): M=(\d+) N=(\d+) K=(\d+),',line)
        if match:launches[tuple(int(v) for v in match.groups()[1:])]=_performance_kernel(match[1]+'(')
    launcher = re.sub(r'\s+', '', (ROOT/'src/gemm.cu').read_text())
    final = []
    for shape, kernel, suite, variant in cases:
        assert re.sub(r'\s+', '', kernel) in launcher, ('Kernel missing from launcher', kernel)
        assert launches[shape]==kernel, ('Warm kernel identity mismatch',shape)
        assert cold_kernels[suite, variant] == {kernel}, (suite, variant, cold_kernels[suite, variant])
        row = warm[shape]
        for label in ('custom', 'cublas'):
            values = samples[shape + (label,)]
            assert len(values) == 11 and all(math.isfinite(x) and x > 0 for x in values)
            assert math.isclose(statistics.median(values), row[label + '_us'], rel_tol=1e-10)
        assert math.isclose(row['time_ratio'], row['custom_us']/row['cublas_us'], rel_tol=1e-10)
        measured = cold[suite, variant]
        control = cold[suite, 'cublas']
        assert measured['samples'] == control['samples'] == 9
        final.append(dict(shape=shape, kernel=kernel, warm_us=row['custom_us'],
                          warm_cublas_us=row['cublas_us'], cold_us=measured['median_us'],
                          cold_cublas_us=control['median_us']))

    # Launch dimensions are retained from the migrated comment tables. Times and
    # kernel identities are checked against the packaged profiler exports.
    layouts = [
        ('gemm_base_kernel', 32, 8, 32, 8),
        ('gemm_smem_kernel', 32, 32, 32, 32),
        ('gemm_smem_reg_kernel', 32, 32, 16, 16),
        ('gemm_smem_reg_half2_kernel', 32, 32, 16, 16),
        ('gemm_f4_kernel', 32, 32, 4, 32),
        ('gemm_wmma_smem_kernel', 16, 16, 32, 1),
        ('gemm_wmma_gmem_kernel', 16, 16, 32, 1),
        ('gemm_wmma_smem_reg_kernel', 32, 32, 32, 1),
        ('gemm_wmma_pipeline_kernel<3>', 64, 64, 128, 1),
        ('gemm_wmma_swizzle_kernel<3>', 64, 64, 128, 1),
    ]
    launch_layout = {name: (tile_n, tile_m, bx, by) for name, tile_n, tile_m, bx, by in layouts}
    order = {name: i for i, (name, *_) in enumerate(layouts)}
    order['gemm_mma_llm_kernel<16, 128, 3>'] = len(order)
    order['gemm_mma_llm_kernel<32, 128, 3>'] = len(order)
    order['cuBLAS (`gemm_cublas`)'] = len(order)
    groups = []
    suite, shape = 'qkv', (8, 6144, 4096)
    logical_m, n, _ = shape
    rows = []
    for (case, variant), value in cold.items():
        if case != suite:
            continue
        if variant == 'cublas':
            kernel, executed_m, grid, block = 'cuBLAS (`gemm_cublas`)', logical_m, 'library', 'library'
        else:
            identities = cold_kernels[suite, variant]
            assert len(identities) == 1
            kernel = next(iter(identities))
            executed_m = logical_m
            if kernel == 'gemm_wmma_gmem_kernel':
                executed_m = 16
            elif kernel in ('gemm_wmma_pipeline_kernel<3>', 'gemm_wmma_swizzle_kernel<3>'):
                executed_m = 64
            if kernel.startswith('gemm_mma_llm_kernel'):
                grid, block = (n // 128, 1), (128, 1)
            else:
                tile_n, tile_m, bx, by = launch_layout[kernel]
                grid = ((n + tile_n - 1) // tile_n, (executed_m + tile_m - 1) // tile_m)
                block = (bx, by)
        rows.append(dict(kernel=kernel, executed_m=executed_m, grid=grid, block=block,
                         gpu_us=value['median_us'], samples=value['samples']))
    rows.sort(key=lambda r: (order[r['kernel']], r['executed_m']))
    groups.append(dict(shape=shape, rows=rows, samples=9,
                       cublas_us=cold[suite, 'cublas']['median_us']))

    raw, units = ncu_rows(SAVED/'m32/all_setups.raw.csv')
    identities, configurations = collections.defaultdict(set), collections.defaultdict(set)
    for row in raw:
        label = re.search(r'(\d\d_[A-Za-z0-9_]+)', row['Kernel Name'])
        if not label:
            label = re.search(r'(\d\d_[A-Za-z0-9_]+)', row.get('thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg', ''))
        assert label
        label = label.group(1)
        if 'cublas' not in label:
            identities[label].add(_performance_kernel(row['Kernel Name']))
            block = tuple(int(v) for v in re.findall(r'\d+', row['Block Size']))
            grid = tuple(int(v) for v in re.findall(r'\d+', row['Grid Size']))
            assert block[2] == grid[2] == 1
            configurations[label].add((grid[:2], block[:2]))
    rows = []
    for value in tables['m32']:
        label = value['variant']
        executed_m = 64 if 'M64' in label else 32
        if 'cublas' in label:
            kernel, grid, block = 'cuBLAS (`gemm_cublas`)', 'library', 'library'
        else:
            assert len(identities[label]) == len(configurations[label]) == 1
            kernel = next(iter(identities[label]))
            grid, block = next(iter(configurations[label]))
        rows.append(dict(kernel=kernel, executed_m=executed_m, grid=grid, block=block,
                         gpu_us=value['gpu_us'], samples=1))
    control = next(r['gpu_us'] for r in rows if r['kernel'].startswith('cuBLAS') and r['executed_m'] == 32)
    rows.sort(key=lambda r: (order[r['kernel']], r['executed_m']))
    groups.insert(1, dict(shape=(32, 6144, 4096), rows=rows, samples=1, cublas_us=control))
    from square_benchmark import load_results
    square_path=ROOT/latest['progressive_square']
    measured=load_results(square_path)
    rows=[]
    for value in measured:
        kernel='cuBLAS (`gemm_cublas`)' if value['id']==8 else value['kernel']
        rows.append(dict(kernel=kernel,executed_m=1024,
                         grid='library' if value['id']==8 else tuple(value['grid'][:2]),
                         block='library' if value['id']==8 else tuple(value['block'][:2]),
                         gpu_us=value['median_us'],samples=9))
    groups.append(dict(shape=(1024,1024,1024),rows=rows,samples=9,cublas_us=measured[8]['median_us']))
    assert [len(group['rows']) for group in groups]==[10,13,10]
    assert len(final)==5
    return dict(final=final,progressive=groups,warm_dir=warm_dir.relative_to(ROOT),
                square_dir=square_path.relative_to(ROOT))


def performance_document(tables, latest=None):
    from performance_plots import refresh_report, write_plots
    path = ROOT/'docs/PERFORMANCE.md'
    existing = path.read_text()
    data = performance_measurements(tables, latest)
    document = refresh_report(existing, data)
    # A benchmark publishing a new snapshot also prepares its matching figures.
    # Check/default calls remain read-only; update_tables writes the assets.
    if latest is not None:
        write_plots(document)
        assert path.read_text() == existing, 'Document changed while preparing figures'
    return document


def check_project():
    reference_manifest(verify=True)
    tables=saved_tables()
    document=performance_document(tables)
    assert (ROOT/'docs/PERFORMANCE.md').read_text()==document
    from performance_plots import check_plots
    check_plots(document)
    assert re.findall(r'^## (.+)$',document,re.M)==[
        '1. Test methods','2. Final implementation versus cuBLAS','3. Progressive kernel comparisons']
    assert not re.search(r'\| (?:Baseline|Optimized|Custom)(?: \xB5s)? \|',document)
    assert 'gemm_mma_swizzle_kernel' not in document
    for p in [ROOT/'README.md',*sorted((ROOT/'docs').glob('*.md'))]:
        text=p.read_text()
        assert not re.search(r'20\d\d-\d\d-\d\d|202610|13\.2\.86',text),p
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if '://' not in link and not link.startswith('#'):assert (p.parent/link.split('#')[0]).exists(),(p,link)
    for p in (ROOT/'scripts').glob('*.sh'):run(['bash','-n',p])
    for folder in ('tools', 'tests'):
        for p in (ROOT/folder).glob('*.py'):compile(p.read_text(),str(p),'exec')
    print('CHECK PASS: saved table calculations, documentation links, shell and Python syntax',flush=True)

def update_tables():
    existing=(ROOT/'docs/PERFORMANCE.md').read_text()
    tables=saved_tables()
    document=performance_document(tables)
    from performance_plots import write_plots
    write_plots(document)
    assert (ROOT/'docs/PERFORMANCE.md').read_text()==existing, 'Document changed while preparing figures'
    (ROOT/'docs/PERFORMANCE.md').write_text(document)
    print('TABLES GENERATED',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['build','test','benchmark','tables','check'])
    p.add_argument('--suite',choices=['current'],default='current')
    a=p.parse_args()
    if a.action=='build':build()
    elif a.action=='test':correctness()
    elif a.action=='benchmark':current_benchmark()
    elif a.action=='tables':update_tables()
    else:check_project()
if __name__=='__main__':main()
