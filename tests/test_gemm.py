"""Correctness checks for the public FP16 gemm_launcher entry point."""
import ctypes
import gc
import json
from pathlib import Path

import torch

GUARD = 64
CANARY = 123.
ATOL = RTOL = .002
MODELS = {
    'llama3_8b': (4096, 14336, 1024, 128256),
    'qwen25_7b': (3584, 18944, 512, 152064),
}
TOKENS = [1, 8, 32, 64, 128, 512, 2048]
EXTRAS = [
    (1,4096,1025), (1,512,1025), (1,1024,8192), (2,512,512),
    (3,576,640), (17,640,672), (31,512,1024), (47,1024,512),
    (63,1024,640), (65,1024,640), (96,1024,1024), (128,576,18944),
    (384,2048,14336), (1664,2048,14336), (832,2048,8256),
    (17,19,0), (0,19,32), (19,0,32),
]


def guarded(shape, fill=float('nan'), offset=GUARD):
    size = 1
    for dim in shape:
        size *= dim
    buffer = torch.full((size+offset+GUARD,), CANARY, device='cuda', dtype=torch.float16)
    tensor = buffer[offset:-GUARD].view(shape)
    tensor.fill_(fill)
    return tensor, buffer


def canaries(buffer, offset=GUARD):
    return bool((buffer[:offset] == CANARY).all() and (buffer[-GUARD:] == CANARY).all())


def invoke(fn, a, b, c, alpha, beta):
    m, k = a.shape
    n = b.shape[1]
    torch.cuda.synchronize()
    fn(a.data_ptr(), b.data_ptr(), c.data_ptr(), m, n, k, alpha, beta)
    torch.cuda.synchronize()


def metrics(c, reference):
    if not c.numel():
        return {'pass': True, 'finite': True, 'violations': 0}
    diff = (c.float()-reference).abs()
    result = {
        'finite': bool(torch.isfinite(c).all()),
        'violations': int((diff > ATOL+RTOL*reference.abs()).sum()),
        'max_abs_error': float(diff.max()),
        'rmse': float(diff.square().mean().sqrt()),
        'relative_l2_error': float(diff.square().sum().sqrt()/reference.square().sum().sqrt().clamp_min(1e-30)),
        'matches_fp32_reference_rounded_fp16_pct': float((c == reference.half()).float().mean())*100,
    }
    result['pass'] = result['finite'] and result['violations'] == 0
    return result


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')


def bundled_cases():
    options = dict(device='cuda', dtype=torch.float16)
    cases = [(torch.tensor([[1.,2.,3.], [4.,5.,6.]], **options),
              torch.tensor([[1.,2.], [3.,4.], [5.,6.]], **options),
              torch.ones((2,2), **options), 1., 0.)]
    for m,n,k,alpha,beta,low,high,zero_c in [
        (16,16,16,1.,0.,-1.,1.,True),
        (16,16,16,1.,1.,-.5,.5,False),
        (32,16,16,.5,.5,-1.,1.,False),
        (16,32,16,1.,1.,-1.,1.,False),
        (16,16,32,0.,1.,-1.,1.,False),
    ]:
        a = torch.empty((m,k), **options).uniform_(low, high)
        b = torch.empty((k,n), **options).uniform_(low, high)
        c = torch.zeros((m,n), **options) if zero_c else torch.empty((m,n), **options).uniform_(low, high)
        cases.append((a,b,c,alpha,beta))
    return cases


def check_bundled(fn):
    results = []
    for index, (a,b,c,alpha,beta) in enumerate(bundled_cases()):
        reference = (alpha*(a.float()@b.float())+beta*c.float()).half()
        protected, buffers = [], []
        for original in (a,b,c):
            tensor, buffer = guarded(original.shape)
            tensor.copy_(original)
            protected.append(tensor)
            buffers.append(buffer)
        invoke(fn, *protected, alpha, beta)
        torch.testing.assert_close(protected[2], reference, atol=.05, rtol=.05)
        assert all(canaries(buffer) for buffer in buffers)
        assert torch.equal(protected[0], a) and torch.equal(protected[1], b)
        results.append({'case': index, 'pass': True})
        print('BUNDLED PASS', index, flush=True)
    return results


def extended_cases():
    cases = [dict(id='control_square_m1024', model='control', op='square', M=1024, N=1024, K=1024)]
    for model, (hidden,ffn,kv,vocab) in MODELS.items():
        for op,n,k in [('qkv',hidden+2*kv,hidden), ('attn_out',hidden,hidden),
                       ('gate_up',2*ffn,hidden), ('down',hidden,ffn), ('lm_head',vocab,hidden)]:
            for m in ([1,32,128] if op == 'lm_head' else TOKENS):
                cases.append(dict(id=f'{model}_{op}_m{m}', model=model, op=op, M=m, N=n, K=k))
    assert len(cases) == 63
    return cases+[dict(id=f'extra_{m}_{n}_{k}', M=m, N=n, K=k, model='extra', op='regression')
                  for m,n,k in EXTRAS]


def check_extended(fn):
    results = []
    for index, case in enumerate(extended_cases()):
        m,n,k = (case[key] for key in ('M','N','K'))
        torch.manual_seed(20261014+index)
        a = torch.randn((m,k), device='cuda', dtype=torch.float16)*.125
        b = torch.randn((k,n), device='cuda', dtype=torch.float16)*.125
        c, buffer = guarded((m,n))
        alpha,beta = (1.,0.) if index < 63 else (.7,-.3)
        if beta:
            c.fill_(2.)
        reference = alpha*(a.float()@b.float())+beta*2.
        invoke(fn, a, b, c, alpha, beta)
        check = metrics(c, reference)
        check['canaries_pass'] = canaries(buffer)
        check['pass'] &= check['canaries_pass']
        results.append(dict(**case, alpha=alpha, beta=beta, **check))
        print('EXPORT', case['id'], check['pass'], flush=True)
        del a,b,c,buffer,reference
        gc.collect()
        torch.cuda.empty_cache()
    # Shift each pointer independently by one FP16 element to exercise fallback.
    for which in ('A','B','C'):
        m,n,k = 17,1024,544
        a = torch.randn((m,k), device='cuda', dtype=torch.float16)*.125
        b = torch.randn((k,n), device='cuda', dtype=torch.float16)*.125
        offset = GUARD+1 if which == 'C' else GUARD
        c, buffer = guarded((m,n), fill=2., offset=offset)
        if which in ('A','B'):
            original = a if which == 'A' else b
            backing = torch.empty(original.numel()+1, device='cuda', dtype=torch.float16)
            shifted = backing[1:].view(original.shape)
            shifted.copy_(original)
            if which == 'A':
                a = shifted
            else:
                b = shifted
        reference = .7*(a.float()@b.float())-.3*2.
        invoke(fn, a, b, c, .7, -.3)
        check = metrics(c, reference)
        check['canaries_pass'] = canaries(buffer, offset)
        check['pass'] &= check['canaries_pass']
        results.append(dict(id='misaligned_'+which, **check))
        print('EXPORT misaligned', which, check['pass'], flush=True)
    return results


def check_precision(fn):
    results = []
    for name,m,n,k,tiny in [
        ('gemv_cancel',1,4096,18944,False), ('decode16_cancel',8,1024,18944,False),
        ('decode32_tiny',32,1024,4096,True), ('prefill_wide_cancel',256,6144,4096,False),
        ('prefill_long_cancel',256,2048,18944,False),
    ]:
        a = torch.full((m,k), 2**-13 if tiny else 256., device='cuda', dtype=torch.float16)
        b = torch.full((k,n), 2**-13 if tiny else 1., device='cuda', dtype=torch.float16)
        if not tiny:
            b[k//2:] = -1.
            b[-1] = -.875
        expected = k*2**-26 if tiny else 32.
        c, buffer = guarded((m,n))
        invoke(fn, a, b, c, 1., 0.)
        passed = bool((c == expected).all()) and canaries(buffer)
        results.append(dict(case=name, M=m, N=n, K=k, expected=expected, exact_pass=passed))
        print(name, passed, flush=True)
    return results


def run(library, output):
    output = Path(output)
    torch.set_num_threads(4)
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    lib = ctypes.CDLL(str(library))
    fn = lib.gemm_launcher
    fn.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_float]*2
    fn.restype = None
    bundled = check_bundled(fn)
    save(output/'bundled.json', bundled)
    checks = check_extended(fn)
    save(output/'extended/export-accuracy.json', checks)
    assert len(checks) == 84 and all(row['pass'] for row in checks)
    precision = check_precision(fn)
    save(output/'extended/extreme-accuracy.json', precision)
    assert len(precision) == 5 and all(row['exact_pass'] for row in precision)
    assert len(bundled) == 6
    save(output/'summary.json', {'bundled': 6, 'extended': 84, 'precision': 5, 'passed': True})
    print('TEST PASS: 6 bundled + 84 extended + 5 precision cases', flush=True)
