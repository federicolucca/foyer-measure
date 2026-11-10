"""
curve_lora — Kaggle kernel (private, GPU T4 x2, internet) for the LoRA points of the learning curve on the fixed test.

The private Kaggle dataset attached to the kernel contains: decision_model.py and systemone_email.py (from this
repository), the consent file consenso_u<USER>.json ({"owner": "<USER>", "train_own": true}), the labelled corpus
(exactly one *.jsonl file) with split_paper3.json and curve_subsets.json from curve_split.py. Create it just before
the push and delete it as soon as the metrics are downloaded (one per session). The kernel writes only metrics and
config to /kaggle/working/results: no weights stay on Kaggle (the weights are never saved, the best epoch stays in
memory).
kernel-metadata.json: is_private true, enable_gpu true, enable_internet true, machine_shape "NvidiaTeslaT4"
(= GPU T4 x2). The kernel takes no environment variables: edit the constants below.

Hyper-parameters are the defaults of decision_model.py (r 16, alpha 32, dropout 0.05, all-linear, lr 2e-4,
3 epochs, effective batch 16, max_len 1024, validation 10% by thread, Tev1 format).
Order of the paper's runs: 0.8B all sizes seed 7 → 4B sizes 0,100,400,all seed 7 → seed 11.
"""
import glob
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import shutil
import subprocess
import sys

UTENTE = '1'
GPUS = 2                          # T4 x2: one process per GPU, the tasks are spread across the two
FLA = True                        # flash-linear-attention gated delta kernel, if it passes the test on the T4

CURVA_BASE = 'togethercomputer/Tev1-0.8B-experimental'   # then 'togethercomputer/Tev1-4B-experimental'
CURVA_MODELLO = 'tev1-0.8b'                               # then 'tev1-4b'
CURVA_SEMI = '7'
CURVA_TAGLIE = '0,50,100,200,300,400,all'
CURVA_CAMPI = 'tipo,urgenza,serve_risposta,azione_48h,e_rumore'
PROVA_FLA = r'''
import inspect, torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
rif = inspect.unwrap(mq.torch_chunk_gated_delta_rule)
torch.manual_seed(0)
def dati():
    B, T, H, D = 1, 300, 4, 128
    q, k, v = (torch.randn(B, T, H, D, device='cuda', dtype=torch.float16, requires_grad=True) for _ in range(3))
    g = (-torch.nn.functional.softplus(torch.randn(B, T, H, device='cuda'))).requires_grad_()
    beta = torch.rand(B, T, H, device='cuda', dtype=torch.float16).requires_grad_()
    return q, k, v, g, beta
x = dati()
y = [t.detach().clone().requires_grad_() for t in x]
o1, _ = chunk_gated_delta_rule(*x[:3], g=x[3], beta=x[4], use_qk_l2norm_in_kernel=True)
o2, _ = rif(*y[:3], g=y[3], beta=y[4], use_qk_l2norm_in_kernel=True)
w = torch.randn_like(o2.float())
(o1.float() * w).sum().backward()
(o2.float() * w).sum().backward()
def rel(a, b):
    return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6))
err = [rel(o1, o2)] + [rel(a.grad, b.grad) for a, b in zip(x, y)]
print('fla test, relative errors (output, gradients q k v g beta):', [round(e, 4) for e in err])
raise SystemExit(0 if all(e < 2e-2 for e in err) else 1)
'''


def sh(cmd):
    print('$', cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=False).returncode


if sh('nvidia-smi') != 0:
    raise SystemExit('No GPU in the kernel: enable GPU T4 (account with a verified phone number).')
if sh(f'{sys.executable} -m pip install -q -U peft transformers accelerate') != 0:
    raise SystemExit('pip failed: does the kernel have internet?')
# the torchao preinstalled on Kaggle (0.10) makes peft fail (it wants > 0.16): the LoRA does not use it
sh(f'{sys.executable} -m pip uninstall -q -y torchao')
if FLA:
    # same maths as the PyTorch path of transformers, Triton kernel; if it does not hold on the T4, it is removed
    with open('/kaggle/working/prova_fla.py', 'w') as f:
        f.write(PROVA_FLA)
    if sh(f'{sys.executable} -m pip install -q flash-linear-attention') != 0 or \
            sh(f'{sys.executable} /kaggle/working/prova_fla.py') != 0:
        print('flash-linear-attention not usable here: PyTorch path', flush=True)
        sh(f'{sys.executable} -m pip uninstall -q -y flash-linear-attention fla-core')
    os.remove('/kaggle/working/prova_fla.py')

dati = os.path.dirname(glob.glob('/kaggle/input/**/decision_model.py', recursive=True)[0])
corpus = glob.glob(os.path.join(dati, '*.jsonl'))
if len(corpus) != 1:
    raise SystemExit(f'the dataset must contain exactly one .jsonl corpus, found {len(corpus)}')
codice = '/kaggle/working/codice'
os.makedirs(codice, exist_ok=True)
for f in ('decision_model.py', 'systemone_email.py'):
    shutil.copy(os.path.join(dati, f), codice)

rc = sh(f'cd {codice} && {sys.executable} -u decision_model.py --user {UTENTE} '
        f'--consent u{UTENTE}={dati}/consenso_u{UTENTE}.json --data {corpus[0]} '
        f'--split {dati}/split_paper3.json --subsets {dati}/curve_subsets.json --system lora '
        f'--model {CURVA_MODELLO} --base-model {CURVA_BASE} --fields {CURVA_CAMPI} --seeds {CURVA_SEMI} '
        f'--sizes {CURVA_TAGLIE} --gpus {GPUS} --out /kaggle/working/results')
shutil.rmtree(codice, ignore_errors=True)
sh('find /kaggle/working -type f | head -50')
sys.exit(rc)
