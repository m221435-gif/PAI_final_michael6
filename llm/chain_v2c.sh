#!/usr/bin/env bash
# v2c: 聞き返しの SFT 文を 3 件足して SFT から通し直す（説明文 v2、断るは言い換え版）。テストは人が確かめた 126 問。
set -e; export HF_HUB_OFFLINE=1; PY=.venv/bin/python; T=data_v2/test_final.jsonl
ev(){ $PY evaluate.py "$@" --data $T 2>&1 | grep "^\[" ; }
$PY train_sft.py --data data_v2/sft_train.jsonl --steps 160 --out runs/v2c_sft-160 2>&1 | grep SAVED; ev --name F_v2c_sft-160 --adapter runs/v2c_sft-160
$PY collect.py --adapter runs/v2c_sft-160 --data data_v2/rl_u2.jsonl --name v2c_sft160 2>&1 | grep -E "^問題|refuse"
cat data/rft_v2c_sft160.jsonl data_v2/sft_train.jsonl | $PY -c "import sys,random; L=[l for l in sys.stdin if l.strip()]; random.Random(0).shuffle(L); open('data_v2/rft_mix_c.jsonl','w').writelines(L)"
$PY train_sft.py --init runs/v2c_sft-160 --data data_v2/rft_mix_c.jsonl --steps 150 --out runs/v2c_rft-150 2>&1 | grep SAVED; ev --name F_v2c_rft-150 --adapter runs/v2c_rft-150
$PY collect.py --adapter runs/v2c_rft-150 --data data_v2/rl_u2.jsonl --name v2c_rft150 --seed 1 2>&1 | grep -E "^問題|refuse"
$PY -c "
import json,collections,random
r=[json.loads(l) for l in open('data/dpo_v2c_rft150.jsonl')]; random.Random(0).shuffle(r); c=collections.Counter(); o=[]
for x in r:
    if c[x['kind']]<30: o.append(x); c[x['kind']]+=1
open('data/dpo_v2c_rft150_bal.jsonl','w').writelines(json.dumps(x,ensure_ascii=False)+'\n' for x in o); print('DPO 組', len(o))"
$PY train_dpo.py --init runs/v2c_rft-150 --data data/dpo_v2c_rft150_bal.jsonl --beta 0.5 --lr 3e-5 --steps 60 --out runs/v2c_dpo-c 2>&1 | grep SAVED; ev --name F_v2c_dpo-c --start runs/v2c_dpo-c_start --adapter runs/v2c_dpo-c
echo CHAIN_V2C_DONE
