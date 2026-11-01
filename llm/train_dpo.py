# DPO: 前の段階のアダプター（RFT）を元モデルに合体させたものを出発点・参照モデルにして、新しい LoRA で DPO する。
#   データ: collect.py が作る data/dpo_<name>.jsonl（同じ問題の 合格 / 不合格 の答えの組。答えはモデル自身のもの、合否はプログラム）
#   使い方: python train_dpo.py --init runs/rft-150 --data data/dpo_rft150.jsonl --steps 80 --out runs/dpo-80
import argparse, json, os, time, torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer
from evaluate import BASE

ap = argparse.ArgumentParser(); ap.add_argument('--init', required=True); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=80); ap.add_argument('--beta', type=float, default=0.1); ap.add_argument('--lr', type=float, default=5e-5); ap.add_argument('--bs', type=int, default=4)
a = ap.parse_args(); t0 = time.time()
tok = AutoTokenizer.from_pretrained(BASE)
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16); m = PeftModel.from_pretrained(m, a.init).merge_and_unload()   # 出発点 = RFT 後のモデル
merged = a.out + '_start'; m.save_pretrained(merged); tok.save_pretrained(merged)   # 評価のとき、この上に DPO のアダプターを載せる
ds = Dataset.from_list([json.loads(l) for l in open(a.data, encoding='utf-8')]).remove_columns(['kind'])
cfg = DPOConfig(output_dir=a.out, max_steps=a.steps, per_device_train_batch_size=a.bs, learning_rate=a.lr, beta=a.beta, lr_scheduler_type='cosine', warmup_steps=2,
                logging_steps=10, save_strategy='no', bf16=True, seed=42, report_to=[], max_length=1024)
peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], task_type='CAUSAL_LM')
tr = DPOTrainer(model=m, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft)   # 参照モデル = アダプターを外した状態 = RFT 後のモデル
tr.train(); tr.save_model(a.out)
json.dump({'start': a.init, 'merged_start': merged, 'data': a.data, 'n': len(ds), 'steps': a.steps, 'beta': a.beta, 'lr': a.lr, 'minutes': round((time.time() - t0) / 60, 1),
           'log': [h for h in tr.state.log_history if 'loss' in h]}, open(os.path.join(a.out, 'train_info.json'), 'w'), ensure_ascii=False, indent=1)
print('SAVED', a.out, f'{(time.time() - t0) / 60:.1f} 分')
