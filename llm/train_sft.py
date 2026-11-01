# SFT（LoRA）: Qwen2.5-1.5B-Instruct に data/sft_train.jsonl を学習させる。損失は答え（assistant）の部分だけ。
#   使い方: python train_sft.py --steps 80 --out runs/sft-80   （設定はソロと同じ: LoRA r16/alpha32、lr 1e-4、乱数 42、bf16）
import argparse, json, glob, os, time
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

BASE = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/'))[0]
ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=80); ap.add_argument('--out', required=True)
ap.add_argument('--data', default='data/sft_train.jsonl'); ap.add_argument('--bs', type=int, default=4); ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--init', default=None, help='続きから学習するアダプター（RFT で使う）')
a = ap.parse_args()


def load(p):
    rows = [json.loads(l) for l in open(p, encoding='utf-8')]
    return Dataset.from_list([{'prompt': r['messages'][:2], 'completion': r['messages'][2:]} for r in rows])   # 会話形式の prompt/completion → 答えだけに損失


ds = load(a.data); t0 = time.time()
cfg = SFTConfig(output_dir=a.out, max_steps=a.steps, per_device_train_batch_size=a.bs, learning_rate=a.lr, lr_scheduler_type='cosine', warmup_steps=2,
                logging_steps=10, save_strategy='no', bf16=True, seed=42, report_to=[], max_length=1024, completion_only_loss=True)
peft = None if a.init else LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], task_type='CAUSAL_LM')
model = BASE
if a.init:
    from peft import AutoPeftModelForCausalLM
    model = AutoPeftModelForCausalLM.from_pretrained(a.init, is_trainable=True)
tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft)
tr.train(); tr.save_model(a.out)
json.dump({'base': BASE, 'data': a.data, 'n': len(ds), 'steps': a.steps, 'bs': a.bs, 'lr': a.lr, 'init': a.init, 'minutes': round((time.time() - t0) / 60, 1),
           'log': [h for h in tr.state.log_history if 'loss' in h]}, open(os.path.join(a.out, 'train_info.json'), 'w'), ensure_ascii=False, indent=1)
print('SAVED', a.out, f'{(time.time() - t0) / 60:.1f} 分')
