#!/usr/bin/env bash
# End-to-end CPU smoke run of the whole training pipeline on the bundled sample corpus (tiny preset).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m data.download --sample
python -m data.clean.pipeline --offline-benchmarks --workers 4
python -m tokenizer.train --vocab-size 4000
PA__DATA__VAL_FRACTION=0.05 python -m data.clean.shard
python -m training.pretrain --preset tiny --seq-len 512 --micro-bs 8 --accum 1 --max-steps ${PRETRAIN_STEPS:-400} \
  --warmup 40 --lr 2e-3 --min-lr 2e-4 --eval-every 200 --eval-batches 10 --sample-every 200 --save-every 200 \
  --log-every 50 --out training/checkpoints/tiny
python -m agent.synth.generate --companies ${SYNTH_COMPANIES:-60} --k 4 --out data/sft
python -m training.sft --init training/checkpoints/tiny/latest.pt --data data/sft --seq-len 1024 --micro-bs 8 \
  --accum 1 --lr 1e-3 --warmup 30 --max-steps ${SFT_STEPS:-600} --eval-every 200 --log-every 50 \
  --out training/checkpoints/sft-tiny
python -m training.eval --checkpoint training/checkpoints/sft-tiny/latest.pt --limit ${EVAL_LIMIT:-150} --out docs/eval
python -m training.eval --policy template --out docs/eval
