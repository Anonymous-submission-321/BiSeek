# BiSeek

BiSeek is a training-free method for streaming video understanding that looks
both ways: **backward**, to answer a question about something that already
happened in the stream (reactive), and **forward**, to stay silent until the
event a standing question waits for actually appears and then speak
(proactive).

One frozen **Qwen3-VL-Embedding-2B** does all the continuous work, and a frozen
**Qwen3-VL-8B-Instruct** is woken only when needed:

- **Backward (reactive).** Every chunk of the incoming stream is encoded once by
  the 2B into a single 2048-d vector and kept in a small semantic memory (4 KiB
  per chunk). When a question arrives, the question vector retrieves a few past
  chunks by cosine similarity; their frames and the most recent frames go to the
  8B, which answers. The memory is an index: retrieved chunks are re-decoded
  from the video at full resolution, so the vectors never have to hold detail.
- **Forward (proactive).** When a standing question is registered, the 8B
  compiles it once into a few concrete, observable visual *proposals*. The 2B
  then scores the live window against those proposals at every step; a sudden
  rise in similarity (a *surge*) opens a short candidate episode, and only then
  is the 8B asked to verify the moment and answer. Most steps never reach the 8B.

Nothing is trained: both models are used as released.

## Installation

Python 3.11 and a CUDA GPU (the main runs used A100 80 GB; 2B + 8B in bf16 fit on one card).

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation   # optional; the configs use flash_attention_2
```

The code runs on the stock `transformers==5.8.1`; no patched library is needed.

## Models

Download into `models/` (paths in `configs/*.json` are relative to the repository root):

```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct  --local-dir models/Qwen3-VL-8B-Instruct
huggingface-cli download Qwen/Qwen3-VL-Embedding-2B --local-dir models/Qwen3-VL-Embedding-2B
```

The embedding model's repository ships a `scripts/qwen3_vl_embedding.py` helper,
which BiSeek loads from `models/Qwen3-VL-Embedding-2B/scripts`.

## Data

Place each benchmark under `data/`:

| Benchmark | Source | Preparation |
|---|---|---|
| StreamingBench | [Hugging Face](https://huggingface.co/datasets/mjuicem/StreamingBench) + [official repo](https://github.com/THUNLP-MT/StreamingBench) | `python biseek/setup_streamingbench.py --dataset-root data/StreamingBench --extract` extracts the videos once and writes the normalized annotation files to `data/StreamingBench/data/annotations/` |
| OVO-Bench | [Hugging Face](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench) + [official repo](https://github.com/JoeLeelyf/OVO-Bench) | `python biseek/setup_ovo_bench.py --root data/OVO-Bench` streams the split tar parts into `data/OVO-Bench/data/chunked_videos/` and checks every clip the annotation file needs |
| StreamArena | [Hugging Face](https://huggingface.co/datasets/hkuzxc/StreamArena) | download as is: `data/StreamArena/question.en.jsonl` and `data/StreamArena/videos/<id>/<id>.mp4` |

Run either setup script with `--help` for the remaining options (archive location, verification only).

## Running

Every script runs the main configuration of the paper by default. Choose GPUs with `GPUS`:

```bash
GPUS=0,1,2,3 scripts/streamingbench_proactive.sh     # StreamingBench Proactive Output
GPUS=0,1,2,3 scripts/streamingbench_reactive.sh      # StreamingBench reactive questions
GPUS=0,1,2,3 scripts/ovo_forward.sh                  # OVO-Bench Forward (SSR + CRR, then REC)
GPUS=0,1,2,3 scripts/ovo_reactive.sh                 # OVO-Bench Real-Time + Backward
GPUS=0,1,2,3 scripts/streamarena_pro.sh              # StreamArena Pro
GPUS=0,1,2,3 TRACK=HR  scripts/streamarena_reactive.sh
GPUS=0,1,2,3 TRACK=RTP scripts/streamarena_reactive.sh
```

Results go to `outputs/<benchmark>/.../main/` (`rank_*.jsonl` per process, plus
`summary.json` where the benchmark is scored automatically). Override the output
directory with `OUT=...` and the configuration with `CONFIG=...`. Extra arguments
are passed through; for a quick check, `--max-samples 5` (or `--max-questions 5`
for StreamArena Pro) limits the number of questions.

### Scoring

- **StreamingBench** and **OVO-Bench** are multiple choice (OVO Forward is
  scored at the official test points); the run writes `summary.json` with the
  benchmark's own metrics.
- **StreamArena** answers are open-ended and are scored by an LLM judge. We use
  a local `gpt-oss-120b` with the benchmark's judge prompts; Pro answers are
  first gated by the benchmark's timing window ([-0.5, +2.0] s around the
  reference). Download it to `models/gpt-oss-120b` and run
  `scripts/judge_streamarena.sh outputs/StreamArena/pro/main ...`. Dequantized
  to bf16 the judge needs about 240 GB of GPU memory (for example 6 x 48 GB).

### Main results

| Benchmark | Metric | BiSeek |
|---|---|---|
| StreamingBench Proactive Output | first response within ±2 s | 47.6 |
| StreamingBench reactive | accuracy | 73.8 |
| OVO-Bench Real-Time / Backward | accuracy | 80.0 / 54.7 |
| OVO-Bench Forward (SSR / CRR / REC) | accuracy | 71.1 / 73.3 / 40.3 |
| StreamArena Pro | timing / timing + content | 20.8 / 20.3 |
| StreamArena HR / RTP | judged accuracy | 18.6 / 30.0 |

StreamArena numbers depend on the judge; ours come from the local gpt-oss-120b above.

## Ablations

The ablations in the paper change one key of a configuration. Copy a file from
`configs/`, change the key, and pass it with `CONFIG=...`:

| Ablation | Key | Values |
|---|---|---|
| Answering / verifying backbone | `answer_model_family`, `answer_model_path` | `qwen3vl` (default, Qwen3-VL-8B-Instruct), `qwen2_5vl` (Qwen2.5-VL-7B-Instruct), `qwen2vl` (Qwen2-VL-7B-Instruct); the 2B embedding model is unchanged |
| Gate / memory encoder | `embedding_backend`, `embedding_model_path` | `qwen3vl` (default), `gme_qwen2vl`, `opsmm`, `vlm2vec`; two-tower `clip`, `siglip`, `jina_clip` |
| Window encoding for the gate | `proposal_match_encoding` | `vit_feature_joint_decoder` (cache A, default: per-frame ViT features cached once, window re-encoded jointly by the decoder), `incremental_kv` (cache B: sliding decoder KV), `cached_frame_mean` (per-frame vectors averaged; the only option for two-tower encoders) |
| Surge threshold | `proposal_surge_threshold` | `0.03` (default); `0.015` for encoders whose similarity moves less |
| Sampling rate | `sample_fps` | `1.0` (default), `2.0` |
| Reactive policy | `reactive_policy` | `retrieve_first` (default: always retrieve, one 8B call per question), `recent_first` (answer from recent frames first, retrieve only when the 8B asks for it) |
| Memory size | `max_memory_vectors` | `-1` (unbounded) or a number of vectors |
| 2B gate alone (no 8B verifier) | `proposal_audit_only` | `true` (StreamArena Pro: `--gate-only`) |

For StreamArena Pro, the window settings are command-line flags with the main
values as defaults (`--sample-fps`, `--proposal-match-encoding`,
`--proposal-surge-threshold`, ...); run `python biseek/eval_streamarena_proactive.py --help`.

## Repository layout

```
biseek/
  eval_streamingbench_proactive.py   StreamingBench Proactive Output
  eval_streamingbench_reactive.py    StreamingBench reactive questions
  eval_ovo_forward_active.py         OVO-Bench Forward
  eval_ovo_streaming.py              OVO-Bench Real-Time + Backward; shared streaming memory, encoders and 8B wrapper
  eval_river_reactive.py             reactive engine shared by the StreamArena HR/RTP runner
  eval_streamarena_proactive.py      StreamArena Pro
  eval_streamarena_reactive.py       StreamArena HR / RTP
  proactive_cascade.py               proposal compilation, surge gate, candidate episodes, 8B verification
  embedding_backends.py              alternative gate encoders for the ablation
  judge_streamarena_local.py         local LLM judge for StreamArena
  score_ovo_bench.py                 OVO-Bench scoring
  setup_streamingbench.py, setup_ovo_bench.py   data preparation
configs/   main configuration of each benchmark
scripts/   one launcher per benchmark
```

## License

Apache License 2.0. The models and benchmarks keep their own licenses.
