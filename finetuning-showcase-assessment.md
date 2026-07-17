# Assessment: Turning medical-llm into a Visual "How LLM Finetuning Works" Educational Project

**Date:** 2026-07-17
**Goal:** An educational project that *shows* LLM finetuning — with pictures, HTML
pages, charts, and walkthroughs.
**Verdict: No rebuild at all. Keep everything — the repo IS the case study.**
The work is purely additive: a presentation/visualization layer on top of what
already exists.

## Why this repo is ideal raw material

Unlike a fresh start, this project already contains a **complete, finished
finetuning run with real artifacts** — exactly what an educational showcase needs
and what a new project would have to produce first:

| Existing artifact | Educational use |
|---|---|
| `outputs/biomistral-medical/trainer_log.jsonl` | Per-step loss, cosine LR schedule, throughput → interactive charts |
| `all_results.json`, `eval_results.json` | Final train vs eval metrics → summary stat tiles |
| `training_loss.png`, `training_eval_loss.png` | Ready-made figures |
| `runs/` TensorBoard events + `tensorboard.sh` | Live-dashboard demo, gradient norms, GPU memory |
| `training_analysis.md` | A **real diagnosed overfitting case** (train 2.13→0.70 while eval 1.35→1.43) — a genuine lesson, not a toy example |
| `checkpoint-100/` vs `checkpoint-135/` | Show checkpointing; compare mid-training vs final adapter |
| `adapter_model.safetensors` (LoRA, rank 32) | Show how small the trained delta is vs the 14 GB base model |
| `configs/*.yaml` + `dataset_info.json` | Small, readable configs → annotated line-by-line in HTML |
| `scripts/prepare_medical_data.py` + `data/processed/*.json` | Real Alpaca-format samples for the "what does training data look like" chapter |
| `scripts/chat.sh` + chat config | Live before/after inference demo |

The medical domain is irrelevant to the lesson — it is simply the worked example.
Nothing needs to be swapped out.

## What to ADD (the actual project)

A `docs/` (or `site/`) layer of self-contained HTML pages, roughly one per
pipeline stage:

1. **Overview** — pipeline diagram (data → tokenization → QLoRA → training →
   evaluation → inference), with links into each chapter.
2. **The data** — Alpaca format explained with real samples from
   `medical_train.json`; the prep script annotated.
3. **QLoRA explained visually** — diagrams: frozen 4-bit base weights + trainable
   low-rank A/B adapter matrices; NF4 quantization; why VRAM drops from ~28 GB
   (fp16 full finetune) to ~16–18 GB.
4. **The training run** — interactive charts built from `trainer_log.jsonl`:
   loss curve, cosine LR schedule with warmup, steps/sec; annotated config.
5. **The overfitting story** — train-vs-eval divergence chart from the real run,
   `training_analysis.md` turned into a visual lesson with its fix
   recommendations (fewer epochs, lower rank, more data).
6. **Results & inference** — adapter size vs base model size; how `chat.yaml`
   attaches the adapter; sample outputs.

Optional extras, in order of value:
- **A second training run with the fixed hyperparameters** (rank 8–16, 1–2
  epochs, full data) → a before/after comparison chapter, completing the
  narrative arc: train → diagnose → fix.
- **Jupyter notebooks** mirroring each chapter for hands-on learners.
- **Gradio side-by-side demo**: base model vs finetuned adapter answering the
  same question.

## What to clean up (minor)

- `data/processed/*.bak` files — delete.
- README — reframe from "medical assistant" to "finetuning walkthrough".
- Root `medical-llm-plan.md` — archive or fold into the docs.

## Structure decision (open)

The HTML layer can be built as (a) self-contained static pages in `docs/`
(viewable locally, publishable via GitHub Pages), (b) Claude Artifacts
(private hosted pages), or (c) notebooks-first with exported HTML. Static
`docs/` is the most durable default for an educational repo.
