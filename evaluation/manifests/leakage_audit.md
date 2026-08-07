# Data Leakage Audit

Date: 2026-08-07
Applies to the `evaluation/` set (150 frames from 10 QA videos).

## Purpose
Determine whether any evaluation frame's **source video** was used to train the
fine-tuned model, so frames can be assigned `evaluation_role` (`dev` / `held_out`
/ `excluded`) and a `leakage_risk` flag.

## 1. What was checked
- `video_id`, frame filename lineage for each sampled frame
- any training / validation / test split manifest in the repo (`data.yaml`,
  `train/`, `val/`, `test/`, split CSVs)
- git history references to the QA videos / `qa_inputs`
- earlier pipeline QA artifacts (`test_videos/`, `docs/`, `tests/`)

## 2. Findings

| Item | Finding |
|---|---|
| Training manifest in repo | **None found** (no `data.yaml`, YOLO label split, or train/val split is tracked) |
| QA source files | `qa_inputs/qa_01..qa_10` created after the model integration (files dated 2026-08-06). |
| Training provenance of the fine-tuned model | **Unknown** — the training dataset is not present in the repo. |
| Can any video be proven unseen / held-out? | **No** — cannot prove a video was not in training. |

## 3. Enforcement (honesty)

Because provenance cannot be verified within the repo:

- Every frame currently defaults to `evaluation_role = unknown`.
- `evaluation_role = unknown` frames are **excluded from FINAL (held-out) metrics**
  unless the reviewer explicitly assigns a role.
- `evaluation_role = excluded` frames are always excluded.
- `leakage_risk = true` frames (set by the reviewer if they recognize a training
  source) are excluded from final metrics by default.
- No `held_out`-only numbers can be reported until the reviewer confirms video
  origin. This is a documented limitation.

## 4. Residual risk
No training manifest exists in the repo, so no frame can be proven untouched.
Human confirmation of each video's provenance is required before any "held-out"
conclusion. This is a documented limitation, not a waived requirement.
