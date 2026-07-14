# Static Source Cover MVP v2 Visual QA

## Scope
Visual QA audit for the Static Source Cover MVP v2 release. The focus is to resolve the original white bubble overflow in `sample2` and establish a controlled person-overlap safety override policy for `sample5`.

---

## Comparison Summary (v1 vs v2)

| Sample | Duplicate Risk (v1) | Duplicate Risk (v2) | Skipped Count (v1) | Skipped Count (v2) | Regression Status | Verdict |
|---|---|---|---:|---:|---|---|
| **sample2** | Medium | **Low** | 2 | **0** | No regression | **PASS** |
| **sample4** | Low | **Low** | 0 | **0** | No regression | **PASS** |
| **sample5** | Medium | **Low** | 5 | **1** | No regression | **PASS** |

---

## Per-Sample Audit Details

### 1. Sample 2 (Bubble Overflow Fix)
- **Verdict**: **PASS**
- **Original BBox**: `{"left": 123.75, "top": 966.0, "right": 568.0, "bottom": 1030.5}`
- **Expanded Cover Box**: `{"left": 61.0, "top": 933.9, "right": 630.8, "bottom": 1062.6}`
- **Expansion Margin**: `expansion_px_x = 62.79`, `expansion_px_y = 32.06`
- **Expansion Reason**: `bubble_margin`
- **Audit Findings**: The adaptive bubble cover margin fully encapsulates the outer white bubble/edges of the original subtitle. The two previously skipped intervals containing minor hand crossovers are now successfully rendered using controlled non-face person overlap override logic.

### 2. Sample 4 (Regression Check)
- **Verdict**: **PASS**
- **Stable BBox**: `{"left": 268.0, "top": 891.5, "right": 455.0, "bottom": 974.0}`
- **Rendered / Skipped**: 25 rendered / 0 skipped
- **Audit Findings**: The green source text remains cleanly covered. No background distortion or central watermark masking was introduced.

### 3. Sample 5 (Person Overlap Policy)
- **Verdict**: **PASS**
- **Stable BBox**: `{"left": 277.0, "top": 942.0, "right": 442.0, "bottom": 1040.0}`
- **Rendered / Skipped**: 24 rendered / 1 skipped
- **Audit Findings**: Under the new static safety override policy, 4 minor track leg crossovers are safely overridden and rendered. Crucially, 1 interval (`interval_6` at 6s-7s) where an athlete completely blocks the subtitle region remains correctly skipped with `keep_reject_reason = true_person_overlap_risk` to protect the moving action from clipping.

---

## Final Verdict

- **Static Source Cover MVP v2**: **PASS**
- **sample2 bubble overflow fix**: **PASS**
- **sample5 person-overlap policy**: **PASS**
- **Static Subtitle Burn fallback**: **PASS**
- **Sidecar generation**: **PASS**
- **Dynamic subtitle replacement**: **OUT OF SCOPE**
- **Production Ready**: **NO** (This remains an MVP; target translations require external integration config).