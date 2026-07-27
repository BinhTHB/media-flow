# Discriminator Metric Analysis — v29 vs v30

## 1. Metric Definition Change

### v29 (`sweep_discriminator_v29.py`)

Uses **intra-frame RGB variance**:

```python
crop = img[y1:y2, x1:x2]
color_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
var = float(np.var(color_crop))
variances.append(var)
avg_variance = float(np.mean(variances))
```

This measures color/texture variation inside individual crops.

### v30 (`sweep_discriminator_r30.py`)

Uses **temporal pixel variance**:

```python
stack = np.stack(crops).astype(np.float32)
var = float(np.mean(np.var(stack, axis=0)))
```

This measures change across aligned crops over a ±0.2-second window (13 crops).

These are different metrics and use different scales. v29's `variance` must not be compared
directly to v30's `temporal_pixel_variance`.

## 2. Contribution Beyond the Band Gate

| Track | y1_norm | Band gate threshold 0.68 | Temporal variance | Discriminator threshold 827.12 | Combined local result |
|---|---:|---|---:|---|---|
| T036 | 0.8594 | pass | 2589.68 | pass | pass |
| T053 | 0.8172 | pass | 386.46 | reject | reject |
| T126 | 0.8250 | pass | 453.22 | reject | reject |

The band gate rejects 25/28 negatives before the discriminator. The discriminator adds exactly
two rejections among those survivors, T053 and T126. T036 passes both local tiers, so it remains
a classifier responsibility. It is incorrect to describe all 10 low-variance GT negatives as
the discriminator's pipeline contribution.

## 3. Safe Margin

```text
positive_min = 961.0449 (T019)
threshold    = 827.12
margin       = 133.9249
```

Reproducer:

```powershell
python -c "import json; d=json.load(open('artifacts/round30/discriminator_42.json')); p=[v['temporal_pixel_variance'] for v in d['per_track'].values() if v['expected_is_subtitle']]; t=827.12; print(f'positive_min={min(p):.4f} threshold={t:.2f} margin={min(p)-t:.4f}')"
# Output: positive_min=961.0449 threshold=827.12 margin=133.9249
```

The threshold has positive margin below the lowest positive. It remains reject-only because
the class ranges overlap: positives [961.04, 4411.30], negatives [17.93, 4078.70].
