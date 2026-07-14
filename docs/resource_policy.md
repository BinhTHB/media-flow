# Resource Policy

`balanced` is the default profile. It favors practical output quality without
making the machine unresponsive. FFmpeg uses NVENC for encoding when available,
while GPU decoding remains opt-in because subtitle, watermark, and Python frame
processing commonly require CPU-resident frames.

## Profiles

- `low`: fewer sampled frames, conservative replacement, simple Gaussian
  watermark blur, and no expensive validation or inpainting.
- `balanced`: `blur_fill_feathered`, configured OCR/ASR GPU devices, NVENC when
  available, and skip/review for hard subtitle cases.
- `quality`: more samples and validation passes; optional `inpaint_light` is
  permitted only when explicitly enabled and the watermark occupies less than
  5% of the frame.

Risky replacement is skipped or sent for review rather than guessed.
Full-frame segmentation and heavy temporal inpainting are disabled by default.
GPU acceleration primarily reduces encode/transcode CPU load; not every FFmpeg
filter runs on GPU.
