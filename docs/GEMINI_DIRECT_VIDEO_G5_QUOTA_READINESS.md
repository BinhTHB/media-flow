# Gemini Direct Video G.5 Quota Readiness

## Baseline

- G.5 direct-video: `cfa9a8c`
- Candidate safety: `2d65d5a`
- Quota and resume: `ede05ac`
- Provider/model: `gemini_video_direct` / `gemini-3.1-flash-lite`

G.5.1 proved that a resume manifest can load. It did not prove completed-stage reuse because no stage had completed before the quota block.

## Credential Handling

`ConfigLoader` resolves `GEMINI_API_KEY` only from the environment. It does not load a key from YAML. Local `config.yml` is a placeholder-only, ignored file. The redacted audit is persisted at `debug/g5_credential_audit.json`.

## Quota and 429 Classification

Quota is scoped to the project/model, not individual API keys. The active quota snapshot is persisted at `debug/g5_active_quota_snapshot.json`; it is not an auto-detected quota when active limits are unavailable.

The recorded HTTP 429 includes `generate_content_free_tier_requests`, which is request-dimension evidence for `RATE_LIMIT_RPM`. A zero `Retry-After` is not permission to burst requests. The direct-video configuration carries minimum request spacing, jitter, unknown-429 backoff, and FPS certification TTL.

## FPS Certification

The existing sample2 evidence recorded countTokens values of 1838 at FPS 1 and 9908 at FPS 6. This proves a count-token effect, but the stored run does not prove successful generation usage for both requests. `debug/g5_fps_certification.json` records this as `PARTIAL`.

A reusable certification must match project identity hash, model, API/runtime versions, payload schema, videoMetadata serialization, media resolution, FPS pair, and endpoint behavior. A changed sample video, expired remote URI, or job ID alone does not invalidate semantic certification.

## Preflight, Promotion, and Resume

The preflight is one minimal text-only generate request before Files API upload or sample execution. A failed preflight blocks upload and sample execution as resumable. The current preflight artifact is blocked because no `GEMINI_API_KEY` is present.

Checkpoint promotion must validate video SHA256, model, prompt/schema versions, FPS, media resolution, and artifact hash before a legacy V1 or V2 checkpoint becomes `COMPLETED_PROMOTED`. Completed checkpoints must log the original completion time, reuse time, identity result, artifact hash, and `network_request_skipped=true`.

Requests run sequentially with one in-flight generate request. Same-stage batching remains enabled. Waiting for a quota-safe slot is progress, not a no-progress failure.

## Sample2 Result

No live sample2 request, Files API upload, FPS smoke, sample4, sample5, model switch, local OCR fallback, or TTS was run in this phase because the environment lacks `GEMINI_API_KEY`. No completed checkpoint was available for valid promotion from the stored manifest.

## Known Limitations

- Active RPM, TPM, and RPD require an AI Studio snapshot or user-confirmed environment values.
- FPS generation usage confirmation remains partial.
- Completed-stage reuse requires a future run to finish at least one stage.

## Capability Verdict

`NOT_DETERMINED`

## Production Ready

`NO`
