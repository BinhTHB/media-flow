# Đánh giá lại dự án media-flow, Phase 8

## Executive summary

`media-flow` hiện có hai nhóm năng lực với mức trưởng thành khác nhau:

1. **Output không phá hủy** đã hoạt động hữu ích: `static_subtitle_burn` tạo được 34 đoạn cho `sample1`, sidecar tạo được cả ASS và SRT.
2. **Output phá hủy** chưa ổn định: `adaptive_render_v5` là baseline tốt nhất đã biết với 8 caption được thay thế, `subtitle_policy_v4` giảm xuống 0, và `subtitle_policy_v6` chỉ phục hồi 3 interval.
3. **Watermark blur** có bằng chứng cross-sample tốt hơn source replacement: `sample1` blur đúng một logo tồn tại 39 frame; `sample3` không blur nhầm person region. Tuy nhiên nó vẫn là thao tác phá hủy và không nên mặc định cho mọi video.

Kết luận định hướng:

- Không rollback toàn bộ safety hoặc static fallback.
- Giữ `adaptive_render_v5` làm **golden compatibility baseline**, không xóa hoặc coi là artifact lỗi thời.
- Rollback về mặt hành vi việc dùng video-level/strict gate để phủ quyết mọi candidate cục bộ đã từng hoạt động.
- Giữ static burn và sidecar thành output độc lập, luôn sẵn có khi target timeline hợp lệ.
- Chỉ đưa thêm safety vào adaptive path theo từng gate, với kiểm thử không-regression so với baseline.
- Không mode nào hiện đủ bằng chứng để gọi toàn hệ thống là Production Ready.

## Phạm vi và bằng chứng audit

### Artifact đã xem

- `output/adaptive_render_v5`
- `output/subtitle_policy_v4`
- `output/subtitle_policy_v6`
- `output/watermark_blur_sample1_v4`
- `output/cross_sample_validation_v2`

### Code đã xem

- `scripts/render_adaptive_subtitles.py`
- `scripts/render_subtitle_policy.py`
- `src/application/services/adaptive_subtitle_region_service.py`
- `src/application/services/subtitle_render_policy.py`
- `src/application/services/watermark_detection_service.py`
- Tests subtitle placement, adaptive rendering, render policy, watermark, benchmark và resource policy.

### Số liệu chính của sample1

| Artifact | Replacement | Static burn | Nhận xét |
|---|---:|---:|---|
| `adaptive_render_v5` | 8 caption | 0 | Baseline tốt nhất đã biết, dùng `expanded_line_bbox`; báo cáo vẫn ghi `production_style_replacement: CONDITIONAL` và leakage chưa được đánh giá |
| `subtitle_policy_v4` | 0 | 34 | Strict/balanced policy chặn replacement nhưng static fallback hữu ích |
| `subtitle_policy_v6` strict | 0 | N/A | Fail-closed đúng cho destructive action |
| `subtitle_policy_v6` adaptive | 3/38 interval | 34 trong full review pack | Phục hồi một phần tại `review_13`, `review_31`, `review_36` |
| `subtitle_policy_v6` sidecar | N/A | 34 timeline item | ASS và SRT đều tồn tại |

`gate_audit_report.json` mới nhất ghi:

- `v5_replaced_count`: 8
- `strict_source_replace_count`: 0
- `adaptive_source_replace_count`: 3
- strict gate: `BLOCK_SOURCE_REPLACE / no_high_confidence_source_replace_intervals`
- adaptive gate: `AUTO_SOURCE_REPLACE`

## Current architecture map

```text
OCR / ASR artifacts
        |
        v
AdaptiveSubtitleRegionService
  - candidate classification
  - geometry and grouping
  - track classification
  - interval/video eligibility
        |
        v
interval_eligibility_report.json
        |
        +-----------------------------+
        |                             |
        v                             v
SubtitleRenderPolicy           render_adaptive_subtitles.py
  - strict decision              - local mask construction
  - adaptive decision            - target text placement
  - static gate                  - FFmpeg render
  - sidecar gate
        |
        v
render_subtitle_policy.py
  - output-mode orchestration
  - adaptive/strict/static/sidecar artifacts
  - review packs and reports

Watermark observations
        |
        v
WatermarkDetectionService
  - temporal grouping
  - persistence and edge scoring
  - person/object/size safety
        |
        v
watermark blur renderer
```

Kiến trúc Application/Infrastructure tổng thể vẫn đúng hướng, nhưng hai script render đang gánh nhiều trách nhiệm cùng lúc: policy orchestration, file I/O, subprocess, report generation và review artifact generation. Đây chưa phải lý do trực tiếp gây regression, nhưng làm khó việc so sánh từng gate và tái sử dụng chính xác baseline.

## Hệ thống hiện làm tốt gì

### 1. Static subtitle burn và sidecar tạo output hữu ích

- `subtitle_policy_v4` và `v6` đều có target timeline 34 đoạn cho `sample1`.
- Static render độc lập với source bbox, source replacement count, dynamic confidence và stable-lane confidence.
- Sidecar ASS/SRT độc lập với destructive replacement.
- Reports đã tách `replacement_gate_result`, `static_render_gate_result`, `sidecar_generation_result`.

### 2. Destructive strict mode fail closed

`strict_source_replace` không mask vùng không chắc chắn và chấp nhận count bằng 0. Đây là hành vi đúng nếu output được ghi nhãn passthrough/safety fallback, không phải source replacement thành công.

### 3. Adaptive path đã được nối lại một phần

`subtitle_policy_v6` cho phép ba floating-caption candidate qua adaptive gate. Điều này chứng minh regression không hoàn toàn do renderer mất khả năng mask; phần lớn nằm ở candidate/gate pipeline.

### 4. Watermark detection có bằng chứng cross-sample hữu ích

- `watermark_blur_sample1_v4`: một logo tại bbox gần góc trái trên, 39 frame, coverage 1.0, watermark score 0.867.
- Blur mode `blur_fill_feathered` có residual reduction 1.0 trên sample1.
- `cross_sample_validation_v2`: sample1 blur một candidate; sample3 blur 0 và ghi nhận đã tránh person-region false blur.
- Watermark service tách persistence, position stability, edge score, subtitle overlap, ASR match và scene attachment.

### 5. Test coverage theo component khá rộng

Tests bao phủ geometry, coordinate transform, track classification, fallback labeling, source mask requirement, resource profile, watermark persistence, person/object risk, static gate independence và sidecar generation.

## Regression analysis

### Regression quan sát được

So với `adaptive_render_v5`, coverage adaptive hiện giảm từ 8 caption xuống 3 interval. `subtitle_policy_v4` từng giảm xuống 0. Vì vậy:

- `adaptive_render_v5` vẫn là baseline chức năng tốt nhất đã biết.
- `subtitle_policy_v6` là phục hồi một phần, chưa tương đương baseline.
- Kết quả 8 của v5 không đồng nghĩa an toàn production: report v5 ghi source leakage là `not_evaluated`.

### Vì sao v5 thay được nhiều hơn

1. **V5 bắt đầu từ tám caption đã có vùng nguồn usable.** Nó xây mask từ `selected_source_bbox`, dùng median/union và `expanded_line_bbox`, thay vì yêu cầu toàn bộ chuỗi candidate/interval/video eligibility mới.
2. **Pipeline mới thay đổi đơn vị đánh giá.** Sample1 hiện được chia thành 38 interval một giây, trong khi v5 dùng tám caption dài hơn. So sánh 8 với 38 không phải cùng mẫu số; cần mapping timestamp chính thức.
3. **Candidate cũ chưa được tái tạo đầy đủ trong report mới.** Nhiều interval mới bị `no_reliable_source_subtitle` hoặc `insufficient_track_evidence`, dù timeline v5 ở cùng vùng thời gian đã có bbox dùng được.
4. **Track confidence bị dùng như điều kiện bắt buộc rộng.** `classify_track` trả `unreliable_track` nếu ít hơn hai điểm hoặc residual score dưới 0.55. Candidate cục bộ có bbox tốt nhưng ít sample có thể bị loại trước render.
5. **Video-level eligibility có thể chặn candidate-level success.** `video_eligibility` đánh dấu skip nếu skip rate trên 40%, reliable presence dưới 30%, hoặc mask area trên 35%. Sample1 đã có skip rate 0.4737 và vì vậy toàn video bị khuyến nghị skip, dù một số interval riêng lẻ vẫn an toàn.
6. **Resource policy bị trộn với quality policy.** Báo cáo balanced ghi “disables expensive dynamic replacement”. Chi phí tính toán không nên tự động được diễn giải thành bằng chứng candidate không an toàn.

### Gate nào quá gắt hoặc đặt sai tầng

#### Video-level skip gate

`skip_rate > 0.40` đang có nguy cơ phủ quyết toàn bộ video. Gate này hợp lý cho auto-enable mặc định, nhưng không hợp lý để chặn mode adaptive được yêu cầu rõ ràng và đã có candidate cục bộ an toàn.

#### Track evidence

`track >= 0.70` và confidence cao phù hợp strict mode, nhưng quá gắt nếu dùng cho mọi floating/bubble candidate. Candidate có coordinate mapping PASS, mask nhỏ, text fit tốt và bbox cục bộ không nên bị loại chỉ vì thiếu nhiều sample trajectory.

#### Scene risk

Scene-text safety là cần thiết, nhưng `candidate_features` cộng centrality vào scene risk. Điều này có thể coi floating bubble ở giữa khung hình là scene text chỉ do vị trí. Cần bằng chứng semantic/temporal bổ sung trước khi block.

#### Person và watermark threshold

Ngưỡng 0.20 trong adaptive policy là fail-closed hợp lý ban đầu, nhưng report hiện chưa chứng minh các score này được hiệu chỉnh trên cùng tập dữ liệu với v5. Không nên nới ngưỡng bằng phỏng đoán; cần replay v5 candidate và đo overlap thật.

#### Mask area

`mask_area_ratio > 0.20` nên tiếp tục là hard review boundary cho adaptive mode cho đến khi benchmark chứng minh khác. V5 chỉ có mask area khoảng 0.046-0.062 trên tám caption, nên nới lên 0.28 không giải thích hoặc khôi phục regression v5.

### Logic bị trộn giữa replacement và fallback

1. `SubtitleRenderPolicy.decide` trả cả `AUTO_SOURCE_REPLACE`, `REVIEW_REQUIRED`, `STATIC_FALLBACK` trong cùng decision space.
2. `ux_decide` có `static_fallback_suppressed_due_to_duplicate_risk`, trong khi explicit static gate lại cho render. Hai lớp policy có thể nói khác nhau.
3. `render_subtitle_policy.py` dùng cùng target timeline và report tổng cho adaptive, static và sidecar; một số legacy fields như `reason_source_replace_not_used` vẫn dựa trên strict count bằng 0 ngay khi adaptive count là 3.
4. Resource profile quyết định có chạy dynamic replacement, trong khi safety gate cũng quyết định candidate có an toàn. Hai nguyên nhân khác nhau bị biểu diễn gần như một kết quả.
5. Passthrough video trong thư mục strict/adaptive có nguy cơ bị hiểu là replacement nếu consumer chỉ nhìn path, không đọc `artifact_type`.

Nguyên tắc cần khóa:

- Replacement decision chỉ trả `REPLACE`, `REVIEW`, `SKIP`.
- Static decision chỉ dựa vào target timeline/timing/output request.
- Sidecar decision chỉ dựa vào target timeline và format generation.
- Resource decision chỉ quyết định có đủ tài nguyên chạy mode hay không, không đổi nhãn safety.

## Giữ, rollback và fallback

### Giữ nguyên

- `adaptive_render_v5` và toàn bộ report/review frames làm golden baseline.
- Local mask, coordinate validation, no-bottom-fallback semantics của adaptive renderer.
- Static render gate độc lập và sidecar fail-open.
- Tách ba gate result trong reports.
- Watermark temporal persistence, edge stability, subtitle overlap và person/object safety.
- Tests geometry, coordinate mapping, static independence, fallback labeling, watermark false-positive prevention.
- `Production Ready: NO`.

### Rollback về hành vi, không rollback toàn bộ code

- Bỏ việc để video-level strict eligibility phủ quyết candidate-level adaptive mode.
- Bỏ coupling giữa balanced resource profile và kết luận “replacement không an toàn”.
- Bỏ mọi report field khiến count strict bằng 0 che mất adaptive count lớn hơn 0.
- Không dùng strict policy làm implementation mặc định bên trong `adaptive_source_replace`.

### Chỉ coi là fallback

- `static_subtitle_burn`: output hữu ích, nhưng không xóa hoặc thay source subtitle.
- Sidecar ASS/SRT: fallback ít rủi ro nhất, không thay đổi pixels.
- Source passthrough: artifact review/safety fallback, không phải replacement.

### Không rollback

- Không rollback static/sidecar chỉ vì adaptive regression.
- Không rollback watermark safety để tăng replacement coverage.
- Không rollback coordinate bounds, clipping checks, explicit mask requirement, scene/person/watermark hard rejects khi có bằng chứng rõ.

## Output mode matrix

| Mode | Mục tiêu | Khi dùng | Production/default | Rủi ro | Verdict hiện tại |
|---|---|---|---|---|---|
| `adaptive_v5_compat` / `adaptive_source_replace` | Thay source subtitle/bubble bằng local mask và target text, không bottom fallback | Opt-in, review/benchmark, khi candidate cục bộ có bbox và safety evidence | Không default; chưa production | Mask sai nội dung cảnh, source leakage, mismatch giữa caption và interval | **CONDITIONAL**: v5 8 caption; v6 3 interval; chưa parity và leakage v5 chưa đánh giá |
| `strict_source_replace` | Chỉ replace khi confidence/safety ở mức cao nhất | Workflow cần fail-closed, chấp nhận count 0 | Không nên là default output duy nhất | Không tạo output dịch hữu ích nếu consumer không lấy static/sidecar; passthrough dễ bị ghi nhãn sai | **CONDITIONAL**: safety PASS, utility FAIL nếu đứng một mình |
| `static_subtitle_burn` | Burn target timeline mà không xóa source | Fallback mặc định khi target timeline hợp lệ | Có thể là default rendered fallback | Duplicate subtitle, che nội dung nếu placement không an toàn | **PASS** về tạo output; **CONDITIONAL** về chất lượng hình ảnh |
| Sidecar `.ass/.srt` | Phụ đề tách rời, không đổi pixels | Luôn tạo khi target timeline hợp lệ | Nên là default artifact | Player support và style khác nhau | **PASS** |
| `watermark_blur` | Blur overlay/logo bền vững | Opt-in hoặc auto chỉ với candidate confidence cao | Không nên mặc định cho mọi video | Xóa nhầm logo hợp lệ, người hoặc scene text | **CONDITIONAL**: sample1/sample3 tốt; cross-sample còn ít auto-positive |

## Recommended next steps

### Bước 1: Đóng băng baseline và định nghĩa phép so sánh

- Giữ nguyên `output/adaptive_render_v5`.
- Tạo manifest golden gồm tám caption, timestamp, source bbox, mask bbox, mask area và review frames.
- Mapping tám caption v5 sang 38 interval mới theo overlap timestamp.
- Không thay threshold cho tới khi biết từng caption v5 bị mất ở stage nào.

### Bước 2: Tạo replay/audit harness trước khi sửa policy

Với từng v5 caption, report:

1. Candidate có được detector mới phát hiện không?
2. Bbox có tương đương theo IoU/center distance không?
3. Bị reject bởi candidate, interval, video, safety hay resource gate?
4. Nếu render lại, mask có source leakage, person overlap, watermark overlap hoặc scene damage không?

Đầu ra cần cho phép so sánh v5 và current bằng cùng timeline/candidate set.

### Bước 3: Khóa contract output mode

- `adaptive_source_replace`: chỉ adaptive replacement, không bottom fallback.
- `strict_source_replace`: strict replacement hoặc explicit passthrough, không claim replacement khi count 0.
- `static_subtitle_burn`: luôn độc lập.
- Sidecars: luôn độc lập.
- Full review pack chỉ là bộ tập hợp artifact, không phải một loại render.

### Bước 4: Gỡ gate sai tầng theo bằng chứng

Ưu tiên thử nghiệm:

1. Cho candidate-level adaptive chạy dù video-level strict result bằng 0.
2. Tách resource skip khỏi safety reject.
3. Cho phép dùng v5-compatible bbox evidence khi track mới thiếu sample nhưng coordinate mapping và local mask vẫn hợp lệ.
4. Giữ hard reject cho clipping, invalid bbox, rõ ràng scene/person/watermark overlap và mask quá lớn.

Mỗi thay đổi chỉ được merge nếu không làm giảm static/sidecar và không tạo unsafe mask trên golden review set.

### Bước 5: Chỉ sau parity mới mở rộng safety

Khi adaptive path đạt parity hợp lý với v5 trên sample1, chạy cross-sample validation. Safety mới được thêm từng gate, mỗi gate phải có:

- fixture positive và negative;
- lý do reject riêng;
- ảnh review;
- delta coverage so với baseline;
- không thay đổi nhãn output.

## Stop-doing list

- Dừng thêm phase/policy mới trước khi có golden replay.
- Dừng dùng tổng số interval khác mẫu số để claim tốt hơn hoặc kém hơn v5.
- Dừng claim PASS cho adaptive chỉ vì output video tồn tại.
- Dừng coi static burn hoặc passthrough là source replacement.
- Dừng dùng resource profile như safety verdict.
- Dừng thay nhiều threshold cùng lúc.
- Dừng gộp strict, adaptive, static và sidecar vào một count hoặc một verdict.
- Dừng auto-blur/auto-mask nếu không có review evidence trên candidate.
- Dừng claim cross-sample subtitle PASS khi auto replacement bằng 0 mà tiêu chí đang đánh giá replacement.
- Không xóa `adaptive_render_v5`.
- Không claim Production Ready.

## Proposed acceptance criteria

### Baseline and audit

- Có manifest mapping đầy đủ 8 caption v5 sang input hiện tại.
- Mỗi caption có kết quả detector, interval gate, video gate, safety gate, resource gate và render result riêng.
- Mọi reject có reason cụ thể, không dùng “policy failed” chung chung.

### Adaptive source replacement

- Không bottom fallback trong adaptive video.
- Không invalid/out-of-frame mask.
- Không replace candidate có person, watermark hoặc scene-text overlap rõ ràng.
- Trên sample1, current adaptive phải khôi phục mọi v5 caption vẫn vượt qua safety audit mới; caption không khôi phục phải có bằng chứng frame-level.
- Count được so sánh trên cùng caption mapping, không so trực tiếp 8 caption với 38 interval.
- Leakage và source-removal quality được đánh giá, không để `not_evaluated`.

### Strict replacement

- Count 0 được chấp nhận.
- Count 0 không chặn static/sidecar.
- Passthrough được ghi nhãn `source_passthrough_safety_fallback`.

### Static and sidecar

- Target timeline hợp lệ luôn tạo ASS và SRT.
- Target timeline hợp lệ luôn tạo static burn làm default rendered output, độc lập với adaptive/strict result.
- Reports không gọi static artifact là replacement.
- Khi duplicate risk tồn tại, vẫn render static nhưng ghi rõ `output_type = static_subtitle_burn_not_source_replacement` và `duplicate_subtitle_risk = true`.
- Placement review không cho static text che vùng nội dung cấm đã định nghĩa.

### Watermark

- Sample1 vẫn blur đúng candidate hiện tại.
- Sample3 vẫn không blur person region.
- Auto-blur chỉ xảy ra khi candidate độc lập với subtitle và scene object.

### Verification

- Unit/integration tests pass.
- Compileall pass.
- Review pack chứa video, reports và contact sheets cho từng mode.
- `Production Ready` vẫn là `NO`.

## Quyết định đề xuất trước Phase tiếp theo

Phase tiếp theo nên là **baseline parity và gate attribution**, không phải thêm feature. Mục tiêu là trả lời có bằng chứng: năm caption v5 chưa được v6 thay thế bị mất ở detector, candidate mapping, track, interval policy, video policy hay safety validation. Chỉ sau khi có mapping này mới quyết định nới gate nào.

### Phạm vi đã được xác nhận

- Mapping/replay audit đủ tám caption v5 trước khi sửa adaptive gate.
- Sau audit, chỉ sửa adaptive gate dựa trên nguyên nhân reject đã chứng minh.
- Candidate-level được override video-level `SKIP` trong `adaptive_source_replace`/`adaptive_v5_compat` khi bbox đúng, mask nhỏ, target text hợp lệ, không đè person/watermark/scene text/menu/object quan trọng và có review frame.
- Override phải report `video_level_gate = SKIP`, `candidate_level_override = true`, `override_reason = local_candidate_safe`.
- Candidate thiếu bằng chứng cục bộ chuyển manual review, không replace.
- Quy tắc override không áp dụng cho `strict_source_replace`.
- Không thay đổi static burn, sidecar, watermark, TTS hoặc strict default.
- Không thêm feature mới.
- Static burn và ASS/SRT luôn được tạo khi target timeline hợp lệ.
- Adaptive chưa đạt parity với tám caption v5 phải nhận verdict `CONDITIONAL` hoặc `FAIL`, không được báo PASS chung chung.
