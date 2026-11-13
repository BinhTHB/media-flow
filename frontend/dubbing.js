(function () {
  "use strict";

  const state = {
    files: [],
    inputPath: "",
    speakers: [],
    selectedPreset: "yolo-4class",
    subtitleAction: "translate",
    dubAudio: false,
    blurExtra: true,
    subtitleLayout: "temporal",
    hideSourceSub: true,
    useCustomSegments: false,
    selectedSegmentId: null,
    stopPlaybackAt: null,
    resizingState: null,
    isScrubbing: false,
    lastActiveSegId: null,
    isPlayingOutput: false,
    activeJobId: null,
    isProcessing: false,
  };

  let activePreviewAudio = null;

  const el = {
    projectName: document.getElementById("projectName"),
    sourceList: document.getElementById("sourceList"),
    clearFilesBtn: document.getElementById("clearFilesBtn"),
    fileInput: document.getElementById("fileInput"),
    dropZone: document.querySelector(".drop-zone"),
    refreshFiles: document.getElementById("refreshFiles"),
    videoShell: document.getElementById("videoShell"),
    videoPlayer: document.getElementById("videoPlayer"),
    videoEmpty: document.getElementById("videoEmpty"),
    presetGrid: document.getElementById("presetGrid"),
    subtitleOptionsPanel: document.getElementById("subtitleOptionsPanel"),
    layoutButtonGroup: document.getElementById("layoutButtonGroup"),
    checkBlurExtra: document.getElementById("checkBlurExtra"),
    checkHideSourceSub: document.getElementById("checkHideSourceSub"),
    sourceLanguage: document.getElementById("sourceLanguage"),
    targetLanguage: document.getElementById("targetLanguage"),
    speakerMode: document.getElementById("speakerMode"),
    planButton: document.getElementById("planButton"),
    exportButton: document.getElementById("exportButton"),
    sideExportButton: document.getElementById("sideExportButton"),
    sideExportText: document.getElementById("sideExportText"),
    formMessage: document.getElementById("formMessage"),
    speakerCount: document.getElementById("speakerCount"),
    themeToggle: document.getElementById("themeToggle"),
    voiceSpeaker: document.getElementById("voiceSpeaker"),
    voiceSelect: document.getElementById("voiceSelect"),
    previewVoice: document.getElementById("previewVoice"),
    stageList: document.getElementById("stageList"),
    planStatus: document.getElementById("planStatus"),
    timelineState: document.getElementById("timelineState"),
    currentTime: document.getElementById("currentTime"),
    totalTime: document.getElementById("totalTime"),
    playToggle: document.getElementById("playToggle"),
    playhead: document.getElementById("playhead"),
    captionOverlay: document.getElementById("captionOverlay"),
    sourceCaption: document.getElementById("sourceCaption"),
    targetCaption: document.getElementById("targetCaption"),
    timelinePanel: document.querySelector(".timeline-panel"),
    timelineRuler: document.querySelector(".timeline-ruler"),
    tracks: document.getElementById("tracks"),
    segmentToolbar: document.getElementById("segmentToolbar"),
    segBadge: document.getElementById("segBadge"),
    segTimeRange: document.getElementById("segTimeRange"),
    btnTrimStartMinus: document.getElementById("btnTrimStartMinus"),
    btnTrimStartPlus: document.getElementById("btnTrimStartPlus"),
    btnPlaySegOnly: document.getElementById("btnPlaySegOnly"),
    btnTrimEndMinus: document.getElementById("btnTrimEndMinus"),
    btnTrimEndPlus: document.getElementById("btnTrimEndPlus"),
    btnAddSegment: document.getElementById("btnAddSegment"),
    scriptList: document.getElementById("scriptList"),
    charactersList: document.getElementById("charactersList"),
  };

  function escapeHtml(value) {
    return String(value || "").replace(/[&<>"']/g, function (char) {
      return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[char];
    });
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) return "—";
    const units = ["B", "KB", "MB", "GB"];
    const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    return (bytes / Math.pow(1024, index)).toFixed(index ? 1 : 0) + " " + units[index];
  }

  function formatTime(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "00:00";
    const total = Math.floor(seconds);
    const minutes = Math.floor(total / 60);
    const secs = total % 60;
    return String(minutes).padStart(2, "0") + ":" + String(secs).padStart(2, "0");
  }

  async function api(path, options) {
    const response = await fetch(path, options);
    let payload = {};
    try { payload = await response.json(); } catch (_) {}
    if (!response.ok) {
      throw new Error(payload.error || "Yêu cầu thất bại (" + response.status + ").");
    }
    return payload;
  }

  function setMessage(message, isError) {
    if (!el.formMessage) return;
    el.formMessage.textContent = message || "";
    el.formMessage.classList.toggle("is-error", Boolean(isError));
  }

  function updatePlanButton() {
    if (!el.planButton) return;
    el.planButton.disabled = !state.inputPath || Boolean(state.isProcessing);
  }

  function updateExportButtonLabel() {
    let text = "⚡ Bắt đầu xử lý (Run Pipeline)";
    const flowName = state.selectedPreset === "yolo-text" ? "yolo-text" : "yolo-4class";
    const parts = [];
    if (state.subtitleAction === "translate") parts.push("Dịch phụ đề");
    else if (state.subtitleAction === "blur") parts.push("Làm mờ phụ đề");
    if (state.dubAudio) parts.push("Lồng tiếng");
    if (!parts.length) parts.push("Giữ nguyên");
    text = `⚡ ${parts.join(" + ")} (Run ${flowName})`;

    const svgIcon = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg> ';
    if (el.exportButton) el.exportButton.innerHTML = svgIcon + "<span>" + text + "</span>";
    if (el.sideExportText) el.sideExportText.textContent = text;
  }

  function updateExportButtonState() {
    const hasVideo = Boolean(state.inputPath);
    const disabled = !hasVideo || Boolean(state.isProcessing);
    const title = state.isProcessing
      ? "Đang có tiến trình xử lý diễn ra..."
      : (hasVideo ? "Bấm để bắt đầu xử lý video theo pipeline backend CLI" : "Vui lòng chọn video trước.");

    if (el.exportButton) {
      el.exportButton.disabled = disabled;
      el.exportButton.title = title;
    }
    if (el.sideExportButton) {
      el.sideExportButton.disabled = disabled;
      el.sideExportButton.title = title;
      el.sideExportButton.classList.toggle("is-ready-action", hasVideo && !state.isProcessing);
    }
  }

  function renderPresetUI() {
    if (el.presetGrid) {
      el.presetGrid.querySelectorAll(".preset-btn").forEach(function (btn) {
        const p = btn.dataset.preset;
        btn.classList.toggle("is-active", state.selectedPreset === p);
      });
    }

    const featureGrid = document.getElementById("featureOptions");
    if (featureGrid) {
      featureGrid.querySelectorAll(".preset-btn").forEach(function (btn) {
        if (btn.dataset.subtitleAction) {
          btn.classList.toggle("is-active", state.subtitleAction === btn.dataset.subtitleAction);
        } else if (btn.dataset.dubAudio) {
          btn.classList.toggle("is-active", Boolean(state.dubAudio));
        }
      });
    }

    const isTransSub = (state.subtitleAction === "translate");

    if (el.subtitleOptionsPanel) {
      el.subtitleOptionsPanel.style.opacity = isTransSub ? "1" : "0.5";
      el.subtitleOptionsPanel.style.pointerEvents = isTransSub ? "auto" : "none";
    }

    const blurExtraGroup = document.getElementById("blurExtraOptionGroup");
    if (blurExtraGroup) {
      blurExtraGroup.style.display = isTransSub ? "flex" : "none";
    }
    const layoutGroup = document.getElementById("layoutOptionGroup");
    if (layoutGroup) {
      layoutGroup.style.display = isTransSub ? "flex" : "none";
    }
    const hideSubGroup = document.getElementById("hideSourceSubGroup");
    if (hideSubGroup) {
      hideSubGroup.style.display = isTransSub ? "flex" : "none";
    }

    if (el.layoutButtonGroup) {
      el.layoutButtonGroup.querySelectorAll(".sub-opt-btn").forEach(function (btn) {
        const isLayoutActive = (btn.dataset.layout === state.subtitleLayout || (state.subtitleLayout === "in-place" && btn.dataset.layout === "temporal"));
        btn.classList.toggle("is-active", isLayoutActive);
        const indicator = btn.querySelector(".opt-indicator");
        if (indicator) indicator.textContent = isLayoutActive ? "●" : "○";
      });
    }

    if (el.checkBlurExtra) {
      el.checkBlurExtra.checked = Boolean(state.blurExtra);
    }

    if (el.checkHideSourceSub) {
      el.checkHideSourceSub.checked = Boolean(state.hideSourceSub);
    }

    if (el.captionOverlay) {
      el.captionOverlay.classList.toggle("layout-in-place", state.subtitleLayout === "temporal" || state.subtitleLayout === "in-place");
      el.captionOverlay.classList.toggle("layout-bottom", state.subtitleLayout === "bottom");
      el.captionOverlay.classList.toggle("hide-source", Boolean(state.hideSourceSub));
    }
  }

  function selectPreset(presetName) {
    state.selectedPreset = presetName;
    renderPresetUI();
    updateExportButtonLabel();
    updateExportButtonState();
    setMessage("Đã chọn flow nhận diện: " + presetName + ". Bấm 'Bắt đầu xử lý' để chạy pipeline.", false);
  }

  function selectSubtitleAction(action) {
    if (["translate", "blur", "preserve"].indexOf(action) === -1) return;
    state.subtitleAction = action;
    renderPresetUI();
    updateExportButtonLabel();
    updateExportButtonState();
    setMessage("Đã chọn xử lý phụ đề: " + action + ".", false);
  }

  function toggleDubAudio(enabled) {
    state.dubAudio = Boolean(enabled);
    renderPresetUI();
    updateExportButtonLabel();
    updateExportButtonState();
    setMessage(state.dubAudio ? "Đã bật lồng tiếng AI." : "Đã tắt lồng tiếng AI.", false);
  }

  function renderSources() {
    if (!state.files.length) {
      el.sourceList.innerHTML = '<div class="empty-state">Chưa có video trong input/test_videos/outputs.</div>';
      return;
    }
    el.sourceList.innerHTML = state.files.map(function (file) {
      const active = file.path === state.inputPath ? " is-active" : "";
      const isProtected = file.path.startsWith("test_videos/");
      const deleteBtn = isProtected ? "" :
        '<button class="source-item-delete" type="button" title="Xóa video này" data-delete-path="' +
        escapeHtml(file.path) + '">✕</button>';

      let badgeHtml = '<span class="source-badge badge-input">GỐC</span>';
      if (file.path.startsWith("test_videos/")) {
        badgeHtml = '<span class="source-badge badge-sample">MẪU</span>';
      } else if (file.path.startsWith("outputs/")) {
        badgeHtml = '<span class="source-badge badge-output">KẾT QUẢ</span>';
      }

      return '<div class="source-item' + active + '" data-path="' + escapeHtml(file.path) + '">' +
        '<div class="source-main">' +
        '<i class="source-thumb" aria-hidden="true"></i><span><strong>' +
        badgeHtml + escapeHtml(file.name) + '</strong><small>' + formatBytes(file.size) +
        '</small></span></div>' +
        '<div class="source-actions">' + deleteBtn + '</div>' +
        '</div>';
    }).join("");

    el.sourceList.querySelectorAll(".source-main").forEach(function (element) {
      element.addEventListener("click", function (event) {
        event.stopPropagation();
        const parent = element.closest(".source-item");
        const filePath = parent ? parent.dataset.path : "";
        const file = state.files.find(function (item) { return item.path === filePath; });
        if (file) selectVideo(file);
      });
    });

    el.sourceList.querySelectorAll(".source-item-delete").forEach(function (button) {
      button.addEventListener("click", function (event) {
        event.stopPropagation();
        const pathToDelete = button.dataset.deletePath;
        if (pathToDelete) deleteFile(pathToDelete);
      });
    });
  }

  async function deleteFile(filePath) {
    const fileName = filePath.split("/").pop() || filePath;
    if (!confirm('Bạn có chắc chắn muốn xóa video "' + fileName + '"?')) {
      return;
    }
    try {
      await api("/api/files?path=" + encodeURIComponent(filePath), { method: "DELETE" });
      if (state.inputPath === filePath) {
        state.inputPath = "";
        state.plan = null;
        state.activeJobId = null;
        el.projectName.textContent = "Chưa chọn video";
        if (el.videoPlayer) {
          try { el.videoPlayer.pause(); } catch (_) {}
          el.videoPlayer.removeAttribute("src");
          el.videoPlayer.load();
        }
        el.videoShell.classList.remove("has-video");
        updatePlanButton();
        updateExportButtonState();
      }
      await loadFiles();
      setMessage('Đã xóa "' + fileName + '".', false);
    } catch (error) {
      setMessage(error.message, true);
    }
  }

  async function clearAllFiles() {
    if (!confirm("Bạn có chắc chắn muốn xóa tất cả file tải lên và kết quả cũ?\n(Các file mẫu trong thư mục test_videos sẽ được bảo vệ)")) {
      return;
    }
    try {
      const res = await api("/api/files", { method: "DELETE" });
      if (state.inputPath && !state.inputPath.startsWith("test_videos/")) {
        state.inputPath = "";
        state.plan = null;
        state.activeJobId = null;
        el.projectName.textContent = "Chưa chọn video";
        if (el.videoPlayer) {
          try { el.videoPlayer.pause(); } catch (_) {}
          el.videoPlayer.removeAttribute("src");
          el.videoPlayer.load();
        }
        el.videoShell.classList.remove("has-video");
        updatePlanButton();
        updateExportButtonState();
      }
      await loadFiles();
      setMessage("Đã xóa lịch sử lưu trữ (" + (res.deleted_count || 0) + " file đã được dọn dẹp).", false);
    } catch (error) {
      setMessage(error.message, true);
    }
  }

  function selectVideo(file) {
    state.inputPath = file.path;
    state.plan = null;
    state.useCustomSegments = false;
    state.isPlayingOutput = false;
    state.activeJobId = null;
    if (el.projectName) el.projectName.textContent = file.name;
    el.videoPlayer.src = "/api/media?path=" + encodeURIComponent(file.path);
    el.videoPlayer.load();
    el.videoShell.classList.add("has-video");
    if (el.timelineState) el.timelineState.textContent = "chưa phân tích";
    if (el.captionOverlay) el.captionOverlay.hidden = true;
    if (el.scriptList) {
      el.scriptList.innerHTML = '<div class="empty-state">Đã chọn video. Sẵn sàng xử lý theo pipeline backend CLI hoặc bấm "Xem trước kịch bản" để chỉnh sửa.</div>';
    }
    if (el.charactersList) {
      el.charactersList.innerHTML = '<div class="empty-state">Chưa có nhân vật. Hệ thống sẽ nhận diện người nói tự động khi chạy pipeline.</div>';
    }
    setMessage("Đã chọn " + file.name + ". Sẵn sàng bấm 'Bắt đầu xử lý' theo pipeline backend CLI.", false);
    updatePlanButton();
    updateExportButtonState();
    renderSources();
  }

  function seekToSegment(startTime, autoPlay) {
    if (Number.isFinite(startTime) && el.videoPlayer) {
      el.videoPlayer.currentTime = Math.max(0, startTime);
      syncPlaybackUI();
      if (autoPlay) {
        el.videoPlayer.play().catch(function () {});
      }
    }
  }

  function renderSpeakers() {
    const list = (state.plan && state.plan.speakers && state.plan.speakers.length)
      ? state.plan.speakers
      : state.speakers;

    if (el.speakerCount) {
      el.speakerCount.textContent = list.length ? (list.length + " nhân vật") : "Auto";
    }

    if (el.voiceSpeaker) {
      el.voiceSpeaker.innerHTML = list.length
        ? list.map(function (speaker) {
            return '<option value="' + escapeHtml(speaker.id) + '">' + escapeHtml(speaker.label) + "</option>";
          }).join("")
        : '<option value="S01">Speaker 01</option>';
    }
  }

  function renderScriptList(segments) {
    if (!el.scriptList) return;
    if (!segments || !segments.length) {
      el.scriptList.innerHTML = '<div class="empty-state">Không tìm thấy câu thoại nào trong video hoặc video không có tiếng nói.</div>';
      return;
    }

    const speakers = (state.plan && state.plan.speakers && state.plan.speakers.length)
      ? state.plan.speakers
      : [{ id: "S01", label: "Speaker 01" }];

    el.scriptList.innerHTML = segments.map(function (seg, index) {
      const startTime = Number(seg.start || 0);
      const endTime = Number(seg.end || 0);
      const timeStr = formatTime(startTime) + " → " + formatTime(endTime);
      const speakerId = seg.speaker_id || seg.speaker || "S01";

      const speakerOptions = speakers.map(function (spk) {
        const sel = (spk.id === speakerId) ? " selected" : "";
        return '<option value="' + escapeHtml(spk.id) + '"' + sel + '>' + escapeHtml(spk.id) + '</option>';
      }).join("");

      return '<div class="script-item-card" id="script-card-' + escapeHtml(seg.id) + '" data-start="' + startTime + '" data-end="' + endTime + '">' +
        '<div class="script-item-meta">' +
          '<div class="script-meta-left">' +
            '<select class="speaker-select" data-seg-id="' + escapeHtml(seg.id) + '" title="Đổi nhân vật nói câu này">' +
              speakerOptions +
              '<option value="__new__">+ Mới</option>' +
            '</select>' +
            '<span class="script-timecode" title="Bấm để phát câu này">' + timeStr + '</span>' +
          '</div>' +
          '<div class="script-meta-right">' +
            '<button class="play-segment-btn" type="button" title="Phát câu này" data-start="' + startTime + '">▶</button>' +
            '<button class="delete-segment-btn" type="button" title="Xóa câu thoại này" data-seg-id="' + escapeHtml(seg.id) + '">✕</button>' +
          '</div>' +
        '</div>' +
        '<input class="script-text-source" data-seg-id="' + escapeHtml(seg.id) + '" value="' + escapeHtml(seg.source_text || "") + '" placeholder="Câu thoại gốc…">' +
        '<textarea class="script-text-target" data-seg-id="' + escapeHtml(seg.id) + '" placeholder="Nhập bản dịch…">' +
          escapeHtml(seg.target_text || "") +
        '</textarea>' +
      '</div>';
    }).join("");

    el.scriptList.querySelectorAll(".speaker-select").forEach(function (sel) {
      sel.addEventListener("change", function () {
        const segId = sel.dataset.segId;
        const seg = (state.plan && state.plan.segments) ? state.plan.segments.find(function (s) { return s.id === segId; }) : null;
        if (!seg) return;

        let targetSpk = sel.value;
        if (targetSpk === "__new__") {
          const nextIdx = (state.plan.speakers ? state.plan.speakers.length : 1) + 1;
          targetSpk = "S" + String(nextIdx).padStart(2, "0");
          if (!state.plan.speakers) state.plan.speakers = [];
          state.plan.speakers.push({
            id: targetSpk,
            label: "Speaker " + String(nextIdx).padStart(2, "0"),
            short: "S" + nextIdx,
            count: 0,
            voice_id: el.voiceSelect ? el.voiceSelect.value : "elevenlabs:nPczCjzI2devNBz1zQrb",
          });
        }

        seg.speaker_id = targetSpk;
        seg.speaker_label = "Speaker " + targetSpk.replace("S", "");
        state.useCustomSegments = true;

        // Cập nhật lại số câu của từng speaker
        const counts = {};
        state.plan.segments.forEach(function (s) {
          const sp = s.speaker_id || "S01";
          counts[sp] = (counts[sp] || 0) + 1;
        });
        if (state.plan.speakers) {
          state.plan.speakers.forEach(function (spk) {
            spk.count = counts[spk.id] || 0;
          });
        }

        renderSpeakers();
        renderCharactersList(state.plan.speakers);
        renderTimelineTracks(state.plan.segments, el.videoPlayer.duration || state.plan.duration || 10);
        renderScriptList(state.plan.segments);
      });
    });

    el.scriptList.querySelectorAll(".play-segment-btn").forEach(function (btn) {
      btn.addEventListener("click", function (event) {
        event.stopPropagation();
        seekToSegment(Number(btn.dataset.start));
      });
    });

    el.scriptList.querySelectorAll(".delete-segment-btn").forEach(function (btn) {
      btn.addEventListener("click", function (event) {
        event.stopPropagation();
        deleteSegment(btn.dataset.segId);
      });
    });

    el.scriptList.querySelectorAll(".script-timecode").forEach(function (tc) {
      tc.addEventListener("click", function (event) {
        event.stopPropagation();
        const card = tc.closest(".script-item-card");
        if (card) seekToSegment(Number(card.dataset.start), false);
      });
    });

    el.scriptList.querySelectorAll(".script-text-source").forEach(function (input) {
      input.addEventListener("input", function () {
        state.useCustomSegments = true;
        const segId = input.dataset.segId;
        const seg = (state.plan && state.plan.segments) ? state.plan.segments.find(function (s) { return s.id === segId; }) : null;
        if (seg) {
          seg.source_text = input.value;
          if (el.tracks) {
            el.tracks.querySelectorAll('.timeline-dynamic .timeline-segment-block[data-seg-id="' + segId + '"]').forEach(function (block) {
              const label = block.querySelector(".seg-label");
              const trackRow = block.closest(".track-row");
              if (label && trackRow && trackRow.textContent.includes("Phụ đề gốc")) {
                label.textContent = seg.source_text;
              }
            });
          }
          const curTime = (el.videoPlayer && el.videoPlayer.currentTime) || 0;
          if (curTime >= seg.start && curTime <= seg.end && el.sourceCaption) {
            el.sourceCaption.textContent = seg.source_text;
            updateCaptionOverlayPosition(seg);
          }
        }
      });
    });

    el.scriptList.querySelectorAll(".script-text-target").forEach(function (textarea) {
      textarea.addEventListener("input", function () {
        state.useCustomSegments = true;
        const segId = textarea.dataset.segId;
        const seg = (state.plan && state.plan.segments) ? state.plan.segments.find(function (s) { return s.id === segId; }) : null;
        if (seg) {
          seg.target_text = textarea.value;
          if (el.tracks) {
            el.tracks.querySelectorAll('.timeline-dynamic .timeline-segment-block[data-seg-id="' + segId + '"]').forEach(function (block) {
              const label = block.querySelector(".seg-label");
              if (label) {
                const isSpeaker = block.closest(".speaker-track-row");
                const spkPrefix = isSpeaker ? ((seg.speaker_id || "S01") + ": ") : "";
                label.textContent = spkPrefix + (seg.target_text || seg.source_text);
              }
            });
          }
          const curTime = (el.videoPlayer && el.videoPlayer.currentTime) || 0;
          if (curTime >= seg.start && curTime <= seg.end && el.targetCaption) {
            el.targetCaption.textContent = seg.target_text;
            updateCaptionOverlayPosition(seg);
          }
        }
      });
    });
  }

  function addNewSegment() {
    if (!state.inputPath) {
      setMessage("Vui lòng chọn video trước khi thêm câu thoại.", true);
      return;
    }
    const curTime = Math.round(((el.videoPlayer && el.videoPlayer.currentTime) || 0) * 10) / 10;
    const duration = (el.videoPlayer && el.videoPlayer.duration) || (state.plan && state.plan.duration) || 10;
    const startTime = Math.min(curTime, Math.max(0, duration - 1));
    const endTime = Math.min(Math.round((startTime + 2.0) * 10) / 10, duration);

    if (!state.plan) {
      state.plan = {
        ready: true,
        duration: duration,
        speakers: [{ id: "S01", label: "Speaker 01", short: "S1", count: 1, voice_id: (el.voiceSelect ? el.voiceSelect.value : "elevenlabs:nPczCjzI2devNBz1zQrb") }],
        segments: [],
      };
    }
    if (!state.plan.segments) state.plan.segments = [];

    const activeSpk = el.voiceSpeaker ? el.voiceSpeaker.value : "S01";
    const newSegId = "seg_custom_" + Date.now();
    const newSeg = {
      id: newSegId,
      start: startTime,
      end: endTime,
      source_text: "Câu thoại mới",
      target_text: "Bản dịch lồng tiếng mới",
      speaker_id: activeSpk,
      speaker_label: "Speaker " + activeSpk.replace("S", ""),
    };

    state.plan.segments.push(newSeg);
    state.plan.segments.sort(function (a, b) { return a.start - b.start; });
    state.useCustomSegments = true;
    renderPlan(state.plan);
    selectSegment(newSegId, false);
    setMessage("Đã thêm câu thoại mới tại mốc " + formatTime(startTime) + ".", false);
  }

  function deleteSegment(segId) {
    if (!state.plan || !state.plan.segments) return;
    const idx = state.plan.segments.findIndex(function (s) { return s.id === segId; });
    if (idx !== -1) {
      state.plan.segments.splice(idx, 1);
      state.useCustomSegments = state.plan.segments.length > 0;
      if (state.selectedSegmentId === segId) {
        state.selectedSegmentId = null;
        if (el.segmentToolbar) el.segmentToolbar.hidden = true;
      }
      renderPlan(state.plan);
      setMessage("Đã xóa câu thoại.", false);
    }
  }

  function renderCharactersList(speakers) {
    if (!el.charactersList) return;
    if (!speakers || !speakers.length) {
      el.charactersList.innerHTML = '<div class="empty-state">Chưa có nhân vật nào được nhận diện.</div>';
      return;
    }

    const currentVoiceSpk = el.voiceSpeaker ? el.voiceSpeaker.value : "";

    el.charactersList.innerHTML = speakers.map(function (speaker) {
      const short = (speaker.id || "SP").toUpperCase();
      let voiceName = speaker.voice_id || "Chưa gán";
      if (el.voiceSelect) {
        const opt = el.voiceSelect.querySelector('option[value="' + speaker.voice_id + '"]');
        if (opt) voiceName = opt.textContent;
      }
      const isSelected = (speaker.id === currentVoiceSpk);
      return '<div class="character-card' + (isSelected ? " is-selected-speaker" : "") + '" data-speaker-id="' + escapeHtml(speaker.id) + '" title="Bấm để chỉnh giọng cho nhân vật này">' +
        '<div class="character-info">' +
          '<div class="character-avatar">' + escapeHtml(short) + '</div>' +
          '<div class="character-details">' +
            '<strong>' + escapeHtml(speaker.label) + '</strong>' +
            '<small>' + (speaker.count ? (speaker.count + " câu thoại") : "0 câu thoại") + ' · Giọng: ' + escapeHtml(voiceName) + '</small>' +
          '</div>' +
        '</div>' +
        '<span class="mini-status">READY</span>' +
      '</div>';
    }).join("");

    el.charactersList.querySelectorAll(".character-card").forEach(function (card) {
      card.addEventListener("click", function () {
        const spkId = card.dataset.speakerId;
        if (spkId && el.voiceSpeaker) {
          el.voiceSpeaker.value = spkId;
          const spk = speakers.find(function (s) { return s.id === spkId; });
          if (spk && spk.voice_id && el.voiceSelect) {
            el.voiceSelect.value = spk.voice_id;
          }
          renderCharactersList(speakers);
        }
      });
    });
  }

  function selectSegment(segId, autoPlay) {
    if (!state.plan || !state.plan.segments) return;
    const seg = state.plan.segments.find(function (s) { return s.id === segId; });
    if (!seg) return;

    state.isPlayingOutput = false;
    state.selectedSegmentId = segId;

    if (el.segmentToolbar) {
      el.segmentToolbar.hidden = false;
      if (el.segBadge) el.segBadge.textContent = seg.speaker_id || seg.speaker || "S01";
      if (el.segTimeRange) el.segTimeRange.textContent = formatTime(seg.start) + " → " + formatTime(seg.end);
    }

    if (el.tracks) {
      el.tracks.querySelectorAll(".timeline-segment-block").forEach(function (b) {
        b.classList.toggle("is-active", b.dataset.segId === segId);
      });
    }

    if (el.scriptList) {
      el.scriptList.querySelectorAll(".script-item-card").forEach(function (card) {
        const isActive = (card.id === ("script-card-" + segId));
        card.classList.toggle("is-active", isActive);
        if (isActive) {
          card.scrollIntoView({ block: "nearest", behavior: "smooth" });
          const ta = card.querySelector(".script-text-target");
          if (ta) ta.focus();
        }
      });
    }

    seekToSegment(seg.start, Boolean(autoPlay));
  }

  function trimSegment(segId, field, delta) {
    if (!state.plan || !state.plan.segments) return;
    const segs = state.plan.segments;
    const idx = segs.findIndex(function (s) { return s.id === segId; });
    if (idx === -1) return;
    const seg = segs[idx];
    state.useCustomSegments = true;

    const prevSeg = idx > 0 ? segs[idx - 1] : null;
    const nextSeg = idx < segs.length - 1 ? segs[idx + 1] : null;

    const currentVal = Number(seg[field] || 0);
    let newVal = Math.round((currentVal + delta) * 10) / 10;
    newVal = Math.max(0, newVal);

    if (field === "start") {
      const minBound = prevSeg ? Number(prevSeg.end) : 0;
      newVal = Math.max(minBound, Math.min(newVal, Number(seg.end) - 0.2));
    } else if (field === "end") {
      const maxBound = nextSeg ? Number(nextSeg.start) : (el.videoPlayer.duration || state.plan.duration || 9999);
      newVal = Math.min(maxBound, Math.max(Number(seg.start) + 0.2, newVal));
    }
    seg[field] = newVal;

    if (el.segTimeRange && state.selectedSegmentId === segId) {
      el.segTimeRange.textContent = formatTime(seg.start) + " → " + formatTime(seg.end);
    }

    const card = document.getElementById("script-card-" + segId);
    if (card) {
      card.dataset.start = seg.start;
      card.dataset.end = seg.end;
      const tc = card.querySelector(".script-timecode");
      if (tc) tc.textContent = formatTime(seg.start) + " → " + formatTime(seg.end);
    }

    renderTimelineTracks(state.plan.segments, el.videoPlayer.duration || state.plan.duration || 10);
    if (el.videoPlayer) {
      const targetTime = (field === "start") ? Number(seg.start) : Number(seg.end);
      el.videoPlayer.currentTime = Math.max(0, targetTime);
      if (el.currentTime) el.currentTime.textContent = formatTime(targetTime);
      updatePlayhead(targetTime, el.videoPlayer.duration || state.plan.duration || 10);
    }
  }

  function playSegmentOnly(segId) {
    if (!state.plan || !state.plan.segments) return;
    const seg = state.plan.segments.find(function (s) { return s.id === segId; });
    if (!seg) return;

    state.stopPlaybackAt = Number(seg.end);
    if (el.videoPlayer) {
      el.videoPlayer.currentTime = Number(seg.start);
      el.videoPlayer.play().catch(function () {});
    }
  }

  function renderTimelineTracks(segments, totalDuration) {
    if (!el.tracks) return;
    const duration = Math.max(1, totalDuration || el.videoPlayer.duration || 10);

    const ruler = el.timelineRuler;
    if (ruler) {
      const step = duration / 4;
      ruler.innerHTML = [0, 1, 2, 3, 4].map(function (i) {
        return '<span>' + formatTime(i * step) + '</span>';
      }).join("");
    }

    if (!segments || !segments.length) {
      el.tracks.innerHTML =
        '<div class="track-row"><span class="track-label">▣ Video</span><div class="track-lane video-clip-lane"><div class="video-clip-placeholder"><i>Chưa chọn video</i></div></div></div>' +
        '<div class="track-row"><span class="track-label">▤ Phụ đề gốc</span><div class="track-lane subtitle-track"><i>Chưa có transcript</i></div></div>' +
        '<div class="track-row"><span class="track-label">◈ Bản dịch</span><div class="track-lane translation-track"><i>Chưa có bản dịch</i></div></div>' +
        '<div class="track-row speaker-track-row"><span class="track-label speaker-purple">◉ Speaker 01</span><div class="track-lane waveform purple"><i>Chưa có câu thoại</i></div></div>';
      return;
    }

    const speakerMap = {};
    segments.forEach(function (seg) {
      const spk = seg.speaker_id || seg.speaker || "S01";
      if (!speakerMap[spk]) speakerMap[spk] = [];
      speakerMap[spk].push(seg);
    });

    const videoName = escapeHtml((state.inputPath || "Video").split("/").pop());
    const timeSpan = "00:00 → " + formatTime(duration);
    let tracksHtml = '<div class="track-row"><span class="track-label">▣ Video</span><div class="track-lane video-clip-lane">' +
      '<div class="video-clip-block" style="left:0%;width:100%;" title="' + videoName + ' (' + timeSpan + ')">' +
        '<span class="clip-name">▣ ' + videoName + '</span>' +
        '<span class="clip-duration">' + timeSpan + '</span>' +
      '</div>' +
    '</div></div>';

    // Track phụ đề gốc
    tracksHtml += '<div class="track-row"><span class="track-label">▤ Phụ đề gốc</span><div class="track-lane timeline-dynamic">';
    segments.forEach(function (seg) {
      const left = ((seg.start / duration) * 100).toFixed(2);
      const width = (Math.max(1.2, ((seg.end - seg.start) / duration) * 100)).toFixed(2);
      const isSelected = (state.selectedSegmentId === seg.id);
      const tip = (seg.speaker_id || seg.speaker || "S01") + ' [' + formatTime(seg.start) + ' → ' + formatTime(seg.end) + ']\nGốc: ' + (seg.source_text || '');
      tracksHtml += '<div class="timeline-segment-block' + (isSelected ? ' is-active' : '') + '" style="left:' + left + '%;width:' + width + '%;" data-seg-id="' + escapeHtml(seg.id) + '" data-start="' + seg.start + '" data-end="' + seg.end + '" title="' + escapeHtml(tip) + '">' +
        '<span class="resize-handle left" data-handle="left" data-seg-id="' + escapeHtml(seg.id) + '"></span>' +
        '<span class="seg-label">' + escapeHtml(seg.source_text) + '</span>' +
        '<span class="resize-handle right" data-handle="right" data-seg-id="' + escapeHtml(seg.id) + '"></span>' +
      '</div>';
    });
    tracksHtml += '</div></div>';

    // Track bản dịch
    tracksHtml += '<div class="track-row"><span class="track-label">◈ Bản dịch</span><div class="track-lane timeline-dynamic">';
    segments.forEach(function (seg) {
      const left = ((seg.start / duration) * 100).toFixed(2);
      const width = (Math.max(1.2, ((seg.end - seg.start) / duration) * 100)).toFixed(2);
      const isSelected = (state.selectedSegmentId === seg.id);
      const tip = (seg.speaker_id || seg.speaker || "S01") + ' [' + formatTime(seg.start) + ' → ' + formatTime(seg.end) + ']\nDịch: ' + (seg.target_text || '');
      tracksHtml += '<div class="timeline-segment-block' + (isSelected ? ' is-active' : '') + '" style="left:' + left + '%;width:' + width + '%;" data-seg-id="' + escapeHtml(seg.id) + '" data-start="' + seg.start + '" data-end="' + seg.end + '" title="' + escapeHtml(tip) + '">' +
        '<span class="resize-handle left" data-handle="left" data-seg-id="' + escapeHtml(seg.id) + '"></span>' +
        '<span class="seg-label">' + escapeHtml(seg.target_text || seg.source_text) + '</span>' +
        '<span class="resize-handle right" data-handle="right" data-seg-id="' + escapeHtml(seg.id) + '"></span>' +
      '</div>';
    });
    tracksHtml += '</div></div>';

    // Các track cho từng Speaker
    Object.keys(speakerMap).forEach(function (spkId) {
      const spkSegs = speakerMap[spkId];
      tracksHtml += '<div class="track-row speaker-track-row"><span class="track-label">◉ ' + escapeHtml(spkId) + '</span><div class="track-lane timeline-dynamic">';
      spkSegs.forEach(function (seg) {
        const left = ((seg.start / duration) * 100).toFixed(2);
        const width = (Math.max(1.2, ((seg.end - seg.start) / duration) * 100)).toFixed(2);
        const isSelected = (state.selectedSegmentId === seg.id);
        const tip = spkId + ' [' + formatTime(seg.start) + ' → ' + formatTime(seg.end) + ']\n' + (seg.target_text || seg.source_text || '');
        tracksHtml += '<div class="timeline-segment-block' + (isSelected ? ' is-active' : '') + '" style="left:' + left + '%;width:' + width + '%;" data-seg-id="' + escapeHtml(seg.id) + '" data-start="' + seg.start + '" data-end="' + seg.end + '" title="' + escapeHtml(tip) + '">' +
          '<span class="resize-handle left" data-handle="left" data-seg-id="' + escapeHtml(seg.id) + '"></span>' +
          '<span class="seg-label">' + escapeHtml(spkId) + ': ' + escapeHtml(seg.target_text || seg.source_text) + '</span>' +
          '<span class="resize-handle right" data-handle="right" data-seg-id="' + escapeHtml(seg.id) + '"></span>' +
        '</div>';
      });
      tracksHtml += '</div></div>';
    });

    el.tracks.innerHTML = tracksHtml;

    el.tracks.querySelectorAll(".timeline-segment-block").forEach(function (block) {
      block.addEventListener("click", function (e) {
        e.stopPropagation();
        const segId = block.dataset.segId;
        if (segId) {
          selectSegment(segId, false);
        } else {
          seekToSegment(Number(block.dataset.start), false);
        }
      });
    });

    el.tracks.querySelectorAll(".resize-handle").forEach(function (handle) {
      handle.addEventListener("mousedown", function (e) {
        e.stopPropagation();
        const segId = handle.dataset.segId;
        const which = handle.dataset.handle;
        if (!segId || !state.plan || !state.plan.segments) return;
        const seg = state.plan.segments.find(function (s) { return s.id === segId; });
        if (!seg) return;

        state.resizingState = {
          segId: segId,
          handle: which,
          initialX: e.clientX,
          origStart: Number(seg.start),
          origEnd: Number(seg.end),
          duration: duration,
        };
        selectSegment(segId, false);
      });
    });
  }

  function updatePlayhead(currentTime, duration) {
    if (!el.playhead || !el.timelinePanel) return;
    const sampleLane = document.querySelector(".track-lane");
    if (!sampleLane) return;
    const laneRect = sampleLane.getBoundingClientRect();
    const panelRect = el.timelinePanel.getBoundingClientRect();
    const maxDur = Math.max(0.1, duration || el.videoPlayer.duration || 1);
    const ratio = Math.min(1, Math.max(0, currentTime / maxDur));
    const playheadX = (laneRect.left - panelRect.left) + (ratio * laneRect.width);
    el.playhead.style.left = Math.round(playheadX) + "px";
  }

  function seekTimelineFromEvent(e) {
    const sampleLane = document.querySelector(".track-lane");
    if (!sampleLane) return;
    const laneRect = sampleLane.getBoundingClientRect();
    const duration = el.videoPlayer.duration || (state.plan && state.plan.duration) || 0;
    if (!duration || duration <= 0) return;
    const clickX = e.clientX - laneRect.left;
    const ratio = Math.min(1, Math.max(0, clickX / laneRect.width));
    const targetTime = ratio * duration;
    el.videoPlayer.currentTime = targetTime;
    if (el.currentTime) el.currentTime.textContent = formatTime(targetTime);
    updatePlayhead(targetTime, duration);
  }

  function bindTimelineInteraction() {
    const ruler = el.timelineRuler;
    const tracks = el.tracks;
    if (!ruler || !tracks) return;

    function onStart(e) {
      if (e.target.closest("button") || e.target.closest("input") || e.target.closest("textarea") || e.target.closest(".resize-handle")) {
        return;
      }
      state.isScrubbing = true;
      seekTimelineFromEvent(e);
    }

    ruler.addEventListener("mousedown", onStart);
    tracks.addEventListener("mousedown", onStart);

    window.addEventListener("mousemove", function (e) {
      if (state.resizingState) {
        const rs = state.resizingState;
        const sampleLane = document.querySelector(".track-lane");
        if (!sampleLane) return;
        const laneRect = sampleLane.getBoundingClientRect();
        const deltaX = e.clientX - rs.initialX;
        const deltaSeconds = (deltaX / laneRect.width) * rs.duration;
        const segs = (state.plan && state.plan.segments) ? state.plan.segments : [];
        const idx = segs.findIndex(function (s) { return s.id === rs.segId; });
        if (idx === -1) return;
        const seg = segs[idx];
        const prevSeg = idx > 0 ? segs[idx - 1] : null;
        const nextSeg = idx < segs.length - 1 ? segs[idx + 1] : null;

        if (rs.handle === "left") {
          let newStart = Math.round((rs.origStart + deltaSeconds) * 10) / 10;
          const minBound = prevSeg ? Number(prevSeg.end) : 0;
          newStart = Math.max(minBound, Math.min(newStart, rs.origEnd - 0.2));
          seg.start = newStart;
        } else if (rs.handle === "right") {
          let newEnd = Math.round((rs.origEnd + deltaSeconds) * 10) / 10;
          const maxBound = nextSeg ? Number(nextSeg.start) : rs.duration;
          newEnd = Math.min(maxBound, Math.max(rs.origStart + 0.2, newEnd));
          seg.end = newEnd;
        }

        if (el.segTimeRange && state.selectedSegmentId === rs.segId) {
          el.segTimeRange.textContent = formatTime(seg.start) + " → " + formatTime(seg.end);
        }

        // Cập nhật vị trí trực quan cho block đang kéo
        el.tracks.querySelectorAll('.timeline-segment-block[data-seg-id="' + rs.segId + '"]').forEach(function (b) {
          b.style.left = ((seg.start / rs.duration) * 100).toFixed(2) + "%";
          b.style.width = (Math.max(1.2, ((seg.end - seg.start) / rs.duration) * 100)).toFixed(2) + "%";
        });
      } else if (state.isScrubbing) {
        seekTimelineFromEvent(e);
      }
    });

    window.addEventListener("mouseup", function () {
      if (state.resizingState) {
        const rs = state.resizingState;
        state.resizingState = null;
        const seg = (state.plan && state.plan.segments) ? state.plan.segments.find(function (s) { return s.id === rs.segId; }) : null;
        if (seg) {
          const card = document.getElementById("script-card-" + seg.id);
          if (card) {
            card.dataset.start = seg.start;
            card.dataset.end = seg.end;
            const tc = card.querySelector(".script-timecode");
            if (tc) tc.textContent = formatTime(seg.start) + " → " + formatTime(seg.end);
          }
          renderTimelineTracks(state.plan.segments, el.videoPlayer.duration || state.plan.duration || 10);
        }
      }
      state.isScrubbing = false;
    });
  }

  function renderStages(stages) {
    if (!el.stageList || !stages || !stages.length) return;
    el.stageList.innerHTML = stages.map(function (stage, index) {
      const status = String(stage.status || "wait");
      const className = status === "ready" ? "is-ready" : status === "blocked" ? "is-blocked" : status === "running" ? "is-running" : "";
      const label = status.toUpperCase();
      return '<li class="' + className + '"><i>' + (index + 1) + '</i><span>' +
        escapeHtml(stage.label) + '</span><b>' + label + '</b></li>';
    }).join("");

    const hasBlocker = stages.some(function (stage) { return stage.status === "blocked"; });
    if (el.planStatus) {
      el.planStatus.textContent = hasBlocker ? "BLOCKED" : "READY";
      el.planStatus.style.color = hasBlocker ? "var(--yellow)" : "var(--green)";
    }
  }

  function renderPlan(plan) {
    state.plan = plan;
    renderStages(plan.stages);
    renderSpeakers();
    renderScriptList(plan.segments);
    renderCharactersList(plan.speakers);
    renderTimelineTracks(plan.segments, plan.duration || el.videoPlayer.duration || 10);

    const segCount = (plan.segments && plan.segments.length) || 0;
    if (el.timelineState) {
      el.timelineState.textContent = plan.ready
        ? (segCount > 0 ? (segCount + " câu thoại sẵn sàng") : "sẵn sàng chạy")
        : "thiếu provider";
    }

    setMessage(plan.message || "Đã tạo kế hoạch.", !plan.ready);
    updateExportButtonState();
  }

  async function loadFiles() {
    const payload = await api("/api/files");
    state.files = (payload.files || []).filter(function (file) {
      return /\.(mp4|mov|mkv|avi|webm|m4v)$/i.test(file.name);
    });
    renderSources();
  }

  async function uploadFile(file) {
    if (!file) return;
    if (file.size > 4 * 1024 * 1024 * 1024) {
      setMessage("Dung lượng video vượt quá giới hạn 4 GB.", true);
      return;
    }
    setMessage("Đang tải " + file.name + "…", false);
    try {
      const payload = await api("/api/uploads", {
        method: "POST",
        headers: {
          "X-Filename": encodeURIComponent(file.name),
          "Content-Type": file.type || "application/octet-stream"
        },
        body: file,
      });
      await loadFiles();
      const uploaded = state.files.find(function (item) { return item.path === payload.path; });
      if (uploaded) selectVideo(uploaded);
      setMessage("Đã tải video lên input/uploads.", false);
    } catch (error) {
      setMessage(error.message, true);
    }
  }

  function updateStageProgress(stage, progress, status) {
    if (!el.stageList) return;
    const items = el.stageList.querySelectorAll("li");
    if (!items || items.length < 6) return;

    const stageIdxMap = {
      "preflight": 0,
      "ingest": 0,
      "asr": 1,
      "processing": 1,
      "composition": 2,
      "translate": 2,
      "translation": 2,
      "tts": 3,
      "compose": 4,
      "finalizing": 5,
      "mux": 5,
      "completed": 6,
    };

    let activeIdx = stageIdxMap[stage] !== undefined ? stageIdxMap[stage] : 0;
    if (progress >= 100 || status === "succeeded") activeIdx = 6;
    else if (progress >= 90) activeIdx = Math.max(activeIdx, 5);
    else if (progress >= 65) activeIdx = Math.max(activeIdx, 4);
    else if (progress >= 45) activeIdx = Math.max(activeIdx, 3);
    else if (progress >= 25) activeIdx = Math.max(activeIdx, 2);
    else if (progress >= 10) activeIdx = Math.max(activeIdx, 1);

    items.forEach(function (li, idx) {
      const b = li.querySelector("b");
      li.classList.remove("is-ready", "is-running", "is-blocked");
      if (status === "failed") {
        if (idx === activeIdx) {
          li.classList.add("is-blocked");
          if (b) b.textContent = "FAIL";
        }
      } else if (idx < activeIdx) {
        li.classList.add("is-ready");
        if (b) b.textContent = "DONE";
      } else if (idx === activeIdx && activeIdx < 6) {
        li.classList.add("is-running");
        if (b) b.textContent = "RUN";
      } else {
        if (b) b.textContent = "WAIT";
      }
    });
  }

  async function createPlan() {
    if (!state.inputPath) {
      setMessage("Hãy chọn video trước.", true);
      return;
    }
    state.isProcessing = true;
    state.isPlayingOutput = false;
    updatePlanButton();
    updateExportButtonState();

    const originalText = el.planButton.firstElementChild ? el.planButton.firstElementChild.textContent : "";
    if (el.planButton.firstElementChild) {
      el.planButton.firstElementChild.textContent = "Đang phân tích lời thoại & dịch thuật…";
    }
    setMessage("Đang trích xuất audio, nhận diện lời thoại (ASR) và dịch sang ngôn ngữ đích…", false);
    try {
      const plan = await api("/api/dubbing/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input_path: state.inputPath,
          preset: state.selectedPreset || "yolo-4class",
          subtitle_action: state.subtitleAction || "translate",
          dub_audio: Boolean(state.dubAudio),
          source_language: el.sourceLanguage ? el.sourceLanguage.value : "auto",
          target_language: el.targetLanguage ? el.targetLanguage.value : "vi",
          voice_id: el.voiceSelect ? el.voiceSelect.value : "elevenlabs:nPczCjzI2devNBz1zQrb",
          speaker_mode: el.speakerMode ? el.speakerMode.value : "single",
          fit_policy: "fit",
          preserve_ambience: true,
        }),
      });
      renderPlan(plan);
    } catch (error) {
      setMessage("Phân tích tự động gặp sự cố (" + error.message + "). Bạn có thể bấm '+ Thêm câu' để tự nhập kịch bản thoại.", true);
      if (!state.plan) {
        const dur = (el.videoPlayer && el.videoPlayer.duration) || 10;
        state.plan = {
          ready: true,
          duration: dur,
          speakers: [{ id: "S01", label: "Speaker 01", short: "S1", count: 0, voice_id: (el.voiceSelect ? el.voiceSelect.value : "elevenlabs:nPczCjzI2devNBz1zQrb") }],
          segments: [],
          stages: [
            { id: "ingest", label: "Nạp và kiểm tra media", status: "ready" },
            { id: "asr", label: "Nhận diện lời thoại", status: "ready" },
            { id: "translate", label: "Dịch kịch bản", status: "ready" },
            { id: "tts", label: "Tạo giọng đọc AI", status: "ready" },
            { id: "compose", label: "Ghép timeline", status: "ready" },
            { id: "mux", label: "Mux video kết quả", status: "ready" },
          ],
        };
        renderPlan(state.plan);
      }
    } finally {
      state.isProcessing = false;
      updatePlanButton();
      updateExportButtonState();
      if (el.planButton.firstElementChild) {
        el.planButton.firstElementChild.textContent = originalText || "Phân tích & tạo kế hoạch";
      }
    }
  }

  async function pollJob(jobId) {
    let pollCount = 0;
    while (pollCount < 600) {
      pollCount++;
      const job = await api("/api/jobs/" + encodeURIComponent(jobId));
      const progress = Number(job.progress || 0);
      if (el.planStatus) el.planStatus.textContent = String(job.status || "running").toUpperCase();
      if (el.timelineState) el.timelineState.textContent = "job " + progress + "%";
      updateStageProgress(job.stage, progress, job.status);
      setMessage(job.message || ("Đang xử lý " + progress + "%…"), job.status === "failed");

      if (job.status === "succeeded") {
        updateStageProgress("completed", 100, "succeeded");
        if (job.output_relative) {
          if (state.activeJobId === jobId && state.inputPath === job.input_relative) {
            state.isPlayingOutput = true;
            if (el.captionOverlay) {
              el.captionOverlay.hidden = true;
              el.captionOverlay.style.display = "none";
            }
            el.videoPlayer.src = "/api/media?path=" + encodeURIComponent(job.output_relative) + "&t=" + Date.now();
            el.videoPlayer.load();
          }
        }
        await loadFiles();
        return job;
      }
      if (job.status === "failed") {
        updateStageProgress(job.stage, progress, "failed");
        throw new Error(job.message || "Dubbing job thất bại.");
      }
      await new Promise(function (resolve) { setTimeout(resolve, 1000); });
    }
    throw new Error("Quá thời gian chờ (10 phút). Tác vụ đã bị ngắt.");
  }

  async function runDubbing() {
    if (!state.inputPath) {
      setMessage("Hãy chọn video trước khi chạy pipeline.", true);
      return;
    }

    const sourceName = String(state.inputPath).split(/[\\/]/).pop() || "video.mp4";
    const stem = sourceName.replace(/\.[^.]+$/, "");
    const preset = state.selectedPreset || "yolo-4class";
    const outputSuffix = preset + "-" + (state.subtitleAction || "translate") + (state.dubAudio ? "-dub" : "");

    state.isProcessing = true;
    updatePlanButton();
    updateExportButtonState();
    setMessage("Đang khởi tạo tiến trình xử lý video (" + preset + ")...", false);

    try {
      const modeValue = el.speakerMode ? el.speakerMode.value : "single";
      const isMulti = (modeValue === "script" || modeValue === "auto");

      const jobPayload = {
        input_path: state.inputPath,
        output_name: stem + "-" + outputSuffix + ".mp4",
        preset: preset,
        subtitle_action: state.subtitleAction || "translate",
        dub_audio: Boolean(state.dubAudio),
        extra_text_action: state.blurExtra ? "blur" : "preserve",
        subtitle_layout: state.subtitleLayout || "temporal",
        hide_source_subtitle: Boolean(state.hideSourceSub !== false),
        device: "auto",
        overwrite: true,
        source_language: el.sourceLanguage ? el.sourceLanguage.value : "auto",
        target_language: el.targetLanguage ? el.targetLanguage.value : "vi",
        voice_id: el.voiceSelect ? el.voiceSelect.value : "elevenlabs:nPczCjzI2devNBz1zQrb",
        speaker_mode: isMulti ? "auto" : "single",
        preserve_ambience: true,
        mix_original_volume: 0.20,
        asr_provider: "auto",
        dubbing_engine: "quality-auto",
        speaker_voices: {},
        use_custom_segments: Boolean(state.useCustomSegments),
      };

      if (state.useCustomSegments && state.plan && state.plan.segments) {
        jobPayload.segments = state.plan.segments;
      }

      if (isMulti && state.plan && state.plan.speakers) {
        state.plan.speakers.forEach(function (s) {
          if (s.id && s.voice_id) {
            jobPayload.speaker_voices[s.id] = s.voice_id;
          }
        });
      }

      const job = await api("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(jobPayload),
      });

      state.activeJobId = job.id;
      await pollJob(job.id);
      setMessage("Xử lý hoàn tất theo flow CLI! Video kết quả đã sẵn sàng.", false);
    } catch (error) {
      setMessage(error.message, true);
    } finally {
      state.isProcessing = false;
      updatePlanButton();
      updateExportButtonState();
    }
  }

  function getVideoRenderedRect() {
    const video = el.videoPlayer;
    if (!video || !video.videoWidth || !video.videoHeight) return null;
    const cw = video.clientWidth;
    const ch = video.clientHeight;
    if (!cw || !ch) return null;

    const videoRatio = video.videoWidth / video.videoHeight;
    const containerRatio = cw / ch;

    let renderW = cw;
    let renderH = ch;
    let offsetX = 0;
    let offsetY = 0;

    if (containerRatio > videoRatio) {
      renderH = ch;
      renderW = renderH * videoRatio;
      offsetX = (cw - renderW) / 2;
    } else {
      renderW = cw;
      renderH = renderW / videoRatio;
      offsetY = (ch - renderH) / 2;
    }

    return {
      left: offsetX,
      top: offsetY,
      width: renderW,
      height: renderH,
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      containerWidth: cw,
      containerHeight: ch,
    };
  }

  function updateCaptionOverlayPosition(curSeg) {
    if (!el.captionOverlay) return;
    if (state.isPlayingOutput) {
      el.captionOverlay.hidden = true;
      el.captionOverlay.style.display = "none";
      return;
    }
    if (!curSeg && state.plan && state.plan.segments) {
      const curTime = (el.videoPlayer && el.videoPlayer.currentTime) || 0;
      curSeg = state.plan.segments.find(function (s) {
        return curTime >= s.start && curTime <= s.end;
      });
    }
    if (!curSeg) {
      el.captionOverlay.hidden = true;
      el.captionOverlay.style.display = "none";
      return;
    }

    const isTemporalOrInPlace = (state.subtitleLayout === "in-place" || state.subtitleLayout === "temporal");
    const activeBbox = (curSeg && curSeg.bbox) || (state.plan && state.plan.default_subtitle_bbox);
    const hasInPlaceBbox = Boolean(
      isTemporalOrInPlace &&
      activeBbox &&
      activeBbox.frame_width &&
      activeBbox.frame_height &&
      activeBbox.x1 !== undefined
    );

    el.captionOverlay.classList.toggle("layout-in-place", hasInPlaceBbox);
    el.captionOverlay.classList.toggle("layout-bottom", !hasInPlaceBbox);
    el.captionOverlay.classList.toggle("hide-source", Boolean(state.hideSourceSub));

    const rect = getVideoRenderedRect();

    if (hasInPlaceBbox && rect) {
      const bbox = activeBbox;
      const scaleX = rect.width / bbox.frame_width;
      const scaleY = rect.height / bbox.frame_height;

      const rawBoxW = (bbox.x2 - bbox.x1) * scaleX;
      const rawBoxH = (bbox.y2 - bbox.y1) * scaleY;
      const cx = rect.left + (((bbox.x1 + bbox.x2) / 2) * scaleX);
      const cy = rect.top + (((bbox.y1 + bbox.y2) / 2) * scaleY);

      const boxW = Math.min(rect.width - 20, Math.max(rawBoxW + 24, rect.width * 0.28));
      let boxLeft = cx - boxW / 2;
      boxLeft = Math.max(rect.left + 8, Math.min(rect.left + rect.width - boxW - 8, boxLeft));

      const boxH = Math.max(26, rawBoxH + 8);
      let boxTop = cy - boxH / 2;
      boxTop = Math.max(rect.top + 6, Math.min(rect.top + rect.height - boxH - 6, boxTop));

      el.captionOverlay.style.position = "absolute";
      el.captionOverlay.style.left = boxLeft.toFixed(1) + "px";
      el.captionOverlay.style.top = boxTop.toFixed(1) + "px";
      el.captionOverlay.style.width = boxW.toFixed(1) + "px";
      el.captionOverlay.style.minHeight = boxH.toFixed(1) + "px";
      el.captionOverlay.style.bottom = "auto";
      el.captionOverlay.style.right = "auto";
      el.captionOverlay.style.transform = "none";
    } else if (rect) {
      const boxW = Math.min(rect.width * 0.88, 760);
      const boxLeft = rect.left + (rect.width - boxW) / 2;
      const bottomPx = (rect.containerHeight - (rect.top + rect.height)) + Math.max(16, rect.height * 0.04);

      el.captionOverlay.style.position = "absolute";
      el.captionOverlay.style.left = boxLeft.toFixed(1) + "px";
      el.captionOverlay.style.width = boxW.toFixed(1) + "px";
      el.captionOverlay.style.bottom = bottomPx.toFixed(1) + "px";
      el.captionOverlay.style.top = "auto";
      el.captionOverlay.style.right = "auto";
      el.captionOverlay.style.minHeight = "auto";
      el.captionOverlay.style.transform = "none";
    } else {
      el.captionOverlay.style.position = "absolute";
      el.captionOverlay.style.left = "6%";
      el.captionOverlay.style.right = "6%";
      el.captionOverlay.style.bottom = "22px";
      el.captionOverlay.style.top = "auto";
      el.captionOverlay.style.width = "auto";
      el.captionOverlay.style.minHeight = "auto";
      el.captionOverlay.style.transform = "none";
    }
  }

  function syncPlaybackUI() {
    if (!el.videoPlayer) return;
    const duration = el.videoPlayer.duration || 1;
    const curTime = el.videoPlayer.currentTime || 0;
    if (el.currentTime) el.currentTime.textContent = formatTime(curTime);

    // Tự động dừng khi chỉ phát phân đoạn đang chọn (Play segment only)
    if (state.stopPlaybackAt !== null && curTime >= state.stopPlaybackAt) {
      el.videoPlayer.pause();
      state.stopPlaybackAt = null;
    }

    if (!state.isScrubbing) {
      updatePlayhead(curTime, duration);
    }

    // Nếu đang phát video kết quả xuất bản thì ẩn overlay preview
    if (state.isPlayingOutput) {
      if (el.captionOverlay) {
        el.captionOverlay.hidden = true;
        el.captionOverlay.style.display = "none";
      }
      return;
    }

    // Live subtitle overlay & highlight segment
    if (state.plan && state.plan.segments && state.plan.segments.length) {
      const curSeg = state.plan.segments.find(function (s) {
        return curTime >= s.start && curTime <= s.end;
      });

      if (curSeg) {
        if (el.captionOverlay) {
          if (el.sourceCaption) el.sourceCaption.textContent = curSeg.source_text || "";
          if (el.targetCaption) el.targetCaption.textContent = curSeg.target_text || "";
          updateCaptionOverlayPosition(curSeg);
          el.captionOverlay.hidden = false;
          el.captionOverlay.style.display = "";
        }

        if (state.lastActiveSegId !== curSeg.id) {
          state.lastActiveSegId = curSeg.id;

          // Highlight card trong danh sách kịch bản (chỉ cuộn 1 lần khi sang câu mới)
          if (el.scriptList) {
            el.scriptList.querySelectorAll(".script-item-card").forEach(function (card) {
              const isActive = (card.id === ("script-card-" + curSeg.id));
              card.classList.toggle("is-active", isActive);
              if (isActive && !state.isScrubbing) {
                card.scrollIntoView({ block: "nearest", behavior: "smooth" });
              }
            });
          }

          // Highlight timeline block
          if (el.tracks) {
            el.tracks.querySelectorAll(".timeline-segment-block").forEach(function (block) {
              const start = Number(block.dataset.start);
              const isActive = Math.abs(start - curSeg.start) < 0.05;
              block.classList.toggle("is-active", isActive);
            });
          }
        }
      } else {
        if (el.captionOverlay) {
          el.captionOverlay.hidden = true;
          el.captionOverlay.style.display = "none";
        }
        if (state.lastActiveSegId !== null) {
          state.lastActiveSegId = null;
          if (el.scriptList) {
            el.scriptList.querySelectorAll(".script-item-card.is-active").forEach(function (card) {
              card.classList.remove("is-active");
            });
          }
          if (el.tracks) {
            el.tracks.querySelectorAll(".timeline-segment-block.is-active").forEach(function (block) {
              block.classList.remove("is-active");
            });
          }
        }
      }
    }
  }

  function bindPlayer() {
    el.videoPlayer.addEventListener("loadedmetadata", function () {
      if (el.totalTime) el.totalTime.textContent = formatTime(el.videoPlayer.duration);
      if (state.plan && state.plan.segments) {
        renderTimelineTracks(state.plan.segments, el.videoPlayer.duration);
      }
      updatePlayhead(el.videoPlayer.currentTime || 0, el.videoPlayer.duration);
      syncPlaybackUI();
    });

    el.videoPlayer.addEventListener("timeupdate", syncPlaybackUI);
    el.videoPlayer.addEventListener("seeked", syncPlaybackUI);
    el.videoPlayer.addEventListener("seeking", syncPlaybackUI);

    el.videoPlayer.addEventListener("play", function () {
      if (el.playToggle) el.playToggle.textContent = "❚❚";
    });
    el.videoPlayer.addEventListener("pause", function () {
      if (el.playToggle) el.playToggle.textContent = "▶";
      state.stopPlaybackAt = null;
    });
    if (el.playToggle) {
      el.playToggle.addEventListener("click", function () {
        if (el.videoPlayer.paused) el.videoPlayer.play().catch(function () {});
        else el.videoPlayer.pause();
      });
    }
    document.querySelectorAll("[data-seek]").forEach(function (button) {
      button.addEventListener("click", function () {
        el.videoPlayer.currentTime = Math.max(0, (el.videoPlayer.currentTime || 0) + Number(button.dataset.seek));
        syncPlaybackUI();
      });
    });
  }

  function bindTabs() {
    document.querySelectorAll("[data-tab]").forEach(function (button) {
      button.addEventListener("click", function () {
        const tab = button.dataset.tab;
        document.querySelectorAll("[data-tab]").forEach(function (item) {
          item.classList.toggle("is-active", item === button);
        });
        const scriptTab = document.getElementById("scriptTab");
        const charactersTab = document.getElementById("charactersTab");
        if (scriptTab) scriptTab.hidden = (tab !== "script");
        if (charactersTab) charactersTab.hidden = (tab !== "characters");
      });
    });
  }

  function bindControls() {
    if (el.refreshFiles) {
      el.refreshFiles.addEventListener("click", function () {
        loadFiles().catch(function (error) { setMessage(error.message, true); });
      });
    }
    if (el.clearFilesBtn) {
      el.clearFilesBtn.addEventListener("click", clearAllFiles);
    }
    if (el.fileInput) {
      el.fileInput.addEventListener("change", function () { uploadFile(el.fileInput.files[0]); });
    }
    if (el.dropZone) {
      el.dropZone.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          if (el.fileInput) el.fileInput.click();
        }
      });
      el.dropZone.addEventListener("dragover", function (event) { event.preventDefault(); el.dropZone.classList.add("is-dragging"); });
      el.dropZone.addEventListener("dragleave", function () { el.dropZone.classList.remove("is-dragging"); });
      el.dropZone.addEventListener("drop", function (event) {
        event.preventDefault();
        el.dropZone.classList.remove("is-dragging");
        uploadFile(event.dataTransfer.files[0]);
      });
    }
    if (el.planButton) {
      el.planButton.addEventListener("click", createPlan);
    }

    // Preset Selection (CLI backend pipeline)
    if (el.presetGrid) {
      el.presetGrid.querySelectorAll(".preset-btn").forEach(function (btn) {
        btn.addEventListener("click", function () {
          const presetName = btn.dataset.preset;
          if (presetName) selectPreset(presetName);
        });
      });
    }

    // Feature options: subtitle action + audio dubbing toggle
    const featureGrid = document.getElementById("featureOptions");
    if (featureGrid) {
      featureGrid.querySelectorAll(".preset-btn").forEach(function (btn) {
        btn.addEventListener("click", function () {
          if (btn.dataset.subtitleAction) {
            selectSubtitleAction(btn.dataset.subtitleAction);
          } else if (btn.dataset.dubAudio) {
            toggleDubAudio(!state.dubAudio);
          }
        });
      });
    }

    if (el.checkBlurExtra) {
      el.checkBlurExtra.addEventListener("change", function () {
        state.blurExtra = el.checkBlurExtra.checked;
        renderPresetUI();
      });
    }

    // Subtitle Layout & Suppression Selection
    if (el.layoutButtonGroup) {
      el.layoutButtonGroup.querySelectorAll(".sub-opt-btn").forEach(function (btn) {
        btn.addEventListener("click", function () {
          state.subtitleLayout = btn.dataset.layout || "temporal";
          renderPresetUI();
          updateCaptionOverlayPosition();
        });
      });
    }

    if (el.checkHideSourceSub) {
      el.checkHideSourceSub.addEventListener("change", function () {
        state.hideSourceSub = el.checkHideSourceSub.checked;
        renderPresetUI();
        updateCaptionOverlayPosition();
      });
    }

    // Timeline Segment Trim Controls
    if (el.btnTrimStartMinus) {
      el.btnTrimStartMinus.addEventListener("click", function () {
        if (state.selectedSegmentId) trimSegment(state.selectedSegmentId, "start", -0.2);
      });
    }
    if (el.btnTrimStartPlus) {
      el.btnTrimStartPlus.addEventListener("click", function () {
        if (state.selectedSegmentId) trimSegment(state.selectedSegmentId, "start", 0.2);
      });
    }
    if (el.btnTrimEndMinus) {
      el.btnTrimEndMinus.addEventListener("click", function () {
        if (state.selectedSegmentId) trimSegment(state.selectedSegmentId, "end", -0.2);
      });
    }
    if (el.btnTrimEndPlus) {
      el.btnTrimEndPlus.addEventListener("click", function () {
        if (state.selectedSegmentId) trimSegment(state.selectedSegmentId, "end", 0.2);
      });
    }
    if (el.btnPlaySegOnly) {
      el.btnPlaySegOnly.addEventListener("click", function () {
        if (state.selectedSegmentId) playSegmentOnly(state.selectedSegmentId);
      });
    }

    // Speaker - Voice mapping
    if (el.voiceSpeaker) {
      el.voiceSpeaker.addEventListener("change", function () {
        const spkId = el.voiceSpeaker.value;
        const spk = (state.plan && state.plan.speakers) ? state.plan.speakers.find(function (s) { return s.id === spkId; }) : null;
        if (spk && spk.voice_id && el.voiceSelect) {
          el.voiceSelect.value = spk.voice_id;
        }
        if (state.plan && state.plan.speakers) {
          renderCharactersList(state.plan.speakers);
        }
      });
    }

    if (el.voiceSelect) {
      el.voiceSelect.addEventListener("change", function () {
        const spkId = el.voiceSpeaker ? el.voiceSpeaker.value : "S01";
        const spk = (state.plan && state.plan.speakers) ? state.plan.speakers.find(function (s) { return s.id === spkId; }) : null;
        if (spk) {
          spk.voice_id = el.voiceSelect.value;
          renderCharactersList(state.plan.speakers);
        }
      });
    }

    if (el.themeToggle) {
      el.themeToggle.addEventListener("click", function () {
        const current = document.documentElement.getAttribute("data-theme") || "dark";
        const next = current === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        localStorage.setItem("mediaflow_theme", next);
      });
    }
    if (el.previewVoice) {
      el.previewVoice.addEventListener("click", async function () {
        const original = el.previewVoice.textContent;
        el.previewVoice.disabled = true;
        el.previewVoice.textContent = "Đang tạo preview…";
        setMessage("Đang tạo audio preview thật cho voice đã chọn…", false);
        try {
          const response = await fetch("/api/dubbing/voice-preview", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({voice_id: el.voiceSelect.value}),
          });
          if (!response.ok) {
            let message = "Không tạo được voice preview.";
            try {
              const payload = await response.json();
              message = payload.error || message;
            } catch (_) {}
            throw new Error(message);
          }
          if (activePreviewAudio) {
            try {
              activePreviewAudio.pause();
              activePreviewAudio.currentTime = 0;
            } catch (_) {}
            activePreviewAudio = null;
          }
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          const audio = new Audio(url);
          activePreviewAudio = audio;
          audio.addEventListener("ended", function () {
            URL.revokeObjectURL(url);
            if (activePreviewAudio === audio) activePreviewAudio = null;
          }, {once: true});
          audio.addEventListener("error", function () {
            URL.revokeObjectURL(url);
            if (activePreviewAudio === audio) activePreviewAudio = null;
          }, {once: true});
          await audio.play();
          setMessage("Đang phát preview thật. Chọn voice khác rồi bấm lại để so sánh.", false);
        } catch (error) {
          setMessage(error.message, true);
        } finally {
          el.previewVoice.disabled = false;
          el.previewVoice.textContent = original;
        }
      });
    }
    if (el.btnAddSegment) {
      el.btnAddSegment.addEventListener("click", addNewSegment);
    }
    if (el.exportButton) {
      el.exportButton.addEventListener("click", runDubbing);
    }
    if (el.sideExportButton) {
      el.sideExportButton.addEventListener("click", runDubbing);
    }

    // Space shortcut to play/pause video
    window.addEventListener("keydown", function (e) {
      if (e.key === " " || e.code === "Space") {
        const active = document.activeElement;
        const tag = active ? active.tagName : "";
        if (tag === "TEXTAREA" || tag === "INPUT") return;
        e.preventDefault();
        if (el.videoPlayer) {
          if (el.videoPlayer.paused) el.videoPlayer.play().catch(function () {});
          else el.videoPlayer.pause();
        }
      }
    });

    window.addEventListener("resize", function () {
      updateCaptionOverlayPosition();
    });

    bindTimelineInteraction();
  }

  async function init() {
    bindControls();
    bindPlayer();
    bindTabs();
    renderPresetUI();
    renderSpeakers();
    updateExportButtonLabel();
    updateExportButtonState();
    await loadFiles();
  }

  init().catch(function (error) { setMessage(error.message, true); });
}());


