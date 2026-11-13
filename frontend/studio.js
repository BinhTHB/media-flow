/* ==========================================================================
   MediaFlow CLI Studio - GUI CLI Runner Logic
   Tập trung 100% vào việc thực thi CLI pipeline và hiển thị logs thời gian thực.
   ========================================================================== */
(function () {
  "use strict";

  // Application State
  const state = {
    theme: document.documentElement.getAttribute("data-theme") || "dark",
    activeTool: "video",
    files: [],
    showCatalog: false,
    selectedFile: null,
    selectedPreset: "reup",
    activeJobId: null,
    pollTimer: null,
    capabilities: null,
  };

  // DOM Elements cache
  const el = {
    // Theme & Navigation
    themeToggle: document.getElementById("themeToggle"),
    runtimeState: document.getElementById("runtimeState"),
    runtimeDetail: document.getElementById("runtimeDetail"),
    navTabs: document.querySelectorAll("[data-nav-tool]"),
    videoSection: document.getElementById("videoSection"),
    sourceSection: document.getElementById("sourceSection"),
    publishSection: document.getElementById("publishSection"),
    ttsSection: document.getElementById("ttsSection"),

    // Step 1: Input Video
    selectedVideoBanner: document.getElementById("selectedVideoBanner"),
    currentMediaPath: document.getElementById("currentMediaPath"),
    currentMediaMeta: document.getElementById("currentMediaMeta"),
    videoStatusBadge: document.getElementById("videoStatusBadge"),
    sourceVideoPlayer: document.getElementById("sourceVideoPlayer"),
    videoPreviewPlaceholder: document.getElementById("videoPreviewPlaceholder"),
    dropZone: document.getElementById("dropZone"),
    fileInput: document.getElementById("fileInput"),
    uploadProgress: document.getElementById("uploadProgress"),
    sourceList: document.getElementById("sourceList"),
    fileFilter: document.getElementById("fileFilter"),
    refreshFiles: document.getElementById("refreshFiles"),
    clearFilesBtn: document.getElementById("clearFilesBtn"),

    // Step 2: Preset
    presetCards: document.querySelectorAll(".preset-option-card"),
    reupOptions: document.getElementById("reupOptions"),
    genericProcessOptions: document.getElementById("genericProcessOptions"),
    reupVisualCheck: document.getElementById("reupVisualCheck"),
    reupAudioCheck: document.getElementById("reupAudioCheck"),
    reupSpeechCaptionCheck: document.getElementById("reupSpeechCaptionCheck"),
    subtitleActionSelect: document.getElementById("subtitleActionSelect"),
    dubAudioCheck: document.getElementById("dubAudioCheck"),
    extraTextActionSelect: document.getElementById("extraTextActionSelect"),

    // Step 3: CLI Parameters
    sourceLanguageSelect: document.getElementById("sourceLanguageSelect"),
    targetLanguageSelect: document.getElementById("targetLanguageSelect"),
    voiceSelect: document.getElementById("voiceSelect"),
    previewVoiceBtn: document.getElementById("previewVoiceBtn"),
    subtitleLayoutSelect: document.getElementById("subtitleLayoutSelect"),
    deviceSelect: document.getElementById("deviceSelect"),
    profileSelect: document.getElementById("profileSelect"),
    variationSelect: document.getElementById("variationSelect"),
    hideSourceSubCheck: document.getElementById("hideSourceSubCheck"),
    overwriteCheck: document.getElementById("overwriteCheck"),

    // Step 4: Run & CLI Preview
    cliCommandPreview: document.getElementById("cliCommandPreview"),
    runJobBtn: document.getElementById("runJobBtn"),
    runButtonText: document.getElementById("runButtonText"),
    cancelJobBtn: document.getElementById("cancelJobBtn"),
    cancelButtonText: document.getElementById("cancelButtonText"),
    videoFormMessage: document.getElementById("videoFormMessage"),

    // Step 5: Monitoring & Terminal Logs
    jobMonitoringSection: document.getElementById("jobMonitoringSection"),
    jobStageTitle: document.getElementById("jobStageTitle"),
    jobStatusText: document.getElementById("jobStatusText"),
    jobFrameLine: document.getElementById("jobFrameLine"),
    jobProgressPercent: document.getElementById("jobProgressPercent"),
    progressFill: document.getElementById("progressFill"),
    jobMetrics: document.getElementById("jobMetrics"),
    metricSubtitle: document.getElementById("metricSubtitle"),
    metricWatermark: document.getElementById("metricWatermark"),
    metricLogo: document.getElementById("metricLogo"),
    metricOther: document.getElementById("metricOther"),
    jobLogsBox: document.getElementById("jobLogsBox"),
    jobLogsArea: document.getElementById("jobLogsArea"),

    // Step 6: Result Card
    jobResultSection: document.getElementById("jobResultSection"),
    resultFilePath: document.getElementById("resultFilePath"),
    resultFileSize: document.getElementById("resultFileSize"),
    resultDownloadBtn: document.getElementById("resultDownloadBtn"),
    resultViewBtn: document.getElementById("resultViewBtn"),

    // Tab 2: TTS
    ttsScript: document.getElementById("ttsScript"),
    ttsVoiceSelect: document.getElementById("ttsVoiceSelect"),
    charCount: document.getElementById("charCount"),
    playTtsBtn: document.getElementById("playTtsBtn"),
    audioPlayerWrapper: document.getElementById("audioPlayerWrapper"),
    ttsAudioPlayer: document.getElementById("ttsAudioPlayer"),
    ttsMessage: document.getElementById("ttsMessage"),
  };

  // Utility helpers
  function escapeHtml(value) {
    return String(value || "").replace(/[&<>"']/g, function (char) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[char];
    });
  }

  function formatBytes(bytes) {
    if (!bytes || bytes <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return (bytes / Math.pow(1024, i)).toFixed(i > 1 ? 1 : 0) + " " + units[i];
  }

  async function api(path, options) {
    const response = await fetch(path, options);
    let payload = {};
    try {
      payload = await response.json();
    } catch (_) {}
    if (!response.ok) {
      throw new Error(payload.error || `Yêu cầu thất bại (${response.status})`);
    }
    return payload;
  }

  // Theme Management
  function initTheme() {
    const savedTheme = localStorage.getItem("mediaflow_theme");
    const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const initialTheme = savedTheme || (prefersDark ? "dark" : "light");
    setTheme(initialTheme);

    if (el.themeToggle) {
      el.themeToggle.addEventListener("click", function () {
        const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
        const nextTheme = currentTheme === "dark" ? "light" : "dark";
        setTheme(nextTheme);
      });
    }
  }

  function setTheme(theme) {
    state.theme = theme;
    document.documentElement.setAttribute("data-theme", theme);
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
      document.documentElement.classList.remove("light");
    } else {
      document.documentElement.classList.add("light");
      document.documentElement.classList.remove("dark");
    }
    try {
      localStorage.setItem("mediaflow_theme", theme);
    } catch (_) {}
  }

  // Navigation Tab Switching
  function initTabs() {
    el.navTabs.forEach(function (btn) {
      btn.addEventListener("click", function () {
        const tool = btn.getAttribute("data-nav-tool");
        if (tool) switchTool(tool);
      });
    });
  }

  function switchTool(tool) {
    state.activeTool = tool;
    el.navTabs.forEach(function (tab) {
      tab.classList.toggle("is-active", tab.getAttribute("data-nav-tool") === tool);
    });
    const sections = {
      video: el.videoSection,
      sources: el.sourceSection,
      publish: el.publishSection,
      tts: el.ttsSection,
    };
    Object.keys(sections).forEach(function (name) {
      const section = sections[name];
      if (!section) return;
      if (name === tool) {
        section.classList.add("is-active");
        section.removeAttribute("hidden");
      } else {
        section.classList.remove("is-active");
        section.setAttribute("hidden", "");
      }
    });
    if (tool === "publish") loadZernioAccounts();
  }

  // Runtime State
  function setRuntime(online, detail) {
    if (!el.runtimeState) return;
    el.runtimeState.classList.toggle("is-online", online);
    el.runtimeState.classList.toggle("is-offline", !online);
    if (el.runtimeDetail) {
      el.runtimeDetail.textContent = detail || (online ? "Local CLI Backend" : "Server Offline");
    }
  }

  function processChoices() {
    const preset = state.selectedPreset || "reup";
    if (preset === "reup") {
      const visual = el.reupVisualCheck ? el.reupVisualCheck.checked : true;
      const audio = el.reupAudioCheck ? el.reupAudioCheck.checked : true;
      return {
        preset: preset,
        subAction: visual ? "translate" : "preserve",
        dubAudio: audio,
        extraAction: visual ? "blur" : "preserve",
      };
    }
    return {
      preset: preset,
      subAction: el.subtitleActionSelect ? el.subtitleActionSelect.value : "translate",
      dubAudio: el.dubAudioCheck ? el.dubAudioCheck.checked : false,
      extraAction: el.extraTextActionSelect ? el.extraTextActionSelect.value : "preserve",
    };
  }

  function syncReupPanel() {
    const reup = state.selectedPreset === "reup";
    if (el.reupOptions) el.reupOptions.hidden = !reup;
    if (el.genericProcessOptions) el.genericProcessOptions.hidden = reup;
  }

  // Update CLI Command Preview Box
  function updateCliCommandPreview() {
    if (!el.cliCommandPreview) return;

    const inputPath = state.selectedFile ? state.selectedFile.path : "<chọn-video-đầu-vào>";
    const choices = processChoices();
    const preset = choices.preset;
    const srcLang = el.sourceLanguageSelect ? el.sourceLanguageSelect.value : "vi";
    const tgtLang = el.targetLanguageSelect ? el.targetLanguageSelect.value : "vi";
    const voiceId = el.voiceSelect ? el.voiceSelect.value : "elevenlabs:nPczCjzI2devNBz1zQrb";
    const subLayout = el.subtitleLayoutSelect ? el.subtitleLayoutSelect.value : "in-place";
    const device = el.deviceSelect ? el.deviceSelect.value : "auto";
    const profile = el.profileSelect ? el.profileSelect.value : "balanced";
    const subAction = choices.subAction;
    const dubAudio = choices.dubAudio;
    const extraAction = choices.extraAction;
    const hideSourceSub = el.hideSourceSubCheck && el.hideSourceSubCheck.checked;
    const overwrite = el.overwriteCheck && el.overwriteCheck.checked;

    const parts = [
      "python -m src.interfaces.cli",
      `--input "${inputPath}"`,
      `--preset ${preset}`,
      `--subtitle-action ${subAction}`,
    ];

    if (preset === "reup" && !dubAudio) parts.push("--no-dub-audio");
    else if (dubAudio) parts.push("--dub-audio");
    if (preset === "reup" && el.reupSpeechCaptionCheck && !el.reupSpeechCaptionCheck.checked) {
      parts.push("--no-speech-captions");
    }
    parts.push(`--extra-text-action ${extraAction}`);

    if (subAction === "translate" || dubAudio) {
      parts.push(`--source-language ${srcLang}`);
      parts.push(`--target-language ${tgtLang}`);
    }
    if (subAction === "translate") {
      parts.push(`--subtitle-layout ${subLayout}`);
    }
    if (dubAudio) {
      parts.push(`--voice-id ${voiceId}`);
    }

    if (device !== "auto") parts.push(`--device ${device}`);
    if (profile !== "balanced") parts.push(`--profile ${profile}`);
    const variation = el.variationSelect ? el.variationSelect.value : "balanced";
    if (variation !== "balanced") parts.push(`--variation ${variation}`);
    if (subAction === "translate" && hideSourceSub) parts.push("--hide-source-subtitle");
    if (overwrite) parts.push("--overwrite");

    syncFlagVisibility(subAction);

    el.cliCommandPreview.textContent = parts.join(" ");
  }

  // "Ẩn sub gốc" chỉ có ý nghĩa khi bước 2b đang dịch phụ đề
  function syncFlagVisibility(subAction) {
    const flagLabel = document.getElementById("hideSourceSubFlag");
    if (flagLabel) {
      flagLabel.hidden = subAction !== "translate";
    }
  }

  // Video Management & File List
  async function loadFileList() {
    if (!el.sourceList) return;
    state.showCatalog = true;
    el.sourceList.hidden = false;
    if (el.fileFilter) el.fileFilter.hidden = false;
    el.sourceList.innerHTML = '<div class="empty-state">Đang tải danh sách video dự án…</div>';
    try {
      const res = await api("/api/files");
      state.files = Array.isArray(res) ? res : (Array.isArray(res.files) ? res.files : []);
      renderFileList();
    } catch (err) {
      el.sourceList.innerHTML = `<div class="empty-state" style="color:var(--red);">Lỗi: ${escapeHtml(err.message)}</div>`;
    }
  }

  function visibleFiles() {
    const query = el.fileFilter ? el.fileFilter.value.trim().toLowerCase() : "";
    if (!query) return state.files;
    return state.files.filter(function (file) {
      const name = (file.name || "").toLowerCase();
      const path = (file.path || "").toLowerCase();
      return name.indexOf(query) !== -1 || path.indexOf(query) !== -1;
    });
  }

  function renderFileList() {
    if (!el.sourceList) return;
    if (el.fileFilter) el.fileFilter.hidden = !state.showCatalog;
    if (!state.showCatalog) {
      el.sourceList.hidden = true;
      return;
    }
    el.sourceList.hidden = false;
    if (!state.files.length) {
      el.sourceList.innerHTML = '<div class="empty-state">Chưa có video nào. Kéo thả file video vào ô trên để tải lên.</div>';
      return;
    }
    const files = visibleFiles();
    if (!files.length) {
      el.sourceList.innerHTML = '<div class="empty-state">Không có file khớp bộ lọc.</div>';
      return;
    }

    el.sourceList.innerHTML = files.map(function (file) {
      const isSelected = state.selectedFile && state.selectedFile.path === file.path;
      const isSample = file.path.startsWith("test_videos/");
      const isOutput = file.path.startsWith("outputs/");
      const isRemovable = !isSample;

      let badgeClass = "";
      let badgeText = "GỐC";
      if (isSample) {
        badgeClass = "sample";
        badgeText = "MẪU";
      } else if (isOutput) {
        badgeClass = "output";
        badgeText = "KẾT QUẢ";
      }

      return `
        <div class="source-item ${isSelected ? "is-selected" : ""}" data-path="${escapeHtml(file.path)}">
          <div class="source-meta">
            <span class="source-name" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span>
            <span class="source-details">${escapeHtml(file.source || "local")} · ${formatBytes(file.size)}</span>
          </div>
          <div class="source-actions">
            <span class="source-badge ${badgeClass}">[${badgeText}]</span>
            ${isRemovable ? `<button class="source-item-delete" type="button" data-delete="${escapeHtml(file.path)}" title="Xóa file này">✕</button>` : ""}
          </div>
        </div>
      `;
    }).join("");

    el.sourceList.querySelectorAll(".source-item").forEach(function (item) {
      item.addEventListener("click", function () {
        const path = item.getAttribute("data-path");
        const found = state.files.find(function (f) { return f.path === path; });
        if (found) {
          selectVideo(found);
        }
      });
    });

    el.sourceList.querySelectorAll("[data-delete]").forEach(function (btn) {
      btn.addEventListener("click", function (e) {
        e.stopPropagation();
        const p = btn.getAttribute("data-delete");
        deleteFile(p);
      });
    });
  }

  function selectVideo(file) {
    state.selectedFile = file;

    // Highlight trong danh sách
    if (el.sourceList) {
      el.sourceList.querySelectorAll(".source-item").forEach(function (item) {
        const match = item.getAttribute("data-path") === file.path;
        item.classList.toggle("is-selected", match);
      });
    }

    // Cập nhật banner video đã chọn
    if (el.currentMediaPath) {
      el.currentMediaPath.textContent = file.name;
    }
    if (el.currentMediaMeta) {
      el.currentMediaMeta.textContent = `${file.path} · ${formatBytes(file.size)}`;
    }
    if (el.videoStatusBadge) {
      el.videoStatusBadge.textContent = "SẴN SÀNG";
      el.videoStatusBadge.className = "banner-badge";
    }

    // Nạp và hiển thị trình phát video preview
    if (el.sourceVideoPlayer) {
      el.sourceVideoPlayer.src = "/api/media?path=" + encodeURIComponent(file.path);
      el.sourceVideoPlayer.removeAttribute("hidden");
      el.sourceVideoPlayer.load();
    }
    if (el.videoPreviewPlaceholder) {
      el.videoPreviewPlaceholder.setAttribute("hidden", "");
    }

    // Kích hoạt nút chạy pipeline
    if (el.runJobBtn) {
      el.runJobBtn.removeAttribute("disabled");
    }

    // Cập nhật câu lệnh CLI tương đương
    updateCliCommandPreview();
  }

  function resetSelectedVideo() {
    state.selectedFile = null;
    if (el.currentMediaPath) el.currentMediaPath.textContent = "Chưa chọn video";
    if (el.currentMediaMeta) el.currentMediaMeta.textContent = "Tải video mới, hoặc bấm Hiện";
    if (el.videoStatusBadge) {
      el.videoStatusBadge.textContent = "CHỜ VIDEO";
      el.videoStatusBadge.className = "banner-badge";
    }
    if (el.sourceVideoPlayer) {
      el.sourceVideoPlayer.pause();
      el.sourceVideoPlayer.removeAttribute("src");
      el.sourceVideoPlayer.setAttribute("hidden", "");
    }
    if (el.videoPreviewPlaceholder) {
      el.videoPreviewPlaceholder.removeAttribute("hidden");
    }
    if (el.runJobBtn) el.runJobBtn.setAttribute("disabled", "");
    updateCliCommandPreview();
  }

  async function deleteFile(path) {
    if (!path) return;
    const name = path.split("/").pop();
    if (!confirm(`Xóa file "${name}" khỏi lưu trữ dự án?`)) return;
    try {
      await api("/api/files?path=" + encodeURIComponent(path), { method: "DELETE" });
      if (state.selectedFile && state.selectedFile.path === path) {
        resetSelectedVideo();
      }
      await loadFileList();
    } catch (err) {
      alert("Lỗi khi xóa file: " + err.message);
    }
  }

  function clearPickerView() {
    state.showCatalog = false;
    if (el.fileFilter) el.fileFilter.value = "";
    resetSelectedVideo();
    renderFileList();
    if (el.videoFormMessage) {
      el.videoFormMessage.textContent = "Đã ẩn danh sách trên màn hình. File trên đĩa vẫn còn.";
      el.videoFormMessage.className = "yup-msg form-message is-success";
    }
  }

  // Upload Management
  function initUpload() {
    if (!el.dropZone || !el.fileInput) return;

    el.fileInput.addEventListener("change", function () {
      if (el.fileInput.files && el.fileInput.files[0]) {
        uploadFile(el.fileInput.files[0]);
      }
    });

    el.dropZone.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        el.fileInput.click();
      }
    });

    el.dropZone.addEventListener("dragover", function (e) {
      e.preventDefault();
      el.dropZone.classList.add("is-dragover");
    });

    el.dropZone.addEventListener("dragleave", function () {
      el.dropZone.classList.remove("is-dragover");
    });

    el.dropZone.addEventListener("drop", function (e) {
      e.preventDefault();
      el.dropZone.classList.remove("is-dragover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        uploadFile(e.dataTransfer.files[0]);
      }
    });
  }

  async function uploadFile(file) {
    if (!file) return;
    const progressSpan = el.uploadProgress;
    const progressFill = progressSpan ? progressSpan.querySelector("i") : null;

    if (progressSpan) progressSpan.removeAttribute("hidden");
    if (progressFill) progressFill.style.width = "15%";

    try {
      const response = await fetch("/api/uploads", {
        method: "POST",
        headers: {
          "X-Filename": encodeURIComponent(file.name),
          "Content-Type": file.type || "application/octet-stream",
        },
        body: file,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(function () { return {}; });
        throw new Error(errorData.error || `Tải file thất bại (${response.status})`);
      }

      if (progressFill) progressFill.style.width = "100%";
      const result = await response.json();
      if (state.showCatalog) await loadFileList();
      const uploaded = state.files.find(function (f) { return f.name === result.name || f.path === result.path; }) || result;
      selectVideo(uploaded);

      if (el.videoFormMessage) {
        el.videoFormMessage.textContent = `Tải lên thành công: ${result.name}`;
        el.videoFormMessage.className = "yup-msg form-message is-success";
      }
    } catch (err) {
      if (el.videoFormMessage) {
        el.videoFormMessage.textContent = `Lỗi tải lên: ${err.message}`;
        el.videoFormMessage.className = "yup-msg form-message is-error";
      }
    } finally {
      setTimeout(function () {
        if (progressSpan) progressSpan.setAttribute("hidden", "");
        if (progressFill) progressFill.style.width = "0%";
      }, 1000);
    }
  }

  // Preset Selection
  function initPresets() {
    el.presetCards.forEach(function (card) {
      card.addEventListener("click", function () {
        const preset = card.getAttribute("data-preset");
        if (!preset) return;

        state.selectedPreset = preset;

        // Cập nhật class is-active
        el.presetCards.forEach(function (c) {
          c.classList.toggle("is-active", c.getAttribute("data-preset") === preset);
        });

        syncReupPanel();
        updateCliCommandPreview();
      });
    });
    syncReupPanel();
  }

  // CLI Parameters Listeners
  function initCliParams() {
    const inputs = [
      el.sourceLanguageSelect,
      el.targetLanguageSelect,
      el.voiceSelect,
      el.subtitleLayoutSelect,
      el.deviceSelect,
      el.profileSelect,
      el.variationSelect,
      el.hideSourceSubCheck,
      el.overwriteCheck,
      el.subtitleActionSelect,
      el.dubAudioCheck,
      el.extraTextActionSelect,
      el.reupVisualCheck,
      el.reupAudioCheck,
      el.reupSpeechCaptionCheck,
    ];

    inputs.forEach(function (input) {
      if (input) {
        input.addEventListener("change", updateCliCommandPreview);
      }
    });

    // Voice quick test button in Step 3
    if (el.previewVoiceBtn && el.voiceSelect) {
      let previewAudio = null;
      el.previewVoiceBtn.addEventListener("click", async function () {
        const voiceId = el.voiceSelect.value;
        const origText = el.previewVoiceBtn.textContent;
        el.previewVoiceBtn.textContent = "⏳ Đang tạo...";
        el.previewVoiceBtn.setAttribute("disabled", "");

        try {
          const res = await fetch("/api/dubbing/voice-preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              voice_id: voiceId,
              text: "Xin chào! Đây là giọng đọc thử nghiệm của hệ thống MediaFlow.",
            }),
          });

          if (!res.ok) {
            const errData = await res.json().catch(function () { return {}; });
            throw new Error(errData.error || `Mã lỗi ${res.status}`);
          }

          const blob = await res.blob();
          const url = URL.createObjectURL(blob);
          if (previewAudio) previewAudio.pause();
          previewAudio = new Audio(url);
          previewAudio.play();
          el.previewVoiceBtn.textContent = "🔊 Đang phát...";
          previewAudio.onended = function () {
            el.previewVoiceBtn.textContent = origText;
            el.previewVoiceBtn.removeAttribute("disabled");
          };
        } catch (err) {
          alert(`Lỗi nghe thử giọng: ${err.message}`);
          el.previewVoiceBtn.textContent = origText;
          el.previewVoiceBtn.removeAttribute("disabled");
        }
      });
    }
  }

  // Job Execution & Realtime Logs
  function setCancelVisible(visible) {
    if (!el.cancelJobBtn) return;
    if (visible) {
      el.cancelJobBtn.removeAttribute("hidden");
      el.cancelJobBtn.removeAttribute("disabled");
      if (el.cancelButtonText) el.cancelButtonText.textContent = "✕ HỦY JOB";
    } else {
      el.cancelJobBtn.setAttribute("hidden", "");
    }
  }

  async function requestCancel() {
    if (!state.activeJobId) return;
    if (el.cancelJobBtn) el.cancelJobBtn.setAttribute("disabled", "");
    if (el.cancelButtonText) el.cancelButtonText.textContent = "ĐANG HỦY…";
    try {
      await api(`/api/jobs/${encodeURIComponent(state.activeJobId)}`, { method: "DELETE" });
      if (el.videoFormMessage) {
        el.videoFormMessage.textContent = "Đã gửi yêu cầu hủy; job sẽ dừng ở checkpoint kế tiếp.";
        el.videoFormMessage.className = "yup-msg form-message";
      }
    } catch (err) {
      if (el.videoFormMessage) {
        el.videoFormMessage.textContent = `Không hủy được job: ${err.message}`;
        el.videoFormMessage.className = "yup-msg form-message is-error";
      }
      if (el.cancelJobBtn) el.cancelJobBtn.removeAttribute("disabled");
      if (el.cancelButtonText) el.cancelButtonText.textContent = "✕ HỦY JOB";
    }
  }

  function initJobRunner() {
    if (!el.runJobBtn) return;

    if (el.cancelJobBtn) {
      el.cancelJobBtn.addEventListener("click", requestCancel);
    }

    el.runJobBtn.addEventListener("click", async function () {
      if (!state.selectedFile) {
        alert("Vui lòng chọn video trước khi bắt đầu.");
        return;
      }

      const choices = processChoices();
      if (choices.preset === "reup" && choices.subAction === "preserve" && !choices.dubAudio) {
        alert("Chọn xử lý hình, tiếng, hoặc cả hai.");
        return;
      }
      const resolvedPreset = choices.preset;
      const subAction = choices.subAction;
      const dubAudio = choices.dubAudio;
      const extraAction = choices.extraAction;
      const srcName = state.selectedFile.name || "video.mp4";
      const stem = srcName.replace(/\.[^.]+$/, "");
      const outputPreset = dubAudio && subAction === "translate"
        ? `${resolvedPreset}-translate-dub`
        : `${resolvedPreset}-${subAction}${dubAudio ? "-dub" : ""}`;

      const payload = {
        input_path: state.selectedFile.path,
        output_name: `${stem}-${outputPreset}.mp4`,
        preset: resolvedPreset,
        subtitle_action: subAction,
        dub_audio: dubAudio,
        extra_text_action: extraAction,
        create_missing_subtitles: el.reupSpeechCaptionCheck ? el.reupSpeechCaptionCheck.checked : true,
        source_language: el.sourceLanguageSelect ? el.sourceLanguageSelect.value : "vi",
        target_language: el.targetLanguageSelect ? el.targetLanguageSelect.value : "vi",
        voice_id: el.voiceSelect ? el.voiceSelect.value : "elevenlabs:nPczCjzI2devNBz1zQrb",
        subtitle_layout: el.subtitleLayoutSelect ? el.subtitleLayoutSelect.value : "in-place",
        hide_source_subtitle: el.hideSourceSubCheck ? el.hideSourceSubCheck.checked : true,
        device: el.deviceSelect ? el.deviceSelect.value : "auto",
        profile: el.profileSelect ? el.profileSelect.value : "balanced",
        variation_level: el.variationSelect ? el.variationSelect.value : "balanced",
        overwrite: el.overwriteCheck ? el.overwriteCheck.checked : true,
      };

      // Cập nhật trạng thái UI
      el.runJobBtn.setAttribute("disabled", "");
      setCancelVisible(true);
      if (el.runButtonText) el.runButtonText.textContent = "⏳ ĐANG CHẠY PIPELINE...";
      if (el.videoStatusBadge) {
        el.videoStatusBadge.textContent = "ĐANG XỬ LÝ";
        el.videoStatusBadge.className = "banner-badge running";
      }
      if (el.videoFormMessage) {
        el.videoFormMessage.textContent = "Đang khởi động tiến trình backend CLI...";
        el.videoFormMessage.className = "yup-msg form-message";
      }

      if (el.jobResultSection) el.jobResultSection.setAttribute("hidden", "");
      showJobMonitor({ status: "queued", progress: 0, message: "Đang khởi tạo tiến trình CLI..." });

      if (el.jobLogsArea) el.jobLogsArea.textContent = "Đang khởi tạo tiến trình CLI...\n";
      if (el.progressFill) el.progressFill.style.width = "0%";
      if (el.jobProgressPercent) el.jobProgressPercent.textContent = "0%";
      if (el.jobStatusText) el.jobStatusText.textContent = "RUNNING";

      try {
        const job = await api("/api/jobs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        state.activeJobId = job.id;
        rememberJob(job.id);
        startJobPolling(job.id);
      } catch (err) {
        el.runJobBtn.removeAttribute("disabled");
        setCancelVisible(false);
        if (el.runButtonText) el.runButtonText.textContent = "⚡ BẮT ĐẦU CHẠY PIPELINE (RUN CLI)";
        if (el.videoStatusBadge) {
          el.videoStatusBadge.textContent = "LỖI";
          el.videoStatusBadge.className = "banner-badge error";
        }
        if (el.videoFormMessage) {
          el.videoFormMessage.textContent = `Không thể tạo job: ${err.message}`;
          el.videoFormMessage.className = "yup-msg form-message is-error";
        }
      }
    });
  }

  function rememberJob(jobId) {
    try {
      if (jobId) sessionStorage.setItem("mediaflow.activeJobId", jobId);
    } catch (_) {
      // Phiên riêng tư có thể chặn storage. Dock vẫn bám job trong bộ nhớ.
    }
  }

  function rememberedJobId() {
    try {
      return sessionStorage.getItem("mediaflow.activeJobId") || "";
    } catch (_) {
      return "";
    }
  }

  function finishJobView(job) {
    setCancelVisible(false);
    if (el.runJobBtn) el.runJobBtn.removeAttribute("disabled");
    if (el.runButtonText) el.runButtonText.textContent = "⚡ BẮT ĐẦU CHẠY PIPELINE (RUN CLI)";
    if (job.status === "succeeded") {
      if (el.videoStatusBadge) {
        el.videoStatusBadge.textContent = "HOÀN THÀNH";
        el.videoStatusBadge.className = "banner-badge success";
      }
      if (el.videoFormMessage) {
        el.videoFormMessage.textContent = "Pipeline CLI đã hoàn tất thành công!";
        el.videoFormMessage.className = "yup-msg form-message is-success";
      }
      if (el.jobResultSection) {
        el.jobResultSection.removeAttribute("hidden");
        if (el.resultFilePath) {
          el.resultFilePath.textContent = `Tệp kết quả: ${job.output_relative || job.output_path || "outputs/output.mp4"}`;
        }
        if (el.resultFileSize) {
          el.resultFileSize.textContent = `Dung lượng: ${formatBytes(job.output_size || 0)}`;
        }
        const mediaUrl = `/api/media?path=${encodeURIComponent(job.output_relative || job.output_name)}`;
        if (el.resultDownloadBtn) {
          el.resultDownloadBtn.href = mediaUrl;
          el.resultDownloadBtn.download = job.output_name || "output.mp4";
        }
        if (el.resultViewBtn) el.resultViewBtn.href = mediaUrl;
      }
      if (state.showCatalog) loadFileList();
      fillPublishMetadata(job.output_relative || job.output_name);
      return;
    }
    if (job.status === "failed") {
      if (el.videoStatusBadge) {
        el.videoStatusBadge.textContent = "THẤT BẠI";
        el.videoStatusBadge.className = "banner-badge error";
      }
      if (el.videoFormMessage) {
        el.videoFormMessage.textContent = `Thực thi thất bại: ${job.message || "Lỗi không xác định"}`;
        el.videoFormMessage.className = "yup-msg form-message is-error";
      }
    }
  }

  function attachJob(job, poll) {
    if (!job || !job.id) return;
    state.activeJobId = job.id;
    rememberJob(job.id);
    showJobMonitor(job);
    const live = job.status === "running" || job.status === "queued";
    if (live) {
      setCancelVisible(true);
      if (el.runJobBtn) el.runJobBtn.setAttribute("disabled", "");
      if (el.runButtonText) el.runButtonText.textContent = "⏳ ĐANG CHẠY PIPELINE...";
      if (poll) startJobPolling(job.id);
      return;
    }
    finishJobView(job);
  }

  function isStaleStoredRun(job) {
    return Boolean(job && job.persisted && (job.status === "running" || job.status === "queued"));
  }

  function startJobPolling(jobId) {
    if (state.pollTimer) clearInterval(state.pollTimer);
    rememberJob(jobId);

    state.pollTimer = setInterval(async function () {
      try {
        const job = await api(`/api/jobs/${jobId}`);
        updateJobUI(job);

        if (job.status === "succeeded" || job.status === "failed") {
          clearInterval(state.pollTimer);
          state.pollTimer = null;
          finishJobView(job);
        }
      } catch (err) {
        // Polling retry
      }
    }, 800);
  }

  function monitorDetail(job) {
    const message = String(job.message || "");
    if (describeFrames(message)) return message;
    const logs = Array.isArray(job.logs) ? job.logs : [];
    for (let index = logs.length - 1; index >= 0; index -= 1) {
      if (describeFrames(logs[index])) return String(logs[index]);
    }
    return message;
  }

  function describeFrames(message) {
    const match = String(message || "").match(/(\d+)\s*\/\s*(\d+)\s*frames/i);
    if (!match) return "";
    return match[1] + " / " + match[2] + " khung";
  }

  function showJobMonitor(job) {
    if (el.jobMonitoringSection) el.jobMonitoringSection.removeAttribute("hidden");
    if (job) updateJobUI(job);
  }

  function updateJobUI(job) {
    const progress = Math.min(100, Math.max(0, job.progress || 0));
    const live = job.status === "running" || job.status === "queued";
    const statusLabels = {
      running: "ĐANG CHẠY",
      queued: "XẾP HÀNG",
      succeeded: "XONG",
      failed: "LỖI",
    };
    if (el.jobMonitoringSection) el.jobMonitoringSection.classList.toggle("is-live", live);
    if (el.progressFill) el.progressFill.style.width = `${progress}%`;
    if (el.jobProgressPercent) el.jobProgressPercent.textContent = `${Math.round(progress)}%`;
    const detail = monitorDetail(job);
    if (el.jobStageTitle) el.jobStageTitle.textContent = detail || "Đang xử lý…";
    if (el.jobFrameLine) el.jobFrameLine.textContent = describeFrames(detail);
    if (el.jobStatusText && statusLabels[job.status]) {
      el.jobStatusText.textContent = statusLabels[job.status];
    }

    const detections = job.detections || {};
    const counts = {
      subtitle: detections.subtitle || 0,
      watermark: detections.watermark || 0,
      logo: detections.logo || 0,
      other: detections.other_text || detections.other || 0,
    };
    if (el.metricSubtitle) el.metricSubtitle.textContent = counts.subtitle;
    if (el.metricWatermark) el.metricWatermark.textContent = counts.watermark;
    if (el.metricLogo) el.metricLogo.textContent = counts.logo;
    if (el.metricOther) el.metricOther.textContent = counts.other;
    if (el.jobMetrics) {
      const any = counts.subtitle || counts.watermark || counts.logo || counts.other;
      el.jobMetrics.hidden = !any;
    }

    if (el.jobLogsBox) {
      const lines = Array.isArray(job.logs) ? job.logs : [];
      el.jobLogsBox.hidden = lines.length === 0;
      if (lines.length && el.jobLogsArea) {
        el.jobLogsArea.textContent = lines.join("\n");
        el.jobLogsArea.scrollTop = el.jobLogsArea.scrollHeight;
      }
    }
  }

  // Real TTS Voice Preview Tab
  function initTTS() {
    if (el.ttsScript && el.charCount) {
      const updateCount = function () {
        const len = el.ttsScript.value.length;
        el.charCount.textContent = `${len} / 300 ký tự`;
      };
      el.ttsScript.addEventListener("input", updateCount);
      updateCount();
    }

    if (!el.playTtsBtn) return;

    el.playTtsBtn.addEventListener("click", async function () {
      const text = el.ttsScript ? el.ttsScript.value.trim() : "";
      const voiceId = el.ttsVoiceSelect ? el.ttsVoiceSelect.value : "";

      if (!text) {
        if (el.ttsMessage) {
          el.ttsMessage.textContent = "Vui lòng nhập văn bản muốn thử giọng.";
          el.ttsMessage.className = "yup-msg is-error";
        }
        return;
      }

      el.playTtsBtn.setAttribute("disabled", "");
      if (el.ttsMessage) {
        el.ttsMessage.textContent = "Đang tổng hợp giọng nói từ provider AI...";
        el.ttsMessage.className = "yup-msg";
      }

      try {
        const response = await fetch("/api/dubbing/voice-preview", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ voice_id: voiceId, text: text }),
        });

        if (!response.ok) {
          let errDetail = `Mã lỗi ${response.status}`;
          try {
            const errJson = await response.json();
            errDetail = errJson.error || errDetail;
          } catch (_) {}
          throw new Error(errDetail);
        }

        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);

        if (el.ttsAudioPlayer) {
          el.ttsAudioPlayer.src = audioUrl;
          if (el.audioPlayerWrapper) el.audioPlayerWrapper.removeAttribute("hidden");
          await el.ttsAudioPlayer.play();
        }

        if (el.ttsMessage) {
          el.ttsMessage.textContent = "Đã tạo preview thành công!";
          el.ttsMessage.className = "yup-msg form-message is-success";
        }
      } catch (err) {
        if (el.ttsMessage) {
          el.ttsMessage.textContent = `Lỗi thử giọng: ${err.message}`;
          el.ttsMessage.className = "yup-msg form-message is-error";
        }
      } finally {
        el.playTtsBtn.removeAttribute("disabled");
      }
    });
  }

  let zernioAccounts = [];

  function initSources() {
    const button = document.getElementById("sourcePollBtn");
    if (!button) return;
    button.addEventListener("click", async function () {
      const message = document.getElementById("sourcePollMessage");
      const log = document.getElementById("sourcePollLog");
      button.setAttribute("disabled", "");
      try {
        await api("/api/ingest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sources: document.getElementById("sourcesText").value,
            limit: Number(document.getElementById("sourceLimit").value || 10),
            then_process: document.getElementById("sourceThenProcess").checked,
          }),
        });
        if (message) message.textContent = "Đang tải. Có thể mất vài phút nếu bật xử lý reup.";
        const timer = setInterval(async function () {
          const status = await api("/api/ingest");
          if (log) log.textContent = (status.log || []).join("\n") || status.message || "Đang chạy…";
          if (status.status === "running") return;
          clearInterval(timer);
          button.removeAttribute("disabled");
          if (message) message.textContent = status.message || status.status;
        }, 2000);
      } catch (err) {
        if (message) message.textContent = err.message;
        button.removeAttribute("disabled");
      }
    });
  }

  async function loadZernioAccounts() {
    const list = document.getElementById("zernioAccountList");
    if (!list) return;
    const payload = await api("/api/zernio/accounts");
    zernioAccounts = payload.accounts || [];
    if (!zernioAccounts.length) {
      list.innerHTML = "<p>Chưa có API key.</p>";
      return;
    }
    list.innerHTML = zernioAccounts.map(function (row) {
      return `<label class="check-inline" style="display:flex;gap:8px;align-items:center;margin:6px 0;">
        <input type="checkbox" class="zernio-pick" value="${row.id}" checked>
        <span>${row.label} · ${row.platform} · ${row.api_key_masked}</span>
        <button type="button" data-check="${row.id}">Kiểm tra</button>
        <button type="button" data-remove="${row.id}">Xóa</button>
      </label>`;
    }).join("");
    list.querySelectorAll("[data-remove]").forEach(function (btn) {
      btn.addEventListener("click", async function () {
        await api(`/api/zernio/accounts/${btn.getAttribute("data-remove")}`, { method: "DELETE" });
        loadZernioAccounts();
      });
    });
    list.querySelectorAll("[data-check]").forEach(function (btn) {
      btn.addEventListener("click", async function () {
        const message = document.getElementById("zernioMessage");
        try {
          const result = await api(`/api/zernio/accounts/${btn.getAttribute("data-check")}/check`, { method: "POST" });
          const names = (result.matched || []).map(function (item) { return item.name || item.id; }).join(", ");
          if (message) message.textContent = names
            ? `Đã nối ${result.platform}: ${names}`
            : `Key hợp lệ nhưng chưa có tài khoản ${result.platform}.`;
        } catch (err) {
          if (message) message.textContent = err.message;
        }
      });
    });
  }

  async function fillPublishMetadata(videoPath) {
    const pathInput = document.getElementById("publishVideoPath");
    const title = document.getElementById("publishTitle");
    const content = document.getElementById("publishContent");
    const tags = document.getElementById("publishTags");
    if (!videoPath) return;
    if (pathInput) pathInput.value = videoPath;
    try {
      const meta = await api("/api/publish/suggest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_path: videoPath }),
      });
      if (title) title.value = meta.title || "";
      if (content) content.value = meta.description || "";
      if (tags) tags.value = (meta.tags || []).join(", ");
    } catch (_) {
      if (title && !title.value) title.value = String(videoPath).split(/[\\/]/).pop() || "";
    }
  }

  function initPublish() {
    const add = document.getElementById("zernioAddBtn");
    const suggest = document.getElementById("publishSuggestBtn");
    const publish = document.getElementById("zernioPublishBtn");
    if (suggest) {
      suggest.addEventListener("click", function () {
        const pathInput = document.getElementById("publishVideoPath");
        fillPublishMetadata(pathInput ? pathInput.value : "");
      });
    }
    if (add) {
      add.addEventListener("click", async function () {
        const message = document.getElementById("zernioMessage");
        try {
          await api("/api/zernio/accounts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              label: document.getElementById("zernioLabel").value,
              platform: document.getElementById("zernioPlatform").value,
              api_key: document.getElementById("zernioKey").value,
            }),
          });
          document.getElementById("zernioKey").value = "";
          if (message) message.textContent = "Đã lưu key.";
          loadZernioAccounts();
        } catch (err) {
          if (message) message.textContent = err.message;
        }
      });
    }
    if (publish) {
      publish.addEventListener("click", async function () {
        const message = document.getElementById("zernioMessage");
        const ids = Array.from(document.querySelectorAll(".zernio-pick:checked")).map(function (box) {
          return box.value;
        });
        try {
          const result = await api("/api/zernio/publish", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              video_path: document.getElementById("publishVideoPath").value,
              title: document.getElementById("publishTitle").value,
              content: document.getElementById("publishContent").value,
              tags: document.getElementById("publishTags") ? document.getElementById("publishTags").value : "",
              account_ids: ids,
            }),
          });
          const lines = (result.results || []).map(function (row) {
            return row.ok ? `${row.label}: ${row.url || row.post_id || "đã gửi"}` : `${row.label || row.id}: ${row.error}`;
          });
          if (message) message.textContent = lines.join(" · ") || "Không có kết quả.";
        } catch (err) {
          if (message) message.textContent = err.message;
        }
      });
    }
  }

  // Application Initialization
  async function init() {
    initTheme();
    initTabs();
    initUpload();
    initPresets();
    initCliParams();
    initJobRunner();
    initSources();
    initPublish();
    initTTS();

    if (el.refreshFiles) {
      el.refreshFiles.addEventListener("click", function () {
        state.showCatalog = true;
        loadFileList();
      });
    }
    if (el.fileFilter) {
      el.fileFilter.addEventListener("input", renderFileList);
    }
    if (el.clearFilesBtn) {
      el.clearFilesBtn.addEventListener("click", clearPickerView);
    }

    // Health check
    try {
      await api("/api/health");
      setRuntime(true, "Local CLI Backend");
    } catch (_) {
      setRuntime(false, "Server Offline");
    }

    renderFileList();
    updateCliCommandPreview();
    await resumeActiveJob();
  }

  async function resumeActiveJob() {
    try {
      const payload = await api("/api/jobs");
      const jobs = Array.isArray(payload.jobs) ? payload.jobs : [];
      const live = jobs.find(function (job) {
        return !job.persisted && (job.status === "running" || job.status === "queued");
      });
      if (live) {
        attachJob(live, true);
        return;
      }
      const savedId = rememberedJobId();
      if (savedId) {
        try {
          const saved = await api(`/api/jobs/${encodeURIComponent(savedId)}`);
          if (!isStaleStoredRun(saved)) {
            attachJob(saved, true);
            return;
          }
        } catch (_) {
          // Job đã bị dọn. Rơi xuống job mới nhất còn đọc được.
        }
      }
      const latest = jobs.find(function (job) {
        return job && job.id && !isStaleStoredRun(job);
      });
      if (latest) attachJob(latest, false);
    } catch (_) {
      // Không có job thì để màn hình trống.
    }
  }

  init().catch(function (err) {
    console.error("Lỗi khởi tạo Studio:", err);
  });
})();
