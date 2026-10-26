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
    selectedFile: null,
    selectedPreset: "yolo-4class",
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
    refreshFiles: document.getElementById("refreshFiles"),
    clearFilesBtn: document.getElementById("clearFilesBtn"),

    // Step 2: Preset
    presetCards: document.querySelectorAll(".preset-option-card"),
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
    hideSourceSubCheck: document.getElementById("hideSourceSubCheck"),
    overwriteCheck: document.getElementById("overwriteCheck"),

    // Step 4: Run & CLI Preview
    cliCommandPreview: document.getElementById("cliCommandPreview"),
    runJobBtn: document.getElementById("runJobBtn"),
    runButtonText: document.getElementById("runButtonText"),
    videoFormMessage: document.getElementById("videoFormMessage"),

    // Step 5: Monitoring & Terminal Logs
    jobMonitoringSection: document.getElementById("jobMonitoringSection"),
    jobStageTitle: document.getElementById("jobStageTitle"),
    jobStatusText: document.getElementById("jobStatusText"),
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
      const match = tab.getAttribute("data-nav-tool") === tool;
      tab.classList.toggle("is-active", match);
    });

    if (tool === "video") {
      if (el.videoSection) {
        el.videoSection.classList.add("is-active");
        el.videoSection.removeAttribute("hidden");
      }
      if (el.ttsSection) {
        el.ttsSection.classList.remove("is-active");
        el.ttsSection.setAttribute("hidden", "");
      }
    } else if (tool === "tts") {
      if (el.ttsSection) {
        el.ttsSection.classList.add("is-active");
        el.ttsSection.removeAttribute("hidden");
      }
      if (el.videoSection) {
        el.videoSection.classList.remove("is-active");
        el.videoSection.setAttribute("hidden", "");
      }
    }
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

  // Update CLI Command Preview Box
  function updateCliCommandPreview() {
    if (!el.cliCommandPreview) return;

    const inputPath = state.selectedFile ? state.selectedFile.path : "<chọn-video-đầu-vào>";
    const preset = state.selectedPreset || "yolo-4class";
    const srcLang = el.sourceLanguageSelect ? el.sourceLanguageSelect.value : "vi";
    const tgtLang = el.targetLanguageSelect ? el.targetLanguageSelect.value : "vi";
    const voiceId = el.voiceSelect ? el.voiceSelect.value : "elevenlabs:EXAVITQu4vr4xnSDxMaL";
    const subLayout = el.subtitleLayoutSelect ? el.subtitleLayoutSelect.value : "in-place";
    const device = el.deviceSelect ? el.deviceSelect.value : "auto";
    const profile = el.profileSelect ? el.profileSelect.value : "balanced";
    const subAction = el.subtitleActionSelect ? el.subtitleActionSelect.value : "translate";
    const dubAudio = el.dubAudioCheck ? el.dubAudioCheck.checked : false;
    const extraAction = el.extraTextActionSelect ? el.extraTextActionSelect.value : "preserve";
    const hideSourceSub = el.hideSourceSubCheck && el.hideSourceSubCheck.checked;
    const overwrite = el.overwriteCheck && el.overwriteCheck.checked;

    const parts = [
      "python -m src.interfaces.cli",
      `--input "${inputPath}"`,
      `--preset ${preset}`,
      `--subtitle-action ${subAction}`,
    ];

    if (dubAudio) parts.push("--dub-audio");
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
    el.sourceList.innerHTML = '<div class="empty-state">Đang tải danh sách video dự án…</div>';
    try {
      const res = await api("/api/files");
      state.files = Array.isArray(res) ? res : (Array.isArray(res.files) ? res.files : []);
      renderFileList();
    } catch (err) {
      el.sourceList.innerHTML = `<div class="empty-state" style="color:var(--red);">Lỗi: ${escapeHtml(err.message)}</div>`;
    }
  }

  function renderFileList() {
    if (!el.sourceList) return;
    if (!state.files.length) {
      el.sourceList.innerHTML = '<div class="empty-state">Chưa có video nào. Kéo thả file video vào ô trên để tải lên!</div>';
      return;
    }

    el.sourceList.innerHTML = state.files.map(function (file) {
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

    // Nếu chưa chọn video nào và có danh sách, tự động chọn file đầu tiên
    if (!state.selectedFile && state.files.length > 0) {
      selectVideo(state.files[0]);
    }
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
    if (el.currentMediaMeta) el.currentMediaMeta.textContent = "Vui lòng bấm chọn một file từ danh sách bên dưới";
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

  async function clearAllFiles() {
    if (!confirm("Bạn có chắc chắn muốn dọn dẹp toàn bộ file tải lên và kết quả cũ trong dự án?")) return;
    try {
      const res = await api("/api/files", { method: "DELETE" });
      resetSelectedVideo();
      await loadFileList();
      if (el.videoFormMessage) {
        el.videoFormMessage.textContent = `Đã dọn dẹp ${res.deleted_count || 0} file khỏi dự án.`;
        el.videoFormMessage.className = "yup-msg form-message is-success";
      }
    } catch (err) {
      alert("Lỗi khi dọn dẹp: " + err.message);
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
      await loadFileList();

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

        // Cập nhật câu lệnh CLI
        updateCliCommandPreview();
      });
    });
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
      el.hideSourceSubCheck,
      el.overwriteCheck,
      el.subtitleActionSelect,
      el.dubAudioCheck,
      el.extraTextActionSelect,
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
  function initJobRunner() {
    if (!el.runJobBtn) return;

    el.runJobBtn.addEventListener("click", async function () {
      if (!state.selectedFile) {
        alert("Vui lòng chọn video trước khi bắt đầu.");
        return;
      }

      const resolvedPreset = state.selectedPreset || "yolo-4class";
      const subAction = el.subtitleActionSelect ? el.subtitleActionSelect.value : "translate";
      const dubAudio = el.dubAudioCheck ? el.dubAudioCheck.checked : false;
      const extraAction = el.extraTextActionSelect ? el.extraTextActionSelect.value : "preserve";
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
        source_language: el.sourceLanguageSelect ? el.sourceLanguageSelect.value : "vi",
        target_language: el.targetLanguageSelect ? el.targetLanguageSelect.value : "vi",
        voice_id: el.voiceSelect ? el.voiceSelect.value : "elevenlabs:EXAVITQu4vr4xnSDxMaL",
        subtitle_layout: el.subtitleLayoutSelect ? el.subtitleLayoutSelect.value : "in-place",
        hide_source_subtitle: el.hideSourceSubCheck ? el.hideSourceSubCheck.checked : true,
        device: el.deviceSelect ? el.deviceSelect.value : "auto",
        profile: el.profileSelect ? el.profileSelect.value : "balanced",
        overwrite: el.overwriteCheck ? el.overwriteCheck.checked : true,
      };

      // Cập nhật trạng thái UI
      el.runJobBtn.setAttribute("disabled", "");
      if (el.runButtonText) el.runButtonText.textContent = "⏳ ĐANG CHẠY PIPELINE...";
      if (el.videoStatusBadge) {
        el.videoStatusBadge.textContent = "ĐANG XỬ LÝ";
        el.videoStatusBadge.className = "banner-badge running";
      }
      if (el.videoFormMessage) {
        el.videoFormMessage.textContent = "Đang khởi động tiến trình backend CLI...";
        el.videoFormMessage.className = "yup-msg form-message";
      }

      // Hiện khung monitoring & terminal logs, ẩn khung kết quả cũ
      if (el.jobMonitoringSection) el.jobMonitoringSection.removeAttribute("hidden");
      if (el.jobResultSection) el.jobResultSection.setAttribute("hidden", "");

      // Cuộn êm đến khung monitoring
      if (el.jobMonitoringSection) {
        el.jobMonitoringSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }

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
        startJobPolling(job.id);
      } catch (err) {
        el.runJobBtn.removeAttribute("disabled");
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

  function startJobPolling(jobId) {
    if (state.pollTimer) clearInterval(state.pollTimer);

    state.pollTimer = setInterval(async function () {
      try {
        const job = await api(`/api/jobs/${jobId}`);
        updateJobUI(job);

        if (job.status === "succeeded" || job.status === "failed") {
          clearInterval(state.pollTimer);
          state.pollTimer = null;

          el.runJobBtn.removeAttribute("disabled");
          if (el.runButtonText) el.runButtonText.textContent = "⚡ BẮT ĐẦU CHẠY PIPELINE (RUN CLI)";

          if (job.status === "succeeded") {
            if (el.videoStatusBadge) {
              el.videoStatusBadge.textContent = "HOÀN THÀNH";
              el.videoStatusBadge.className = "banner-badge success";
            }
            if (el.jobStatusText) el.jobStatusText.textContent = "COMPLETED";
            if (el.videoFormMessage) {
              el.videoFormMessage.textContent = "Pipeline CLI đã hoàn tất thành công!";
              el.videoFormMessage.className = "yup-msg form-message is-success";
            }

            // Hiển thị Card kết quả
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
              if (el.resultViewBtn) {
                el.resultViewBtn.href = mediaUrl;
              }
            }

            // Tải lại danh sách file để thấy video kết quả mới trong dự án
            await loadFileList();
          } else {
            if (el.videoStatusBadge) {
              el.videoStatusBadge.textContent = "THẤT BẠI";
              el.videoStatusBadge.className = "banner-badge error";
            }
            if (el.jobStatusText) el.jobStatusText.textContent = "FAILED";
            if (el.videoFormMessage) {
              el.videoFormMessage.textContent = `Thực thi thất bại: ${job.message || "Lỗi không xác định"}`;
              el.videoFormMessage.className = "yup-msg form-message is-error";
            }
          }
        }
      } catch (err) {
        // Polling retry
      }
    }, 800);
  }

  function updateJobUI(job) {
    const progress = Math.min(100, Math.max(0, job.progress || 0));
    if (el.progressFill) el.progressFill.style.width = `${progress}%`;
    if (el.jobProgressPercent) el.jobProgressPercent.textContent = `${Math.round(progress)}%`;
    if (el.jobStageTitle) el.jobStageTitle.textContent = job.message || "Đang xử lý…";

    if (job.detections) {
      if (el.metricSubtitle) el.metricSubtitle.textContent = job.detections.subtitle || 0;
      if (el.metricWatermark) el.metricWatermark.textContent = job.detections.watermark || 0;
      if (el.metricLogo) el.metricLogo.textContent = job.detections.logo || 0;
      if (el.metricOther) el.metricOther.textContent = job.detections.other_text || job.detections.other || 0;
    }

    if (job.logs && Array.isArray(job.logs) && el.jobLogsArea) {
      el.jobLogsArea.textContent = job.logs.join("\n");
      el.jobLogsArea.scrollTop = el.jobLogsArea.scrollHeight;
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

  // Application Initialization
  async function init() {
    initTheme();
    initTabs();
    initUpload();
    initPresets();
    initCliParams();
    initJobRunner();
    initTTS();

    if (el.refreshFiles) {
      el.refreshFiles.addEventListener("click", loadFileList);
    }
    if (el.clearFilesBtn) {
      el.clearFilesBtn.addEventListener("click", clearAllFiles);
    }

    // Health check
    try {
      await api("/api/health");
      setRuntime(true, "Local CLI Backend");
    } catch (_) {
      setRuntime(false, "Server Offline");
    }

    // Load video files and update preview
    await loadFileList();
    updateCliCommandPreview();
  }

  init().catch(function (err) {
    console.error("Lỗi khởi tạo Studio:", err);
  });
})();
