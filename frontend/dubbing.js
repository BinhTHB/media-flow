(function () {
  "use strict";

  const state = {
    files: [],
    inputPath: "",
    speakers: [],
    plan: null,
    capabilities: null,
  };

  const el = {
    runtime: document.getElementById("runtimeState"),
    projectName: document.getElementById("projectName"),
    sourceList: document.getElementById("sourceList"),
    fileInput: document.getElementById("fileInput"),
    dropZone: document.querySelector(".drop-zone"),
    refreshFiles: document.getElementById("refreshFiles"),
    videoShell: document.getElementById("videoShell"),
    videoPlayer: document.getElementById("videoPlayer"),
    videoEmpty: document.getElementById("videoEmpty"),
    sourceLanguage: document.getElementById("sourceLanguage"),
    targetLanguage: document.getElementById("targetLanguage"),
    preserveAmbience: document.getElementById("preserveAmbience"),
    fitPolicy: document.getElementById("fitPolicy"),
    planButton: document.getElementById("planButton"),
    exportButton: document.getElementById("exportButton"),
    formMessage: document.getElementById("formMessage"),
    capabilityBadge: document.getElementById("capabilityBadge"),
    speakerList: document.getElementById("speakerList"),
    speakerCount: document.getElementById("speakerCount"),
    speakerHint: document.getElementById("speakerHint"),
    addSpeaker: document.getElementById("addSpeaker"),
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
    pipelineNote: document.getElementById("pipelineNote"),
    captionOverlay: document.getElementById("captionOverlay"),
  };

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, function (char) {
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
    el.formMessage.textContent = message || "";
    el.formMessage.classList.toggle("is-error", Boolean(isError));
  }

  function updatePlanButton() {
    el.planButton.disabled = !state.inputPath;
  }

  function setRuntime(online) {
    el.runtime.classList.toggle("is-online", online);
    el.runtime.classList.toggle("is-offline", !online);
    el.runtime.lastChild.textContent = online ? " Runtime sẵn sàng" : " Mất kết nối";
  }

  function renderSources() {
    if (!state.files.length) {
      el.sourceList.innerHTML = '<div class="empty-state">Chưa có video trong input/test_videos/outputs.</div>';
      return;
    }
    el.sourceList.innerHTML = state.files.map(function (file) {
      const active = file.path === state.inputPath ? " is-active" : "";
      return '<button class="source-item' + active + '" type="button" data-path="' +
        escapeHtml(file.path) + '">' +
        '<i class="source-thumb" aria-hidden="true"></i><span><strong>' +
        escapeHtml(file.name) + '</strong><small>' + formatBytes(file.size) +
        '</small></span></button>';
    }).join("");
    el.sourceList.querySelectorAll("[data-path]").forEach(function (button) {
      button.addEventListener("click", function () {
        const file = state.files.find(function (item) { return item.path === button.dataset.path; });
        if (file) selectVideo(file);
      });
    });
  }

  function selectVideo(file) {
    state.inputPath = file.path;
    state.plan = null;
    el.projectName.textContent = file.name;
    el.videoPlayer.src = "/api/media?path=" + encodeURIComponent(file.path);
    el.videoPlayer.load();
    el.videoShell.classList.add("has-video");
    el.timelineState.textContent = "chưa phân tích";
    el.captionOverlay.hidden = true;
    setMessage("Đã chọn " + file.name + ". Có thể lập kế hoạch dubbing.", false);
    updatePlanButton();
    renderSources();
  }

  function renderSpeakers() {
    el.speakerCount.textContent = String(state.speakers.length);
    el.voiceSpeaker.innerHTML = state.speakers.length
      ? state.speakers.map(function (speaker) {
          return '<option value="' + escapeHtml(speaker.id) + '">' + escapeHtml(speaker.label) + "</option>";
        }).join("")
      : '<option value="S01">Speaker 01</option>';
    if (!state.speakers.length) {
      el.speakerList.innerHTML = '<div class="empty-state">Chưa có dữ liệu speaker. Kết quả thật sẽ xuất hiện sau khi có ASR/diarization.</div>';
      el.speakerHint.textContent = "MVP hiện chỉ dựng giao diện và hợp đồng pipeline.";
      return;
    }
    el.speakerHint.textContent = "Speaker thủ công; diarization tự động sẽ thay thế danh sách này khi được wiring.";
    el.speakerList.innerHTML = state.speakers.map(function (speaker) {
      return '<div class="speaker-card"><i class="speaker-avatar">' + escapeHtml(speaker.short) +
        '</i><span><strong>' + escapeHtml(speaker.label) + '</strong><small>' +
        escapeHtml(speaker.meta) + '</small></span><i class="speaker-wave" aria-hidden="true"></i></div>';
    }).join("");
  }

  function renderStages(stages) {
    if (!stages || !stages.length) return;
    el.stageList.innerHTML = stages.map(function (stage, index) {
      const status = String(stage.status || "wait");
      const className = status === "ready" ? "is-ready" : status === "blocked" ? "is-blocked" : status === "running" ? "is-running" : "";
      const label = status.toUpperCase();
      return '<li class="' + className + '"><i>' + (index + 1) + '</i><span>' +
        escapeHtml(stage.label) + '</span><b>' + label + '</b></li>';
    }).join("");
    const hasBlocker = stages.some(function (stage) { return stage.status === "blocked"; });
    el.planStatus.textContent = hasBlocker ? "BLOCKED" : "READY";
    el.planStatus.style.color = hasBlocker ? "var(--yellow)" : "var(--green)";
  }

  function renderPlan(plan) {
    state.plan = plan;
    renderStages(plan.stages);
    el.timelineState.textContent = plan.ready ? "sẵn sàng chạy" : "thiếu provider";
    if (plan.blockers && plan.blockers.length) {
      el.pipelineNote.innerHTML = '<span class="note-icon">!</span><span>' +
        escapeHtml(plan.blockers[0]) + '</span>';
    }
    setMessage(plan.message || "Đã tạo kế hoạch.", !plan.ready);
    el.exportButton.disabled = !plan.ready;
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
    setMessage("Đang tải " + file.name + "…", false);
    try {
      const payload = await api("/api/uploads", {
        method: "POST",
        headers: {"X-Filename": file.name, "Content-Type": file.type || "application/octet-stream"},
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

  async function createPlan() {
    if (!state.inputPath) {
      setMessage("Hãy chọn video trước.", true);
      return;
    }
    el.planButton.disabled = true;
    el.planButton.firstElementChild.textContent = "Đang kiểm tra…";
    try {
      const plan = await api("/api/dubbing/plan", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          input_path: state.inputPath,
          source_language: el.sourceLanguage.value,
          target_language: el.targetLanguage.value,
          voice_id: el.voiceSelect.value,
          speaker_mode: "auto",
          fit_policy: el.fitPolicy.value,
          preserve_ambience: el.preserveAmbience.checked,
        }),
      });
      renderPlan(plan);
    } catch (error) {
      setMessage(error.message, true);
    } finally {
      el.planButton.disabled = false;
      el.planButton.firstElementChild.textContent = "Phân tích & tạo kế hoạch";
    }
  }

  async function pollJob(jobId) {
    while (true) {
      const job = await api("/api/jobs/" + encodeURIComponent(jobId));
      const progress = Number(job.progress || 0);
      el.planStatus.textContent = String(job.status || "running").toUpperCase();
      el.timelineState.textContent = "job " + progress + "%";
      setMessage(job.message || ("Đang xử lý " + progress + "%…"), job.status === "failed");
      if (job.status === "succeeded") {
        if (job.output_relative) {
          el.videoPlayer.src = "/api/media?path=" + encodeURIComponent(job.output_relative);
          el.videoPlayer.load();
        }
        await loadFiles();
        return job;
      }
      if (job.status === "failed") {
        throw new Error(job.message || "Dubbing job thất bại.");
      }
      await new Promise(function (resolve) { setTimeout(resolve, 1000); });
    }
  }

  async function runDubbing() {
    if (!state.inputPath || !state.plan || !state.plan.ready) {
      setMessage("Hãy phân tích và tạo kế hoạch READY trước.", true);
      return;
    }
    const sourceName = String(state.inputPath).split(/[\\/]/).pop() || "video.mp4";
    const stem = sourceName.replace(/\.[^.]+$/, "");
    el.exportButton.disabled = true;
    setMessage("Đang xếp job dubbing thật…", false);
    try {
      const job = await api("/api/jobs", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          input_path: state.inputPath,
          output_name: stem + "-dubbed.mp4",
          preset: "dub-audio",
          device: "auto",
          overwrite: true,
          source_language: el.sourceLanguage.value,
          target_language: el.targetLanguage.value,
          voice_id: el.voiceSelect.value,
          asr_provider: "auto",
          dubbing_engine: "quality-auto",
        }),
      });
      await pollJob(job.id);
      setMessage("Dubbing hoàn tất. Đang phát file output.", false);
    } catch (error) {
      setMessage(error.message, true);
    } finally {
      el.exportButton.disabled = !(state.plan && state.plan.ready);
    }
  }

  async function loadCapabilities() {
    try {
      state.capabilities = await api("/api/dubbing/capabilities");
      const ready = Boolean(state.capabilities.ready);
      el.capabilityBadge.textContent = ready ? "Local · READY" : "Local · UI + CONTRACT";
      el.capabilityBadge.classList.toggle("is-ready", ready);
    } catch (error) {
      el.capabilityBadge.textContent = "Provider unavailable";
      setMessage(error.message, true);
    }
  }

  function bindPlayer() {
    el.videoPlayer.addEventListener("loadedmetadata", function () {
      el.totalTime.textContent = formatTime(el.videoPlayer.duration);
    });
    el.videoPlayer.addEventListener("timeupdate", function () {
      const duration = el.videoPlayer.duration || 1;
      const ratio = Math.min(1, Math.max(0, el.videoPlayer.currentTime / duration));
      el.currentTime.textContent = formatTime(el.videoPlayer.currentTime);
      el.playhead.style.left = (9 + ratio * 86) + "%";
    });
    el.videoPlayer.addEventListener("play", function () { el.playToggle.textContent = "❚❚"; });
    el.videoPlayer.addEventListener("pause", function () { el.playToggle.textContent = "▶"; });
    el.playToggle.addEventListener("click", function () {
      if (el.videoPlayer.paused) el.videoPlayer.play().catch(function () {});
      else el.videoPlayer.pause();
    });
    document.querySelectorAll("[data-seek]").forEach(function (button) {
      button.addEventListener("click", function () {
        el.videoPlayer.currentTime = Math.max(0, (el.videoPlayer.currentTime || 0) + Number(button.dataset.seek));
      });
    });
    document.querySelectorAll(".play-line").forEach(function (button) {
      button.addEventListener("click", function () {
        el.videoPlayer.currentTime = Number(button.dataset.time || 0);
        el.videoPlayer.play().catch(function () {});
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
        document.getElementById("scriptTab").hidden = tab !== "script";
        document.getElementById("charactersTab").hidden = tab !== "characters";
      });
    });
  }

  function bindControls() {
    el.refreshFiles.addEventListener("click", function () { loadFiles().catch(function (error) { setMessage(error.message, true); }); });
    el.fileInput.addEventListener("change", function () { uploadFile(el.fileInput.files[0]); });
    el.dropZone.addEventListener("dragover", function (event) { event.preventDefault(); el.dropZone.classList.add("is-dragging"); });
    el.dropZone.addEventListener("dragleave", function () { el.dropZone.classList.remove("is-dragging"); });
    el.dropZone.addEventListener("drop", function (event) {
      event.preventDefault();
      el.dropZone.classList.remove("is-dragging");
      uploadFile(event.dataTransfer.files[0]);
    });
    el.planButton.addEventListener("click", createPlan);
    el.addSpeaker.addEventListener("click", function () {
      const id = "S0" + (state.speakers.length + 1);
      state.speakers.push({id: id, short: id.slice(-2), label: "Speaker " + id.slice(-2), meta: "Thủ công · chưa có voice"});
      renderSpeakers();
      setMessage("Đã thêm speaker thủ công. Chưa có diarization để nhận diện tự động.", false);
    });
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
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.addEventListener("ended", function () { URL.revokeObjectURL(url); }, {once: true});
        audio.addEventListener("error", function () { URL.revokeObjectURL(url); }, {once: true});
        await audio.play();
        setMessage("Đang phát preview thật. Chọn voice khác rồi bấm lại để A/B bằng tai.", false);
      } catch (error) {
        setMessage(error.message, true);
      } finally {
        el.previewVoice.disabled = false;
        el.previewVoice.textContent = original;
      }
    });
    el.exportButton.addEventListener("click", runDubbing);
  }

  async function init() {
    bindControls();
    bindPlayer();
    bindTabs();
    renderSpeakers();
    try {
      await api("/api/health");
      setRuntime(true);
    } catch (_) {
      setRuntime(false);
    }
    await Promise.all([loadFiles(), loadCapabilities()]);
  }

  init().catch(function (error) { setMessage(error.message, true); });
}());
