"use strict";

const state = {
  inputPath: "",
  preset: "blur-all",
  device: "auto",
  activeJobId: null,
  pollTimer: null,
  files: []
};

const elements = {
  runtimeState: document.getElementById("runtimeState"),
  fileInput: document.getElementById("fileInput"),
  dropZone: document.getElementById("dropZone"),
  uploadProgress: document.getElementById("uploadProgress"),
  sourceList: document.getElementById("sourceList"),
  refreshFiles: document.getElementById("refreshFiles"),
  presetGrid: document.getElementById("presetGrid"),
  deviceControl: document.getElementById("deviceControl"),
  outputName: document.getElementById("outputName"),
  overwrite: document.getElementById("overwrite"),
  runButton: document.getElementById("runButton"),
  formMessage: document.getElementById("formMessage"),
  videoStage: document.getElementById("videoStage"),
  videoPlayer: document.getElementById("videoPlayer"),
  emptyPreview: document.getElementById("emptyPreview"),
  stageBadge: document.getElementById("stageBadge"),
  mediaMeta: document.getElementById("mediaMeta"),
  jobStateIcon: document.getElementById("jobStateIcon"),
  jobStateTitle: document.getElementById("jobStateTitle"),
  jobStateMessage: document.getElementById("jobStateMessage"),
  progressValue: document.getElementById("progressValue"),
  progressBar: document.getElementById("progressBar"),
  resultStatus: document.getElementById("resultStatus"),
  metricSubtitle: document.getElementById("metricSubtitle"),
  metricWatermark: document.getElementById("metricWatermark"),
  metricLogo: document.getElementById("metricLogo"),
  metricOther: document.getElementById("metricOther"),
  detailPreset: document.getElementById("detailPreset"),
  detailDevice: document.getElementById("detailDevice"),
  detailDuration: document.getElementById("detailDuration"),
  detailSize: document.getElementById("detailSize"),
  jobLogs: document.getElementById("jobLogs"),
  copyLogs: document.getElementById("copyLogs"),
  recentJobs: document.getElementById("recentJobs")
};

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, function (character) {
    return {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;"
    }[character];
  });
}

async function api(path, options) {
  const response = await fetch(path, options);
  let payload = {};
  try {
    payload = await response.json();
  } catch (_) {
    payload = {};
  }
  if (!response.ok) {
    throw new Error(payload.error || "Yêu cầu thất bại.");
  }
  return payload;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "—";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return (bytes / Math.pow(1024, index)).toFixed(index > 1 ? 1 : 0) + " " + units[index];
}

function formatSeconds(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 60) return seconds.toFixed(1) + " giây";
  const minutes = Math.floor(seconds / 60);
  return minutes + " phút " + Math.round(seconds % 60) + " giây";
}

function setMessage(message, isError) {
  elements.formMessage.textContent = message || "";
  elements.formMessage.classList.toggle("is-error", Boolean(isError));
}

function setPreview(path, isOutput) {
  if (!path) return;
  elements.videoPlayer.src = "/api/media?path=" + encodeURIComponent(path);
  elements.videoPlayer.load();
  elements.videoStage.classList.add("has-video");
  elements.stageBadge.hidden = !isOutput;
  elements.mediaMeta.textContent = (isOutput ? "OUTPUT · " : "SOURCE · ") + path;
}

function updateRunState() {
  const busy = Boolean(state.activeJobId);
  elements.runButton.disabled = !state.inputPath || busy;
  elements.runButton.querySelector("span").textContent = busy ? "Đang xử lý…" : "Chạy MediaFlow";
}

function selectSource(path) {
  state.inputPath = path;
  Array.from(elements.sourceList.querySelectorAll(".source-item")).forEach(function (button) {
    button.classList.toggle("is-active", button.dataset.path === path);
  });
  const selected = state.files.find(function (item) { return item.path === path; });
  setPreview(path, false);
  if (selected) {
    elements.mediaMeta.textContent = "SOURCE · " + selected.name + " · " + formatBytes(selected.size);
    const stem = selected.name.replace(/\.[^.]+$/, "");
    elements.outputName.value = stem + "-mediaflow.mp4";
  }
  setMessage("");
  updateRunState();
}

function renderSources() {
  if (!state.files.length) {
    elements.sourceList.innerHTML = '<div class="empty-row">Chưa có video trong input/test_videos.</div>';
    return;
  }
  elements.sourceList.innerHTML = state.files.map(function (item) {
    const active = item.path === state.inputPath ? " is-active" : "";
    return '<button class="source-item' + active + '" type="button" data-path="' +
      escapeHtml(item.path) + '"><span>' + escapeHtml(item.name) +
      '</span><small>' + formatBytes(item.size) + '</small></button>';
  }).join("");
  Array.from(elements.sourceList.querySelectorAll(".source-item")).forEach(function (button) {
    button.addEventListener("click", function () { selectSource(button.dataset.path); });
  });
}

async function loadFiles() {
  elements.refreshFiles.disabled = true;
  try {
    const payload = await api("/api/files");
    state.files = payload.files || [];
    renderSources();
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    elements.refreshFiles.disabled = false;
  }
}

async function uploadFile(file) {
  if (!file) return;
  elements.uploadProgress.hidden = false;
  elements.dropZone.classList.add("is-uploading");
  setMessage("Đang tải video vào dự án…");
  try {
    const payload = await api("/api/uploads", {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Filename": encodeURIComponent(file.name)
      },
      body: file
    });
    await loadFiles();
    selectSource(payload.path);
    setMessage("Đã thêm " + payload.name + ".");
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    elements.uploadProgress.hidden = true;
    elements.dropZone.classList.remove("is-uploading");
    elements.fileInput.value = "";
  }
}

function setJobVisual(status, progress, message) {
  const normalized = status || "idle";
  elements.jobStateIcon.className = "job-state-icon";
  elements.resultStatus.className = "result-status";

  if (normalized === "running" || normalized === "queued") {
    elements.jobStateIcon.classList.add("is-running");
    elements.resultStatus.classList.add("is-running");
    elements.jobStateTitle.textContent = normalized === "queued" ? "Đã xếp hàng" : "Đang xử lý";
  } else if (normalized === "succeeded") {
    elements.jobStateIcon.classList.add("is-success");
    elements.resultStatus.classList.add("is-success");
    elements.jobStateTitle.textContent = "Hoàn tất";
  } else if (normalized === "failed") {
    elements.jobStateIcon.classList.add("is-failed");
    elements.resultStatus.classList.add("is-failed");
    elements.jobStateTitle.textContent = "Thất bại";
  } else {
    elements.jobStateTitle.textContent = "Sẵn sàng";
  }

  elements.resultStatus.textContent = normalized.toUpperCase();
  elements.jobStateMessage.textContent = message || "";
  elements.progressValue.textContent = Math.round(progress || 0) + "%";
  elements.progressBar.style.width = Math.max(0, Math.min(100, progress || 0)) + "%";
}

function renderJob(job) {
  setJobVisual(job.status, job.progress, job.message);
  elements.detailPreset.textContent = job.preset || "—";
  elements.detailDevice.textContent = job.device || "—";
  elements.detailDuration.textContent = formatSeconds(job.duration_seconds);
  elements.detailSize.textContent = formatBytes(job.output_size);

  const detections = job.detections || {};
  elements.metricSubtitle.textContent = detections.subtitle ?? "—";
  elements.metricWatermark.textContent = detections.watermark ?? "—";
  elements.metricLogo.textContent = detections.logo ?? "—";
  elements.metricOther.textContent = detections.other_burn_in_text ?? "—";

  elements.jobLogs.textContent = (job.logs && job.logs.length)
    ? job.logs.join("\n")
    : "Đang chờ nhật ký…";
  elements.jobLogs.scrollTop = elements.jobLogs.scrollHeight;

  if (job.status === "succeeded" && job.output_relative) {
    setPreview(job.output_relative, true);
  }
}

function renderRecent(jobs) {
  if (!jobs.length) {
    elements.recentJobs.innerHTML = '<div class="empty-row">Chưa có tác vụ.</div>';
    return;
  }
  elements.recentJobs.innerHTML = jobs.map(function (job) {
    const action = job.status === "succeeded"
      ? '<button type="button" data-output="' + escapeHtml(job.output_relative) + '">Xem output</button>'
      : '<span>' + escapeHtml(job.status) + '</span>';
    return '<article class="recent-row"><strong>' + escapeHtml(job.output_name) +
      '</strong><span>' + escapeHtml(job.preset) + '</span><span>' +
      formatSeconds(job.duration_seconds) + '</span>' + action + '</article>';
  }).join("");

  Array.from(elements.recentJobs.querySelectorAll("button[data-output]")).forEach(function (button) {
    button.addEventListener("click", function () { setPreview(button.dataset.output, true); });
  });
}

async function loadJobs() {
  try {
    const payload = await api("/api/jobs");
    renderRecent(payload.jobs || []);
  } catch (_) {
    return;
  }
}

async function pollJob() {
  if (!state.activeJobId) return;
  try {
    const job = await api("/api/jobs/" + encodeURIComponent(state.activeJobId));
    renderJob(job);
    await loadJobs();

    if (job.status === "succeeded" || job.status === "failed") {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      state.activeJobId = null;
      updateRunState();
      if (job.status === "succeeded") {
        setMessage("Đã tạo " + job.output_name + ".");
        await loadFiles();
      } else {
        setMessage(job.message || "Pipeline thất bại.", true);
      }
    }
  } catch (error) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    state.activeJobId = null;
    updateRunState();
    setMessage(error.message, true);
  }
}

async function startJob() {
  if (!state.inputPath || state.activeJobId) return;
  setMessage("");
  elements.runButton.disabled = true;
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        input_path: state.inputPath,
        output_name: elements.outputName.value,
        preset: state.preset,
        device: state.device,
        overwrite: elements.overwrite.checked
      })
    });
    state.activeJobId = job.id;
    renderJob(job);
    updateRunState();
    state.pollTimer = setInterval(pollJob, 1200);
    await pollJob();
  } catch (error) {
    state.activeJobId = null;
    updateRunState();
    setMessage(error.message, true);
  }
}

function bindControls() {
  elements.refreshFiles.addEventListener("click", loadFiles);
  elements.fileInput.addEventListener("change", function () {
    uploadFile(elements.fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach(function (eventName) {
    elements.dropZone.addEventListener(eventName, function (event) {
      event.preventDefault();
      elements.dropZone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach(function (eventName) {
    elements.dropZone.addEventListener(eventName, function (event) {
      event.preventDefault();
      elements.dropZone.classList.remove("is-dragging");
    });
  });
  elements.dropZone.addEventListener("drop", function (event) {
    uploadFile(event.dataTransfer.files[0]);
  });

  Array.from(elements.presetGrid.querySelectorAll("[data-preset]")).forEach(function (button) {
    button.addEventListener("click", function () {
      state.preset = button.dataset.preset;
      Array.from(elements.presetGrid.querySelectorAll("[data-preset]")).forEach(function (item) {
        item.classList.toggle("is-active", item === button);
      });
      elements.detailPreset.textContent = state.preset;
    });
  });

  Array.from(elements.deviceControl.querySelectorAll("[data-device]")).forEach(function (button) {
    button.addEventListener("click", function () {
      state.device = button.dataset.device;
      Array.from(elements.deviceControl.querySelectorAll("[data-device]")).forEach(function (item) {
        item.classList.toggle("is-active", item === button);
      });
      elements.detailDevice.textContent = state.device;
    });
  });

  elements.runButton.addEventListener("click", startJob);
  elements.copyLogs.addEventListener("click", async function () {
    try {
      await navigator.clipboard.writeText(elements.jobLogs.textContent);
      elements.copyLogs.textContent = "Đã chép";
      setTimeout(function () { elements.copyLogs.textContent = "Sao chép"; }, 1200);
    } catch (_) {
      setMessage("Không thể sao chép nhật ký.", true);
    }
  });
}

async function init() {
  bindControls();
  updateRunState();
  try {
    await api("/api/health");
    elements.runtimeState.classList.add("is-online");
    elements.runtimeState.querySelector("span:last-child").textContent = "Runtime sẵn sàng";
  } catch (error) {
    elements.runtimeState.classList.add("is-offline");
    elements.runtimeState.querySelector("span:last-child").textContent = "Mất kết nối";
    setMessage(error.message, true);
  }
  await Promise.all([loadFiles(), loadJobs()]);
}

init();
