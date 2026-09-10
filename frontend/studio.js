/* MediaFlow AI Studio shell. Provider actions stay honest until adapters are configured. */
(function () {
  "use strict";

  const definitions = {
    tts: {
      kicker: "TEXT TO SPEECH",
      title: "Tạo voice từ script",
      description: "Nhập text, chọn voice profile và kiểm tra contract trước khi nối provider thật.",
      input: "Text + voice profile",
      processing: "TTS adapter (chưa cấu hình)",
      output: "MP3/WAV artifact",
    },
    subtitle: {
      kicker: "SUBTITLE WORKSPACE",
      title: "Đọc và dịch subtitle",
      description: "Nạp SRT/VTT, giữ timestamp và chuẩn bị bản dịch cho timeline video.",
      input: "SRT/VTT + ngôn ngữ đích",
      processing: "Subtitle parser + translator",
      output: "SRT/VTT đã đồng bộ",
    },
    stt: {
      kicker: "SPEECH TO TEXT",
      title: "Transcript theo timestamp",
      description: "Chuẩn bị transcript có speaker và mốc thời gian để ghép vào pipeline.",
      input: "Audio/video + ngôn ngữ",
      processing: "ASR + diarization (chưa cấu hình)",
      output: "Transcript JSON/SRT",
    },
    clone: {
      kicker: "VOICE CLONE",
      title: "Quản lý voice profile",
      description: "Tạo profile từ sample giọng nói; file chỉ dùng trong project local.",
      input: "Voice sample + profile name",
      processing: "Voice clone adapter (chưa cấu hình)",
      output: "Voice profile",
    },
    dubbing: {
      kicker: "VIDEO DUBBING",
      title: "Dịch và lồng tiếng video",
      description: "Mở workspace timeline để quản lý transcript, nhân vật, voice và fit thời lượng.",
      input: "Video + speaker map",
      processing: "ASR → translate → TTS → compose",
      output: "Video MP4 đã mux audio",
    },
  };

  const state = {
    tool: new URLSearchParams(window.location.search).get("tool") || "tts",
    capabilities: null,
  };

  const el = {
    runtime: document.getElementById("runtimeState"),
    runtimeTitle: document.getElementById("runtimeTitle"),
    runtimeDetail: document.getElementById("runtimeDetail"),
    toolKicker: document.getElementById("toolKicker"),
    toolTitle: document.getElementById("toolTitle"),
    toolDescription: document.getElementById("toolDescription"),
    inputContract: document.getElementById("inputContract"),
    processingContract: document.getElementById("processingContract"),
    outputContract: document.getElementById("outputContract"),
    toolForm: document.getElementById("toolForm"),
    toolMessage: document.getElementById("toolMessage"),
    capabilityList: document.getElementById("capabilityList"),
    toolStatus: document.getElementById("toolStatus"),
  };

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, function (char) {
      return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[char];
    });
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
    el.toolMessage.textContent = message || "";
    el.toolMessage.classList.toggle("is-error", Boolean(isError));
  }

  function setRuntime(online, detail) {
    el.runtime.classList.toggle("is-online", online);
    el.runtime.classList.toggle("is-offline", !online);
    el.runtime.lastChild.textContent = online ? " Runtime sẵn sàng" : " Mất kết nối";
    el.runtimeTitle.textContent = online ? "Local runtime" : "Runtime unavailable";
    el.runtimeDetail.textContent = detail || (online ? "API local đang hoạt động." : "Khởi động server để dùng workspace.");
  }

  function capabilityStatus(tool) {
    const providers = (state.capabilities && state.capabilities.providers) || {};
    const mapping = {
      tts: ["tts"],
      subtitle: ["translation", "composer"],
      stt: ["asr", "diarization"],
      clone: ["tts"],
      dubbing: ["asr", "translation", "tts", "composer"],
    };
    const keys = mapping[tool] || ["tts"];
    const available = keys.filter(function (key) { return providers[key] && providers[key].available; }).length;
    return {available: available, total: keys.length, providers: providers, keys: keys};
  }

  function renderCapabilities() {
    const providers = (state.capabilities && state.capabilities.providers) || {};
    const labels = {
      asr: "ASR",
      diarization: "Speaker diarization",
      translation: "Translation",
      tts: "TTS",
      composer: "AudioComposer",
    };
    const keys = ["asr", "diarization", "translation", "tts", "composer"];
    if (!state.capabilities) {
      el.capabilityList.innerHTML = '<div class="empty-state">Đang tải capability…</div>';
      return;
    }
    el.capabilityList.innerHTML = keys.map(function (key) {
      const provider = providers[key] || {};
      const ready = Boolean(provider.available);
      const status = ready ? "READY" : "BLOCKED";
      return '<div class="capability-item"><span><strong>' + labels[key] +
        '</strong><small>' + escapeHtml(provider.reason || "Chưa có thông tin.") +
        '</small></span><b class="' + (ready ? "ready" : "") + '">' + status + "</b></div>";
    }).join("");
  }

  function renderToolStatus() {
    const result = capabilityStatus(state.tool);
    const ready = result.available === result.total && result.total > 0;
    el.toolStatus.textContent = ready ? "READY" : "PARTIAL";
    el.toolStatus.classList.toggle("is-ready", ready);
    el.toolStatus.classList.toggle("is-blocked", !ready);
  }

  function inputMarkup(kind) {
    if (kind === "tts") {
      return '<h3>Quick TTS preview</h3>' +
        '<label class="form-label"><span>SCRIPT</span><textarea id="ttsText" placeholder="Nhập câu thoại để thử contract…">Cuối cùng chúng ta cũng gặp nhau.</textarea></label>' +
        '<div class="form-row"><label class="form-label"><span>VOICE</span><select id="ttsVoice"><option>vi-VN-HoaiMyNeural</option><option>vi-VN-NamMinhNeural</option></select></label>' +
        '<label class="form-label"><span>FORMAT</span><select id="ttsFormat"><option>MP3</option><option>WAV</option></select></label></div>' +
        '<div class="form-actions"><button class="primary-button tool-action" type="button">Tạo preview</button><button class="secondary-button" type="button" data-clear>Clear</button></div>';
    }
    if (kind === "subtitle") {
      return '<h3>Subtitle input</h3>' +
        '<label class="file-drop"><input type="file" accept=".srt,.vtt,text/vtt,application/x-subrip"><strong>Chọn SRT hoặc VTT</strong><small data-file-name>Kéo thả file subtitle vào đây</small></label>' +
        '<div class="form-row"><label class="form-label"><span>SOURCE</span><select><option>Auto detect</option><option>中文</option><option>English</option><option>Tiếng Việt</option></select></label>' +
        '<label class="form-label"><span>TARGET</span><select><option>Tiếng Việt</option><option>English</option><option>中文</option></select></label></div>' +
        '<div class="form-actions"><button class="primary-button tool-action" type="button">Phân tích subtitle</button></div>';
    }
    if (kind === "stt") {
      return '<h3>Transcript input</h3>' +
        '<label class="file-drop"><input type="file" accept="audio/*,video/*"><strong>Chọn audio hoặc video</strong><small data-file-name>Hỗ trợ file trong project hoặc upload local</small></label>' +
        '<div class="form-row"><label class="form-label"><span>LANGUAGE</span><select><option>Auto detect</option><option>中文</option><option>English</option><option>Tiếng Việt</option></select></label>' +
        '<label class="form-label"><span>OUTPUT</span><select><option>SRT + JSON</option><option>TXT</option></select></label></div>' +
        '<div class="form-actions"><button class="primary-button tool-action" type="button">Tạo transcript</button></div>';
    }
    if (kind === "clone") {
      return '<h3>Voice profile</h3>' +
        '<label class="form-label"><span>PROFILE NAME</span><input type="text" placeholder="Ví dụ: Narrator nữ 01"></label>' +
        '<label class="file-drop"><input type="file" accept="audio/*"><strong>Chọn voice sample</strong><small data-file-name>Khuyến nghị sample sạch, 10–30 giây</small></label>' +
        '<div class="form-actions"><button class="primary-button tool-action" type="button">Tạo profile</button></div>';
    }
    return '<h3>Video Dubbing Workspace</h3><p class="empty-state">Mọi control chính nằm trên timeline: source video, subtitle, speaker map, voice assignment và fit policy.</p>' +
      '<div class="form-actions"><a class="primary-button" href="/dubbing.html">Mở workspace <span>→</span></a></div>';
  }

  function bindFormActions() {
    el.toolForm.querySelectorAll(".file-drop input").forEach(function (input) {
      input.addEventListener("change", function () {
        const file = input.files && input.files[0];
        const label = input.closest(".file-drop").querySelector("[data-file-name]");
        if (label) label.textContent = file ? file.name + " · " + Math.round(file.size / 1024) + " KB" : "Chưa chọn file";
      });
    });
    el.toolForm.querySelectorAll("[data-clear]").forEach(function (button) {
      button.addEventListener("click", function () {
        const text = el.toolForm.querySelector("textarea");
        if (text) text.value = "";
        setMessage("", false);
      });
    });
    el.toolForm.querySelectorAll(".tool-action").forEach(function (button) {
      button.addEventListener("click", function () {
        const result = capabilityStatus(state.tool);
        const ready = result.available === result.total && result.total > 0;
        if (ready) {
          setMessage("Provider đã sẵn sàng; bước tiếp theo là nối job runner vào contract này.", false);
        } else {
          setMessage("UI đã sẵn sàng, nhưng provider thật chưa được cấu hình trong local runtime.", true);
        }
      });
    });
  }

  function renderTool(tool) {
    if (!definitions[tool]) tool = "tts";
    state.tool = tool;
    const definition = definitions[tool];
    el.toolKicker.textContent = definition.kicker;
    el.toolTitle.textContent = definition.title;
    el.toolDescription.textContent = definition.description;
    el.inputContract.textContent = definition.input;
    el.processingContract.textContent = definition.processing;
    el.outputContract.textContent = definition.output;
    el.toolForm.innerHTML = inputMarkup(tool);
    document.querySelectorAll("[data-tool-link]").forEach(function (link) {
      const linkTool = link.dataset.toolLink;
      link.classList.toggle("is-active", linkTool === tool);
    });
    bindFormActions();
    renderToolStatus();
  }

  async function loadCapabilities() {
    try {
      state.capabilities = await api("/api/dubbing/capabilities");
      renderCapabilities();
      renderToolStatus();
      const providers = state.capabilities.providers || {};
      const available = Object.keys(providers).filter(function (key) { return providers[key].available; }).length;
      setRuntime(true, available + "/" + Object.keys(providers).length + " local providers available.");
    } catch (error) {
      setRuntime(false, error.message);
      renderCapabilities();
      renderToolStatus();
    }
  }

  async function init() {
    renderTool(state.tool);
    try {
      await api("/api/health");
      setRuntime(true, "API local đang hoạt động.");
    } catch (_) {
      setRuntime(false);
    }
    await loadCapabilities();
  }

  init().catch(function (error) { setMessage(error.message, true); });
}());
