const STORAGE_KEY = "rag-chat-settings-v1";
const HISTORY_KEY = "rag-chat-history-v1";

const elements = {
  layout: document.querySelector(".layout"),
  historyList: document.getElementById("historyList"),
  messages: document.getElementById("messages"),
  emptyState: document.getElementById("emptyState"),
  composer: document.getElementById("composer"),
  questionInput: document.getElementById("questionInput"),
  sendBtn: document.getElementById("sendBtn"),
  messageTemplate: document.getElementById("messageTemplate"),
  settingsModal: document.getElementById("settingsModal"),
  settingsForm: document.getElementById("settingsForm"),
  openSettings: document.getElementById("openSettings"),
  openSettingsFromSide: document.getElementById("openSettingsFromSide"),
  closeSettings: document.getElementById("closeSettings"),
  cancelSettings: document.getElementById("cancelSettings"),
  toggleHistory: document.getElementById("toggleHistory"),
  newChatBtn: document.getElementById("newChatBtn"),
  apiKey: document.getElementById("apiKey"),
  baseUrl: document.getElementById("baseUrl"),
  modelName: document.getElementById("modelName"),
  collectionName: document.getElementById("collectionName"),
  topK: document.getElementById("topK"),
  temperature: document.getElementById("temperature"),
};

const state = {
  loading: false,
  historyOpen: false,
  activeConversationId: null,
  conversations: [],
  autoScrollTimer: null,
  settings: {
    apiKey: "",
    baseUrl: "https://api.deepseek.com",
    model: "deepseek-chat",
    collectionName: "regulation_chunks_v1",
    topK: 5,
    temperature: 0.2,
  },
};

function htmlEscape(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function autoResizeInput() {
  elements.questionInput.style.height = "auto";
  elements.questionInput.style.height = `${Math.min(elements.questionInput.scrollHeight, 180)}px`;
}

function scrollToBottom() {
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function startAutoScroll() {
  stopAutoScroll();
  state.autoScrollTimer = window.setInterval(scrollToBottom, 80);
}

function stopAutoScroll() {
  if (state.autoScrollTimer) {
    window.clearInterval(state.autoScrollTimer);
    state.autoScrollTimer = null;
  }
}

function setLoading(isLoading) {
  state.loading = isLoading;
  elements.sendBtn.disabled = isLoading;
  elements.sendBtn.textContent = isLoading ? "..." : "发送";
}

function setHistoryOpen(isOpen) {
  state.historyOpen = isOpen;
  elements.layout.classList.toggle("history-open", isOpen);
  elements.toggleHistory.classList.toggle("active", isOpen);
}

function formatTime(ts) {
  return new Date(ts).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function createConversation() {
  const now = Date.now();
  return {
    id: `conv_${now}_${Math.random().toString(36).slice(2, 8)}`,
    title: "新对话",
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

function getActiveConversation() {
  return state.conversations.find((item) => item.id === state.activeConversationId) || null;
}

function saveConversations() {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(state.conversations.slice(0, 50)));
}

function loadConversations() {
  const raw = localStorage.getItem(HISTORY_KEY);
  if (raw) {
    try {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) {
        state.conversations = parsed
          .filter((item) => item && typeof item === "object")
          .map((item) => ({
            id: String(item.id || createConversation().id),
            title: String(item.title || "新对话"),
            createdAt: Number(item.createdAt || Date.now()),
            updatedAt: Number(item.updatedAt || Date.now()),
            messages: Array.isArray(item.messages)
              ? item.messages
                  .filter((m) => m && (m.role === "user" || m.role === "assistant"))
                  .map((m) => ({
                    role: m.role,
                    content: String(m.content || ""),
                    references: Array.isArray(m.references) ? m.references : [],
                  }))
              : [],
          }));
      }
    } catch {
      localStorage.removeItem(HISTORY_KEY);
    }
  }

  if (state.conversations.length === 0) {
    const conv = createConversation();
    state.conversations = [conv];
  }
  state.conversations.sort((a, b) => b.updatedAt - a.updatedAt);
  state.activeConversationId = state.conversations[0].id;
}

function selectConversation(conversationId) {
  const exists = state.conversations.some((c) => c.id === conversationId);
  if (!exists) return;
  state.activeConversationId = conversationId;
  renderHistory();
  renderConversation();
}

function ensureTitle(conversation, userText) {
  if (conversation.title !== "新对话") return;
  const cleaned = String(userText || "").trim();
  if (!cleaned) return;
  conversation.title = cleaned.slice(0, 24);
}

function addMessage(conversation, message) {
  conversation.messages.push(message);
  conversation.updatedAt = Date.now();
}

function saveAndRenderHistory() {
  state.conversations.sort((a, b) => b.updatedAt - a.updatedAt);
  saveConversations();
  renderHistory();
}

function renderHistory() {
  const html = state.conversations
    .map((conv) => {
      const activeClass = conv.id === state.activeConversationId ? "active" : "";
      return `
        <button class="history-item ${activeClass}" data-conv-id="${htmlEscape(conv.id)}">
          <div class="history-title">${htmlEscape(conv.title || "新对话")}</div>
          <div class="history-meta">${htmlEscape(formatTime(conv.updatedAt || Date.now()))}</div>
        </button>
      `;
    })
    .join("");
  elements.historyList.innerHTML = html;
}

function refsHtml(references) {
  if (!Array.isArray(references) || references.length === 0) {
    return "";
  }
  return references
    .map((ref) => {
      const title = `[R${ref.rank}] ${ref.doc_name || "未命名文档"} | ${ref.section_path || "-"}`;
      const pages = `${ref.page_start ?? "-"} / ${ref.page_end ?? "-"}`;
      return `
        <details class="ref-card">
          <summary>${htmlEscape(title)}</summary>
          <div class="ref-body">
            <div class="meta">
              doc_id: ${htmlEscape(ref.doc_id || "-")}<br>
              page_start/page_end: ${htmlEscape(pages)}<br>
              score: ${Number(ref.score || 0).toFixed(4)}
            </div>
            <div class="block"><strong>匹配片段</strong>\n${htmlEscape(ref.matched_text || "")}</div>
            <div class="block"><strong>上文</strong>\n${htmlEscape(ref.context_before || "（无）")}</div>
            <div class="block"><strong>下文</strong>\n${htmlEscape(ref.context_after || "（无）")}</div>
          </div>
        </details>
      `;
    })
    .join("");
}

function createMessageNode({ role, content, references = [], typing = false }) {
  const node = elements.messageTemplate.content.firstElementChild.cloneNode(true);
  node.classList.add(role);

  const bubble = node.querySelector(".bubble");
  if (typing) {
    bubble.innerHTML = `<span class="typing"><i></i><i></i><i></i></span>`;
  } else {
    bubble.textContent = content || "";
  }

  const refs = node.querySelector(".refs");
  refs.innerHTML = refsHtml(references);
  return node;
}

function renderConversation() {
  const conv = getActiveConversation();
  elements.messages.innerHTML = "";

  if (!conv || conv.messages.length === 0) {
    elements.emptyState.style.display = "grid";
    return;
  }

  elements.emptyState.style.display = "none";
  conv.messages.forEach((msg) => {
    const node = createMessageNode({
      role: msg.role,
      content: msg.content,
      references: msg.references || [],
      typing: false,
    });
    elements.messages.appendChild(node);
  });
  scrollToBottom();
}

function animateAnswer(bubble, text) {
  return new Promise((resolve) => {
    const fullText = String(text || "");
    if (!fullText) {
      bubble.textContent = "";
      resolve();
      return;
    }

    bubble.textContent = "";
    let index = 0;

    const tick = () => {
      index = Math.min(fullText.length, index + 3);
      bubble.textContent = fullText.slice(0, index);
      scrollToBottom();
      if (index >= fullText.length) {
        resolve();
      } else {
        window.requestAnimationFrame(tick);
      }
    };

    window.requestAnimationFrame(tick);
  });
}

function readSettingsFromForm() {
  return {
    apiKey: elements.apiKey.value.trim(),
    baseUrl: elements.baseUrl.value.trim() || "https://api.deepseek.com",
    model: elements.modelName.value.trim() || "deepseek-chat",
    collectionName: elements.collectionName.value.trim() || "regulation_chunks_v1",
    topK: Math.min(20, Math.max(1, Number(elements.topK.value) || 5)),
    temperature: Math.min(1.5, Math.max(0, Number(elements.temperature.value) || 0.2)),
  };
}

function writeSettingsToForm(settings) {
  elements.apiKey.value = settings.apiKey || "";
  elements.baseUrl.value = settings.baseUrl || "https://api.deepseek.com";
  elements.modelName.value = settings.model || "deepseek-chat";
  elements.collectionName.value = settings.collectionName || "regulation_chunks_v1";
  elements.topK.value = String(settings.topK ?? 5);
  elements.temperature.value = String(settings.temperature ?? 0.2);
}

function saveSettings(settings) {
  state.settings = settings;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  writeSettingsToForm(settings);
}

function loadLocalSettings() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    writeSettingsToForm(state.settings);
    return;
  }

  try {
    const parsed = JSON.parse(raw);
    state.settings = { ...state.settings, ...parsed };
  } catch {
    localStorage.removeItem(STORAGE_KEY);
  }
  writeSettingsToForm(state.settings);
}

async function loadServerDefaults() {
  try {
    const resp = await fetch("/api/config");
    if (!resp.ok) return;
    const cfg = await resp.json();
    state.settings = {
      ...state.settings,
      baseUrl: cfg.default_base_url || state.settings.baseUrl,
      model: cfg.default_model || state.settings.model,
      collectionName: cfg.default_collection || state.settings.collectionName,
    };
    if (Array.isArray(cfg.available_collections) && cfg.available_collections.length > 0) {
      const exists = cfg.available_collections.includes(state.settings.collectionName);
      if (!exists) {
        state.settings.collectionName = cfg.available_collections[0];
      }
    }
    writeSettingsToForm(state.settings);
  } catch {
    // Keep local defaults.
  }
}

async function askQuestion(question) {
  const payload = {
    question,
    api_key: state.settings.apiKey,
    base_url: state.settings.baseUrl,
    model: state.settings.model,
    collection_name: state.settings.collectionName,
    top_k: state.settings.topK,
    temperature: state.settings.temperature,
  };

  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.detail || "请求失败");
  }
  return data;
}

function newConversation() {
  const conv = createConversation();
  state.conversations.unshift(conv);
  state.activeConversationId = conv.id;
  saveAndRenderHistory();
  renderConversation();
  elements.questionInput.focus();
}

function bindEvents() {
  elements.questionInput.addEventListener("input", autoResizeInput);

  elements.questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      elements.composer.requestSubmit();
    }
  });

  elements.composer.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.loading) return;

    const question = elements.questionInput.value.trim();
    if (!question) return;

    const conv = getActiveConversation();
    if (!conv) return;

    ensureTitle(conv, question);
    addMessage(conv, { role: "user", content: question, references: [] });
    saveAndRenderHistory();
    renderConversation();

    elements.questionInput.value = "";
    autoResizeInput();

    const pending = createMessageNode({
      role: "assistant",
      content: "",
      references: [],
      typing: true,
    });
    elements.messages.appendChild(pending);

    setLoading(true);
    startAutoScroll();

    try {
      const data = await askQuestion(question);
      const answer = data.answer || "";
      const refs = Array.isArray(data.references) ? data.references : [];
      const bubble = pending.querySelector(".bubble");
      await animateAnswer(bubble, answer);
      pending.querySelector(".refs").innerHTML = refsHtml(refs);

      addMessage(conv, {
        role: "assistant",
        content: answer,
        references: refs,
      });
      saveAndRenderHistory();
    } catch (error) {
      const errText = error instanceof Error ? error.message : String(error);
      const bubble = pending.querySelector(".bubble");
      bubble.textContent = `请求失败：${errText}`;
      pending.querySelector(".refs").innerHTML = "";

      addMessage(conv, {
        role: "assistant",
        content: `请求失败：${errText}`,
        references: [],
      });
      saveAndRenderHistory();
    } finally {
      stopAutoScroll();
      setLoading(false);
      scrollToBottom();
    }
  });

  elements.toggleHistory.addEventListener("click", () => {
    setHistoryOpen(!state.historyOpen);
  });

  elements.newChatBtn.addEventListener("click", () => {
    newConversation();
  });

  elements.historyList.addEventListener("click", (event) => {
    const target = event.target.closest(".history-item");
    if (!target) return;
    const conversationId = target.getAttribute("data-conv-id");
    if (!conversationId) return;
    selectConversation(conversationId);
  });

  const openModal = () => elements.settingsModal.showModal();
  const closeModal = () => elements.settingsModal.close();

  elements.openSettings.addEventListener("click", openModal);
  elements.openSettingsFromSide.addEventListener("click", openModal);
  elements.closeSettings.addEventListener("click", closeModal);
  elements.cancelSettings.addEventListener("click", closeModal);

  elements.settingsForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const settings = readSettingsFromForm();
    saveSettings(settings);
    closeModal();
  });
}

async function bootstrap() {
  loadLocalSettings();
  await loadServerDefaults();
  loadConversations();
  bindEvents();
  renderHistory();
  renderConversation();
  autoResizeInput();
}

bootstrap();
