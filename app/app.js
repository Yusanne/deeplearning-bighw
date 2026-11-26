const STORAGE_KEY = "citefaith-annotation-v1";
const WORLD_NAMES = {
  plus: { title: "原始世界", hint: "支持答案的证据", className: "world-plus" },
  del: { title: "删除世界", hint: "移除金标证据页", className: "world-del" },
  swap: { title: "替换世界", hint: "相似但结论冲突", className: "world-swap" },
  benign: { title: "良性世界", hint: "加入无关页面", className: "world-benign" },
};

const state = {
  rows: [],
  llm: {},
  index: 0,
  annotator: "A",
  annotations: {},
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function defaultAnnotation() {
  return {
    swap_validity: "",
    positive_evidence_valid: "",
    counterfactual_answer: "",
    counterfactual_answerable: "",
    benign_valid: "",
    fact_occurrence_count: "",
    text_issue: "none",
    confidence: "",
    notes: "",
    updated_at: null,
  };
}

function getAnnotation(row) {
  if (!state.annotations[row.qid]) state.annotations[row.qid] = {};
  if (!state.annotations[row.qid][state.annotator]) {
    state.annotations[row.qid][state.annotator] = defaultAnnotation();
  }
  return state.annotations[row.qid][state.annotator];
}

function isComplete(annotation) {
  return Boolean(
    annotation.swap_validity &&
    annotation.positive_evidence_valid &&
    annotation.benign_valid &&
    annotation.fact_occurrence_count &&
    annotation.confidence,
  );
}

function persist() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({
    annotations: state.annotations,
    annotator: state.annotator,
    index: state.index,
  }));
  $("saveState").textContent = "已自动保存";
}

function restore() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (!saved) return;
    state.annotations = saved.annotations || {};
    state.annotator = saved.annotator || "A";
    state.index = Number.isInteger(saved.index) ? saved.index : 0;
    $("annotatorSelect").value = state.annotator;
  } catch (error) {
    console.warn("无法读取本地标注缓存", error);
  }
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 2200);
}

function llmAnswer(qid, world) {
  const item = state.llm[`${qid}__${world}`];
  if (!item) return null;
  const raw = item.raw ?? item.content ?? item;
  if (typeof raw === "object") return raw.answer || JSON.stringify(raw);
  try {
    const parsed = JSON.parse(raw);
    return parsed.answer || raw;
  } catch (_) {
    return String(raw).slice(0, 500);
  }
}

function formatPages(pages, openFirst = false) {
  if (!pages || !pages.length) {
    return '<p class="small-note">没有页面内容。</p>';
  }
  return pages.map((page, index) => `
    <details class="page" ${openFirst && index === 0 ? "open" : ""}>
      <summary>Page ${escapeHtml(page.page)} · ${escapeHtml(page.chars || String(page.text || "").length)} chars</summary>
      <div class="page-text">${escapeHtml(page.text || "")}</div>
    </details>
  `).join("");
}

function renderWorld(row, world) {
  const meta = WORLD_NAMES[world];
  const pages = row.worlds?.[world] || [];
  const answer = llmAnswer(row.qid, world);
  const answerBlock = answer
    ? `<div class="world-answer"><strong>模型输出</strong><br />${escapeHtml(answer)}</div>`
    : `<div class="world-answer"><strong>模型输出</strong><br /><span class="small-note">未加载模型输出文件</span></div>`;
  let facts = "";
  if (world === "swap") {
    const swap = row.swap || {};
    facts = `<div class="swap-facts">
      <div class="fact"><strong>规则代理</strong>${escapeHtml(swap.rule || "未记录")}</div>
      <div class="fact"><strong>代理结果</strong>${escapeHtml(swap.valid_proxy ? "可能有效" : "待人工确认")}</div>
    </div>`;
  }
  return `<article class="world-card ${meta.className}">
    <div class="world-head"><span class="world-name">${meta.title}</span><span class="world-hint">${meta.hint}</span></div>
    <div class="world-body">${answerBlock}${facts}${formatPages(pages, world === "plus")}</div>
  </article>`;
}

function choice(name, value, title, description, selected) {
  return `<label class="choice ${selected === value ? "selected" : ""}">
    <input type="radio" name="${name}" value="${escapeHtml(value)}" ${selected === value ? "checked" : ""} />
    <span class="choice-text"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(description)}</span></span>
  </label>`;
}

function renderAnnotationPanel(row, annotation) {
  const yesNo = (name, selected, yesTitle, noTitle, help) => `
    <div class="choice-row">
      ${choice(name, "yes", yesTitle, help, selected)}
      ${choice(name, "no", noTitle, "标记为否，后续在分歧复核时处理", selected)}
      ${choice(name, "uncertain", "不确定", "暂时无法判断，保留原因到备注", selected)}
    </div>`;
  return `<aside class="card annotation-card">
    <div class="card-head"><div><h3>人工标注</h3><p class="annotation-subtitle">当前：标注人 ${state.annotator}。先独立完成，再与另一位成员讨论分歧。</p></div><span class="tag teal">${isComplete(annotation) ? "已完成" : "进行中"}</span></div>

    <div class="field">
      <div class="field-label">1. 替换证据有效性 <span class="field-help">核心门槛</span></div>
      <div class="choice-row">
        ${choice("swap_validity", "flip", "Flip：结论确实改变", "E− 支持一个不同的、可判定的答案", annotation.swap_validity)}
        ${choice("swap_validity", "unanswerable", "变为不可回答", "E− 不能支持新答案，但原答案不应继续成立", annotation.swap_validity)}
        ${choice("swap_validity", "invalid", "Invalid：替换无效", "主题不符、结论未变或文本明显不自然", annotation.swap_validity)}
      </div>
    </div>

    <div class="field">
      <div class="field-label">2. 正证据页是否真的支持答案</div>
      ${yesNo("positive_evidence_valid", annotation.positive_evidence_valid, "支持", "不支持", "答案或必要信息可以在正证据页中定位")}
    </div>

    <div class="field">
      <div class="field-label">3. 反事实答案 <span class="field-help">独立作答，不看模型答案</span></div>
      <input id="counterfactualAnswer" class="text-input" value="${escapeHtml(annotation.counterfactual_answer)}" placeholder="例如：Friday 4:45pm / 证据不足" />
    </div>

    <div class="field">
      <div class="field-label">4. 替换后是否可回答</div>
      ${yesNo("counterfactual_answerable", annotation.counterfactual_answerable, "可回答", "不可回答", "E− 中存在足够信息得出新的事实结论")}
    </div>

    <div class="field">
      <div class="field-label">5. 良性页是否真的不改变事实</div>
      ${yesNo("benign_valid", annotation.benign_valid, "良性有效", "不能作为良性页", "加入页面与主题相关，但不单独支持或冲突于答案")}
    </div>

    <div class="field">
      <div class="field-label">6. 事实在全文中的出现次数</div>
      <div class="choice-row">
        ${choice("fact_occurrence_count", "1", "仅 1 处", "支持 single-support 主分析", annotation.fact_occurrence_count)}
        ${choice("fact_occurrence_count", "2+", "2 处或更多", "存在冗余支撑，单独分层", annotation.fact_occurrence_count)}
        ${choice("fact_occurrence_count", "unknown", "无法判断", "保留原因到备注", annotation.fact_occurrence_count)}
      </div>
    </div>

    <div class="field">
      <div class="field-label">7. 文本/版面问题</div>
      <select id="textIssue" class="select" style="width:100%">
        <option value="none" ${annotation.text_issue === "none" ? "selected" : ""}>无明显问题</option>
        <option value="text_not_extractable" ${annotation.text_issue === "text_not_extractable" ? "selected" : ""}>无文本层/扫描件</option>
        <option value="answer_format_needs_review" ${annotation.text_issue === "answer_format_needs_review" ? "selected" : ""}>答案格式不一致</option>
        <option value="visual_or_layout_dependent" ${annotation.text_issue === "visual_or_layout_dependent" ? "selected" : ""}>依赖图表或复杂版面</option>
      </select>
    </div>

    <div class="field">
      <div class="field-label">8. 标注信心</div>
      <div class="choice-row">
        ${choice("confidence", "high", "高", "可以直接作为最终标签", annotation.confidence)}
        ${choice("confidence", "medium", "中", "有一点歧义，但倾向明确", annotation.confidence)}
        ${choice("confidence", "low", "低", "建议进入分歧复核", annotation.confidence)}
      </div>
    </div>

    <div class="field">
      <div class="field-label">备注</div>
      <textarea id="annotationNotes" placeholder="记录证据定位、分歧原因、为何判定 invalid 等。">${escapeHtml(annotation.notes)}</textarea>
    </div>

    <div class="annotation-footer">
      <span class="shortcut-note">快捷键：←/→ 切题<br />1=flip · 2=不可回答 · 3=invalid</span>
      <button id="markNext" class="mark-button">保存并下一题 →</button>
    </div>
  </aside>`;
}

function renderSidebar() {
  const list = $("questionList");
  $("loadedCount").textContent = state.rows.length;
  if (!state.rows.length) {
    list.innerHTML = '<div class="empty-state compact">请先加载<br />week1/results/worlds</div>';
    return;
  }
  list.innerHTML = state.rows.map((row, index) => {
    const annotation = getAnnotation(row);
    const done = isComplete(annotation);
    return `<button class="question-item ${index === state.index ? "active" : ""}" data-index="${index}">
      <span class="question-number">${String(index + 1).padStart(2, "0")}</span>
      <span class="question-title" title="${escapeHtml(row.question)}">${escapeHtml(row.qid)}</span>
      <span class="status-dot ${done ? "done" : ""}"></span>
    </button>`;
  }).join("");
  list.querySelectorAll(".question-item").forEach((button) => {
    button.addEventListener("click", () => {
      state.index = Number(button.dataset.index);
      persist();
      render();
    });
  });
}

function updateProgress() {
  const completed = state.rows.filter((row) => isComplete(getAnnotation(row))).length;
  const total = state.rows.length;
  $("progressText").textContent = `${completed} / ${total} 已完成`;
  $("progressBar").style.width = `${total ? Math.round(completed / total * 100) : 0}%`;
  $("prevButton").disabled = state.index <= 0;
  $("nextButton").disabled = state.index >= total - 1;
}

function render() {
  renderSidebar();
  updateProgress();
  if (!state.rows.length) return;
  const row = state.rows[state.index];
  const annotation = getAnnotation(row);
  const positivePages = (row.worlds?.plus || []).map((page) => page.page);
  const swapPage = row.swap_page ?? row.swap?.page ?? "未记录";
  $("mainPanel").innerHTML = `<div class="case-header">
    <div><p class="section-kicker">${escapeHtml(row.qid)} · ${escapeHtml(row.doc_id)}</p><h2>${escapeHtml(row.question)}</h2></div>
    <div class="case-meta"><span class="tag teal">金标页 ${escapeHtml(positivePages.join(", ") || "未记录")}</span><span class="tag amber">swap 页 ${escapeHtml(swapPage)}</span><span class="tag">${escapeHtml(row.answer_format || "unknown")}</span></div>
  </div>
  <div class="grid">
    <div class="evidence-column">
      <section class="card"><div class="card-head"><div><h3>原始问题与参考答案</h3><p class="small-note">先确认题目能否被明确判定，再查看四个世界。</p></div></div><div class="gold-answer">${escapeHtml(row.answer || "未记录参考答案")}</div></section>
      <section class="card"><div class="card-head"><div><h3>四个证据世界</h3><p class="small-note">展开页面阅读全文。替换世界的“规则代理”只作提示，最终标签以人工判断为准。</p></div></div><div class="worlds">${Object.keys(WORLD_NAMES).map((world) => renderWorld(row, world)).join("")}</div></section>
    </div>
    ${renderAnnotationPanel(row, annotation)}
  </div>`;
  bindAnnotationEvents(row, annotation);
}

function updateField(row, annotation, field, value) {
  annotation[field] = value;
  annotation.updated_at = new Date().toISOString();
  persist();
  renderSidebar();
  updateProgress();
  if (field === "swap_validity" || field === "positive_evidence_valid" || field === "benign_valid" || field === "fact_occurrence_count" || field === "confidence") {
    const panel = $("mainPanel").querySelector(".annotation-card");
    if (panel) panel.outerHTML = renderAnnotationPanel(row, annotation);
    bindAnnotationEvents(row, annotation);
  }
}

function bindAnnotationEvents(row, annotation) {
  $("mainPanel").querySelectorAll('input[type="radio"]').forEach((input) => {
    input.addEventListener("change", () => updateField(row, annotation, input.name, input.value));
  });
  const answer = $("counterfactualAnswer");
  if (answer) answer.addEventListener("input", () => updateField(row, annotation, "counterfactual_answer", answer.value));
  const notes = $("annotationNotes");
  if (notes) notes.addEventListener("input", () => updateField(row, annotation, "notes", notes.value));
  const textIssue = $("textIssue");
  if (textIssue) textIssue.addEventListener("change", () => updateField(row, annotation, "text_issue", textIssue.value));
  const markNext = $("markNext");
  if (markNext) markNext.addEventListener("click", () => {
    persist();
    if (state.index < state.rows.length - 1) {
      state.index += 1;
      persist();
      render();
    } else {
      showToast("已到最后一题，当前标注已保存");
    }
  });
}

async function readJsonFiles(fileList) {
  const files = [...fileList].filter((file) => file.name.toLowerCase().endsWith(".json"));
  return Promise.all(files.map(async (file) => {
    try { return JSON.parse(await file.text()); } catch (error) { return { __error: file.name, error: String(error) }; }
  }));
}

async function loadWorldFiles(fileList) {
  const parsed = await readJsonFiles(fileList);
  const rows = parsed.filter((item) => item && item.qid && item.worlds);
  rows.sort((a, b) => a.qid.localeCompare(b.qid, undefined, { numeric: true }));
  state.rows = rows;
  state.index = Math.min(state.index, Math.max(0, rows.length - 1));
  persist();
  render();
  showToast(`已加载 ${rows.length} 道题`);
}

async function loadLlmFiles(fileList) {
  const files = [...fileList].filter((file) => file.name.toLowerCase().endsWith(".json"));
  let loaded = 0;
  for (const file of files) {
    const match = file.name.match(/^(.+)__(plus|del|swap|benign)\.json$/i);
    if (!match) continue;
    try {
      state.llm[`${match[1]}__${match[2].toLowerCase()}`] = JSON.parse(await file.text());
      loaded += 1;
    } catch (error) {
      console.warn("无法读取模型输出", file.name, error);
    }
  }
  render();
  showToast(`已加载 ${loaded} 条模型输出`);
}

function downloadJson(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function exportAnnotations(currentOnly) {
  const rows = currentOnly && state.rows.length ? [state.rows[state.index]] : state.rows;
  const output = rows.map((row) => ({
    qid: row.qid,
    doc_id: row.doc_id,
    question: row.question,
    gold_answer: row.answer,
    gold_pages_textview: row.gold_pages_textview,
    swap_page: row.swap_page,
    proxy: row.swap,
    annotations: state.annotations[row.qid] || {},
  }));
  const suffix = currentOnly && rows[0] ? `_${rows[0].qid}` : "_all";
  downloadJson(`citefaith_annotations_${state.annotator}${suffix}.json`, {
    schema_version: "citefaith-pilot-annotation-v1",
    exported_at: new Date().toISOString(),
    annotator: currentOnly ? state.annotator : "A+B",
    items: output,
  });
  showToast(currentOnly ? "当前题目已导出" : "全部标注已导出");
}

$("worldsInput").addEventListener("change", (event) => loadWorldFiles(event.target.files));
$("llmInput").addEventListener("change", (event) => loadLlmFiles(event.target.files));
$("annotatorSelect").addEventListener("change", (event) => {
  state.annotator = event.target.value;
  persist();
  render();
});
$("prevButton").addEventListener("click", () => { if (state.index > 0) { state.index -= 1; persist(); render(); } });
$("nextButton").addEventListener("click", () => { if (state.index < state.rows.length - 1) { state.index += 1; persist(); render(); } });
$("exportCurrent").addEventListener("click", () => exportAnnotations(true));
$("exportAll").addEventListener("click", () => exportAnnotations(false));

document.addEventListener("keydown", (event) => {
  const tag = document.activeElement?.tagName;
  if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) return;
  if (event.key === "ArrowLeft" && state.index > 0) { state.index -= 1; persist(); render(); }
  if (event.key === "ArrowRight" && state.index < state.rows.length - 1) { state.index += 1; persist(); render(); }
  if (event.key === "1" && state.rows.length) { updateField(state.rows[state.index], getAnnotation(state.rows[state.index]), "swap_validity", "flip"); render(); }
  if (event.key === "2" && state.rows.length) { updateField(state.rows[state.index], getAnnotation(state.rows[state.index]), "swap_validity", "unanswerable"); render(); }
  if (event.key === "3" && state.rows.length) { updateField(state.rows[state.index], getAnnotation(state.rows[state.index]), "swap_validity", "invalid"); render(); }
});

restore();
updateProgress();
