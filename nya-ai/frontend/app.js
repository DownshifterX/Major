const API = ""; // same-origin, backend serves this frontend

const state = {
  view: "dashboard",
  cases: [],
  historicalCases: [],
  currentCaseId: null,
  currentCase: null,
  currentTab: "overview",
  documents: [],
  analyses: {},      // analysis_type -> result object (with .citations)
  citationsMap: {},  // ref_key -> citation, scoped to last-rendered analysis
  loadingAnalysis: null,
  compareTargetId: null,
};

const main = document.getElementById("main");

// ---------------------------------------------------------------- API ----
async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : undefined,
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return res.json();
}
const getJSON = (path) => api(path);
const postJSON = (path, body) => api(path, { method: "POST", body: JSON.stringify(body) });
const patchJSON = (path, body) => api(path, { method: "PATCH", body: JSON.stringify(body) });

// ------------------------------------------------------------- helpers ---
function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}
function esc(s) {
  if (s === undefined || s === null) return "";
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function nl2br(s) { return esc(s).replace(/\n/g, "<br/>"); }

function refChips(refs) {
  if (!refs || !refs.length) return "";
  return refs.map(r => `<span class="ref-chip" data-ref="${esc(r)}">${esc(r)}</span>`).join("");
}

function attachRefChipHandlers(scopeEl) {
  scopeEl.querySelectorAll(".ref-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const citation = state.citationsMap[chip.dataset.ref];
      if (citation) openDrawer(citation);
    });
  });
}

function openDrawer(citation) {
  document.getElementById("drawer-content").innerHTML = `
    <h4>${esc(citation.case_name || citation.document_filename || "Source")}</h4>
    <div class="drawer-meta">
      ${citation.source_type === "historical_case" ? "Historical case" : "Current case document"}
      ${citation.document_filename ? " · " + esc(citation.document_filename) : ""}
      ${citation.page_number ? " · page " + esc(citation.page_number) : ""}
      ${citation.similarity_score != null ? " · relevance " + citation.similarity_score.toFixed(3) : ""}
    </div>
    <div class="drawer-excerpt">${esc(citation.excerpt || "(no excerpt captured)")}</div>
  `;
  document.getElementById("drawer").classList.add("open");
  document.getElementById("drawer-backdrop").classList.add("open");
}
function closeDrawer() {
  document.getElementById("drawer").classList.remove("open");
  document.getElementById("drawer-backdrop").classList.remove("open");
}
document.getElementById("drawer-close").addEventListener("click", closeDrawer);
document.getElementById("drawer-backdrop").addEventListener("click", closeDrawer);

function setCitationsMap(citations) {
  state.citationsMap = {};
  (citations || []).forEach(c => { state.citationsMap[c.ref_key] = c; });
}

// -------------------------------------------------------------- sidebar --
async function refreshSidebar() {
  state.cases = await getJSON("/cases");
  const list = document.getElementById("case-list");
  list.innerHTML = "";
  state.cases.forEach(c => {
    const item = el(`<div class="case-list-item ${c.id === state.currentCaseId ? "active" : ""}">${esc(c.title)}</div>`);
    item.addEventListener("click", () => openCase(c.id));
    list.appendChild(item);
  });
  document.getElementById("nav-dashboard").classList.toggle("active", state.view === "dashboard");
  document.getElementById("nav-corpus").classList.toggle("active", state.view === "corpus");
  document.getElementById("nav-search").classList.toggle("active", state.view === "search");
}
document.getElementById("nav-dashboard").addEventListener("click", () => { state.view = "dashboard"; state.currentCaseId = null; render(); });
document.getElementById("nav-corpus").addEventListener("click", () => { state.view = "corpus"; state.currentCaseId = null; render(); });
document.getElementById("nav-search").addEventListener("click", () => { state.view = "search"; state.currentCaseId = null; render(); });
document.getElementById("btn-new-case").addEventListener("click", () => { state.view = "new-case"; state.currentCaseId = null; render(); });

// ------------------------------------------------------------ dashboard --
async function renderDashboard() {
  const cases = state.cases;
  const analyzed = cases.filter(c => c.status === "analyzed").length;
  const indexed = cases.filter(c => c.status === "indexed").length;

  main.innerHTML = `
    <div class="page-header">
      <div>
        <h1>Dashboard</h1>
        <div class="sub">Case intelligence overview</div>
      </div>
      <button class="btn primary" id="dash-new-case">+ New case</button>
    </div>

    <div class="grid cols-3" style="margin-bottom:30px;">
      <div class="card"><div class="stat-num">${cases.length}</div><div class="stat-label">Total cases</div></div>
      <div class="card"><div class="stat-num">${indexed}</div><div class="stat-label">Documents indexed</div></div>
      <div class="card"><div class="stat-num">${analyzed}</div><div class="stat-label">Fully analyzed</div></div>
    </div>

    <h3 style="font-family:var(--serif); font-weight:400; font-size:17px; margin-bottom:14px;">Recent cases</h3>
    <div class="grid cols-2" id="dash-case-grid"></div>
  `;
  document.getElementById("dash-new-case").addEventListener("click", () => { state.view = "new-case"; render(); });

  const grid = document.getElementById("dash-case-grid");
  if (!cases.length) {
    grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1;">
      <div class="empty-title">No cases yet</div>
      Create your first case to upload documents and run source-grounded analysis.
    </div>`;
    return;
  }
  cases.forEach(c => {
    const card = el(`<div class="card case-card">
      <div class="title">${esc(c.title)}</div>
      <div class="meta">${esc(c.case_type || "Unspecified type")} · <span class="status-pill ${c.status}">${esc(c.status)}</span></div>
    </div>`);
    card.addEventListener("click", () => openCase(c.id));
    grid.appendChild(card);
  });
}

// -------------------------------------------------------------- corpus --
async function renderCorpus() {
  state.historicalCases = await getJSON("/historical-cases");
  main.innerHTML = `
    <div class="page-header">
      <div>
        <h1>Historical Case Corpus</h1>
        <div class="sub">The precedent library used for retrieval</div>
      </div>
    </div>
    <div class="synthetic-note" style="background:#e8f0fe; color:#1a73e8; border-color:#c5d7f8;">Real historical precedents from the Indian legal corpus (Indian Kanoon). Includes Supreme Court, High Court, and Tribunal judgments. Grounded retrieval is powered by this corpus.</div>
    <div class="grid cols-2" id="corpus-grid"></div>
  `;
  const grid = document.getElementById("corpus-grid");
  state.historicalCases.forEach(hc => {
    grid.appendChild(el(`
      <div class="card">
        <div class="title" style="font-weight:600; margin-bottom:6px;">${esc(hc.case_name)}</div>
        <div class="meta" style="color:var(--muted); font-size:12px; margin-bottom:10px;">
          ${esc(hc.court || "")} ${hc.year ? "· " + hc.year : ""} ${hc.citation ? "· " + esc(hc.citation) : ""}
        </div>
        <div style="font-size:13px; margin-bottom:8px;">${esc(hc.case_type || "")}</div>
        <div style="font-size:13px; color:var(--muted);">${esc((hc.summary || "").slice(0, 220))}…</div>
      </div>
    `));
  });
}

// ------------------------------------------------------------ new case --
function renderNewCase() {
  main.innerHTML = `
    <div class="page-header"><div><h1>Register New Case</h1><div class="sub">Import from an online judgment URL or create manually</div></div></div>
    
    <div class="grid cols-2" style="gap:24px; max-width:1050px;">
      <div class="card">
        <h3 style="margin-top:0;">Option A: Import from URL (Indian Kanoon / Court)</h3>
        <p style="font-size:13px; color:var(--muted); margin-bottom:14px;">Paste an Indian Kanoon or public court judgment URL. Nya AI will acquire the document, clean the text, extract its legal profile, and index it automatically.</p>
        <label>Judgment URL</label>
        <input type="url" id="f-url" placeholder="https://indiankanoon.org/doc/154031036/" />
        <label>Case Type</label>
        <input type="text" id="f-url-type" placeholder="e.g. Criminal, Civil, Constitutional" value="Criminal" />
        <div class="form-actions" style="margin-top:16px;">
          <button class="btn primary" id="f-url-submit">Acquire &amp; Index Case</button>
        </div>
        <div id="url-status" style="margin-top:10px;"></div>
      </div>

      <div class="card">
        <h3 style="margin-top:0;">Option B: Enter Case Details Manually</h3>
        <label>Case title</label>
        <input type="text" id="f-title" placeholder="e.g. State v. A. Nair — Alleged Theft" />
        <label>Case type</label>
        <input type="text" id="f-type" placeholder="e.g. Criminal - Theft, Civil - Contract" />
        <label>Facts</label>
        <textarea id="f-facts" placeholder="What happened, in your own words"></textarea>
        <label>Charges / legal issues</label>
        <textarea id="f-issues" placeholder="What is being alleged or disputed"></textarea>
        <label>Arguments (one per line)</label>
        <textarea id="f-arguments" placeholder="Prosecution: ...&#10;Defence: ..."></textarea>
        <label>Evidence on hand (one item per line)</label>
        <textarea id="f-evidence" placeholder="e.g. CCTV footage, signed statement"></textarea>
        <div class="form-actions">
          <button class="btn primary" id="f-submit">Create Case</button>
          <button class="btn ghost" id="f-cancel">Cancel</button>
        </div>
      </div>
    </div>
  `;

  document.getElementById("f-cancel").addEventListener("click", () => { state.view = "dashboard"; render(); });
  
  // URL Import handler
  document.getElementById("f-url-submit").addEventListener("click", async () => {
    const url = document.getElementById("f-url").value.trim();
    if (!url) { alert("Please enter a valid judgment URL."); return; }
    const status = document.getElementById("url-status");
    status.innerHTML = `<div class="loading-line">Acquiring document, extracting legal profile & indexing…</div>`;
    try {
      const c = await postJSON("/cases/from-url", {
        url,
        case_type: document.getElementById("f-url-type").value.trim() || "General",
      });
      await openCase(c.id);
    } catch (err) {
      status.innerHTML = `<div style="color:var(--danger); font-size:13px;">Acquisition failed: ${esc(err.message)}</div>`;
    }
  });

  // Manual Form handler
  document.getElementById("f-submit").addEventListener("click", async () => {
    const title = document.getElementById("f-title").value.trim();
    if (!title) { alert("Please give the case a title."); return; }
    const payload = {
      title,
      case_type: document.getElementById("f-type").value.trim(),
      facts: document.getElementById("f-facts").value.trim(),
      charges_issues: document.getElementById("f-issues").value.trim(),
      arguments_text: document.getElementById("f-arguments").value.trim(),
      evidence_text: document.getElementById("f-evidence").value.trim(),
    };
    const c = await postJSON("/cases", payload);
    await openCase(c.id);
  });
}

// --------------------------------------------------------- case workspace
async function openCase(caseId) {
  state.currentCaseId = caseId;
  state.view = "case";
  state.currentTab = "overview";
  state.analyses = {};
  await loadCaseData();
  render();
}

async function loadCaseData() {
  state.currentCase = await getJSON(`/cases/${state.currentCaseId}`);
  state.documents = await getJSON(`/cases/${state.currentCaseId}/documents`);
}

const TABS = [
  ["overview", "Overview"],
  ["documents", "Documents"],
  ["similar", "Similar Cases"],
  ["graph", "Citation Graph"],
  ["legal", "Legal Analysis"],
  ["prosecution-defence", "Arguments"],
  ["evidence", "Evidence"],
  ["contradictions", "Contradictions"],
  ["compare", "Case Comparison"],
  ["brief", "Pre-Hearing Brief"],
];

function renderCaseWorkspace() {
  const c = state.currentCase;
  main.innerHTML = `
    <div class="page-header">
      <div>
        <h1>${esc(c.title)}</h1>
        <div class="sub">${esc(c.case_type || "Unspecified type")} ${c.legal_domain ? "· " + esc(c.legal_domain) : ""} ${c.case_number ? "· #" + esc(c.case_number) : ""}</div>
      </div>
      <div style="display:flex; align-items:center; gap:12px;">
        <span class="status-pill ${c.status}">${esc(c.status)}</span>
        <button class="btn ghost" id="btn-export-pdf" style="display:none;">Export PDF</button>
        <button class="btn primary" id="btn-analyze">Analyze case</button>
      </div>
    </div>
    <div class="tabs" id="tabs"></div>
    <div id="tab-content"></div>
  `;
  const tabsEl = document.getElementById("tabs");
  TABS.forEach(([key, label]) => {
    const t = el(`<div class="tab ${state.currentTab === key ? "active" : ""}">${label}</div>`);
    t.addEventListener("click", () => { state.currentTab = key; renderTabContent(); syncTabHighlight(); });
    tabsEl.appendChild(t);
  });
  document.getElementById("btn-analyze").addEventListener("click", runFullAnalysis);
  
  const btnExport = document.getElementById("btn-export-pdf");
  if (c.status === "analyzed") {
      btnExport.style.display = "inline-flex";
      btnExport.addEventListener("click", () => {
          window.open(`${API}/cases/${state.currentCaseId}/brief/export`, "_blank");
      });
  }
  
  renderTabContent();
}
function syncTabHighlight() {
  [...document.querySelectorAll(".tab")].forEach((t, i) => t.classList.toggle("active", TABS[i][0] === state.currentTab));
}

async function runFullAnalysis() {
  const btn = document.getElementById("btn-analyze");
  btn.disabled = true; btn.textContent = "Analyzing…";
  try {
    const result = await postJSON(`/cases/${state.currentCaseId}/analyze`, {});
    state.analyses = result;
    await loadCaseData();
    renderCaseWorkspace();
    if (state.currentTab === "overview") { state.currentTab = "similar"; renderCaseWorkspace(); }
  } catch (e) {
    alert("Analysis failed: " + e.message);
  } finally {
    btn.disabled = false; btn.textContent = "Analyze case";
  }
}

function renderTabContent() {
  const holder = document.getElementById("tab-content");
  switch (state.currentTab) {
    case "overview": return renderOverviewTab(holder);
    case "documents": return renderDocumentsTab(holder);
    case "similar": return renderAnalysisTab(holder, "similar_cases", "/similar-cases", renderSimilarCases);
    case "graph": return renderGraphTab(holder);
    case "legal": return renderAnalysisTab(holder, "legal_provisions", "/legal-provisions", renderLegalProvisions);
    case "prosecution-defence": return renderAnalysisTab(holder, "prosecution_defence", "/prosecution-defence", renderProsecutionDefence);
    case "evidence": return renderAnalysisTab(holder, "evidence_gaps", "/evidence-gaps", renderEvidenceGaps);
    case "contradictions": return renderAnalysisTab(holder, "contradictions", "/contradictions", renderContradictions);
    case "compare": return renderCompareTab(holder);
    case "brief": return renderAnalysisTab(holder, "brief", "/brief", renderBrief);
  }
}

async function renderGraphTab(holder) {
  holder.innerHTML = `<div class="loading-line">Building precedent citation graph…</div>`;
  try {
    const data = await getJSON(`/cases/${state.currentCaseId}/citation-graph`);
    const nodes = data.nodes || [];
    const links = data.links || [];
    const metrics = data.metrics || {};

    holder.innerHTML = `
      <div class="page-header" style="margin-bottom:16px;">
        <div>
          <h2 style="font-family:var(--serif); font-size:18px; margin:0;">Citation &amp; Precedent Authority Network</h2>
          <div class="sub">Inter-case citation relationships across the historical precedent corpus</div>
        </div>
      </div>

      <div class="grid cols-3" style="margin-bottom:20px;">
        <div class="card"><div class="stat-num">${metrics.node_count || nodes.length}</div><div class="stat-label">Connected Precedents</div></div>
        <div class="card"><div class="stat-num">${metrics.edge_count || links.length}</div><div class="stat-label">Citation Paths</div></div>
        <div class="card"><div class="stat-num">${metrics.focus_cited_by_count || 0}</div><div class="stat-label">Times Cited Authority</div></div>
      </div>

      <div class="card" style="margin-bottom:20px;">
        <h3 style="margin-top:0;">Precedent Authority Ranking</h3>
        <div style="font-size:12.5px; color:var(--muted); margin-bottom:12px;">Precedents ranked by in-degree citation frequency in case law.</div>
        <div class="grid cols-2" style="gap:12px;">
          ${nodes.slice(0, 10).map(n => `
            <div style="border:1px solid var(--border); padding:10px 14px; border-radius:4px; background:#fff;">
              <div style="font-weight:600; font-size:13.5px; color:var(--primary);">${esc(n.label)}</div>
              <div style="font-size:12px; color:var(--muted); margin-top:2px;">${esc(n.court || "")} ${n.year ? "· " + n.year : ""}</div>
              <div style="margin-top:6px; font-size:12px;">
                <span class="score-badge" style="background:#e8f0fe; color:#1a73e8;">Cited ${n.in_degree} times</span>
                ${n.out_degree ? `<span class="score-badge" style="background:#f1f3f4; color:#5f6368;">Cites ${n.out_degree} cases</span>` : ""}
              </div>
            </div>
          `).join("")}
        </div>
      </div>

      <div class="card">
        <h3 style="margin-top:0;">Citation Relationships</h3>
        <table style="width:100%; border-collapse:collapse; font-size:12.5px; margin-top:8px;">
          <thead>
            <tr style="text-align:left; border-bottom:2px solid var(--border); color:var(--muted);">
              <th style="padding:6px 8px;">Citing Case</th>
              <th style="padding:6px 8px;">Relation</th>
              <th style="padding:6px 8px;">Cited Authority</th>
            </tr>
          </thead>
          <tbody>
            ${links.slice(0, 15).map(l => {
              const srcNode = nodes.find(n => n.id === l.source) || { label: l.source };
              const tgtNode = nodes.find(n => n.id === l.target) || { label: l.target };
              return `
                <tr style="border-bottom:1px solid #f0f0f0;">
                  <td style="padding:8px; font-weight:500;">${esc(srcNode.label)}</td>
                  <td style="padding:8px; color:var(--accent); font-family:monospace;">— cites →</td>
                  <td style="padding:8px; font-weight:500;">${esc(tgtNode.label)}</td>
                </tr>
              `;
            }).join("")}
          </tbody>
        </table>
      </div>
    `;
  } catch (e) {
    holder.innerHTML = `<div class="empty-state"><div class="empty-title">Couldn't load citation graph</div>${esc(e.message)}</div>`;
  }
}

function renderOverviewTab(holder) {
  const c = state.currentCase;
  holder.innerHTML = `
    <div class="grid cols-2">
      <div class="card"><h3>Facts</h3><div>${nl2br(c.facts) || "<span style='color:var(--muted)'>Not provided</span>"}</div></div>
      <div class="card"><h3>Charges / Issues</h3><div>${nl2br(c.charges_issues) || "<span style='color:var(--muted)'>Not provided</span>"}</div></div>
      <div class="card"><h3>Arguments</h3><div>${nl2br(c.arguments_text) || "<span style='color:var(--muted)'>Not provided</span>"}</div></div>
      <div class="card"><h3>Evidence on hand</h3><div>${nl2br(c.evidence_text) || "<span style='color:var(--muted)'>Not provided</span>"}</div></div>
    </div>
    <p class="disclaimer">Nya(AI) is a decision-support tool for legal professionals. It does not determine guilt, innocence, or sentencing, and every generated claim should be checked against the linked source before relying on it.</p>
  `;
}

// ------------------------------------------------------------ documents --
function renderDocumentsTab(holder) {
  holder.innerHTML = `
    <div class="upload-zone" id="upload-zone">
      <div style="font-weight:600; margin-bottom:4px;">Upload a case document</div>
      <div style="font-size:12.5px;">.pdf, .txt, or .md — click to choose a file</div>
      <input type="file" id="file-input" style="display:none" accept=".pdf,.txt,.md" />
    </div>
    <div id="upload-status"></div>
    <div class="card" id="doc-list-card"><h3>Documents (${state.documents.length})</h3><div id="doc-list"></div></div>
  `;
  document.getElementById("upload-zone").addEventListener("click", () => document.getElementById("file-input").click());
  document.getElementById("file-input").addEventListener("change", handleUpload);

  const listEl = document.getElementById("doc-list");
  if (!state.documents.length) {
    listEl.innerHTML = `<div style="color:var(--muted); font-size:13px;">No documents uploaded yet.</div>`;
  } else {
    state.documents.forEach(d => {
      listEl.appendChild(el(`
        <div class="doc-row">
          <div>${esc(d.filename)} <span style="color:var(--muted); font-size:11.5px;">· ${esc(d.doc_type)} · ${d.num_chunks} chunks</span></div>
          <button class="btn ghost small" data-del="${d.id}">Remove</button>
        </div>
      `));
    });
    listEl.querySelectorAll("[data-del]").forEach(btn => {
      btn.addEventListener("click", async () => {
        await api(`/cases/${state.currentCaseId}/documents/${btn.dataset.del}`, { method: "DELETE" });
        await loadCaseData();
        renderDocumentsTab(holder);
      });
    });
  }
}

async function handleUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const status = document.getElementById("upload-status");
  status.innerHTML = `<div class="loading-line">Uploading and indexing ${esc(file.name)}…</div>`;
  const formData = new FormData();
  formData.append("file", file);
  formData.append("doc_type", "general");
  try {
    await api(`/cases/${state.currentCaseId}/documents`, { method: "POST", body: formData });
    await loadCaseData();
    status.innerHTML = "";
    renderDocumentsTab(document.getElementById("tab-content"));
    refreshSidebar();
  } catch (err) {
    status.innerHTML = `<div style="color:var(--danger); font-size:13px;">Upload failed: ${esc(err.message)}</div>`;
  }
}

// -------------------------------------------------------- generic runner
async function renderAnalysisTab(holder, key, endpointSuffix, renderer) {
  holder.innerHTML = `<div class="loading-line">Loading ${key.replace("_", " ")}…</div>`;
  try {
    let result = state.analyses[key];
    if (!result) {
      result = await getJSON(`/cases/${state.currentCaseId}${endpointSuffix}`);
      state.analyses[key] = result;
    }
    setCitationsMap(result.citations);
    holder.innerHTML = "";
    renderer(holder, result);
    attachRefChipHandlers(holder);
  } catch (e) {
    holder.innerHTML = `<div class="empty-state"><div class="empty-title">Couldn't load this analysis</div>${esc(e.message)}</div>`;
  }
}

function renderSimilarCases(holder, result) {
  const cases = result.similar_cases || [];
  if (!cases.length) {
    holder.appendChild(el(`<div class="empty-state"><div class="empty-title">No similar cases retrieved yet</div>Run "Analyze case" from the top of the page.</div>`));
    return;
  }
  cases.forEach(sc => {
    const b = sc.score_breakdown || {};
    const breakdownHtml = sc.score_breakdown ? `
      <div style="display:flex; flex-wrap:wrap; gap:6px; margin:8px 0;">
        <span class="score-badge" style="background:#e8f0fe; color:#1a73e8;" title="Embedding semantic cosine similarity">Semantic: ${b.semantic != null ? b.semantic : '—'}</span>
        <span class="score-badge" style="background:#fce8e6; color:#c5221f;" title="BM25 keyword overlap">BM25: ${b.lexical != null ? b.lexical : '—'}</span>
        <span class="score-badge" style="background:#e6f4ea; color:#137333;" title="Statutory section match">Statute: ${b.statute_overlap != null ? b.statute_overlap : '—'}</span>
        <span class="score-badge" style="background:#fef7e0; color:#b06000;" title="Precedent citation frequency">Authority: ${b.citation_authority != null ? b.citation_authority : '—'}</span>
      </div>
    ` : "";

    holder.appendChild(el(`
      <div class="result-card">
        <div class="rc-title">${esc(sc.case_name)} ${sc.relevance_score != null ? `<span class="score-badge">relevance ${sc.relevance_score}</span>` : ""} ${sc.rerank_total_score != null ? `<span class="score-badge" style="background:var(--primary); color:#fff;">rerank ${sc.rerank_total_score}</span>` : ""}</div>
        <div class="rc-meta">${esc(sc.court || "")} ${sc.year ? "· " + sc.year : ""}</div>
        ${breakdownHtml}
        <div class="rc-body">${esc(sc.why_relevant)}</div>
        <div style="margin-top:8px; font-style:italic; color:var(--muted); font-size:12.6px;">"${esc((sc.matching_excerpt || "").slice(0, 220))}…"</div>
        <div style="margin-top:8px;">${refChips([sc.ref_key])}</div>
      </div>
    `));
  });
}

function renderLegalProvisions(holder, result) {
  const provisions = result.legal_provisions || [];
  provisions.forEach(p => {
    holder.appendChild(el(`
      <div class="result-card">
        <div class="rc-title">${esc(p.provision)}</div>
        <div class="rc-body">${esc(p.why_relevant)}</div>
        <div style="margin-top:8px;">${refChips(p.source_refs)}</div>
      </div>
    `));
  });
}

function renderProsecutionDefence(holder, result) {
  const wrap = el(`<div class="pd-columns">
    <div class="pd-side">
      <div class="column-heading">Prosecution</div>
      <ul id="pd-pros-args"></ul>
      <div class="column-heading" style="margin-top:14px;">Precedents cited</div>
      <div id="pd-pros-refs"></div>
      <div class="column-heading" style="margin-top:14px;">Gaps</div>
      <ul id="pd-pros-gaps"></ul>
    </div>
    <div class="pd-side">
      <div class="column-heading">Defence</div>
      <ul id="pd-def-args"></ul>
      <div class="column-heading" style="margin-top:14px;">Precedents cited</div>
      <div id="pd-def-refs"></div>
      <div class="column-heading" style="margin-top:14px;">Gaps</div>
      <ul id="pd-def-gaps"></ul>
    </div>
  </div>`);
  holder.appendChild(wrap);
  const p = result.prosecution || {}, d = result.defence || {};
  (p.arguments || []).forEach(a => wrap.querySelector("#pd-pros-args").appendChild(el(`<li>${esc(a)}</li>`)));
  (d.arguments || []).forEach(a => wrap.querySelector("#pd-def-args").appendChild(el(`<li>${esc(a)}</li>`)));
  (p.gaps || []).forEach(a => wrap.querySelector("#pd-pros-gaps").appendChild(el(`<li>${esc(a)}</li>`)));
  (d.gaps || []).forEach(a => wrap.querySelector("#pd-def-gaps").appendChild(el(`<li>${esc(a)}</li>`)));
  wrap.querySelector("#pd-pros-refs").innerHTML = refChips(p.supporting_precedents) || `<span style="color:var(--muted); font-size:12px;">None</span>`;
  wrap.querySelector("#pd-def-refs").innerHTML = refChips(d.supporting_precedents) || `<span style="color:var(--muted); font-size:12px;">None</span>`;
  if (result.note) holder.appendChild(el(`<p class="disclaimer">${esc(result.note)}</p>`));
}

function renderEvidenceGaps(holder, result) {
  const present = el(`<div class="evidence-col"><div class="tag-present">Evidence present</div><ul></ul></div>`);
  (result.evidence_present || []).forEach(e => present.querySelector("ul").appendChild(el(`<li>${esc(e)}</li>`)));
  const gaps = el(`<div class="evidence-col"><div class="tag-verify">Potential evidence to verify</div><ul></ul></div>`);
  (result.potential_evidence_to_verify || []).forEach(e => gaps.querySelector("ul").appendChild(el(`<li>${esc(e)}</li>`)));
  holder.appendChild(present);
  holder.appendChild(gaps);
}

function renderContradictions(holder, result) {
  const items = result.contradictions || [];
  items.forEach(c => {
    holder.appendChild(el(`
      <div class="result-card">
        <div class="rc-title">${esc(c.type.replace(/_/g, " "))}</div>
        <div class="rc-body">${esc(c.description)}</div>
        ${c.values_found && c.values_found.length ? `<div style="margin-top:6px; font-size:12.5px; color:var(--muted);">Values found: ${c.values_found.map(esc).join(", ")}</div>` : ""}
        <div style="margin-top:8px;">${refChips(c.source_refs)}</div>
      </div>
    `));
  });
}

function renderBrief(holder, result) {
  const sections = [
    ["Case overview", result.case_overview],
    ["Key facts", result.key_facts],
    ["Legal issues", result.legal_issues],
  ];
  sections.forEach(([title, body]) => {
    holder.appendChild(el(`<div class="analysis-block"><h3>${esc(title)}</h3><div>${nl2br(body)}</div></div>`));
  });

  const laws = el(`<div class="analysis-block"><h3>Relevant laws</h3></div>`);
  (result.relevant_laws || []).forEach(p => laws.appendChild(el(`<div class="result-card"><div class="rc-title">${esc(p.provision)}</div><div class="rc-body">${esc(p.why_relevant)}</div></div>`)));
  if (!(result.relevant_laws || []).length) laws.appendChild(el(`<div style="color:var(--muted); font-size:13px;">Run Legal Analysis first.</div>`));
  holder.appendChild(laws);

  const precedents = el(`<div class="analysis-block"><h3>Relevant precedents</h3></div>`);
  (result.relevant_precedents || []).forEach(p => precedents.appendChild(el(`<div class="result-card"><div class="rc-title">${esc(p.case_name)}</div><div class="rc-body">${esc(p.why_relevant)}</div></div>`)));
  if (!(result.relevant_precedents || []).length) precedents.appendChild(el(`<div style="color:var(--muted); font-size:13px;">Run Similar Cases first.</div>`));
  holder.appendChild(precedents);

  const pd = el(`<div class="analysis-block"><h3>Prosecution vs Defence</h3></div>`);
  const pdWrap = el(`<div class="pd-columns"><div class="pd-side"><div class="column-heading">Prosecution</div><ul></ul></div><div class="pd-side"><div class="column-heading">Defence</div><ul></ul></div></div>`);
  ((result.prosecution_arguments || {}).arguments || []).forEach(a => pdWrap.querySelector(".pd-side:nth-child(1) ul").appendChild(el(`<li>${esc(a)}</li>`)));
  ((result.defence_arguments || {}).arguments || []).forEach(a => pdWrap.querySelector(".pd-side:nth-child(2) ul").appendChild(el(`<li>${esc(a)}</li>`)));
  pd.appendChild(pdWrap);
  holder.appendChild(pd);

  const evi = el(`<div class="analysis-block"><h3>Evidence summary & gaps</h3></div>`);
  const eviWrap = el(`<div><div class="tag-present">Present</div><ul></ul><div class="tag-verify" style="margin-top:10px;">To verify</div><ul></ul></div>`);
  (result.evidence_summary || []).forEach(a => eviWrap.querySelector("ul:nth-of-type(1)").appendChild(el(`<li>${esc(a)}</li>`)));
  (result.evidence_gaps || []).forEach(a => eviWrap.querySelector("ul:nth-of-type(2)").appendChild(el(`<li>${esc(a)}</li>`)));
  evi.appendChild(eviWrap);
  holder.appendChild(evi);

  const contra = el(`<div class="analysis-block"><h3>Contradictions</h3></div>`);
  (result.contradictions || []).forEach(c => contra.appendChild(el(`<div class="result-card"><div class="rc-title">${esc((c.type||"").replace(/_/g," "))}</div><div class="rc-body">${esc(c.description)}</div></div>`)));
  holder.appendChild(contra);

  const questions = el(`<div class="analysis-block"><h3>Questions to examine</h3><ul></ul></div>`);
  (result.questions_to_examine || []).forEach(q => questions.querySelector("ul").appendChild(el(`<li>${esc(q)}</li>`)));
  holder.appendChild(questions);

  if (result.warning) holder.appendChild(el(`<p class="disclaimer" style="color:var(--warn);">${esc(result.warning)}</p>`));
  holder.appendChild(el(`<p class="disclaimer">This brief is assembled entirely from the case data and the analyses you have run — it is a starting point for a legal professional, not a final legal conclusion.</p>`));
}

// -------------------------------------------------------------- compare --
async function renderCompareTab(holder) {
  if (!state.historicalCases.length) state.historicalCases = await getJSON("/historical-cases");
  holder.innerHTML = `
    <div class="card" style="margin-bottom:20px;">
      <label>Choose a historical case to compare against</label>
      <select id="compare-select">
        <option value="">Select a case…</option>
        ${state.historicalCases.map(hc => `<option value="${hc.id}">${esc(hc.case_name)}</option>`).join("")}
      </select>
      <div class="form-actions"><button class="btn primary" id="compare-run">Compare</button></div>
    </div>
    <div id="compare-result"></div>
  `;
  document.getElementById("compare-run").addEventListener("click", async () => {
    const hcId = document.getElementById("compare-select").value;
    if (!hcId) return;
    const resultHolder = document.getElementById("compare-result");
    resultHolder.innerHTML = `<div class="loading-line">Comparing…</div>`;
    try {
      const result = await postJSON(`/cases/${state.currentCaseId}/compare`, { historical_case_id: hcId });
      setCitationsMap(result.citations);
      resultHolder.innerHTML = "";
      renderComparisonResult(resultHolder, result);
      attachRefChipHandlers(resultHolder);
    } catch (e) {
      resultHolder.innerHTML = `<div class="empty-state">${esc(e.message)}</div>`;
    }
  });
}

function renderComparisonResult(holder, result) {
  const cur = result.current_case || {}, hist = result.historical_case || {};
  holder.appendChild(el(`
    <div class="grid cols-2" style="margin-bottom:20px;">
      <div class="card">
        <h3>Current case</h3>
        <div style="font-size:13px;"><b>Facts:</b> ${esc(cur.facts || "—")}</div>
        <div style="font-size:13px; margin-top:6px;"><b>Issues:</b> ${esc(cur.issues || "—")}</div>
      </div>
      <div class="card">
        <h3>${esc(hist.case_name)}</h3>
        <div style="font-size:13px;"><b>Summary:</b> ${esc((hist.summary || "").slice(0, 300))}…</div>
        <div style="font-size:13px; margin-top:6px;"><b>Outcome:</b> ${esc(hist.outcome || "—")}</div>
      </div>
    </div>
  `));
  const sim = el(`<div class="analysis-block"><h3>Similarities</h3><ul></ul></div>`);
  (result.similarities || []).forEach(s => sim.querySelector("ul").appendChild(el(`<li>${esc(s)}</li>`)));
  holder.appendChild(sim);

  const diff = result.differences || {};
  holder.appendChild(el(`
    <div class="analysis-block"><h3>Differences</h3>
      <div class="grid cols-2">
        <div><div class="column-heading">Unique to current case</div><ul>${(diff.unique_to_current_case||[]).map(x=>`<li>${esc(x)}</li>`).join("")}</ul></div>
        <div><div class="column-heading">Unique to historical case</div><ul>${(diff.unique_to_historical_case||[]).map(x=>`<li>${esc(x)}</li>`).join("")}</ul></div>
      </div>
    </div>
  `));

  holder.appendChild(el(`<div class="analysis-block"><h3>Legal significance</h3><div>${esc(result.legal_significance || "")}</div></div>`));
  const app = el(`<div class="analysis-block"><h3>Applicability considerations</h3><ul></ul></div>`);
  (result.applicability_considerations || []).forEach(a => app.querySelector("ul").appendChild(el(`<li>${esc(a)}</li>`)));
  holder.appendChild(app);
}

// ------------------------------------------------------------- router ---
async function render() {
  await refreshSidebar();
  if (state.view === "dashboard") return renderDashboard();
  if (state.view === "corpus") return renderCorpus();
  if (state.view === "search") return renderSearch();
  if (state.view === "new-case") return renderNewCase();
  if (state.view === "case") return renderCaseWorkspace();
}

// --------------------------------------------------------- legal research search
async function renderSearch() {
  main.innerHTML = `
    <div class="page-header">
      <div>
        <h1>Legal Research</h1>
        <div class="sub">Hybrid semantic + BM25 search across the Indian legal precedent corpus</div>
      </div>
    </div>
    <div class="card" style="margin-bottom:20px;">
      <div style="display:flex; gap:12px; align-items:center;">
        <input type="text" id="search-q" placeholder="e.g. Section 302 IPC murder appeal Supreme Court" style="flex:1; font-size:15px;" />
        <button class="btn primary" id="search-btn" style="white-space:nowrap;">Search Corpus</button>
      </div>
      <div style="font-size:12px; color:var(--muted); margin-top:8px;">Searches all ${0} indexed chunks with hybrid retrieval (semantic embedding + BM25 lexical).</div>
    </div>
    <div id="search-results"></div>
  `;

  const doSearch = async () => {
    const q = document.getElementById("search-q").value.trim();
    if (!q) return;
    const resultsEl = document.getElementById("search-results");
    resultsEl.innerHTML = `<div class="loading-line">Searching…</div>`;
    try {
      const results = await getJSON(`/api/search?q=${encodeURIComponent(q)}&limit=15`);
      if (!results.length) {
        resultsEl.innerHTML = `<div class="empty-state"><div class="empty-title">No results found</div>Try broader search terms or ensure the corpus has been ingested.</div>`;
        return;
      }
      resultsEl.innerHTML = "";
      results.forEach(r => {
        resultsEl.appendChild(el(`
          <div class="result-card">
            <div class="rc-title">
              ${esc(r.case_name)}
              ${r.doc_url ? `<a href="${esc(r.doc_url)}" target="_blank" style="font-size:12px; color:var(--accent); font-weight:400; margin-left:8px;">↗ Source</a>` : ""}
            </div>
            <div class="rc-meta">${esc(r.court || "")} ${r.year ? "· " + r.year : ""}</div>
            <div style="display:flex; flex-wrap:wrap; gap:6px; margin:8px 0;">
              <span class="score-badge" style="background:#e8f0fe; color:#1a73e8;" title="Embedding cosine similarity">Semantic: ${r.semantic_score}</span>
              <span class="score-badge" style="background:#fce8e6; color:#c5221f;" title="BM25 lexical match">BM25: ${r.lexical_score}</span>
              <span class="score-badge" style="background:#e6f4ea; color:#137333;" title="Combined score">Hybrid: ${r.hybrid_score}</span>
            </div>
            <div class="rc-body" style="font-style:italic; color:var(--muted); font-size:12.6px;">"${esc((r.excerpt || "").slice(0, 250))}…"</div>
          </div>
        `));
      });
    } catch (e) {
      resultsEl.innerHTML = `<div class="empty-state"><div class="empty-title">Search failed</div>${esc(e.message)}</div>`;
    }
  };

  document.getElementById("search-btn").addEventListener("click", doSearch);
  document.getElementById("search-q").addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });
}

render();
