import { useState, useRef, useEffect } from "react";
import "./App.css";

const API = "http://localhost:8000";

/* ─── Types ───────────────────────────────────────────────────────────────── */
interface Language {
  value: string;
  label: string;
  script: string;
}

interface ClassifyResult {
  category: string;
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  word_count: number;
  method: string;
  icon: string;
}

interface Session {
  id: string;
  name: string;
  article_count: number;
  message_count: number;
}

interface Article {
  id: string;
  category: string;
  word_count: number;
  source_type: string;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  mode?: string;
}

/* ─── Constants ───────────────────────────────────────────────────────────── */
const LANGUAGES: Language[] = [
  { value: "hindi",   label: "Hindi",   script: "हि" },
  { value: "bengali", label: "Bengali", script: "বা" },
  { value: "tamil",   label: "Tamil",   script: "த"  },
  { value: "telugu",  label: "Telugu",  script: "తె" },
  { value: "english", label: "English", script: "En" },
];

// Maps UI language value → short code the backend /classify/image expects
const LANG_CODE: Record<string, string> = {
  hindi:   "hi",
  bengali: "bn",
  tamil:   "ta",
  telugu:  "te",
  english: "en",
};

const CAT_META: Record<string, { color: string }> = {
  "World & International":   { color: "#3B82F6" },
  "Politics & Governance":   { color: "#EF4444" },
  "Business & Finance":      { color: "#F59E0B" },
  "Technology":              { color: "#8B5CF6" },
  "Sports":                  { color: "#10B981" },
  "Health & Medicine":       { color: "#EC4899" },
  "Entertainment & Culture": { color: "#F97316" },
  "Lifestyle & Society":     { color: "#D946EF" },
};

/* ─── Sub-components ──────────────────────────────────────────────────────── */
function BotIcon({
  size = 40,
  pulse = false,
  glow = false,
}: {
  size?: number;
  pulse?: boolean;
  glow?: boolean;
}) {
  return (
    <div
      className={`bot-sphere ${pulse ? "bot-pulse" : ""} ${glow ? "bot-glow" : ""}`}
      style={{ width: size, height: size }}
      aria-hidden="true"
    >
      <div className="bot-sheen" />
      <div className="bot-eye bot-eye-l" />
      <div className="bot-eye bot-eye-r" />
      <div className="bot-mouth" />
    </div>
  );
}

function ProbBar({
  name,
  value,
  color,
  isTop,
  delay = 0,
}: {
  name: string;
  value: number;
  color: string;
  isTop: boolean;
  delay?: number;
}) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setWidth(value * 100), delay + 80);
    return () => clearTimeout(t);
  }, [value, delay]);

  return (
    <div className={`prob-row ${isTop ? "prob-top" : ""}`}>
      <div className="prob-label">
        <span style={isTop ? { color } : {}}>{name}</span>
        <span className="prob-pct" style={isTop ? { color } : {}}>
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div className="prob-track">
        <div
          className="prob-fill"
          style={{
            width: `${width}%`,
            background: isTop ? color : undefined,
            transition: `width 0.9s cubic-bezier(0.34,1.56,0.64,1) ${delay}ms`,
          }}
        />
      </div>
    </div>
  );
}

function Bubble({ msg }: { msg: ChatMessage }) {
  return (
    <div className={`bubble-wrap ${msg.role}`}>
      {msg.role === "assistant" && (
        <div className="bubble-avatar">
          <BotIcon size={26} />
        </div>
      )}
      <div className={`bubble ${msg.role}`}>
        {msg.content}
        {msg.mode && msg.role === "assistant" && (
          <div className="bubble-meta">via {msg.mode}</div>
        )}
      </div>
    </div>
  );
}

/* ─── Main App ────────────────────────────────────────────────────────────── */
export default function App() {
  /* ── State ── */
  const [sessionId, setSessionId]       = useState<string | null>(null);
  const [sessions, setSessions]         = useState<Session[]>([]);
  const [articles, setArticles]         = useState<Article[]>([]);
  const [activeArticle, setActiveArt]   = useState<string | null>(null);

  const [step, setStep]               = useState<number>(1);
  const [inputMode, setInputMode]     = useState<"text" | "image">("text");
  const [language, setLanguage]       = useState<string>("hindi");
  const [inputText, setInputText]     = useState<string>("");
  const [dragOver, setDragOver]       = useState<boolean>(false);
  const [imgFile, setImgFile]         = useState<File | null>(null);
  const [imgPreview, setImgPreview]   = useState<string | null>(null);

  const [loading, setLoading]         = useState<boolean>(false);
  const [result, setResult]           = useState<ClassifyResult | null>(null);
  const [ocrText, setOcrText]         = useState<string | null>(null);

  const [chatOpen, setChatOpen]       = useState<boolean>(false);
  const [chatMsgs, setChatMsgs]       = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput]     = useState<string>("");
  const [chatLoading, setChatLoading] = useState<boolean>(false);

  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  const [newSessionName, setNSName]   = useState<string>("");
  const [showNSInput, setShowNSInput] = useState<boolean>(false);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileRef    = useRef<HTMLInputElement>(null);
  const nsInputRef = useRef<HTMLInputElement>(null);

  /* ── Init ── */
  useEffect(() => {
    const stored = localStorage.getItem("ml_session_id");
    if (stored) {
      setSessionId(stored);
      fetchSessionData(stored);
    } else {
      createSession();
    }
    fetchSessions();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMsgs]);

  useEffect(() => {
    if (showNSInput) nsInputRef.current?.focus();
  }, [showNSInput]);

  /* ── API helpers ── */
  async function createSession(name = "") {
    try {
      const r = await fetch(`${API}/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name || `Session ${new Date().toLocaleDateString("en-IN")}`,
        }),
      });
      const d = await r.json();
      setSessionId(d.session_id);
      localStorage.setItem("ml_session_id", d.session_id);
      setChatMsgs([]);
      setArticles([]);
      setResult(null);
      setStep(1);
      setActiveArt(null);
      await fetchSessions();
      return d.session_id as string;
    } catch (e) {
      console.error("createSession error:", e);
    }
  }

  async function fetchSessions() {
    try {
      const r = await fetch(`${API}/sessions`);
      const d = await r.json();
      setSessions(d.sessions || []);
    } catch (e) {
      console.error("fetchSessions error:", e);
    }
  }

  async function fetchSessionData(sid: string) {
    try {
      const r = await fetch(`${API}/session/${sid}`);
      if (!r.ok) return;
      const d = await r.json();
      setArticles(d.articles || []);
      if (d.messages?.length) {
        setChatMsgs(
          d.messages.map((m: { role: "user" | "assistant"; content: string }) => ({
            role: m.role,
            content: m.content,
          }))
        );
      }
    } catch (e) {
      console.error("fetchSessionData error:", e);
    }
  }

  async function deleteSession(sid: string, e: React.MouseEvent) {
    e.stopPropagation();
    if (!confirm("Delete this session and all its data?")) return;
    await fetch(`${API}/session/${sid}`, { method: "DELETE" });
    if (sid === sessionId) {
      localStorage.removeItem("ml_session_id");
      await createSession();
    }
    fetchSessions();
  }

  function switchSession(sid: string) {
    setSessionId(sid);
    localStorage.setItem("ml_session_id", sid);
    fetchSessionData(sid);
    setResult(null);
    setStep(1);
    setChatMsgs([]);
  }

  /* ── Classification ── */
  async function classify() {
    // Snapshot mutable values at call-time to avoid stale-closure bugs
    // where React state changes mid-await could alter which branch runs.
    const currentMode = inputMode;
    const currentText = inputText;
    const currentFile = imgFile;

    if (!currentText.trim() && !currentFile) return;

    // If the user previously OCR'd an image and the textarea now has that
    // extracted text, but they're still in "image" mode with a file loaded,
    // we must send the IMAGE to the backend — not the textarea text.
    // The mode is the single source of truth.
    if (currentMode === "text" && !currentText.trim()) return;
    if (currentMode === "image" && !currentFile) return;

    setLoading(true);
    setOcrText(null);

    try {
      if (currentMode === "text") {
        /* ── Text path ──
           POST /classify/text
           Response: { category, label, confidence, probabilities,
                       word_count, method, icon, article_id? }
        */
        const r = await fetch(`${API}/classify/text`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          // Match old working code: send { text } only.
          // session_id is optional — include it so articles are saved,
          // but the backend accepts it as Optional[str] = None.
          body: JSON.stringify({ text: currentText, session_id: sessionId || null }),
        });
        if (!r.ok) {
          const err = await r.json();
          alert("Classification error: " + (err.detail || r.statusText));
          return;
        }
        const data = await r.json();
        // Text response shape: { category, label, confidence, probabilities,
        //                        word_count, method, icon, article_id? }
        // article_id is at top level — strip it before storing as result.
        const { article_id, ...classifyResult } = data as ClassifyResult & { article_id?: string };
        setResult(classifyResult);
        if (article_id) setActiveArt(article_id);

      } else {
        /* ── Image path ──
           POST /classify/image  (multipart)
           Response: { success, extracted_text, classification: {...}, article_id? }
                  or { error: "..." }

           IMPORTANT: we use the FILE from the snapshot (currentFile), NOT
           anything from inputText. Setting inputText after the response is
           only for display; it must NOT trigger a second classify() call.
        */
        const form = new FormData();
        form.append("file", currentFile!);
        // Backend expects short code: "hi", "ta", "bn", "te", "en"
        // NOT the full word. This was the core mismatch causing wrong results.
        form.append("language", LANG_CODE[language] ?? "hi");
        // session_id is optional — send it if available so article is saved
        if (sessionId) form.append("session_id", sessionId);

        const r = await fetch(`${API}/classify/image`, { method: "POST", body: form });
        const data = await r.json();

        if (!r.ok || data.error) {
          alert("OCR / Classification error: " + (data.error || data.detail || r.statusText));
          return;
        }

        // Image response shape (from main.py):
        // { success, extracted_text, classification: { category, label,
        //   confidence, probabilities, word_count, method, icon }, article_id? }
        // Result lives under data.classification — not at top level.
        const classification = data.classification as ClassifyResult;
        setResult(classification);
        if (data.article_id) setActiveArt(data.article_id as string);

        // Store extracted text for the OCR strip word-count display only.
        // Do NOT put it in inputText — that would corrupt subsequent text classifications.
        setOcrText((data.extracted_text as string) || null);
      }

      await fetchSessionData(sessionId!);
      setStep(3);
    } catch (e) {
      alert("Network error: " + (e instanceof Error ? e.message : String(e)));
    } finally {
      setLoading(false);
    }
  }

  /* ── Chat ── */
  async function sendChat() {
    if (!chatInput.trim() || !sessionId) return;
    const msg = chatInput.trim();
    setChatInput("");
    setChatMsgs((p) => [...p, { role: "user", content: msg }]);
    setChatLoading(true);
    try {
      const r = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: msg,
          session_id: sessionId,
          article_id: activeArticle,
        }),
      });
      const d = await r.json();
      setChatMsgs((p) => [
        ...p,
        { role: "assistant", content: d.answer, mode: d.mode },
      ]);
    } catch {
      setChatMsgs((p) => [
        ...p,
        {
          role: "assistant",
          content: "API connection failed. Is the server running on port 8000?",
          mode: "error",
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  }

  /* ── Image helpers ── */
  function setImage(file: File | null | undefined) {
    if (!file || !file.type.startsWith("image/")) return;
    setImgFile(file);
    setImgPreview(URL.createObjectURL(file));
    setInputMode("image");
  }

  /* ── Derived values ── */
  const currentLang =
    LANGUAGES.find((l) => l.value === language) || LANGUAGES[0];
  const sortedProbs: [string, number][] = result?.probabilities
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : [];
  const catColor = result
    ? CAT_META[result.category]?.color || "#8B5CF6"
    : "#8B5CF6";
  const assistantCount = chatMsgs.filter((m) => m.role === "assistant").length;

  /* ──────────────────────── RENDER ──────────────────────── */
  return (
    <div className="app">
      <div className="app-bg" />

      {/* ════ SIDEBAR ════ */}
      <aside className={`sidebar ${sidebarOpen ? "sb-open" : "sb-closed"}`}>
        <div className="sb-head">
          <div className="sb-logo">
            <span className="sb-logo-icon">📰</span>
            {sidebarOpen && <span className="sb-logo-text">NewsAI</span>}
          </div>
          <button
            className="sb-toggle"
            onClick={() => setSidebarOpen((v) => !v)}
            title="Toggle sidebar"
          >
            {sidebarOpen ? "◀" : "▶"}
          </button>
        </div>

        {/* New session input */}
        {sidebarOpen && showNSInput ? (
          <form
            className="ns-form"
            onSubmit={async (e) => {
              e.preventDefault();
              await createSession(newSessionName);
              setNSName("");
              setShowNSInput(false);
            }}
          >
            <input
              ref={nsInputRef}
              className="ns-input"
              value={newSessionName}
              onChange={(e) => setNSName(e.target.value)}
              placeholder="Session name…"
            />
            <button type="submit" className="ns-confirm">✓</button>
            <button
              type="button"
              className="ns-cancel"
              onClick={() => setShowNSInput(false)}
            >
              ✕
            </button>
          </form>
        ) : (
          <button
            className="new-sess-btn"
            onClick={() =>
              sidebarOpen ? setShowNSInput(true) : createSession()
            }
            title="New session"
          >
            <span className="new-sess-plus">+</span>
            {sidebarOpen && <span>New Session</span>}
          </button>
        )}

        {/* Session list */}
        <div className="sb-sessions">
          {sidebarOpen && <div className="sb-section-label">Sessions</div>}
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`sess-item ${s.id === sessionId ? "sess-active" : ""}`}
              onClick={() => switchSession(s.id)}
              title={!sidebarOpen ? s.name : undefined}
            >
              <div className="sess-dot" />
              {sidebarOpen && (
                <>
                  <div className="sess-info">
                    <div className="sess-name">{s.name}</div>
                    <div className="sess-meta">
                      {s.article_count}A · {s.message_count}M
                    </div>
                  </div>
                  <button
                    className="sess-del"
                    onClick={(e) => deleteSession(s.id, e)}
                    title="Delete"
                  >
                    ✕
                  </button>
                </>
              )}
            </div>
          ))}
        </div>

        {/* Article list */}
        {sidebarOpen && articles.length > 0 && (
          <div className="sb-articles">
            <div className="sb-section-label">Articles</div>
            {articles
              .slice(-6)
              .reverse()
              .map((a) => (
                <div
                  key={a.id}
                  className={`art-item ${a.id === activeArticle ? "art-active" : ""}`}
                  style={
                    { "--ac": CAT_META[a.category]?.color || "#8B5CF6" } as React.CSSProperties
                  }
                  onClick={() => {
                    setActiveArt(a.id);
                    setChatOpen(true);
                  }}
                >
                  <div className="art-cat">{a.category}</div>
                  <div className="art-meta">
                    {a.word_count}w · {a.source_type}
                  </div>
                </div>
              ))}
          </div>
        )}
      </aside>

      {/* ════ MAIN ════ */}
      <main className="main">
        {/* Header */}
        <header className="hdr">
          <div>
            <div className="hdr-title">
              ML News Classifier
              <span className="hdr-badge">v2.0</span>
            </div>
            <div className="hdr-sub">
              OCR · Translation · 8 Categories · AI Chat
            </div>
          </div>
          <div className="hdr-steps">
            {(
              [
                ["01", "Language"],
                ["02", "Input"],
                ["03", "Results"],
              ] as [string, string][]
            ).map(([n, label], i) => (
              <div
                key={n}
                className={`hdr-step ${
                  step === i + 1
                    ? "step-cur"
                    : step > i + 1
                    ? "step-done"
                    : ""
                }`}
                onClick={() => step > i + 1 && setStep(i + 1)}
              >
                <span className="hdr-step-n">
                  {step > i + 1 ? "✓" : n}
                </span>
                <span className="hdr-step-label">{label}</span>
              </div>
            ))}
          </div>
        </header>

        <div className="content">

          {/* ══ STEP 1: Language ══ */}
          {step === 1 && (
            <section className="section fadeUp">
              <div className="card">
                <div className="card-head">
                  <div className="card-num">01</div>
                  <div>
                    <div className="card-title">Select Language</div>
                    <div className="card-desc">
                      Choose the language of your news article
                    </div>
                  </div>
                </div>
                <div className="lang-grid">
                  {LANGUAGES.map((lang, i) => (
                    <button
                      key={lang.value}
                      className={`lang-btn ${
                        language === lang.value ? "lang-selected" : ""
                      }`}
                      style={{ animationDelay: `${i * 35}ms` }}
                      onClick={() => {
                        setLanguage(lang.value);
                        setStep(2);
                      }}
                    >
                      <span className="lang-script">{lang.script}</span>
                      <span className="lang-label">{lang.label}</span>
                    </button>
                  ))}
                </div>
              </div>
            </section>
          )}

          {/* ══ STEP 2: Input ══ */}
          {step >= 2 && (
            <section className="section fadeUp">
              <div className="card">
                <div className="card-head">
                  <div className="card-num">02</div>
                  <div style={{ flex: 1 }}>
                    <div className="card-title">News Input</div>
                    <div className="card-desc">
                      Paste text or upload a scanned image
                    </div>
                  </div>
                  <button className="lang-chip" onClick={() => setStep(1)}>
                    <span>{currentLang.script}</span>
                    <span>{currentLang.label}</span>
                    <span className="chip-change">change</span>
                  </button>
                </div>

                {/* Mode toggle */}
                <div className="mode-toggle">
                  <button
                    className={`mode-btn ${inputMode === "text" ? "mode-on" : ""}`}
                    onClick={() => setInputMode("text")}
                  >
                    <span>📝</span> Text
                  </button>
                  <button
                    className={`mode-btn ${inputMode === "image" ? "mode-on" : ""}`}
                    onClick={() => setInputMode("image")}
                  >
                    <span>🖼</span> Image / OCR
                  </button>
                </div>

                {inputMode === "text" ? (
                  <div className="textarea-wrap">
                    <textarea
                      className="news-textarea"
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      onKeyDown={(e) =>
                        e.key === "Enter" && e.ctrlKey && classify()
                      }
                      placeholder="Paste your news article here… (Ctrl + Enter to classify)"
                      rows={6}
                    />
                    <div className="word-count">
                      {inputText.split(/\s+/).filter(Boolean).length} words
                    </div>
                  </div>
                ) : (
                  <div
                    className={`drop-zone ${dragOver ? "dz-over" : ""}`}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragOver(true);
                    }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setDragOver(false);
                      setImage(e.dataTransfer.files?.[0]);
                    }}
                    onClick={() => fileRef.current?.click()}
                  >
                    <input
                      ref={fileRef}
                      type="file"
                      accept="image/*"
                      style={{ display: "none" }}
                      onChange={(e) => setImage(e.target.files?.[0])}
                    />
                    {imgPreview ? (
                      <>
                        <img
                          src={imgPreview}
                          alt="preview"
                          className="img-preview"
                        />
                        <div className="dz-overlay">Click to change</div>
                      </>
                    ) : (
                      <div className="dz-placeholder">
                        <div className="dz-icon">🖼</div>
                        <div className="dz-text">
                          Drop image or click to upload
                        </div>
                        <div className="dz-sub">
                          PNG, JPG, HEIC · max 20 MB
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* OCR result strip — shown after image classification */}
                {ocrText && inputMode === "image" && (
                  <div className="ocr-strip">
                    <span className="ocr-label">OCR</span>
                    Extracted {ocrText.split(/\s+/).filter(Boolean).length} words from image
                  </div>
                )}

                <button
                  className={`classify-btn ${loading ? "btn-loading" : ""}`}
                  onClick={classify}
                  disabled={loading || (inputMode === "text" ? !inputText.trim() : !imgFile)}
                >
                  {loading ? (
                    <>
                      <span className="spinner" /> Analyzing…
                    </>
                  ) : (
                    "🔍 Classify Article"
                  )}
                </button>
              </div>
            </section>
          )}

          {/* ══ STEP 3: Result ══ */}
          {step >= 3 && result && (
            <section className="section fadeUp">
              {/* Hero */}
              <div
                className="result-hero"
                style={
                  {
                    "--cc":   catColor,
                    "--cc20": catColor + "20",
                    "--cc10": catColor + "10",
                  } as React.CSSProperties
                }
              >
                <div className="result-icon">{result.icon}</div>
                <div className="result-info">
                  <div className="result-cat" style={{ color: catColor }}>
                    {result.category}
                  </div>
                  <div className="result-stats">
                    <span className="result-conf">
                      {(result.confidence * 100).toFixed(1)}% confidence
                    </span>
                    <span className="result-sep">·</span>
                    <span>{result.word_count} words</span>
                    <span className="result-sep">·</span>
                    <span>via {result.method}</span>
                  </div>
                </div>
                {/* SVG confidence ring */}
                <svg
                  className="conf-ring"
                  viewBox="0 0 48 48"
                  width="88"
                  height="88"
                >
                  <circle
                    cx="24" cy="24" r="20"
                    fill="none"
                    stroke="rgba(255,255,255,0.08)"
                    strokeWidth="4"
                  />
                  <circle
                    cx="24" cy="24" r="20"
                    fill="none"
                    stroke={catColor}
                    strokeWidth="4"
                    strokeDasharray={`${result.confidence * 125.66} 125.66`}
                    strokeLinecap="round"
                    transform="rotate(-90 24 24)"
                    style={{
                      transition:
                        "stroke-dasharray 1.4s cubic-bezier(0.34,1.56,0.64,1)",
                    }}
                  />
                  <text
                    x="24" y="28"
                    textAnchor="middle"
                    fill={catColor}
                    fontSize="10"
                    fontWeight="700"
                    fontFamily="JetBrains Mono, monospace"
                  >
                    {(result.confidence * 100).toFixed(0)}%
                  </text>
                </svg>
              </div>

              {/* Probability bars */}
              <div className="prob-panel">
                <div className="prob-title">Category Distribution</div>
                {sortedProbs.map(([name, val], i) => (
                  <ProbBar
                    key={name}
                    name={name}
                    value={val}
                    color={CAT_META[name]?.color || "#8B5CF6"}
                    isTop={name === result.category}
                    delay={i * 55}
                  />
                ))}
                <div className="result-cta">
                  <button
                    className="cta-primary"
                    onClick={() => setChatOpen(true)}
                  >
                    <BotIcon size={18} />
                    Chat about this article
                  </button>
                  <button
                    className="cta-secondary"
                    onClick={() => {
                      setStep(2);
                      setResult(null);
                      setImgPreview(null);
                      setImgFile(null);
                      setOcrText(null);
                      setInputText("");
                      setInputMode("text");
                    }}
                  >
                    ↺ Classify another
                  </button>
                </div>
              </div>
            </section>
          )}
        </div>
      </main>

      {/* ════ FAB CHAT BUTTON ════ */}
      <button
        className={`fab ${chatOpen ? "fab-active" : ""}`}
        onClick={() => setChatOpen((v) => !v)}
        title="Open chat assistant"
      >
        <BotIcon
          size={30}
          pulse={!chatOpen && articles.length > 0}
          glow={chatOpen}
        />
        {!chatOpen && assistantCount > 0 && (
          <span className="fab-badge">{assistantCount}</span>
        )}
      </button>

      {/* ════ CHAT PANEL ════ */}
      {chatOpen && (
        <div className="chat-backdrop" onClick={() => setChatOpen(false)} />
      )}
      <div className={`chat-panel ${chatOpen ? "cp-open" : ""}`}>
        <div className="cp-head">
          <BotIcon size={38} pulse glow />
          <div className="cp-head-info">
            <div className="cp-title">News Assistant</div>
            <div className="cp-sub">
              Powered by local AI · {sessions.length} sessions
            </div>
          </div>
          <button className="cp-close" onClick={() => setChatOpen(false)}>
            ✕
          </button>
        </div>

        {/* Session switcher inside chat */}
        <div className="cp-sessions">
          {sessions.slice(0, 4).map((s) => (
            <button
              key={s.id}
              className={`cp-sess ${s.id === sessionId ? "cp-sess-on" : ""}`}
              onClick={() => switchSession(s.id)}
            >
              {s.name.slice(0, 12)}
              {s.name.length > 12 ? "…" : ""}
            </button>
          ))}
          <button
            className="cp-sess cp-sess-new"
            onClick={() => {
              setChatOpen(false);
              createSession();
            }}
          >
            + New
          </button>
        </div>

        {activeArticle && (
          <div className="cp-ctx">
            Focused on:{" "}
            <span className="cp-ctx-name">selected article</span>
            <button onClick={() => setActiveArt(null)}>× all</button>
          </div>
        )}

        <div className="cp-msgs">
          {chatMsgs.length === 0 && (
            <div className="cp-empty">
              <BotIcon size={52} pulse glow />
              <p>Ask me anything about your uploaded articles!</p>
              <div className="suggestions">
                {[
                  "Summarize this article",
                  "Who is involved?",
                  "What happened?",
                  "Key facts & figures",
                ].map((q) => (
                  <button
                    key={q}
                    className="sug-btn"
                    onClick={() => setChatInput(q)}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}
          {chatMsgs.map((m, i) => (
            <Bubble key={i} msg={m} />
          ))}
          {chatLoading && (
            <div className="bubble-wrap assistant">
              <div className="bubble-avatar">
                <BotIcon size={26} />
              </div>
              <div className="bubble assistant typing">
                <span />
                <span />
                <span />
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="cp-input-row">
          <input
            className="cp-input"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={(e) =>
              e.key === "Enter" && !e.shiftKey && sendChat()
            }
            placeholder="Ask about the news…"
          />
          <button
            className="cp-send"
            onClick={sendChat}
            disabled={chatLoading || !chatInput.trim()}
          >
            ➤
          </button>
        </div>
      </div>
    </div>
  );
}
