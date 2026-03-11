import { useState } from "react";

const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

const EXAMPLE_QUESTIONS = [
  "How do I unblock someone in chat?",
  "How many Shooting Gallery coins do I need?",
  "What is the difference between wounded conversion and battlefield treatment?",
  "How should I protect myself from enemy rally attacks?",
  "What is the best troop composition for PvP?",
  "When is the next update coming?",
];

const initialPanel = {
  answer: "",
  sources: [],
  model_used: null,
  latency_seconds: null,
  tokens_used: null,
  status: "idle",
  error: "",
};

function formatMetric(label, value, suffix = "") {
  if (value === null || value === undefined || value === "") {
    return `${label}: N/A`;
  }
  return `${label}: ${value}${suffix}`;
}

function Panel({ title, tone, data, loading }) {
  return (
    <section className={`panel panel-${tone}`}>
      <div className="panel-head">
        <div>
          <p className="eyebrow">{tone === "rag" ? "Vector Retrieval" : "Full Prompt Stuffing"}</p>
          <h2>{title}</h2>
        </div>
        <span className={`status-pill status-${loading ? "loading" : data.status}`}>{loading ? "loading" : data.status}</span>
      </div>

      <div className="metrics">
        <span>{formatMetric("Latency", data.latency_seconds, "s")}</span>
        <span>{formatMetric("Tokens", data.tokens_used)}</span>
        <span>{formatMetric("Model", data.model_used)}</span>
      </div>

      <div className="answer-card">
        {loading ? (
          <div className="skeleton">
            <span />
            <span />
            <span />
          </div>
        ) : data.error ? (
          <p className="error-text">{data.error}</p>
        ) : data.answer ? (
          <p className="answer-text">{data.answer}</p>
        ) : (
          <p className="placeholder">Submit a question to compare both approaches.</p>
        )}
      </div>

      {tone === "rag" && data.sources?.length > 0 ? (
        <div className="sources">
          <p>Sources</p>
          <ul>
            {data.sources.map((source, index) => (
              <li key={`${source.title}-${index}`}>
                <span>{source.title || "Untitled"}</span>
                {typeof source.similarity === "number" ? <strong>{source.similarity.toFixed(3)}</strong> : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}

export default function App() {
  const [question, setQuestion] = useState(EXAMPLE_QUESTIONS[0]);
  const [rag, setRag] = useState(initialPanel);
  const [longContext, setLongContext] = useState(initialPanel);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function fetchMethod(path) {
    const response = await fetch(`${API_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) {
      throw new Error(`${path} failed with ${response.status}`);
    }
    return response.json();
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (!question.trim() || isSubmitting) {
      return;
    }

    setIsSubmitting(true);
    setRag({ ...initialPanel, status: "loading" });
    setLongContext({ ...initialPanel, status: "loading" });

    const [ragResult, longContextResult] = await Promise.allSettled([
      fetchMethod("/api/rag"),
      fetchMethod("/api/long-context"),
    ]);

    setRag(
      ragResult.status === "fulfilled"
        ? { ...initialPanel, ...ragResult.value }
        : { ...initialPanel, status: "error", error: ragResult.reason.message },
    );
    setLongContext(
      longContextResult.status === "fulfilled"
        ? { ...initialPanel, ...longContextResult.value }
        : { ...initialPanel, status: "error", error: longContextResult.reason.message },
    );
    setIsSubmitting(false);
  }

  return (
    <main className="app-shell">
      <div className="backdrop backdrop-left" />
      <div className="backdrop backdrop-right" />
      <header className="hero">
        <p className="hero-kicker">GoG CS Bot</p>
        <h1>RAG vs Long Context</h1>
        <p className="hero-subtitle">A comparative experiment on AI customer service approaches</p>
      </header>

      <section className="chip-row" aria-label="Example questions">
        {EXAMPLE_QUESTIONS.map((item) => (
          <button key={item} type="button" className="chip" onClick={() => setQuestion(item)}>
            {item}
          </button>
        ))}
      </section>

      <section className="panel-grid">
        <Panel title="RAG Response" tone="rag" data={rag} loading={isSubmitting && rag.status === "loading"} />
        <Panel
          title="Long Context Response"
          tone="long"
          data={longContext}
          loading={isSubmitting && longContext.status === "loading"}
        />
      </section>

      <form className="composer" onSubmit={handleSubmit}>
        <label className="composer-label" htmlFor="question">
          Ask one question, send it to both systems
        </label>
        <div className="composer-row">
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask about Guns of Glory mechanics, FAQ, or support topics"
            rows={3}
          />
          <button type="submit" disabled={isSubmitting}>
            {isSubmitting ? "Comparing..." : "Ask Both"}
          </button>
        </div>
      </form>
    </main>
  );
}
