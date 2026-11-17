import { useEffect, useRef, useState } from "react";

export default function App() {
  const [docs, setDocs] = useState([]);
  const [status, setStatus] = useState("");
  const [messages, setMessages] = useState([]); // {role, content, sources?}
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef(null);

  const loadDocs = () =>
    fetch("/api/documents").then((r) => r.json()).then(setDocs).catch(() => {});

  useEffect(() => { loadDocs(); }, []);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  async function upload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setStatus(`Uploading & indexing ${file.name}...`);
    const fd = new FormData();
    fd.append("file", file);
    try {
      const r = await fetch("/api/upload", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "Upload failed");
      setStatus(`Indexed ${j.filename} (${j.chunks} chunks)`);
      loadDocs();
    } catch (err) {
      setStatus("Error: " + err.message);
    }
    e.target.value = "";
  }

  async function send(e) {
    e.preventDefault();
    const question = input.trim();
    if (!question || busy) return;
    const history = messages.map(({ role, content }) => ({ role, content }));
    setMessages((m) => [...m, { role: "user", content: question }, { role: "assistant", content: "", sources: [] }]);
    setInput("");
    setBusy(true);

    const patch = (fn) =>
      setMessages((m) => { const c = [...m]; c[c.length - 1] = fn(c[c.length - 1]); return c; });

    try {
      const r = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history }),
      });
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop();
        for (const p of parts) {
          const data = p.replace(/^data: /, "");
          if (data === "[DONE]") continue;
          const ev = JSON.parse(data);
          if (ev.sources) patch((x) => ({ ...x, sources: ev.sources }));
          if (ev.token) patch((x) => ({ ...x, content: x.content + ev.token }));
          if (ev.error) patch((x) => ({ ...x, content: x.content + "\n[error] " + ev.error }));
        }
      }
    } catch (err) {
      patch((x) => ({ ...x, content: "Request failed: " + err.message }));
    }
    setBusy(false);
  }

  return (
    <div className="app">
      <aside>
        <h2>Documents</h2>
        <label className="btn">
          Upload file
          <input type="file" accept=".pdf,.docx,.txt,.md,.csv" onChange={upload} hidden />
        </label>
        <p className="status">{status}</p>
        <ul>
          {docs.map((d) => (
            <li key={d.filename}>{d.filename} <small>({d.chunks})</small></li>
          ))}
        </ul>
      </aside>

      <main>
        <div className="chat">
          {messages.length === 0 && <p className="hint">Upload a document, then ask a question about it.</p>}
          {messages.map((m, i) => (
            <div key={i} className={"msg " + m.role}>
              <div className="bubble">{m.content || (busy ? "..." : "")}</div>
              {m.sources?.length > 0 && <div className="src">Sources: {m.sources.join(", ")}</div>}
            </div>
          ))}
          <div ref={endRef} />
        </div>
        <form onSubmit={send}>
          <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask about your documents..." />
          <button disabled={busy}>Send</button>
        </form>
      </main>
    </div>
  );
}
