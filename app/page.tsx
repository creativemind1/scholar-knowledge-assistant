"use client";

import { useState, useRef, useEffect, useCallback } from "react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: { text: string; page: number }[];
  isStreaming?: boolean;
}

interface Status {
  loaded: boolean;
  scholarName?: string;
  totalChunks?: number;
  uploadedAt?: string;
}

export default function Home() {
  const [status, setStatus] = useState<Status>({ loaded: false });
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isQuerying, setIsQuerying] = useState(false);
  const [uploadState, setUploadState] = useState<{
    uploading: boolean;
    progress: number;
    total: number;
    message: string;
    error: string;
    done: boolean;
  }>({
    uploading: false,
    progress: 0,
    total: 0,
    message: "",
    error: "",
    done: false,
  });
  const [scholarName, setScholarName] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [showSources, setShowSources] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatInputRef = useRef<HTMLTextAreaElement>(null);

  const checkStatus = useCallback(async () => {
    const res = await fetch("/api/status");
    const data = await res.json();
    setStatus(data);
  }, []);

  useEffect(() => {
    checkStatus();
  }, [checkStatus]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleFileUpload = async (file: File) => {
    if (!file.name.endsWith(".pdf")) {
      setUploadState((s) => ({ ...s, error: "Please upload a PDF file." }));
      return;
    }

    const name = scholarName.trim() || file.name.replace(".pdf", "");
    const formData = new FormData();
    formData.append("pdf", file);
    formData.append("scholarName", name);

    setUploadState({
      uploading: true,
      progress: 0,
      total: 0,
      message: "Reading PDF...",
      error: "",
      done: false,
    });

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const event = JSON.parse(line);
            if (event.type === "start") {
              setUploadState((s) => ({
                ...s,
                total: event.total,
                message: "Starting embedding...",
              }));
            } else if (event.type === "progress") {
              setUploadState((s) => ({
                ...s,
                progress: event.current,
                total: event.total,
                message: event.message,
              }));
            } else if (event.type === "done") {
              setUploadState((s) => ({
                ...s,
                uploading: false,
                done: true,
                message: `✓ ${event.totalChunks} chunks embedded from ${event.pages} pages`,
              }));
              await checkStatus();
            } else if (event.type === "error") {
              setUploadState((s) => ({
                ...s,
                uploading: false,
                error: event.message,
              }));
            }
          } catch {}
        }
      }

      if (buffer.trim()) {
        try {
          const event = JSON.parse(buffer);
          if (event.type === "done") {
            setUploadState((s) => ({
              ...s,
              uploading: false,
              done: true,
              message: `✓ ${event.totalChunks} chunks embedded from ${event.pages} pages`,
            }));
            await checkStatus();
          }
        } catch {}
      }
    } catch {
      setUploadState((s) => ({
        ...s,
        uploading: false,
        error: "Upload failed. Make sure Ollama is running locally.",
      }));
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  };

  const handleQuery = async () => {
    const question = input.trim();
    if (!question || isQuerying) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: question,
    };

    const assistantMsg: Message = {
      id: (Date.now() + 1).toString(),
      role: "assistant",
      content: "",
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setIsQuerying(true);

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        const errorText = await res.text();
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id
              ? { ...m, content: `Error: ${errorText}`, isStreaming: false }
              : m,
          ),
        );
        return;
      }

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const event = JSON.parse(line);
            if (event.type === "sources") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsg.id
                    ? { ...m, sources: event.sources }
                    : m,
                ),
              );
            } else if (event.type === "chunk") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsg.id
                    ? { ...m, content: m.content + event.text }
                    : m,
                ),
              );
            } else if (event.type === "answer") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsg.id
                    ? {
                        ...m,
                        content: event.text,
                        isStreaming: false,
                      }
                    : m,
                ),
              );
            } else if (event.type === "done") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsg.id ? { ...m, isStreaming: false } : m,
                ),
              );
            } else if (event.type === "error") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsg.id
                    ? {
                        ...m,
                        content: `Error: ${event.message}`,
                        isStreaming: false,
                      }
                    : m,
                ),
              );
            }
          } catch {}
        }
      }

      if (buffer.trim()) {
        try {
          const event = JSON.parse(buffer);
          if (event.type === "done") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsg.id ? { ...m, isStreaming: false } : m,
              ),
            );
          } else if (event.type === "error") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsg.id
                  ? {
                      ...m,
                      content: `Error: ${event.message}`,
                      isStreaming: false,
                    }
                  : m,
              ),
            );
          }
        } catch {}
      }
    } catch {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsg.id
            ? {
                ...m,
                content: "Failed to get answer. Check Ollama connection.",
                isStreaming: false,
              }
            : m,
        ),
      );
    } finally {
      setIsQuerying(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleQuery();
    }
  };

  const progressPercent =
    uploadState.total > 0
      ? Math.round((uploadState.progress / uploadState.total) * 100)
      : 0;

  return (
    <div className="min-h-screen bg-[#0F1923] text-[#F2E8D9] font-sans">
      {/* Header */}
      <header className="border-b border-[#C9A84C]/20 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-[#C9A84C]/20 border border-[#C9A84C]/40 flex items-center justify-center">
            <span className="text-[#C9A84C] text-sm">✦</span>
          </div>
          <div>
            <h1 className="text-[#F2E8D9] font-serif text-lg font-semibold tracking-wide">
              Scholar RAG
            </h1>
            <p className="text-[#F2E8D9]/40 text-xs tracking-wider uppercase">
              Biography Intelligence
            </p>
          </div>
        </div>

        {status.loaded && (
          <div className="flex items-center gap-2 bg-[#C9A84C]/10 border border-[#C9A84C]/30 rounded-full px-4 py-1.5">
            <span className="w-2 h-2 rounded-full bg-[#8BA888] animate-pulse"></span>
            <span className="text-[#C9A84C] text-sm font-medium">
              {status.scholarName}
            </span>
            <span className="text-[#F2E8D9]/30 text-xs">
              · {status.totalChunks} chunks
            </span>
          </div>
        )}
      </header>

      <div className="flex h-[calc(100vh-65px)]">
        {/* Left sidebar — Upload */}
        <aside className="w-80 border-r border-[#C9A84C]/10 p-6 flex flex-col gap-6 overflow-y-auto">
          <div>
            <h2 className="font-serif text-[#C9A84C] text-sm uppercase tracking-widest mb-1">
              Phase 1
            </h2>
            <h3 className="font-serif text-[#F2E8D9] text-xl mb-2">
              Load Biography
            </h3>
            <p className="text-[#F2E8D9]/50 text-sm leading-relaxed">
              Upload a PDF of any scholar's biography. The text will be
              extracted, chunked, and embedded using your local Ollama model.
            </p>
          </div>

          {/* Scholar name input */}
          <div>
            <label className="text-[#F2E8D9]/60 text-xs uppercase tracking-widest block mb-2">
              Scholar Name
            </label>
            <input
              type="text"
              value={scholarName}
              onChange={(e) => setScholarName(e.target.value)}
              placeholder="e.g. Maulana Ashraf Ali Thanvi"
              className="w-full bg-[#F2E8D9]/5 border border-[#C9A84C]/20 rounded-lg px-4 py-2.5 text-[#F2E8D9] text-sm placeholder-[#F2E8D9]/25 focus:outline-none focus:border-[#C9A84C]/50 transition-colors"
            />
          </div>

          {/* Drop zone */}
          <div
            className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
              dragOver
                ? "border-[#C9A84C] bg-[#C9A84C]/10"
                : "border-[#C9A84C]/20 hover:border-[#C9A84C]/40 hover:bg-white/[0.02]"
            }`}
            onDrop={handleDrop}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={(e) =>
                e.target.files?.[0] && handleFileUpload(e.target.files[0])
              }
            />
            <div className="text-3xl mb-3">📜</div>
            <p className="text-[#F2E8D9]/60 text-sm mb-1">Drop PDF here</p>
            <p className="text-[#F2E8D9]/30 text-xs">or click to browse</p>
          </div>

          {/* Progress */}
          {uploadState.uploading && (
            <div className="space-y-3">
              <div className="flex justify-between text-xs text-[#F2E8D9]/50">
                <span>Embedding chunks</span>
                <span>{progressPercent}%</span>
              </div>
              <div className="w-full bg-[#F2E8D9]/10 rounded-full h-1.5">
                <div
                  className="bg-[#C9A84C] h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
              <p className="text-[#F2E8D9]/40 text-xs">{uploadState.message}</p>
            </div>
          )}

          {uploadState.error && (
            <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-3">
              <p className="text-red-400 text-xs">{uploadState.error}</p>
            </div>
          )}

          {uploadState.done && !uploadState.uploading && (
            <div className="bg-[#8BA888]/10 border border-[#8BA888]/30 rounded-lg p-3">
              <p className="text-[#8BA888] text-xs">{uploadState.message}</p>
            </div>
          )}

          {/* Ollama setup note */}
          <div className="mt-auto bg-white/[0.02] border border-[#F2E8D9]/10 rounded-xl p-4">
            <h4 className="text-[#C9A84C] text-xs uppercase tracking-widest mb-2">
              Requires Ollama
            </h4>
            <div className="space-y-1.5 text-xs text-[#F2E8D9]/50">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[#C9A84C]/50"></span>
                <code className="text-[#C9A84C]/70">ollama serve</code>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[#C9A84C]/50"></span>
                <code className="text-[#C9A84C]/70">ollama pull mistral</code>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[#C9A84C]/50"></span>
                <code className="text-[#C9A84C]/70">
                  ollama pull nomic-embed-text
                </code>
              </div>
            </div>
          </div>
        </aside>

        {/* Main chat area */}
        <main className="flex-1 flex flex-col min-w-0">
          {/* Phase 2 header */}
          <div className="px-6 py-3 border-b border-[#C9A84C]/10 flex items-center gap-3">
            <h2 className="font-serif text-[#C9A84C] text-sm uppercase tracking-widest">
              Phase 2
            </h2>
            <span className="text-[#F2E8D9]/20">·</span>
            <h3 className="font-serif text-[#F2E8D9]/80 text-sm">
              Ask the Biography
            </h3>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
            {messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto">
                {!status.loaded ? (
                  <>
                    <div className="text-5xl mb-4 opacity-40">📚</div>
                    <h3 className="font-serif text-[#F2E8D9]/60 text-xl mb-2">
                      No biography loaded yet
                    </h3>
                    <p className="text-[#F2E8D9]/30 text-sm leading-relaxed">
                      Upload a PDF in the left panel to begin. The system will
                      extract, chunk, and embed the content using your local
                      Mistral model.
                    </p>
                  </>
                ) : (
                  <>
                    <div className="text-4xl mb-4">✦</div>
                    <h3 className="font-serif text-[#C9A84C] text-xl mb-2">
                      {status.scholarName} is ready
                    </h3>
                    <p className="text-[#F2E8D9]/40 text-sm leading-relaxed mb-6">
                      Ask anything about this scholar's life, works, teachings,
                      or legacy.
                    </p>
                    <div className="space-y-2 w-full">
                      {[
                        "What were their most important works?",
                        "Who were their teachers and influences?",
                        "What is their lasting contribution?",
                      ].map((q) => (
                        <button
                          key={q}
                          onClick={() => {
                            setInput(q);
                            chatInputRef.current?.focus();
                          }}
                          className="w-full text-left px-4 py-3 bg-[#F2E8D9]/5 border border-[#C9A84C]/15 rounded-xl text-[#F2E8D9]/60 text-sm hover:bg-[#C9A84C]/10 hover:border-[#C9A84C]/30 hover:text-[#F2E8D9]/80 transition-all"
                        >
                          {q}
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}

            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-2xl w-full ${msg.role === "user" ? "ml-12" : "mr-4"}`}
                >
                  {msg.role === "user" ? (
                    <div className="bg-[#C9A84C]/15 border border-[#C9A84C]/25 rounded-2xl rounded-tr-sm px-5 py-3 ml-auto w-fit max-w-full">
                      <p className="text-[#F2E8D9]/90 text-sm leading-relaxed">
                        {msg.content}
                      </p>
                    </div>
                  ) : (
                    <div className="flex gap-3">
                      <div className="w-0.5 bg-gradient-to-b from-[#C9A84C]/60 to-[#C9A84C]/5 rounded-full flex-shrink-0 mt-1" />
                      <div className="flex-1 min-w-0">
                        <div className="text-[#F2E8D9]/85 text-sm leading-7 whitespace-pre-wrap">
                          {msg.content}
                          {msg.isStreaming && (
                            <span className="inline-block w-1.5 h-4 bg-[#C9A84C]/70 ml-0.5 animate-pulse rounded-sm align-middle" />
                          )}
                        </div>

                        {msg.sources &&
                          msg.sources.length > 0 &&
                          !msg.isStreaming && (
                            <div className="mt-3">
                              <button
                                onClick={() =>
                                  setShowSources(
                                    showSources === msg.id ? null : msg.id,
                                  )
                                }
                                className="text-xs text-[#C9A84C]/60 hover:text-[#C9A84C] transition-colors flex items-center gap-1.5"
                              >
                                <span>
                                  {showSources === msg.id ? "▾" : "▸"}
                                </span>
                                {msg.sources.length} source
                                {msg.sources.length > 1 ? "s" : ""} from
                                biography
                              </button>

                              {showSources === msg.id && (
                                <div className="mt-2 space-y-2">
                                  {msg.sources.map((src, i) => (
                                    <div
                                      key={i}
                                      className="bg-[#F2E8D9]/3 border-l-2 border-[#C9A84C]/30 pl-3 py-2 pr-3 rounded-r-lg"
                                    >
                                      <div className="text-[#C9A84C]/50 text-xs mb-1 uppercase tracking-widest">
                                        Page ~{src.page}
                                      </div>
                                      <p className="text-[#F2E8D9]/50 text-xs leading-relaxed italic">
                                        {src.text}
                                      </p>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="px-6 py-4 border-t border-[#C9A84C]/10">
            <div
              className={`flex gap-3 items-end bg-[#F2E8D9]/5 border rounded-2xl px-4 py-3 transition-colors ${
                status.loaded
                  ? "border-[#C9A84C]/20 focus-within:border-[#C9A84C]/40"
                  : "border-[#F2E8D9]/10 opacity-50"
              }`}
            >
              <textarea
                ref={chatInputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={!status.loaded || isQuerying}
                placeholder={
                  status.loaded
                    ? `Ask about ${status.scholarName}...`
                    : "Upload a biography PDF to begin"
                }
                rows={1}
                className="flex-1 bg-transparent text-[#F2E8D9] text-sm placeholder-[#F2E8D9]/25 focus:outline-none resize-none leading-relaxed"
                style={{ minHeight: "24px", maxHeight: "120px" }}
                onInput={(e) => {
                  const el = e.currentTarget;
                  el.style.height = "auto";
                  el.style.height = Math.min(el.scrollHeight, 120) + "px";
                }}
              />
              <button
                onClick={handleQuery}
                disabled={!status.loaded || !input.trim() || isQuerying}
                className="w-8 h-8 rounded-xl bg-[#C9A84C] flex items-center justify-center transition-all hover:bg-[#C9A84C]/80 disabled:opacity-30 disabled:cursor-not-allowed flex-shrink-0"
              >
                {isQuerying ? (
                  <span className="w-3 h-3 border border-[#0F1923]/50 border-t-[#0F1923] rounded-full animate-spin" />
                ) : (
                  <span className="text-[#0F1923] text-xs font-bold">↑</span>
                )}
              </button>
            </div>
            <p className="text-center text-[#F2E8D9]/20 text-xs mt-2">
              Answers grounded in uploaded biography · Powered by local Mistral
              via Ollama
            </p>
          </div>
        </main>
      </div>
    </div>
  );
}
