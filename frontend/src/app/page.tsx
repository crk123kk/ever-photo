"use client";

import { useState, useCallback } from "react";
import Upload from "@/components/Upload";
import Result from "@/components/Result";

type AppState = "idle" | "ready" | "processing" | "done" | "error";

export default function Home() {
  const [state, setState] = useState<AppState>("idle");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [restoredUrl, setRestoredUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelected = useCallback((file: File) => {
    setSelectedFile(file);
    setOriginalUrl(URL.createObjectURL(file));
    setRestoredUrl(null);
    setError(null);
    setState("ready");
  }, []);

  const handleRestore = async () => {
    if (!selectedFile) return;
    setState("processing");
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const res = await fetch("/api/restore", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `请求失败 (${res.status})`);
      }

      const blob = await res.blob();
      setRestoredUrl(URL.createObjectURL(blob));
      setState("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "未知错误");
      setState("error");
    }
  };

  const handleReset = () => {
    setState("idle");
    setSelectedFile(null);
    setOriginalUrl(null);
    setRestoredUrl(null);
    setError(null);
  };

  return (
    <div className="flex flex-col flex-1 items-center bg-gradient-to-b from-gray-50 to-white">
      <header className="w-full py-8 text-center">
        <h1 className="text-3xl font-bold text-gray-900">Ever Photo</h1>
        <p className="mt-2 text-gray-500">AI 驱动的老照片修复工具</p>
      </header>

      <main className="w-full max-w-4xl px-4 pb-16 space-y-6">
        {(state === "idle" || state === "ready") && (
          <Upload
            onFileSelected={handleFileSelected}
            disabled={state === "processing"}
          />
        )}

        {state === "ready" && selectedFile && (
          <div className="flex justify-center">
            <button
              onClick={handleRestore}
              className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium text-lg"
            >
              开始修复
            </button>
          </div>
        )}

        {state === "processing" && (
          <div className="flex flex-col items-center py-16 space-y-4">
            <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-gray-600 text-lg">AI 正在修复照片，请稍候...</p>
            <p className="text-gray-400 text-sm">通常需要 10-30 秒</p>
          </div>
        )}

        {state === "done" && originalUrl && restoredUrl && (
          <>
            <Result originalUrl={originalUrl} restoredUrl={restoredUrl} />
            <div className="flex justify-center">
              <button
                onClick={handleReset}
                className="px-6 py-2 border border-gray-300 text-gray-600 rounded-lg hover:bg-gray-50 transition-colors"
              >
                修复另一张照片
              </button>
            </div>
          </>
        )}

        {state === "error" && (
          <div className="text-center py-8 space-y-4">
            <p className="text-red-600">{error}</p>
            <button
              onClick={handleReset}
              className="px-6 py-2 border border-gray-300 text-gray-600 rounded-lg hover:bg-gray-50 transition-colors"
            >
              重试
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
