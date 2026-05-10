"use client";

import { useState, useCallback, useRef } from "react";
import Upload from "@/components/Upload";
import Result from "@/components/Result";
import ParamsPanel, { type Params } from "@/components/ParamsPanel";

type AppState = "idle" | "processing" | "done" | "error";

const DEFAULT_PARAMS: Params = {
  scratch_enabled: true,
  scratch_threshold: 10,
  scratch_kernel_size: 15,
  face_enabled: true,
  face_model: "gfpgan",
  fidelity_weight: 0.5,
  upscale_enabled: true,
  upscale_factor: 2,
};

export default function Home() {
  const [state, setState] = useState<AppState>("idle");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [restoredUrl, setRestoredUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [step, setStep] = useState("");
  const [params, setParams] = useState<Params>(DEFAULT_PARAMS);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const cleanup = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  const handleRestoreWithFile = useCallback(async (file: File) => {
    cleanup();
    setState("processing");
    setError(null);
    setProgress(0);
    setStep("Submitting...");

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("params", JSON.stringify(params));

      const submitRes = await fetch("/api/restore", {
        method: "POST",
        body: formData,
      });

      if (!submitRes.ok) {
        const text = await submitRes.text();
        throw new Error(text || `Submit failed (${submitRes.status})`);
      }

      const { task_id } = await submitRes.json();
      setStep("Queued");

      pollingRef.current = setInterval(async () => {
        try {
          const statusRes = await fetch(`/api/status/${task_id}`);
          if (!statusRes.ok) return;
          const data = await statusRes.json();

          setProgress(data.progress);
          setStep(data.step || "");

          if (data.status === "done") {
            cleanup();
            const resultRes = await fetch(`/api/result/${task_id}`);
            if (!resultRes.ok) throw new Error("Failed to fetch result");
            const blob = await resultRes.blob();
            setRestoredUrl(URL.createObjectURL(blob));
            setState("done");
          } else if (data.status === "failed") {
            cleanup();
            setError(data.error || "Restoration failed");
            setState("error");
          }
        } catch {
          // Network hiccup, keep polling
        }
      }, 1500);
    } catch (e) {
      cleanup();
      setError(e instanceof Error ? e.message : "Unknown error");
      setState("error");
    }
  }, [cleanup, params]);

  const handleFileSelected = useCallback((file: File) => {
    setSelectedFile(file);
    setOriginalUrl(URL.createObjectURL(file));
    setRestoredUrl(null);
    setError(null);
    setProgress(0);
    setStep("");
    handleRestoreWithFile(file);
  }, [handleRestoreWithFile]);

  const handleRestore = useCallback(() => {
    if (!selectedFile) return;
    handleRestoreWithFile(selectedFile);
  }, [selectedFile, handleRestoreWithFile]);

  const handlePickFile = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleDownload = () => {
    if (!restoredUrl) return;
    const a = document.createElement("a");
    a.href = restoredUrl;
    a.download = "restored_photo.png";
    a.click();
  };

  return (
    <div className="flex flex-col flex-1 bg-gradient-to-b from-gray-50 to-white">
      <header className="w-full py-8 text-center">
        <h1 className="text-3xl font-bold text-gray-900">Ever Photo</h1>
        <p className="mt-2 text-gray-500">AI 驱动的老照片修复工具</p>
      </header>

      <main className="w-full max-w-6xl mx-auto px-4 pb-16 flex-1">
        {/* Hidden file input for "选择照片" / "选择其他照片" */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFileSelected(file);
            e.target.value = "";
          }}
        />

        <div className="flex gap-6">
          {/* Left: Upload / Progress / Result */}
          <div className="flex-1 min-w-0 space-y-6">
            {state === "idle" && (
              <Upload onFileSelected={handleFileSelected} />
            )}

            {state === "processing" && (
              <div className="flex flex-col items-center py-16 space-y-4 w-full max-w-md mx-auto">
                <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
                <p className="text-gray-600 text-lg">{step || "Preparing..."}</p>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div
                    className="bg-blue-600 h-2.5 rounded-full transition-all duration-500"
                    style={{ width: `${Math.round(progress * 100)}%` }}
                  />
                </div>
                <p className="text-gray-400 text-sm">
                  {Math.round(progress * 100)}%
                </p>
              </div>
            )}

            {state === "done" && originalUrl && restoredUrl && (
              <Result originalUrl={originalUrl} restoredUrl={restoredUrl} />
            )}

            {state === "error" && (
              <div className="text-center py-8 space-y-4">
                <p className="text-red-600">{error}</p>
              </div>
            )}
          </div>

          {/* Right: Params + Actions (always visible) */}
          <div className="w-72 shrink-0">
            <div className="sticky top-8 bg-white border border-gray-200 rounded-xl p-5">
              <h2 className="text-sm font-semibold text-gray-800 mb-4">修复参数</h2>
              <ParamsPanel
                params={params}
                onChange={setParams}
                onStart={handleRestore}
                onPickFile={handlePickFile}
                onDownload={handleDownload}
                state={state}
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
