"use client";

import { useCallback, useRef, useState } from "react";

interface UploadProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

export default function Upload({ onFileSelected, disabled }: UploadProps) {
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      setPreview(URL.createObjectURL(file));
      onFileSelected(file);
    },
    [onFileSelected]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      className={`
        relative flex flex-col items-center justify-center
        w-full min-h-[320px] rounded-2xl border-2 border-dashed
        cursor-pointer transition-all duration-200
        ${
          dragOver
            ? "border-blue-500 bg-blue-500/10"
            : "border-gray-300 hover:border-gray-400 bg-gray-50 hover:bg-gray-100"
        }
        ${disabled ? "opacity-50 pointer-events-none" : ""}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={onChange}
        disabled={disabled}
      />

      {preview ? (
        <img
          src={preview}
          alt="Preview"
          className="max-h-[400px] max-w-full object-contain rounded-lg"
        />
      ) : (
        <div className="text-center p-8">
          <svg
            className="mx-auto h-16 w-16 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>
          <p className="mt-4 text-lg text-gray-600">
            拖拽照片到这里，或点击选择文件
          </p>
          <p className="mt-1 text-sm text-gray-400">
            支持 JPG、PNG 格式
          </p>
        </div>
      )}
    </div>
  );
}
