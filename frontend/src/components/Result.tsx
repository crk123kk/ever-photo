"use client";

import { useState } from "react";
import ComparisonSlider from "./ComparisonSlider";

interface ResultProps {
  originalUrl: string;
  restoredUrl: string;
}

export default function Result({ originalUrl, restoredUrl }: ResultProps) {
  const [viewMode, setViewMode] = useState<"slider" | "grid">("slider");

  return (
    <div className="space-y-6">
      {/* View mode toggle */}
      <div className="flex justify-center gap-2">
        <button
          onClick={() => setViewMode("slider")}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            viewMode === "slider"
              ? "bg-blue-600 text-white"
              : "bg-gray-100 text-gray-600 hover:bg-gray-200"
          }`}
        >
          滑块对比
        </button>
        <button
          onClick={() => setViewMode("grid")}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            viewMode === "grid"
              ? "bg-blue-600 text-white"
              : "bg-gray-100 text-gray-600 hover:bg-gray-200"
          }`}
        >
          并排对比
        </button>
      </div>

      {viewMode === "slider" ? (
        <ComparisonSlider originalUrl={originalUrl} restoredUrl={restoredUrl} />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-gray-500 text-center">
              修复前
            </h3>
            <div className="rounded-xl overflow-hidden bg-gray-100 flex items-center justify-center">
              <img
                src={originalUrl}
                alt="Original"
                className="max-h-[500px] w-full object-contain"
              />
            </div>
          </div>
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-gray-500 text-center">
              修复后
            </h3>
            <div className="rounded-xl overflow-hidden bg-gray-100 flex items-center justify-center">
              <img
                src={restoredUrl}
                alt="Restored"
                className="max-h-[500px] w-full object-contain"
              />
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
