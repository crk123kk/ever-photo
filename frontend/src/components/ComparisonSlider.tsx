"use client";

import { useRef, useState, useCallback } from "react";

interface ComparisonSliderProps {
  originalUrl: string;
  restoredUrl: string;
}

export default function ComparisonSlider({
  originalUrl,
  restoredUrl,
}: ComparisonSliderProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState(50);
  const [dragging, setDragging] = useState(false);

  const handleMove = useCallback((clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setPosition(pct);
  }, []);

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      setDragging(true);
      handleMove(e.clientX);
      const onMove = (ev: MouseEvent) => handleMove(ev.clientX);
      const onUp = () => {
        setDragging(false);
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [handleMove],
  );

  const onTouchStart = useCallback(
    (e: React.TouchEvent) => {
      handleMove(e.touches[0].clientX);
      const onMove = (ev: TouchEvent) => handleMove(ev.touches[0].clientX);
      const onEnd = () => {
        window.removeEventListener("touchmove", onMove);
        window.removeEventListener("touchend", onEnd);
      };
      window.addEventListener("touchmove", onMove);
      window.addEventListener("touchend", onEnd);
    },
    [handleMove],
  );

  return (
    <div
      ref={containerRef}
      className="relative w-full overflow-hidden rounded-xl cursor-ew-resize select-none bg-gray-100"
      onMouseDown={onMouseDown}
      onTouchStart={onTouchStart}
    >
      {/* Original image (full width, underneath) */}
      <img
        src={originalUrl}
        alt="Before"
        className="w-full block"
        draggable={false}
      />

      {/* Restored image (clipped to right of divider) */}
      <div
        className="absolute inset-0"
        style={{ clipPath: `inset(0 0 0 ${position}%)` }}
      >
        <img
          src={restoredUrl}
          alt="After"
          className="w-full block"
          draggable={false}
        />
      </div>

      {/* Divider line + handle */}
      <div
        className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg pointer-events-none"
        style={{ left: `${position}%` }}
      >
        <div
          className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full shadow-md flex items-center justify-center transition-transform ${dragging ? "scale-110" : ""}`}
        >
          <svg
            className="w-4 h-4 text-gray-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8 9l-3 3 3 3m8-6l3 3-3 3"
            />
          </svg>
        </div>
      </div>

      {/* Labels */}
      <span className="absolute top-3 left-3 bg-black/50 text-white text-xs px-2 py-1 rounded pointer-events-none">
        修复前
      </span>
      <span className="absolute top-3 right-3 bg-black/50 text-white text-xs px-2 py-1 rounded pointer-events-none">
        修复后
      </span>
    </div>
  );
}
