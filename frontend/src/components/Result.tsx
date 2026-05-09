"use client";

interface ResultProps {
  originalUrl: string;
  restoredUrl: string;
}

export default function Result({ originalUrl, restoredUrl }: ResultProps) {
  const handleDownload = () => {
    const a = document.createElement("a");
    a.href = restoredUrl;
    a.download = "restored_photo.png";
    a.click();
  };

  return (
    <div className="space-y-6">
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

      <div className="flex justify-center">
        <button
          onClick={handleDownload}
          className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
            />
          </svg>
          下载修复结果
        </button>
      </div>
    </div>
  );
}
