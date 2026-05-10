"use client";

export interface Params {
  denoise_enabled: boolean;
  denoise_task: "denoise" | "deblur_motion" | "deblur_defocus";
  denoise_strength: number;
  scratch_enabled: boolean;
  scratch_threshold: number;
  scratch_kernel_size: number;
  face_enabled: boolean;
  face_model: "gfpgan" | "codeformer";
  fidelity_weight: number;
  colorize_enabled: boolean;
  colorize_strength: number;
  upscale_enabled: boolean;
  upscale_factor: number;
}

interface ParamsPanelProps {
  params: Params;
  onChange: (params: Params) => void;
  onStart: () => void;
  onPickFile: () => void;
  onDownload: () => void;
  state: "idle" | "ready" | "processing" | "done" | "error";
}

function Hint({ children }: { children: React.ReactNode }) {
  return (
    <p className="mt-1 text-[11px] leading-snug text-gray-400">{children}</p>
  );
}

export default function ParamsPanel({
  params,
  onChange,
  onStart,
  onPickFile,
  onDownload,
  state,
}: ParamsPanelProps) {
  const update = (key: keyof Params, value: Params[keyof Params]) =>
    onChange({ ...params, [key]: value });

  const isProcessing = state === "processing";
  const hasResult = state === "done";
  const hasFile = state !== "idle";

  return (
    <div className="w-full space-y-5">
      {/* Section: Denoise/Deblur */}
      <fieldset className="space-y-3">
        <div className="flex items-center justify-between">
          <legend className="text-sm font-medium text-gray-700">去噪/去模糊</legend>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={params.denoise_enabled}
              onChange={(e) => update("denoise_enabled", e.target.checked)}
              disabled={isProcessing}
              className="sr-only peer"
            />
            <div className="w-9 h-5 bg-gray-200 peer-checked:bg-blue-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-disabled:opacity-50" />
          </label>
        </div>
        {params.denoise_enabled && (
          <div className="pl-0.5 space-y-3">
            <div className="space-y-1.5">
              <span className="text-xs text-gray-500">处理模式</span>
              <div className="flex gap-2">
                {([
                  ["denoise", "去噪"],
                  ["deblur_motion", "运动去模糊"],
                  ["deblur_defocus", "散焦去模糊"],
                ] as const).map(([value, label]) => (
                  <button
                    key={value}
                    onClick={() => update("denoise_task", value)}
                    disabled={isProcessing}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors disabled:opacity-50 ${
                      params.denoise_task === value
                        ? "bg-blue-600 text-white"
                        : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <Hint>
                {params.denoise_task === "denoise"
                  ? "去除照片噪点，适合颗粒感明显的老照片。"
                  : params.denoise_task === "deblur_motion"
                  ? "修复因相机抖动造成的运动模糊。"
                  : "修复因对焦不准造成的散焦模糊。"}
              </Hint>
            </div>
            <label className="block">
              <span className="text-xs text-gray-500">
                强度 <span className="tabular-nums">{params.denoise_strength.toFixed(2)}</span>
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={params.denoise_strength}
                onChange={(e) => update("denoise_strength", Number(e.target.value))}
                disabled={isProcessing}
                className="w-full mt-1.5 accent-blue-600 disabled:opacity-50"
              />
              <Hint>0 = 不处理；1 = 完全去噪/去模糊。建议 0.5-0.8，过高可能丢失细节。</Hint>
            </label>
          </div>
        )}
      </fieldset>

      {/* Section: Scratch */}
      <fieldset className="space-y-3">
        <div className="flex items-center justify-between">
          <legend className="text-sm font-medium text-gray-700">划痕修复</legend>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={params.scratch_enabled}
              onChange={(e) => update("scratch_enabled", e.target.checked)}
              disabled={isProcessing}
              className="sr-only peer"
            />
            <div className="w-9 h-5 bg-gray-200 peer-checked:bg-blue-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-disabled:opacity-50" />
          </label>
        </div>
        {params.scratch_enabled && (
          <label className="block pl-0.5">
            <span className="text-xs text-gray-500">
              灵敏度 <span className="tabular-nums">{params.scratch_threshold}</span>
            </span>
            <input
              type="range"
              min={1}
              max={50}
              value={params.scratch_threshold}
              onChange={(e) =>
                update("scratch_threshold", Number(e.target.value))
              }
              disabled={isProcessing}
              className="w-full mt-1.5 accent-blue-600 disabled:opacity-50"
            />
            <Hint>
              值越小检测越灵敏，会识别更多细微划痕；值越大只检测明显损伤。调低可能误检纹理细节，调高可能漏检细小划痕。
            </Hint>
          </label>
        )}
      </fieldset>

      {/* Section: Face */}
      <fieldset className="space-y-3">
        <div className="flex items-center justify-between">
          <legend className="text-sm font-medium text-gray-700">人脸修复</legend>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={params.face_enabled}
              onChange={(e) => update("face_enabled", e.target.checked)}
              disabled={isProcessing}
              className="sr-only peer"
            />
            <div className="w-9 h-5 bg-gray-200 peer-checked:bg-blue-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-disabled:opacity-50" />
          </label>
        </div>
        {params.face_enabled && (
          <div className="pl-0.5 space-y-3">
            <div className="space-y-1.5">
              <span className="text-xs text-gray-500">修复模型</span>
              <div className="flex gap-2">
                <button
                  onClick={() => update("face_model", "gfpgan")}
                  disabled={isProcessing}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors disabled:opacity-50 ${
                    params.face_model === "gfpgan"
                      ? "bg-blue-600 text-white"
                      : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                  }`}
                >
                  GFPGAN
                </button>
                <button
                  onClick={() => update("face_model", "codeformer")}
                  disabled={isProcessing}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors disabled:opacity-50 ${
                    params.face_model === "codeformer"
                      ? "bg-blue-600 text-white"
                      : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                  }`}
                >
                  CodeFormer
                </button>
              </div>
              <Hint>
                {params.face_model === "gfpgan"
                  ? "GFPGAN 效果稳定自然，适合大多数老照片。"
                  : "CodeFormer 支持保真度调节，0.5 以下偏保守，适合需精细控制的场景。"}
              </Hint>
            </div>
            <label className="block">
              <span className="text-xs text-gray-500">
                保真度 <span className="tabular-nums">{params.fidelity_weight.toFixed(2)}</span>
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={params.fidelity_weight}
                onChange={(e) =>
                  update("fidelity_weight", Number(e.target.value))
                }
                disabled={isProcessing}
                className="w-full mt-1.5 accent-blue-600 disabled:opacity-50"
              />
              <Hint>
                {params.face_model === "codeformer"
                  ? "0 = 最大程度保留原貌，修复保守；1 = 最大程度增强，面部更清晰。建议 0.3 以下。"
                  : "保真度仅在使用 CodeFormer 时生效。"}
              </Hint>
            </label>
          </div>
        )}
      </fieldset>

      {/* Section: Colorize */}
      <fieldset className="space-y-3">
        <div className="flex items-center justify-between">
          <legend className="text-sm font-medium text-gray-700">上色</legend>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={params.colorize_enabled}
              onChange={(e) => update("colorize_enabled", e.target.checked)}
              disabled={isProcessing}
              className="sr-only peer"
            />
            <div className="w-9 h-5 bg-gray-200 peer-checked:bg-blue-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-disabled:opacity-50" />
          </label>
        </div>
        {params.colorize_enabled && (
          <div className="pl-0.5 space-y-1.5">
            <label className="block">
              <span className="text-xs text-gray-500">
                上色强度 <span className="tabular-nums">{params.colorize_strength.toFixed(2)}</span>
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={params.colorize_strength}
                onChange={(e) => update("colorize_strength", Number(e.target.value))}
                disabled={isProcessing}
                className="w-full mt-1.5 accent-blue-600 disabled:opacity-50"
              />
              <Hint>为黑白或褪色照片自动上色。0 = 保留原色；1 = 完全上色。适合黑白老照片，彩色照片请谨慎使用。</Hint>
            </label>
          </div>
        )}
      </fieldset>

      {/* Section: Upscale */}
      <fieldset className="space-y-3">
        <div className="flex items-center justify-between">
          <legend className="text-sm font-medium text-gray-700">超分辨率</legend>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={params.upscale_enabled}
              onChange={(e) => update("upscale_enabled", e.target.checked)}
              disabled={isProcessing}
              className="sr-only peer"
            />
            <div className="w-9 h-5 bg-gray-200 peer-checked:bg-blue-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-disabled:opacity-50" />
          </label>
        </div>
        {params.upscale_enabled && (
          <div className="pl-0.5 space-y-1.5">
            <div className="flex gap-2">
              {[2, 4].map((factor) => (
                <button
                  key={factor}
                  onClick={() => update("upscale_factor", factor)}
                  disabled={isProcessing}
                  className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 ${
                    params.upscale_factor === factor
                      ? "bg-blue-600 text-white"
                      : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                  }`}
                >
                  {factor}x
                </button>
              ))}
            </div>
            <Hint>
              2x 放大 2 倍，速度快，适合一般用途；4x 放大 4 倍，细节更丰富但处理时间更长、输出文件更大。
            </Hint>
          </div>
        )}
      </fieldset>

      {/* Divider */}
      <div className="border-t border-gray-100" />

      {/* Actions: always visible */}
      <div className="space-y-2.5">
        <button
          onClick={onStart}
          disabled={!hasFile || isProcessing}
          className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition-colors font-medium text-base"
        >
          {isProcessing ? "修复中..." : hasResult ? "重新修复" : "开始修复"}
        </button>

        {hasResult && (
          <button
            onClick={onDownload}
            className="w-full px-6 py-2.5 bg-white border border-blue-600 text-blue-600 rounded-lg hover:bg-blue-50 transition-colors font-medium flex items-center justify-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            下载修复结果
          </button>
        )}

        <button
          onClick={onPickFile}
          className="w-full px-6 py-2 text-sm text-gray-500 hover:text-gray-700 transition-colors"
        >
          {hasFile ? "选择其他照片" : "选择照片"}
        </button>
      </div>
    </div>
  );
}
