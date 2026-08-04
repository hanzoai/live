import { useEffect, useRef } from "react";
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@hanzo/ui";
import { Spinner } from "@hanzo/gui";
import { Play, Square, Upload } from "@hanzogui/lucide-icons-2";
import type { VideoSourceMode } from "../hooks/useVideoSource";
import { PIPELINES } from "../data/pipelines";

interface InputAndControlsPanelProps {
  localStream: MediaStream | null;
  isInitializing: boolean;
  error: string | null;
  mode: VideoSourceMode;
  onModeChange: (mode: VideoSourceMode) => void;
  isStreaming: boolean;
  isConnecting: boolean;
  isPipelineLoading: boolean;
  canStartStream: boolean;
  onStartStream: () => void;
  onStopStream: () => void;
  onVideoFileUpload?: (file: File) => Promise<boolean>;
  pipelineId: string;
}

export function InputAndControlsPanel({
  localStream,
  isInitializing,
  error,
  mode,
  onModeChange,
  isStreaming,
  isConnecting,
  isPipelineLoading,
  canStartStream,
  onStartStream,
  onStopStream,
  onVideoFileUpload,
  pipelineId,
}: InputAndControlsPanelProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  // Get pipeline category, default to video-input
  const pipelineCategory = PIPELINES[pipelineId]?.category || "video-input";
  const busy = isPipelineLoading || isConnecting;

  useEffect(() => {
    if (videoRef.current && localStream) {
      videoRef.current.srcObject = localStream;
    }
  }, [localStream]);

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (file && onVideoFileUpload) {
      try {
        await onVideoFileUpload(file);
      } catch (uploadError) {
        console.error("Video upload failed:", uploadError);
      }
    }
    // Reset the input value so the same file can be selected again
    event.target.value = "";
  };

  return (
    <Card height="100%">
      <CardHeader>
        <CardTitle>Input &amp; Controls</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="app-stack app-stack--lg">
          <div className="app-stack">
            <h3 className="app-section-title">Mode</h3>
            <Select
              value={pipelineCategory === "video-input" ? mode : "text"}
              onValueChange={value => {
                if (pipelineCategory === "video-input" && value) {
                  onModeChange(value as VideoSourceMode);
                }
              }}
            >
              <SelectTrigger disabled={isStreaming}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {pipelineCategory === "video-input" ? (
                  <>
                    <SelectItem value="video">Video</SelectItem>
                    <SelectItem value="camera">Camera</SelectItem>
                  </>
                ) : (
                  <SelectItem value="text">Text</SelectItem>
                )}
              </SelectContent>
            </Select>
          </div>

          {pipelineCategory === "video-input" && (
            <div className="app-stack">
              <h3 className="app-section-title">Input</h3>
              <div className="app-preview">
                {isInitializing ? (
                  <div className="app-preview-note">
                    {mode === "camera"
                      ? "Requesting camera access…"
                      : "Initializing video…"}
                  </div>
                ) : error ? (
                  <div className="app-preview-note app-preview-note--error">
                    <p>
                      {mode === "camera"
                        ? "Camera access failed:"
                        : "Video error:"}
                    </p>
                    <p>{error}</p>
                  </div>
                ) : localStream ? (
                  <video ref={videoRef} autoPlay muted playsInline />
                ) : (
                  <div className="app-preview-note">
                    {mode === "camera" ? "Camera Preview" : "Video Preview"}
                  </div>
                )}

                {mode === "video" && onVideoFileUpload && (
                  <>
                    <input
                      type="file"
                      accept="video/*"
                      onChange={handleFileUpload}
                      className="app-upload-input"
                      id="video-upload"
                      disabled={isStreaming || isConnecting}
                    />
                    <label
                      htmlFor="video-upload"
                      className="app-upload"
                      data-disabled={
                        isStreaming || isConnecting ? "true" : "false"
                      }
                    >
                      <Upload size={16} />
                    </label>
                  </>
                )}
              </div>
            </div>
          )}

          <div className="app-stack">
            <h3 className="app-section-title">Controls</h3>
            <Button
              onPress={isStreaming ? onStopStream : onStartStream}
              variant={isStreaming ? "destructive" : "default"}
              size="sm"
              width="100%"
              disabled={busy || (!canStartStream && !isStreaming)}
            >
              {busy ? (
                <Spinner size="small" />
              ) : isStreaming ? (
                <Square size={16} />
              ) : (
                <Play size={16} />
              )}
              {busy ? "" : isStreaming ? "Stop" : "Start"}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
