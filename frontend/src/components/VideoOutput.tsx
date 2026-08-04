import { useEffect, useRef, useState, useCallback } from "react";
import { Card, CardContent } from "@hanzo/ui";
import { Spinner } from "@hanzo/gui";
import { Pause, Play } from "@hanzogui/lucide-icons-2";

interface VideoOutputProps {
  remoteStream: MediaStream | null;
  isPipelineLoading?: boolean;
  isConnecting?: boolean;
  pipelineError?: string | null;
  isPlaying?: boolean;
  onPlayPauseToggle?: () => void;
}

export function VideoOutput({
  remoteStream,
  isPipelineLoading = false,
  isConnecting = false,
  pipelineError = null,
  isPlaying = true,
  onPlayPauseToggle,
}: VideoOutputProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [showOverlay, setShowOverlay] = useState(false);
  const [isFadingOut, setIsFadingOut] = useState(false);
  const overlayTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    if (videoRef.current && remoteStream) {
      videoRef.current.srcObject = remoteStream;
    }
  }, [remoteStream]);

  const triggerPlayPause = useCallback(() => {
    if (!onPlayPauseToggle || !remoteStream) return;

    onPlayPauseToggle();

    // Show overlay and immediately start fade out animation
    setShowOverlay(true);
    setIsFadingOut(false);

    if (overlayTimeoutRef.current) {
      clearTimeout(overlayTimeoutRef.current);
    }

    // Start fade out immediately (CSS transition handles the timing)
    requestAnimationFrame(() => setIsFadingOut(true));

    // Remove overlay after animation completes (400ms transition)
    overlayTimeoutRef.current = setTimeout(() => {
      setShowOverlay(false);
      setIsFadingOut(false);
    }, 400) as unknown as number;
  }, [onPlayPauseToggle, remoteStream]);

  // Handle spacebar press for play/pause
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code !== "Space" || !remoteStream) return;

      // Don't trigger if user is typing in an input/textarea/select or any contenteditable element
      const target = e.target as HTMLElement;
      const isInputFocused =
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.tagName === "SELECT" ||
        target.isContentEditable;

      if (!isInputFocused) {
        // Prevent default spacebar behavior (page scroll)
        e.preventDefault();
        triggerPlayPause();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [remoteStream, triggerPlayPause]);

  // Cleanup timeout on unmount
  useEffect(
    () => () => {
      if (overlayTimeoutRef.current) clearTimeout(overlayTimeoutRef.current);
    },
    []
  );

  return (
    <Card height="100%" flex={1}>
      <CardContent flex={1} minH={0}>
        <div className="app-stage-body">
          {remoteStream ? (
            <div className="app-video-frame" onClick={triggerPlayPause}>
              <video
                ref={videoRef}
                className="app-video"
                autoPlay
                muted
                playsInline
              />
              {showOverlay && (
                <div className="app-video-overlay">
                  <div
                    className="app-video-badge"
                    data-fading={isFadingOut ? "true" : "false"}
                  >
                    {isPlaying ? <Play size={48} /> : <Pause size={48} />}
                  </div>
                </div>
              )}
            </div>
          ) : pipelineError ? (
            <div className="app-notice app-notice--error">
              <p>Pipeline Error</p>
              <p>{pipelineError}</p>
            </div>
          ) : isPipelineLoading || isConnecting ? (
            <div className="app-notice">
              <Spinner size="small" />
              <p>{isPipelineLoading ? "Loading pipeline…" : "Connecting…"}</p>
            </div>
          ) : (
            <div className="app-notice">Click "Start" when you are ready</div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
