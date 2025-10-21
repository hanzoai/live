import { useState, useEffect, useCallback, useRef } from "react";
import { loadPipeline, getPipelineStatus } from "../lib/api";
import type { PipelineStatusResponse, PipelineLoadParams } from "../lib/api";

interface UsePipelineOptions {
  pollInterval?: number; // milliseconds
  maxTimeout?: number; // milliseconds
}

export function usePipeline(options: UsePipelineOptions = {}) {
  const { pollInterval = 2000, maxTimeout = 45000 } = options;

  const [status, setStatus] =
    useState<PipelineStatusResponse["status"]>("not_loaded");
  const [pipelineInfo, setPipelineInfo] =
    useState<PipelineStatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pollTimeoutRef = useRef<number | null>(null);
  const loadTimeoutRef = useRef<number | null>(null);
  const isPollingRef = useRef(false);

  // Check initial pipeline status
  const checkStatus = useCallback(async () => {
    try {
      const statusResponse = await getPipelineStatus();
      setStatus(statusResponse.status);
      setPipelineInfo(statusResponse);

      if (statusResponse.status === "error") {
        setError(statusResponse.error || "Unknown pipeline error");
      } else {
        setError(null);
      }
    } catch (err) {
      console.error("Failed to get pipeline status:", err);
      setError(
        err instanceof Error ? err.message : "Failed to get pipeline status"
      );
    }
  }, []);

  // Stop polling
  const stopPolling = useCallback(() => {
    isPollingRef.current = false;
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
    }
  }, []);

  // Start polling for status updates
  const startPolling = useCallback(() => {
    if (isPollingRef.current) return;

    isPollingRef.current = true;

    const poll = async () => {
      if (!isPollingRef.current) return;

      try {
        const statusResponse = await getPipelineStatus();
        setStatus(statusResponse.status);
        setPipelineInfo(statusResponse);

        if (statusResponse.status === "error") {
          setError(statusResponse.error || "Unknown pipeline error");
        } else {
          setError(null);
        }

        // Stop polling if loaded or error
        if (
          statusResponse.status === "loaded" ||
          statusResponse.status === "error"
        ) {
          stopPolling();
          return;
        }
      } catch (err) {
        console.error("Polling error:", err);
        setError(
          err instanceof Error ? err.message : "Failed to get pipeline status"
        );
      }

      if (isPollingRef.current) {
        pollTimeoutRef.current = setTimeout(poll, pollInterval) as unknown as number;
      }
    };

    poll();
  }, [pollInterval, stopPolling]);

  // Load pipeline
  const triggerLoad = useCallback(
    async (
      pipelineId?: string,
      loadParams?: PipelineLoadParams
    ): Promise<boolean> => {
      if (isLoading) {
        console.log("Pipeline already loading");
        return false;
      }

      try {
        setIsLoading(true);
        setError(null);

        // Start the load request
        await loadPipeline({
          pipeline_id: pipelineId,
          load_params: loadParams,
        });

        // Start polling for updates
        startPolling();

        // Set up timeout for the load operation
        const timeoutPromise = new Promise<boolean>((_, reject) => {
          loadTimeoutRef.current = setTimeout(() => {
            reject(
              new Error(
                `Pipeline load timeout after ${maxTimeout / 1000} seconds`
              )
            );
          }, maxTimeout) as unknown as number;
        });

        // Wait for pipeline to be loaded or error
        const loadPromise = new Promise<boolean>((resolve, reject) => {
          const checkComplete = async () => {
            try {
              const currentStatus = await getPipelineStatus();
              if (currentStatus.status === "loaded") {
                resolve(true);
              } else if (currentStatus.status === "error") {
                reject(
                  new Error(currentStatus.error || "Pipeline load failed")
                );
              } else {
                // Continue polling
                setTimeout(checkComplete, pollInterval);
              }
            } catch (err) {
              reject(err);
            }
          };
          checkComplete();
        });

        // Race between load completion and timeout
        const result = await Promise.race([loadPromise, timeoutPromise]);

        // Clear timeout if load completed
        if (loadTimeoutRef.current) {
          clearTimeout(loadTimeoutRef.current);
          loadTimeoutRef.current = null;
        }

        stopPolling();
        return result;
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Failed to load pipeline";
        console.error("Pipeline load error:", errorMessage);
        setError(errorMessage);

        stopPolling();

        // Clear timeout on error
        if (loadTimeoutRef.current) {
          clearTimeout(loadTimeoutRef.current);
          loadTimeoutRef.current = null;
        }

        return false;
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading, maxTimeout, pollInterval, startPolling, stopPolling]
  );

  // Load pipeline with proper state management
  const loadPipelineAsync = useCallback(
    async (
      pipelineId?: string,
      loadParams?: PipelineLoadParams
    ): Promise<boolean> => {
      // Always trigger load - let the backend decide if reload is needed
      return await triggerLoad(pipelineId, loadParams);
    },
    [triggerLoad]
  );

  // Initial status check on mount
  useEffect(() => {
    checkStatus();
  }, [checkStatus]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopPolling();
      if (loadTimeoutRef.current) {
        clearTimeout(loadTimeoutRef.current);
      }
    };
  }, [stopPolling]);

  return {
    status,
    pipelineInfo,
    isLoading,
    error,
    loadPipeline: loadPipelineAsync,
    checkStatus,
    isLoaded: status === "loaded",
    isError: status === "error",
  };
}
