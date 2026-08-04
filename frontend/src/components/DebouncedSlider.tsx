import { useState, useEffect, useRef } from "react";
import { Slider } from "@hanzo/ui";

interface DebouncedSliderProps {
  value: number[];
  onValueChange: (value: number[]) => void;
  onValueCommit?: (value: number[]) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  debounceMs?: number;
}

/**
 * A slider that debounces the expensive half of its output.
 * `onValueChange` fires immediately so the UI tracks the thumb;
 * `onValueCommit` fires once the drag settles, for the API call.
 */
export function DebouncedSlider({
  value,
  onValueChange,
  onValueCommit,
  min = 0,
  max = 100,
  step = 1,
  disabled = false,
  debounceMs = 100,
}: DebouncedSliderProps) {
  const [localValue, setLocalValue] = useState<number[]>(value);
  const commitTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  useEffect(
    () => () => {
      if (commitTimeoutRef.current) clearTimeout(commitTimeoutRef.current);
    },
    []
  );

  const handleValueChange = (newValue: number[]) => {
    setLocalValue(newValue);
    onValueChange(newValue);

    if (!onValueCommit) return;
    if (commitTimeoutRef.current) clearTimeout(commitTimeoutRef.current);
    commitTimeoutRef.current = setTimeout(
      () => onValueCommit(newValue),
      debounceMs
    ) as unknown as number;
  };

  return (
    <Slider
      value={localValue}
      onValueChange={handleValueChange}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
    />
  );
}
