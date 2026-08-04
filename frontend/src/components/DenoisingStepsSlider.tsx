import { useState, useEffect } from "react";
import { Button } from "@hanzo/ui";
import { Plus, Minus } from "@hanzogui/lucide-icons-2";
import { SliderWithInput } from "./SliderWithInput";
import { LabelWithTooltip } from "./LabelWithTooltip";

interface DenoisingStepsSliderProps {
  value: number[];
  onChange: (value: number[]) => void;
  disabled?: boolean;
  defaultValues?: number[];
  tooltip?: string;
}

const MIN_SLIDERS = 1;
const MAX_SLIDERS = 10;
const MIN_VALUE = 0;
const MAX_VALUE = 1000;
const DEFAULT_VALUES = [700, 500];

export function DenoisingStepsSlider({
  value,
  onChange,
  disabled = false,
  defaultValues = DEFAULT_VALUES,
  tooltip,
}: DenoisingStepsSliderProps) {
  const [localValue, setLocalValue] = useState<number[]>(
    value.length > 0 ? value : defaultValues
  );
  const [validationError, setValidationError] = useState<string>("");

  // Sync with external value changes
  useEffect(() => {
    if (value.length > 0) {
      setLocalValue(value);
    }
  }, [value]);

  const validateSteps = (steps: number[]): string => {
    for (let i = 1; i < steps.length; i++) {
      if (steps[i] >= steps[i - 1]) {
        return `Step ${i + 1} must be lower than Step ${i}`;
      }
    }
    return "";
  };

  const calculateBoundaryValue = (
    index: number,
    attemptedValue: number
  ): number => {
    // If we violated the constraint with the previous step, set to previous step - 1
    if (index > 0 && attemptedValue >= localValue[index - 1]) {
      return localValue[index - 1] - 1;
    }
    // If we violated the constraint with the next step, set to next step + 1
    if (
      index < localValue.length - 1 &&
      attemptedValue <= localValue[index + 1]
    ) {
      return localValue[index + 1] + 1;
    }
    return attemptedValue;
  };

  const handleStepValueChange = (index: number, newValue: number) => {
    const updatedValue = [...localValue];
    updatedValue[index] = newValue;

    const error = validateSteps(updatedValue);
    setValidationError(error);

    if (!error) {
      setLocalValue(updatedValue);
      return;
    }

    const boundaryValue = calculateBoundaryValue(index, newValue);
    const boundedValue = [...localValue];
    boundedValue[index] = Math.max(
      MIN_VALUE,
      Math.min(MAX_VALUE, boundaryValue)
    );
    setLocalValue(boundedValue);
  };

  const handleStepCommit = (index: number, newValue: number) => {
    const updatedValue = [...localValue];
    updatedValue[index] = newValue;
    onChange(updatedValue);
  };

  const commit = (updatedValue: number[]) => {
    const error = validateSteps(updatedValue);
    setValidationError(error);
    if (error) return;
    setLocalValue(updatedValue);
    onChange(updatedValue);
  };

  const addSlider = () => {
    if (localValue.length >= MAX_SLIDERS) return;
    // Add a new slider with a value lower than the last one
    const lastValue = localValue[localValue.length - 1];
    commit([...localValue, Math.max(MIN_VALUE, lastValue - 100)]);
  };

  const removeSlider = (index: number) => {
    if (localValue.length <= MIN_SLIDERS) return;
    commit(localValue.filter((_, i) => i !== index));
  };

  const resetToDefaults = () => {
    setValidationError("");
    setLocalValue(defaultValues);
    onChange(defaultValues);
  };

  return (
    <div className="app-stack">
      <div className="app-row app-row--between">
        <LabelWithTooltip label="Denoising Step List" tooltip={tooltip} />
        <div className="app-row">
          <Button
            variant="outline"
            size="sm"
            onPress={resetToDefaults}
            disabled={disabled}
          >
            Reset
          </Button>
          <Button
            variant="outline"
            size="icon-sm"
            onPress={addSlider}
            disabled={disabled || localValue.length >= MAX_SLIDERS}
          >
            <Plus size={12} />
          </Button>
        </div>
      </div>

      {validationError && <div className="app-alert">{validationError}</div>}

      {localValue.map((stepValue, index) => (
        <SliderWithInput
          key={index}
          label={`Step ${index + 1}:`}
          value={stepValue}
          onValueChange={next => handleStepValueChange(index, next)}
          onValueCommit={next => handleStepCommit(index, next)}
          min={MIN_VALUE}
          max={MAX_VALUE}
          step={1}
          incrementAmount={1}
          disabled={disabled}
          inputParser={v => parseInt(v) || MIN_VALUE}
          renderExtraButton={() =>
            localValue.length > MIN_SLIDERS ? (
              <Button
                variant="ghost"
                size="icon-sm"
                onPress={() => removeSlider(index)}
                disabled={disabled}
              >
                <Minus size={14} />
              </Button>
            ) : null
          }
        />
      ))}
    </div>
  );
}
