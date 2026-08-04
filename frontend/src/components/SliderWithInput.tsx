import { Button, Input } from "@hanzo/ui";
import { Plus, Minus } from "@hanzogui/lucide-icons-2";
import { DebouncedSlider } from "./DebouncedSlider";
import { LabelWithTooltip } from "./LabelWithTooltip";

interface SliderWithInputProps {
  label?: string;
  tooltip?: string;
  value: number;
  onValueChange: (value: number) => void;
  onValueCommit?: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  incrementAmount?: number;
  disabled?: boolean;
  labelClassName?: string;
  debounceMs?: number;
  valueFormatter?: (value: number) => number;
  inputParser?: (value: string) => number;
  renderExtraButton?: () => React.ReactNode;
}

/**
 * One numeric parameter: a labelled stepper field for precision and a debounced
 * slider for feel. Both write the same value through the same two callbacks.
 */
export function SliderWithInput({
  label,
  tooltip,
  value,
  onValueChange,
  onValueCommit,
  min = 0,
  max = 100,
  step = 1,
  incrementAmount = step,
  disabled = false,
  labelClassName = "app-label app-label--w16",
  debounceMs = 100,
  valueFormatter = v => v,
  inputParser = v => parseFloat(v) || min,
  renderExtraButton,
}: SliderWithInputProps) {
  const emit = (next: number) => {
    const formatted = valueFormatter(next);
    onValueChange(formatted);
    onValueCommit?.(formatted);
  };

  return (
    <div className="app-stack">
      <div className="app-row">
        {label && (
          <LabelWithTooltip
            label={label}
            tooltip={tooltip}
            className={labelClassName}
          />
        )}
        <div className="app-stepper">
          <Button
            variant="ghost"
            size="icon-sm"
            onPress={() => emit(Math.max(min, value - incrementAmount))}
            disabled={disabled}
          >
            <Minus size={14} />
          </Button>
          <div className="app-stepper-field">
            <Input
              type="number"
              value={String(value)}
              onChangeText={text =>
                emit(Math.max(min, Math.min(max, inputParser(text))))
              }
              disabled={disabled}
              textAlign="center"
              borderWidth={0}
              height={28}
              min={min}
              max={max}
              step={step}
            />
          </div>
          <Button
            variant="ghost"
            size="icon-sm"
            onPress={() => emit(Math.min(max, value + incrementAmount))}
            disabled={disabled}
          >
            <Plus size={14} />
          </Button>
          {renderExtraButton?.()}
        </div>
      </div>
      <DebouncedSlider
        value={[value]}
        onValueChange={next => onValueChange(valueFormatter(next[0]))}
        onValueCommit={next => onValueCommit?.(valueFormatter(next[0]))}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        debounceMs={debounceMs}
      />
    </div>
  );
}
