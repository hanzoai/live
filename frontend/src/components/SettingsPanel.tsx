import { useState, useEffect } from "react";
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Input,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Switch,
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@hanzo/ui";
import { Hammer, Info, Minus, Plus, RotateCcw } from "@hanzogui/lucide-icons-2";
import { PIPELINES } from "../data/pipelines";
import { PARAMETER_METADATA } from "../data/parameterMetadata";
import { DenoisingStepsSlider } from "./DenoisingStepsSlider";
import { LabelWithTooltip } from "./LabelWithTooltip";
import { SliderWithInput } from "./SliderWithInput";
import { getDefaultDenoisingSteps, getDefaultResolution } from "../lib/utils";
import type { PipelineId } from "../types";

const MIN_DIMENSION = 16;
const MAX_DIMENSION = 2048;
const MAX_SEED = 2147483647;

interface SettingsPanelProps {
  pipelineId: PipelineId;
  onPipelineIdChange?: (pipelineId: PipelineId) => void;
  isStreaming?: boolean;
  resolution?: { height: number; width: number };
  onResolutionChange?: (resolution: { height: number; width: number }) => void;
  seed?: number;
  onSeedChange?: (seed: number) => void;
  denoisingSteps?: number[];
  onDenoisingStepsChange?: (denoisingSteps: number[]) => void;
  noiseScale?: number;
  onNoiseScaleChange?: (noiseScale: number) => void;
  noiseController?: boolean;
  onNoiseControllerChange?: (enabled: boolean) => void;
  manageCache?: boolean;
  onManageCacheChange?: (enabled: boolean) => void;
  onResetCache?: () => void;
}

/** One integer parameter: label, [−] value [+], and its validation message. */
function NumberField({
  label,
  tooltip,
  value,
  min,
  max,
  error,
  disabled,
  onChange,
}: {
  label: string;
  tooltip?: string;
  value: number;
  min: number;
  max: number;
  error: string | null;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <div className="app-stack">
      <div className="app-row">
        <LabelWithTooltip
          label={label}
          tooltip={tooltip}
          className="app-label app-label--w14"
        />
        <div className="app-stepper" data-invalid={error ? "true" : "false"}>
          <Button
            variant="ghost"
            size="icon-sm"
            onPress={() => onChange(Math.max(min, value - 1))}
            disabled={disabled}
          >
            <Minus size={14} />
          </Button>
          <div className="app-stepper-field">
            <Input
              type="number"
              value={String(value)}
              onChangeText={text => {
                const parsed = parseInt(text);
                if (!isNaN(parsed)) onChange(parsed);
              }}
              disabled={disabled}
              textAlign="center"
              borderWidth={0}
              height={28}
              min={min}
              max={max}
            />
          </div>
          <Button
            variant="ghost"
            size="icon-sm"
            onPress={() => onChange(Math.min(max, value + 1))}
            disabled={disabled}
          >
            <Plus size={14} />
          </Button>
        </div>
      </div>
      {error && <p className="app-error">{error}</p>}
    </div>
  );
}

export function SettingsPanel({
  pipelineId,
  onPipelineIdChange,
  isStreaming = false,
  resolution,
  onResolutionChange,
  seed = 42,
  onSeedChange,
  denoisingSteps = [700, 500],
  onDenoisingStepsChange,
  noiseScale = 0.7,
  onNoiseScaleChange,
  noiseController = true,
  onNoiseControllerChange,
  manageCache = true,
  onManageCacheChange,
  onResetCache,
}: SettingsPanelProps) {
  // Use pipeline-specific default if resolution is not provided
  const effectiveResolution = resolution || getDefaultResolution(pipelineId);
  // Local state for noise scale for immediate UI feedback
  const [localNoiseScale, setLocalNoiseScale] = useState<number>(noiseScale);

  // Validation error states
  const [heightError, setHeightError] = useState<string | null>(null);
  const [widthError, setWidthError] = useState<string | null>(null);
  const [seedError, setSeedError] = useState<string | null>(null);

  // Sync with external value changes
  useEffect(() => {
    setLocalNoiseScale(noiseScale);
  }, [noiseScale]);

  const tunable =
    pipelineId === "longlive" || pipelineId === "streamdiffusionv2";
  const minDimension = tunable ? MIN_DIMENSION : 1;

  /** Range check shared by every numeric field: the message, or null. */
  const rangeError = (value: number, min: number, max: number) =>
    value < min
      ? `Must be at least ${min}`
      : value > max
        ? `Must be at most ${max}`
        : null;

  const handleResolutionChange = (
    dimension: "height" | "width",
    value: number
  ) => {
    const error = rangeError(value, minDimension, MAX_DIMENSION);
    (dimension === "height" ? setHeightError : setWidthError)(error);
    // Always update the value (even if invalid)
    onResolutionChange?.({ ...effectiveResolution, [dimension]: value });
  };

  const handleSeedChange = (value: number) => {
    setSeedError(rangeError(value, 0, MAX_SEED));
    // Always update the value (even if invalid)
    onSeedChange?.(value);
  };

  const currentPipeline = PIPELINES[pipelineId];

  return (
    <Card height="100%" flex={1}>
      <CardHeader>
        <CardTitle>Settings</CardTitle>
      </CardHeader>
      <CardContent flex={1} minH={0}>
        <div className="app-scroll">
          <div className="app-stack app-stack--lg">
            <div className="app-stack">
              <h3 className="app-section-title">Pipeline ID</h3>
              <Select
                value={pipelineId}
                onValueChange={value => {
                  if (value in PIPELINES)
                    onPipelineIdChange?.(value as PipelineId);
                }}
              >
                <SelectTrigger disabled={isStreaming}>
                  <SelectValue placeholder="Select a pipeline" />
                </SelectTrigger>
                <SelectContent>
                  {Object.keys(PIPELINES).map(id => (
                    <SelectItem key={id} value={id}>
                      {id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {currentPipeline && (
              <Card>
                <CardContent>
                  <div className="app-stack">
                    <h4 className="app-section-title">{currentPipeline.name}</h4>
                    <div className="app-row">
                      {currentPipeline.about && (
                        <TooltipProvider delay={200}>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Badge variant="outline">
                                <Info size={14} />
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent maxW={320}>
                              {currentPipeline.about}
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      )}
                      {currentPipeline.modified && (
                        <TooltipProvider delay={200}>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Badge variant="outline">
                                <Hammer size={14} />
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent maxW={320}>
                              This pipeline contains modifications based on the
                              original project.
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      )}
                      {currentPipeline.projectUrl && (
                        <a
                          href={currentPipeline.projectUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="app-link"
                        >
                          <Badge variant="outline">Project Page</Badge>
                        </a>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {tunable && (
              <div className="app-stack">
                <h3 className="app-section-title">Parameters</h3>
                <NumberField
                  label={PARAMETER_METADATA.height.label}
                  tooltip={PARAMETER_METADATA.height.tooltip}
                  value={effectiveResolution.height}
                  min={minDimension}
                  max={MAX_DIMENSION}
                  error={heightError}
                  disabled={isStreaming}
                  onChange={v => handleResolutionChange("height", v)}
                />
                <NumberField
                  label={PARAMETER_METADATA.width.label}
                  tooltip={PARAMETER_METADATA.width.tooltip}
                  value={effectiveResolution.width}
                  min={minDimension}
                  max={MAX_DIMENSION}
                  error={widthError}
                  disabled={isStreaming}
                  onChange={v => handleResolutionChange("width", v)}
                />
                <NumberField
                  label={PARAMETER_METADATA.seed.label}
                  tooltip={PARAMETER_METADATA.seed.tooltip}
                  value={seed}
                  min={0}
                  max={MAX_SEED}
                  error={seedError}
                  disabled={isStreaming}
                  onChange={handleSeedChange}
                />
              </div>
            )}

            {tunable && (
              <div className="app-stack">
                <div className="app-row app-row--between">
                  <LabelWithTooltip
                    label={PARAMETER_METADATA.manageCache.label}
                    tooltip={PARAMETER_METADATA.manageCache.tooltip}
                  />
                  <Switch
                    checked={manageCache}
                    onCheckedChange={next => onManageCacheChange?.(next)}
                  />
                </div>

                <div className="app-row app-row--between">
                  <LabelWithTooltip
                    label={PARAMETER_METADATA.resetCache.label}
                    tooltip={PARAMETER_METADATA.resetCache.tooltip}
                  />
                  <Button
                    onPress={() => onResetCache?.()}
                    disabled={manageCache}
                    variant="outline"
                    size="icon-sm"
                  >
                    <RotateCcw size={14} />
                  </Button>
                </div>
              </div>
            )}

            {tunable && (
              <DenoisingStepsSlider
                value={denoisingSteps}
                onChange={next => onDenoisingStepsChange?.(next)}
                defaultValues={getDefaultDenoisingSteps(pipelineId)}
                tooltip={PARAMETER_METADATA.denoisingSteps.tooltip}
              />
            )}

            {pipelineId === "streamdiffusionv2" && (
              <div className="app-stack">
                <div className="app-row app-row--between">
                  <LabelWithTooltip
                    label={PARAMETER_METADATA.noiseController.label}
                    tooltip={PARAMETER_METADATA.noiseController.tooltip}
                  />
                  <Switch
                    checked={noiseController}
                    onCheckedChange={next => onNoiseControllerChange?.(next)}
                    disabled={isStreaming}
                  />
                </div>

                <SliderWithInput
                  label={PARAMETER_METADATA.noiseScale.label}
                  tooltip={PARAMETER_METADATA.noiseScale.tooltip}
                  value={localNoiseScale}
                  onValueChange={setLocalNoiseScale}
                  onValueCommit={next => onNoiseScaleChange?.(next)}
                  min={0.0}
                  max={1.0}
                  step={0.01}
                  incrementAmount={0.01}
                  disabled={noiseController}
                  labelClassName="app-label app-label--w20"
                  valueFormatter={v => Math.round(v * 100) / 100}
                  inputParser={v => parseFloat(v) || 0.0}
                />
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
