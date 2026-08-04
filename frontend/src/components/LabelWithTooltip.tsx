import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@hanzo/ui";

interface LabelWithTooltipProps {
  label: string;
  tooltip?: string;
  className?: string;
  htmlFor?: string;
}

/** A label that grows a hover hint when one is supplied. */
export function LabelWithTooltip({
  label,
  tooltip,
  className = "app-label",
  htmlFor,
}: LabelWithTooltipProps) {
  if (!tooltip) {
    return (
      <label htmlFor={htmlFor} className={className}>
        {label}
      </label>
    );
  }

  return (
    <TooltipProvider delay={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <label htmlFor={htmlFor} className={`${className} app-label--help`}>
            {label}
          </label>
        </TooltipTrigger>
        <TooltipContent maxW={320}>{tooltip}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
