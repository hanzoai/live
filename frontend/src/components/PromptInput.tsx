import { useState } from "react";
import { Button, Input } from "@hanzo/ui";
import { ArrowUp } from "@hanzogui/lucide-icons-2";

interface PromptInputProps {
  currentPrompt: string;
  onPromptChange?: (prompt: string) => void;
  onPromptSubmit?: (prompt: string) => void;
  disabled?: boolean;
}

export function PromptInput({
  currentPrompt,
  onPromptChange,
  onPromptSubmit,
  disabled = false,
}: PromptInputProps) {
  const [isProcessing, setIsProcessing] = useState(false);

  const handleSubmit = () => {
    if (!currentPrompt.trim()) return;

    setIsProcessing(true);
    onPromptSubmit?.(currentPrompt.trim());
    setTimeout(() => setIsProcessing(false), 1000);
  };

  return (
    <div className="app-prompt">
      <div className="app-prompt-field">
        <Input
          placeholder="blooming flowers"
          value={currentPrompt}
          onChangeText={text => onPromptChange?.(text)}
          onKeyPress={e => {
            if ((e as unknown as React.KeyboardEvent).key === "Enter")
              handleSubmit();
          }}
          disabled={disabled}
          borderWidth={0}
        />
      </div>
      <Button
        onPress={handleSubmit}
        disabled={disabled || !currentPrompt.trim() || isProcessing}
        size="icon-sm"
        rounded={9999}
      >
        {isProcessing ? "..." : <ArrowUp size={16} />}
      </Button>
    </div>
  );
}
