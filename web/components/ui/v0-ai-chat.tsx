"use client";

import { useEffect, useRef, useCallback, useState, type ReactNode } from "react";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { ArrowUpIcon, Paperclip, Sparkles } from "lucide-react";

interface UseAutoResizeTextareaProps {
  minHeight: number;
  maxHeight?: number;
}

export function useAutoResizeTextarea({ minHeight, maxHeight }: UseAutoResizeTextareaProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const adjustHeight = useCallback(
    (reset?: boolean) => {
      const textarea = textareaRef.current;
      if (!textarea) return;
      if (reset) { textarea.style.height = `${minHeight}px`; return; }
      textarea.style.height = `${minHeight}px`;
      const newHeight = Math.max(minHeight, Math.min(textarea.scrollHeight, maxHeight ?? Number.POSITIVE_INFINITY));
      textarea.style.height = `${newHeight}px`;
    },
    [minHeight, maxHeight],
  );
  useEffect(() => { const t = textareaRef.current; if (t) t.style.height = `${minHeight}px`; }, [minHeight]);
  useEffect(() => { const h = () => adjustHeight(); window.addEventListener("resize", h); return () => window.removeEventListener("resize", h); }, [adjustHeight]);
  return { textareaRef, adjustHeight };
}

export interface ComposerAction {
  icon?: ReactNode;
  label: string;
  onClick: () => void;
}

export interface ChatComposerProps {
  placeholder?: string;
  disabled?: boolean;
  onSend: (text: string) => void;
  /** Paperclip button; omitted when not given. */
  onAttach?: () => void;
  /** The answer model, shown as a chip where v0 shows "Project". */
  model?: string;
  /** Pills under the box. Clicking a string sends it; an action runs its handler. */
  suggestions?: string[];
  actions?: ComposerAction[];
  /** Small note under the box, e.g. a validation hint. */
  notice?: ReactNode;
  autoFocus?: boolean;
  className?: string;
}

/** The v0 chat box: dark rounded field, attach button, model chip, arrow send that lights up when there is text. */
export function ChatComposer({ placeholder = "Ask a question...", disabled, onSend, onAttach, model, suggestions, actions, notice, autoFocus, className }: ChatComposerProps) {
  const [value, setValue] = useState("");
  const { textareaRef, adjustHeight } = useAutoResizeTextarea({ minHeight: 60, maxHeight: 200 });

  const submit = () => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
    adjustHeight(true);
  };
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
  };

  return (
    <div className={cn("w-full", className)}>
      <div className="relative rounded-xl border border-border bg-card">
        <div className="overflow-y-auto">
          <Textarea
            ref={textareaRef}
            value={value}
            disabled={disabled}
            autoFocus={autoFocus}
            onChange={(e) => { setValue(e.target.value); adjustHeight(); }}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            aria-label={placeholder}
            className={cn(
              "w-full px-4 py-3 resize-none border-none bg-transparent text-foreground text-sm",
              "focus:outline-none focus-visible:ring-0 focus-visible:ring-offset-0 focus-visible:border-transparent",
              "placeholder:text-muted-foreground placeholder:text-sm min-h-[60px] rounded-none dark:bg-transparent",
            )}
            style={{ overflow: "hidden" }}
          />
        </div>

        <div className="flex items-center justify-between p-3">
          <div className="flex min-w-0 items-center gap-2">
            {onAttach ? (
              <button type="button" onClick={onAttach} className="group flex items-center gap-1 rounded-lg p-2 text-foreground transition-colors hover:bg-accent" aria-label="Attach a document">
                <Paperclip className="h-4 w-4" />
                <span className="hidden text-xs text-muted-foreground transition-opacity group-hover:inline">Attach</span>
              </button>
            ) : null}
          </div>
          <div className="flex items-center gap-2">
            {model ? (
              <span className="flex items-center gap-1.5 rounded-lg border border-dashed border-border px-2 py-1 font-mono text-xs text-muted-foreground" title="Answer model">
                <Sparkles className="h-3.5 w-3.5" />
                {model}
              </span>
            ) : null}
            <button
              type="button"
              onClick={submit}
              disabled={disabled}
              className={cn(
                "flex items-center justify-between gap-1 rounded-lg border border-border px-1.5 py-1.5 text-sm transition-colors hover:border-ring hover:bg-accent",
                value.trim() ? "border-primary bg-primary text-primary-foreground hover:bg-primary" : "text-muted-foreground",
              )}
            >
              <ArrowUpIcon className={cn("h-4 w-4", value.trim() ? "text-primary-foreground" : "text-muted-foreground")} />
              <span className="sr-only">Send</span>
            </button>
          </div>
        </div>
      </div>

      {notice ? <div className="mt-2 text-center text-xs text-muted-foreground">{notice}</div> : null}

      {(actions && actions.length) || (suggestions && suggestions.length) ? (
        <div className="mt-4 flex flex-wrap items-center justify-center gap-3">
          {actions?.map((a) => (
            <button key={a.label} type="button" onClick={a.onClick} className="flex items-center gap-2 rounded-full border border-border bg-card px-4 py-2 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground">
              {a.icon}
              <span className="text-xs">{a.label}</span>
            </button>
          ))}
          {suggestions?.map((s) => (
            <button key={s} type="button" onClick={() => onSend(s)} className="flex items-center gap-2 rounded-full border border-border bg-card px-4 py-2 text-left text-muted-foreground transition-colors hover:bg-accent hover:text-foreground">
              <span className="text-xs">{s}</span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
