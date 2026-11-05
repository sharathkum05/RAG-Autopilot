"use client";

import { memo, useEffect, useRef, useState, type ComponentType } from "react";
import { cn } from "@/lib/utils";

/**
 * Claude-style collapsible step group: a shimmering header while pending, nested rows that appear as steps
 * arrive, an elapsed-time readout when done. Adapted for live-fed steps: every provided row is shown and the
 * last one shimmers while the group is pending.
 */

const SHIMMER_KEY = "an-tool-group-shimmer";

function ensureShimmerStyles() {
  if (typeof document === "undefined" || document.getElementById(SHIMMER_KEY)) return;
  const style = document.createElement("style");
  style.id = SHIMMER_KEY;
  style.textContent = `
@keyframes an-tg-shimmer { from { background-position: 100% center; } to { background-position: 0% center; } }
.an-tg-shimmer { display: inline-block; background-size: 250% 100%; background-clip: text; -webkit-background-clip: text; color: transparent;
  background-image: linear-gradient(90deg, var(--an-tg-shimmer-base, #8a8a8a) 0%, var(--an-tg-shimmer-base, #8a8a8a) 40%, var(--an-tg-shimmer-hi, #262626) 50%, var(--an-tg-shimmer-base, #8a8a8a) 60%, var(--an-tg-shimmer-base, #8a8a8a) 100%);
  background-repeat: no-repeat; animation: an-tg-shimmer 2s linear infinite; }
.dark .an-tg-shimmer { --an-tg-shimmer-base: #6b6b6b; --an-tg-shimmer-hi: #ededed; }
@media (prefers-reduced-motion: reduce) { .an-tg-shimmer { animation: none; color: inherit; background: none; } }
`;
  document.head.appendChild(style);
}

function IconChevronRight({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M9 6l6 6l-6 6" />
    </svg>
  );
}
function IconFile({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M14 3v4a1 1 0 0 0 1 1h4" />
      <path d="M17 21h-10a2 2 0 0 1 -2 -2v-14a2 2 0 0 1 2 -2h7l5 5v11a2 2 0 0 1 -2 2z" />
    </svg>
  );
}
function IconSearch({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="10" cy="10" r="7" />
      <path d="M21 21l-6 -6" />
    </svg>
  );
}
function IconTerminal({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M5 7l5 5l-5 5" />
      <path d="M12 19h7" />
    </svg>
  );
}

export type NestedToolCategory = "file" | "search" | "command" | "generic";
const CATEGORY_ICONS: Record<NestedToolCategory, ComponentType<{ className?: string }>> = { file: IconFile, search: IconSearch, command: IconTerminal, generic: IconFile };

export type NestedTool = {
  category?: NestedToolCategory;
  icon?: ComponentType<{ className?: string }>;
  title: string;
  subtitle?: string;
  isError?: boolean;
  /** Per-row timing, shown right-aligned when present. */
  elapsed?: string;
};

function NestedToolRow({ tool, isShimmer }: { tool: NestedTool; isShimmer?: boolean }) {
  const Icon = tool.icon ?? CATEGORY_ICONS[tool.category ?? "generic"];
  return (
    <div className="flex h-7 items-center gap-2 text-sm text-foreground/80">
      <Icon className={cn("h-3.5 w-3.5 shrink-0", tool.isError ? "text-destructive" : "text-muted-foreground")} />
      <span className={cn("truncate", isShimmer && "an-tg-shimmer", tool.isError && "text-destructive")}>{tool.title}</span>
      {tool.subtitle ? <span className="min-w-0 flex-1 truncate text-muted-foreground">{tool.subtitle}</span> : null}
      {tool.elapsed ? <span className="shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">{tool.elapsed}</span> : null}
    </div>
  );
}

export type ToolGroupState = "completed" | "pending" | "interrupted";

export type ToolGroupProps = {
  state?: ToolGroupState;
  description?: string;
  nestedTools?: NestedTool[];
  completeLabel: string;
  shimmerLabel?: string;
  interruptedLabel: string;
  /** Pre-formatted elapsed time, e.g. "6s" or "1m 12s". */
  elapsedTime?: string;
  defaultOpen?: boolean;
  maxVisibleTools?: number;
  showElapsed?: boolean;
  className?: string;
};

export const ToolGroup = memo(function ToolGroup({
  state = "completed", description, nestedTools = [], completeLabel, shimmerLabel, interruptedLabel, elapsedTime, defaultOpen, maxVisibleTools = 6, showElapsed = true, className,
}: ToolGroupProps) {
  useEffect(() => { ensureShimmerStyles(); }, []);
  const isPending = state === "pending";
  const hasNestedTools = nestedTools.length > 0;
  const [expanded, setExpanded] = useState(defaultOpen ?? isPending);
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => { if (isPending && listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight; }, [isPending, nestedTools.length]);

  if (state === "interrupted") {
    return <div className={cn("flex h-8 items-center gap-2 text-sm text-muted-foreground", className)}><span>{interruptedLabel}</span></div>;
  }

  const subtitle = isPending ? "" : description ? (description.length > 60 ? `${description.slice(0, 57)}...` : description) : "";
  const headerLabel = isPending ? (shimmerLabel ?? completeLabel) : completeLabel;
  const maskThreshold = maxVisibleTools;
  const streamHeight = Math.max(1, maxVisibleTools) * 28;

  return (
    <div className={cn("w-full", className)}>
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        disabled={!hasNestedTools}
        className={cn("group flex h-8 w-full items-center gap-2 text-left text-sm text-foreground/80", hasNestedTools ? "cursor-pointer" : "cursor-default")}
      >
        {hasNestedTools ? <IconChevronRight className={cn("h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform", expanded ? "rotate-90" : "rotate-0")} /> : null}
        <span className={cn("shrink-0", isPending && "an-tg-shimmer")}>{headerLabel}</span>
        {subtitle ? <span className="min-w-0 flex-1 truncate text-muted-foreground">{subtitle}</span> : null}
        {showElapsed && elapsedTime ? <span className="ml-auto shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">{elapsedTime}</span> : null}
      </button>

      {hasNestedTools && expanded ? (
        <div className="relative">
          {isPending && nestedTools.length > maskThreshold ? <div className="pointer-events-none absolute inset-x-0 top-0 z-10 h-10 bg-gradient-to-b from-background to-transparent" /> : null}
          <div
            ref={listRef}
            className={cn("pl-5", nestedTools.length > 1 ? "space-y-1" : "space-y-0", isPending && nestedTools.length > maskThreshold && "overflow-y-auto")}
            style={isPending && nestedTools.length > maskThreshold ? { height: `${streamHeight}px` } : undefined}
          >
            {nestedTools.map((tool, idx) => <NestedToolRow key={idx} tool={tool} isShimmer={isPending && idx === nestedTools.length - 1} />)}
          </div>
        </div>
      ) : null}
    </div>
  );
});

/** "1.2s", "48s", "1m 12s" */
export function formatElapsed(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s < 10 ? s.toFixed(1) : Math.round(s)}s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}
