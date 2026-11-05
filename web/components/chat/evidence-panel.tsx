"use client";

import Image from "next/image";
import { ChevronRight } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import { PAGE_H, PAGE_W, type EvidenceChunk } from "@/lib/types";

export function PageThumb({ chunk, className }: { chunk: EvidenceChunk; className?: string }) {
  return (
    <div className={cn("relative aspect-[360/465] w-full overflow-hidden rounded border border-border bg-white", className)}>
      {chunk.image ? (
        <Image src={chunk.image} alt={`${chunk.doc}, page ${chunk.pno}`} fill sizes="120px" className="object-cover" unoptimized />
      ) : (
        <span className="absolute inset-0 grid place-items-center px-1 text-center font-mono text-[9px] leading-tight text-neutral-500">no page render</span>
      )}
      {chunk.boxes.map((b, i) => {
        // Boxes are either normalised 0..1 (backend) or in ViDoRe page pixels (mock data).
        const norm = b.every((v) => v <= 1.0001);
        const w = norm ? 1 : PAGE_W, h = norm ? 1 : PAGE_H;
        return (
          <span
            key={i}
            aria-hidden="true"
            className="absolute rounded-[2px] border-2 border-evidence bg-evidence/25"
            style={{ left: `${(b[0] / w) * 100}%`, top: `${(b[1] / h) * 100}%`, width: `${((b[2] - b[0]) / w) * 100}%`, height: `${((b[3] - b[1]) / h) * 100}%` }}
          />
        );
      })}
    </div>
  );
}

function Snippet({ text, highlight }: { text: string; highlight?: string }) {
  if (!highlight || !text.includes(highlight)) return <>{text}</>;
  const [before, after] = text.split(highlight);
  return (
    <>
      {before}
      <mark className="rounded-sm bg-evidence/35 px-0.5 text-foreground">{highlight}</mark>
      {after}
    </>
  );
}

function Score({ label, value }: { label: string; value: string }) {
  return (
    <span className="rounded-full border border-border bg-card px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
      {label} <b className="font-medium text-foreground">{value}</b>
    </span>
  );
}

interface EvidencePanelProps {
  messageId: string;
  chunks: EvidenceChunk[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
  activeIndex: number | null;
}

/** "Answered from" dropdown under an assistant message: the chunks, pages, highlighted regions and scores. */
export function EvidencePanel({ messageId, chunks, open, onOpenChange, activeIndex }: EvidencePanelProps) {
  return (
    <Collapsible open={open} onOpenChange={onOpenChange} className="rounded-xl border border-border bg-card">
      <CollapsibleTrigger className="group flex w-full items-center gap-2 px-3.5 py-2.5 text-left text-[13px] text-muted-foreground hover:text-foreground">
        <ChevronRight className="size-3.5 shrink-0 transition-transform group-data-[panel-open]:rotate-90" />
        <span className="min-w-0 flex-1 truncate">
          <b className="font-medium text-foreground">Answered from</b>{" "}
          {chunks.map((c, i) => `[${i + 1}] ${c.doc} p.${c.pno}`).join(" · ")}
        </span>
        <span className="hidden shrink-0 font-mono text-[11px] sm:inline">chunks and scores</span>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="grid gap-2.5 px-3.5 pb-3.5">
          {chunks.map((c, i) => (
            <div
              key={c.id}
              id={`chunk-${messageId}-${i}`}
              className={cn(
                "grid grid-cols-[88px_minmax(0,1fr)] gap-3.5 rounded-lg border border-border bg-secondary/40 p-3 transition-colors",
                activeIndex === i && "border-foreground/60 ring-1 ring-foreground/30",
              )}
            >
              <PageThumb chunk={c} />
              <div className="min-w-0">
                <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-[12.5px] text-muted-foreground">
                  <b className="font-semibold text-foreground">[{i + 1}] {c.doc}</b>
                  <span>page {c.pno}</span>
                  <span className="font-mono text-[11px]">{c.id} · {c.kind}</span>
                </div>
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {c.scores.visualRank != null && <Score label="visual" value={`#${c.scores.visualRank} · ${c.scores.visualScore?.toFixed(1)}`} />}
                  {c.scores.textRank != null && <Score label="text" value={`#${c.scores.textRank} · ${c.scores.textScore?.toFixed(2)}`} />}
                  {c.scores.fusedRank != null && <Score label="rank" value={`#${c.scores.fusedRank}`} />}
                  {c.scores.rerank != null && <Score label="rerank" value={c.scores.rerank.toFixed(2)} />}
                </div>
                <p className="mt-2 max-w-[70ch] text-[13px] leading-relaxed text-foreground">
                  <Snippet text={c.snippet} highlight={c.highlight} />
                </p>
              </div>
            </div>
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
