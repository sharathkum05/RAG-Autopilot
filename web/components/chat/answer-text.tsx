"use client";

import { memo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

/** `[n]` citation markers become `#cite-n` links so the markdown parser carries them through intact. */
const linkify = (text: string) => text.replace(/\[(\d{1,2})\]/g, (_m, n: string) => `[${n}](#cite-${n})`);

const CITE_BUTTON =
  "mx-0.5 inline-flex h-[19px] min-w-[19px] -translate-y-px items-center justify-center rounded-md border border-border bg-secondary px-1 align-middle font-mono text-[11px] font-medium text-foreground hover:border-foreground";

interface AnswerTextProps {
  text: string;
  className?: string;
  onHover?: (i: number | null) => void;
  onClick?: (i: number) => void;
}

/** Answer markdown (GFM) with `[n]` citation markers rendered as buttons that point at the evidence panel. */
export const AnswerText = memo(function AnswerText({ text, className, onHover, onClick }: AnswerTextProps) {
  const components: Components = {
    a: ({ href, children }) => {
      const m = /^#cite-(\d+)$/.exec(href ?? "");
      if (m) {
        const i = Number(m[1]) - 1;
        return (
          <button type="button" className={CITE_BUTTON} onMouseEnter={() => onHover?.(i)} onMouseLeave={() => onHover?.(null)} onClick={() => onClick?.(i)}>
            {m[1]}
          </button>
        );
      }
      return <a href={href} target="_blank" rel="noreferrer" className="underline underline-offset-2 hover:text-foreground">{children}</a>;
    },
    p: ({ children }) => <p className="my-2.5 first:mt-0 last:mb-0">{children}</p>,
    ul: ({ children }) => <ul className="my-2.5 list-disc space-y-1 pl-5 first:mt-0 last:mb-0">{children}</ul>,
    ol: ({ children }) => <ol className="my-2.5 list-decimal space-y-1 pl-5 first:mt-0 last:mb-0">{children}</ol>,
    li: ({ children }) => <li className="pl-0.5 [&>p]:my-0">{children}</li>,
    strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
    em: ({ children }) => <em className="italic">{children}</em>,
    h1: ({ children }) => <h3 className="mb-1.5 mt-4 text-[15px] font-semibold first:mt-0">{children}</h3>,
    h2: ({ children }) => <h3 className="mb-1.5 mt-4 text-[15px] font-semibold first:mt-0">{children}</h3>,
    h3: ({ children }) => <h4 className="mb-1 mt-3 text-[15px] font-semibold first:mt-0">{children}</h4>,
    h4: ({ children }) => <h4 className="mb-1 mt-3 text-[15px] font-semibold first:mt-0">{children}</h4>,
    code: ({ children, className: lang }) => (
      <code className={cn("rounded bg-secondary px-1 py-0.5 font-mono text-[13px] [pre_&]:bg-transparent [pre_&]:p-0", lang)}>{children}</code>
    ),
    pre: ({ children }) => <pre className="my-2.5 overflow-x-auto rounded-lg border border-border bg-secondary/60 p-3 font-mono text-[13px] leading-6">{children}</pre>,
    blockquote: ({ children }) => <blockquote className="my-2.5 border-l-2 border-border pl-3 text-muted-foreground">{children}</blockquote>,
    hr: () => <hr className="my-3 border-border" />,
    table: ({ children }) => (
      <div className="my-2.5 overflow-x-auto rounded-lg border border-border">
        <table className="w-full border-collapse text-[13px] leading-6">{children}</table>
      </div>
    ),
    thead: ({ children }) => <thead className="bg-secondary/60 text-left">{children}</thead>,
    th: ({ children }) => <th className="border-b border-border px-2.5 py-1.5 font-medium">{children}</th>,
    td: ({ children }) => <td className="border-b border-border px-2.5 py-1.5 align-top last:border-b-0">{children}</td>,
  };
  return (
    <div className={cn("max-w-[72ch] text-[15px] leading-7 text-foreground", className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {linkify(text)}
      </ReactMarkdown>
    </div>
  );
});
