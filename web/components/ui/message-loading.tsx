// Three bouncing dots. Plain CSS (Tailwind's animate-bounce with staggered delays) rather than chained SVG <animate>
// ids: those ids are global to the page, so a second loader on screen broke the chain and froze the dots.
function MessageLoading() {
  return (
    <span className="inline-flex h-6 items-center gap-1 text-foreground" aria-label="Thinking" role="img">
      {[0, 150, 300].map((delay) => (
        <span key={delay} className="size-1.5 animate-bounce rounded-full bg-current" style={{ animationDelay: `${delay}ms`, animationDuration: "900ms" }} />
      ))}
    </span>
  );
}

export { MessageLoading };
