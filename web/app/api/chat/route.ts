import { proxy } from "@/lib/proxy";
import { pickAnswer } from "@/lib/mock-answers";
import type { ChatEvent } from "@/lib/types";

/**
 * Mock chat endpoint. Streams newline-delimited JSON events: start, step*, token*, evidence, done.
 * Steps mirror the real pipeline so the UI shows retrieval happening before the answer streams.
 * The Modal backend streams the same event shapes, with real retrieval and Claude behind them.
 */
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const STEPS: [string, string, number][] = [
  ["Encoding question", "ColQwen2.5 · bge-m3 · BM25", 38],
  ["Visual search", "matt_pages · MaxSim · top 50", 97],
  ["Text search", "matt_chunks · dense + BM25 · top 50", 56],
  ["Fusing", "reciprocal rank fusion → 20 pages", 2],
  ["Reranking", "bge-reranker-v2-m3 → top 5", 388],
];

export async function POST(req: Request) {
  const forwarded = await proxy(req, "/api/chat");
  if (forwarded) return forwarded;
  const { question, conversationId } = (await req.json()) as { question: string; conversationId: string };
  const ans = pickAnswer(question ?? "");
  const enc = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (e: ChatEvent) => controller.enqueue(enc.encode(JSON.stringify(e) + "\n"));
      const t0 = Date.now();
      send({ type: "start", conversationId });
      const retrieving = ans.meta.retrievalMs > 0;
      for (const [name, detail, ms] of STEPS) {
        if (!retrieving) break;
        send({ type: "step", step: { name, detail, state: "active" } });
        await sleep(Math.min(ms, 300) + 120);
        send({ type: "step", step: { name, detail, state: "done", ms } });
      }
      send({ type: "step", step: { name: "Answering", detail: "claude-opus-5-5 · top 5 pages as citable documents", state: "active" } });
      await sleep(350);
      for (const word of ans.text.split(" ")) {
        send({ type: "token", text: word + " " });
        await sleep(18);
      }
      send({ type: "step", step: { name: "Answering", detail: "claude-opus-5-5 · top 5 pages as citable documents", state: "done", ms: Date.now() - t0 } });
      send({ type: "evidence", meta: { ...ans.meta, ttrMs: Date.now() - t0 } });
      send({ type: "done" });
      controller.close();
    },
  });
  return new Response(stream, { headers: { "Content-Type": "application/x-ndjson; charset=utf-8", "Cache-Control": "no-cache" } });
}
