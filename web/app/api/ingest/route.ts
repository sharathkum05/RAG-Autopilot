import { proxy } from "@/lib/proxy";
import { NextResponse } from "next/server";
import type { IngestFile, IngestStatus, IngestStep } from "@/lib/types";

/**
 * Mock ingestion endpoint. POST accepts files and starts a job whose stages advance with wall-clock time;
 * GET ?job= reports progress. The Modal backend implements the same two calls.
 */
const STEPS: [string, string, number][] = [
  ["Router", "file type from magic bytes", 60],
  ["Parse", "Docling · layout · OCR · tables", 1800],
  ["Chunk", "sections · tables · figures", 300],
  ["Embed text", "bge-m3 + BM25", 900],
  ["Embed pages", "ColQwen2.5", 1100],
  ["Index", "qdrant + page store", 300],
];

interface Job { startedAt: number; files: IngestFile[] }
const jobs = new Map<string, Job>();

export async function POST(req: Request) {
  const forwarded = await proxy(req, "/api/ingest");
  if (forwarded) return forwarded;
  const form = await req.formData();
  const files: IngestFile[] = form.getAll("files").filter((f): f is File => f instanceof File).map((f) => ({
    name: f.name, size: f.size, type: (f.name.split(".").pop() ?? "").toUpperCase().slice(0, 4),
  }));
  if (!files.length) return NextResponse.json({ error: "no files" }, { status: 400 });
  const jobId = `job_${Math.random().toString(16).slice(2, 10)}`;
  jobs.set(jobId, { startedAt: Date.now(), files });
  return NextResponse.json({ jobId });
}

export async function GET(req: Request) {
  const forwarded = await proxy(req, "/api/ingest");
  if (forwarded) return forwarded;
  const jobId = new URL(req.url).searchParams.get("job") ?? "";
  const job = jobs.get(jobId);
  if (!job) return NextResponse.json({ error: "unknown job" }, { status: 404 });
  const elapsed = Date.now() - job.startedAt;
  const total = STEPS.reduce((a, s) => a + s[2], 0);
  let acc = 0;
  const steps: IngestStep[] = STEPS.map(([name, detail, ms]) => {
    const start = acc; acc += ms;
    const state: IngestStep["state"] = elapsed >= acc ? "done" : elapsed >= start ? "active" : "pending";
    return { name, detail, state, ms: state === "done" ? ms : undefined };
  });
  const done = elapsed >= total;
  const body: IngestStatus = {
    jobId,
    files: job.files.map((f, i) => (done ? { ...f, chunks: 18 + ((f.size + i * 7) % 40) } : f)),
    steps,
    progress: Math.min(1, elapsed / total),
    done,
    conversationId: done ? `conv_pg_${jobId.slice(4, 8)}` : undefined,
  };
  return NextResponse.json(body);
}
