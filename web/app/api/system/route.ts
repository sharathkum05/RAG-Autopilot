import { proxy } from "@/lib/proxy";
import { NextResponse } from "next/server";

/**
 * Runtime metrics. The Modal container implements this with psutil + pynvml (CPU %, RAM, GPU util, GPU memory,
 * loaded models, uptime, cold starts) sampled every 5 s into a ring buffer; this mock returns a plausible snapshot.
 */
const started = Date.now();
let i = 0;
export async function GET(req: Request) {
  const forwarded = await proxy(req, "/api/system");
  if (forwarded) return forwarded;
  i += 1;
  const wobble = (base: number, amp: number) => +(base + amp * Math.sin(i / 3) + amp * 0.5 * Math.cos(i / 7)).toFixed(1);
  return NextResponse.json({
    source: "mock", sampled_at: new Date().toISOString(),
    container: { id: "ta-01HXK9…", gpu: "NVIDIA L4 (24 GB)", region: "us-east", uptime_s: Math.floor((Date.now() - started) / 1000) + 1842, cold_starts_today: 3, state: "warm", scale_to_zero_after_s: 300 },
    cpu: { percent: wobble(18, 9), cores: 4 },
    memory: { used_gb: wobble(6.4, 0.3), total_gb: 16 },
    gpu: { util_percent: wobble(11, 10), memory_used_gb: wobble(13.2, 0.4), memory_total_gb: 24, temperature_c: wobble(48, 3) },
    models_loaded: [{ name: "colqwen2.5-v0.2", memory_gb: 7.4 }, { name: "bge-m3", memory_gb: 2.3 }, { name: "bge-reranker-v2-m3", memory_gb: 2.2 }],
    requests: { last_hour: 37, p50_ms: 196, p95_ms: 548, errors: 0 },
  });
}
