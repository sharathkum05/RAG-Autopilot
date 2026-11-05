import { cookies } from "next/headers";
import { BYOK_COOKIE, decryptKey } from "@/lib/byok";

/**
 * When BACKEND_URL is set, every /api route forwards to the FastAPI backend (streaming bodies pass through) and
 * attaches the session's bring-your-own Anthropic key as a header. When it is unset, the route runs its mock.
 */
/**
 * Read-only data that only changes when an eval runs, cached at Vercel's edge: after the first visit these pages load
 * instantly even while the GPU backend is scaled to zero, and a stale copy is served while a fresh one is fetched.
 * Only Vercel's CDN reads this header; browsers still ask the edge each time.
 */
const EDGE_CACHE: [prefix: string, policy: string][] = [
  ["/api/evals", "s-maxage=3600, stale-while-revalidate=604800"],
  ["/api/answer-evals", "s-maxage=3600, stale-while-revalidate=604800"],
  ["/api/observability", "s-maxage=60, stale-while-revalidate=3600"],
];

export async function proxy(req: Request, path: string): Promise<Response | null> {
  const base = process.env.BACKEND_URL;
  if (!base) return null;
  const url = new URL(req.url);
  const target = `${base.replace(/\/$/, "")}${path}${url.search}`;
  const headers = new Headers();
  const ct = req.headers.get("content-type");
  if (ct) headers.set("content-type", ct);
  try {
    const tok = (await cookies()).get(BYOK_COOKIE)?.value;
    const key = tok ? decryptKey(tok) : null;
    if (key) headers.set("x-anthropic-key", key);
  } catch { /* no cookie store in this context */ }
  const init: RequestInit & { duplex?: "half" } = { method: req.method, headers, cache: "no-store" };
  if (req.method !== "GET" && req.method !== "HEAD") { init.body = req.body; init.duplex = "half"; }
  const res = await fetch(target, init);
  const out = new Headers();
  for (const h of ["content-type", "cache-control"]) { const v = res.headers.get(h); if (v) out.set(h, v); }
  const edge = req.method === "GET" && res.ok ? EDGE_CACHE.find(([p]) => path.startsWith(p))?.[1] : undefined;
  if (edge) out.set("Vercel-CDN-Cache-Control", edge);
  return new Response(res.body, { status: res.status, headers: out });
}
