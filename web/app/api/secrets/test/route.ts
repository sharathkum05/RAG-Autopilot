import { proxy } from "@/lib/proxy";
import { NextResponse } from "next/server";
import { SERVICES } from "@/lib/secrets";

/** Pings a service with the server-held key. Mock: "not configured" unless every env var is set; the backend does real pings. */
export async function POST(req: Request) {
  const forwarded = await proxy(req, "/api/secrets/test");
  if (forwarded) return forwarded;
  const { service } = (await req.json()) as { service: string };
  const s = SERVICES.find((x) => x.id === service);
  if (!s) return NextResponse.json({ error: "unknown service" }, { status: 400 });
  const missing = s.env.filter((k) => !process.env[k]);
  const t0 = Date.now();
  await new Promise((r) => setTimeout(r, 300 + Math.random() * 400));
  if (missing.length) return NextResponse.json({ service, ok: false, message: `not configured: ${missing.join(", ")}`, latency_ms: Date.now() - t0 });
  return NextResponse.json({ service, ok: true, message: "reachable (mock ping)", latency_ms: Date.now() - t0 });
}
