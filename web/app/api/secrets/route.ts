import { proxy } from "@/lib/proxy";
import { NextResponse } from "next/server";
import { SERVICES } from "@/lib/secrets";

const mask = (v: string | undefined) => (v ? `${"•".repeat(8)}${v.slice(-4)}` : null);

/** Which secrets the server has. Only presence and the last four characters are returned. */
export async function GET(req: Request) {
  const forwarded = await proxy(req, "/api/secrets");
  if (forwarded) return forwarded;
  return NextResponse.json({
    services: SERVICES.map((s) => ({
      id: s.id, name: s.name, purpose: s.purpose, stored: s.stored,
      keys: s.env.map((k) => ({ name: k, set: Boolean(process.env[k]), masked: mask(process.env[k]) })),
    })),
  });
}

/** Saving keys only works against the real backend; the mock has nowhere to put them. */
export async function POST(req: Request) {
  const forwarded = await proxy(req, "/api/secrets");
  if (forwarded) return forwarded;
  return NextResponse.json({ error: "Set BACKEND_URL to save keys; the mock routes cannot store secrets." }, { status: 501 });
}
