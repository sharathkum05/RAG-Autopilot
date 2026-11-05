import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { BYOK_COOKIE, decryptKey, encryptKey } from "@/lib/byok";

export async function GET() {
  const jar = await cookies();
  const tok = jar.get(BYOK_COOKIE)?.value;
  const key = tok ? decryptKey(tok) : null;
  return NextResponse.json({
    set: Boolean(key), last4: key ? key.slice(-4) : null, scope: "this browser session",
    warning: process.env.APP_SECRET ? null : "APP_SECRET is not set; the development fallback is in use",
  });
}

export async function POST(req: Request) {
  const { key } = (await req.json()) as { key?: string };
  const k = (key ?? "").trim();
  if (!/^sk-ant-/.test(k)) return NextResponse.json({ error: "That does not look like an Anthropic key (expected sk-ant-…)." }, { status: 400 });
  const jar = await cookies();
  jar.set(BYOK_COOKIE, encryptKey(k), { httpOnly: true, sameSite: "lax", secure: process.env.NODE_ENV === "production", path: "/", maxAge: 60 * 60 * 24 * 7 });
  return NextResponse.json({ set: true, last4: k.slice(-4) });
}

export async function DELETE() {
  const jar = await cookies();
  jar.delete(BYOK_COOKIE);
  return NextResponse.json({ set: false });
}
