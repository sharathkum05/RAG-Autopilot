import { createCipheriv, createDecipheriv, createHash, randomBytes } from "node:crypto";

/**
 * Bring-your-own Anthropic key for a demo session: AES-256-GCM in an httpOnly cookie, never stored server-side,
 * forwarded to the backend per request. APP_SECRET must be set in production.
 */
export const BYOK_COOKIE = "matt_anthropic_key";
const secret = () => createHash("sha256").update(process.env.APP_SECRET ?? "dev-only-secret-change-me").digest();

export function encryptKey(plain: string): string {
  const iv = randomBytes(12);
  const c = createCipheriv("aes-256-gcm", secret(), iv);
  const enc = Buffer.concat([c.update(plain, "utf8"), c.final()]);
  return Buffer.concat([iv, c.getAuthTag(), enc]).toString("base64url");
}

export function decryptKey(token: string): string | null {
  try {
    const buf = Buffer.from(token, "base64url");
    const iv = buf.subarray(0, 12), tag = buf.subarray(12, 28), data = buf.subarray(28);
    const d = createDecipheriv("aes-256-gcm", secret(), iv);
    d.setAuthTag(tag);
    return Buffer.concat([d.update(data), d.final()]).toString("utf8");
  } catch {
    return null;
  }
}
