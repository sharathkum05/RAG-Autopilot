/**
 * Feature flags, read at build time from NEXT_PUBLIC_FLAG_* env vars ("1" turns one on). Everything defaults to off,
 * so a flag is enabled by setting the variable in Vercel and redeploying.
 */
export const FLAGS = {
  /** The Retrieval Evals tab (/evals): page-level retrieval metrics per config. Hidden for now. */
  retrievalEvals: process.env.NEXT_PUBLIC_FLAG_RETRIEVAL_EVALS === "1",
} as const;
