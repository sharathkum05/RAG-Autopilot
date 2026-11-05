/** Services whose keys the server holds. Values never leave the server; the UI sees presence and last four characters. */
export const SERVICES = [
  { id: "anthropic", name: "Anthropic", env: ["ANTHROPIC_API_KEY"], stored: "Modal secret · Vercel env", purpose: "answer generation" },
  { id: "qdrant", name: "Qdrant", env: ["QDRANT_URL", "QDRANT_API_KEY"], stored: "Modal secret · Vercel env", purpose: "vector store" },
  { id: "supabase", name: "Supabase", env: ["SUPABASE_URL", "SUPABASE_SERVICE_KEY"], stored: "Modal secret · Vercel env", purpose: "metadata and page images" },
  { id: "langfuse", name: "Langfuse", env: ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"], stored: "Modal secret · Vercel env", purpose: "traces, cost, TTR" },
  { id: "huggingface", name: "Hugging Face", env: ["HF_TOKEN"], stored: "Modal secret", purpose: "model downloads (optional)" },
  { id: "openai", name: "OpenAI", env: ["OPENAI_API_KEY"], stored: "Modal secret", purpose: "answer-eval judge (optional)" },
  { id: "modal", name: "Modal", env: ["MODAL_API_URL"], stored: "Vercel env", purpose: "query API endpoint" },
] as const;

export type ServiceId = (typeof SERVICES)[number]["id"];
