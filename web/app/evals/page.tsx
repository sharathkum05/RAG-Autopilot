import { redirect } from "next/navigation";
import { FLAGS } from "@/lib/flags";
import { RetrievalEvals } from "./retrieval-evals";

/** Behind the retrievalEvals flag: when it is off, the route sends visitors to Evals Pipeline instead. */
export default function EvalsPage() {
  if (!FLAGS.retrievalEvals) redirect("/answer-evals");
  return <RetrievalEvals />;
}
