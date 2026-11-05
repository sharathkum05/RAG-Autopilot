export type Source = "playground" | "dataset" | "compare";
export type Confidence = "high" | "medium" | "low";

/** Page images in the ViDoRe V3 corpus are 1700 × 2200; bounding boxes are in those pixels. */
export const PAGE_W = 1700;
export const PAGE_H = 2200;

export interface EvidenceScores {
  visualRank?: number;
  visualScore?: number;
  textRank?: number;
  textScore?: number;
  fusedRank?: number;
  rerank?: number;
}

export interface EvidenceChunk {
  id: string;
  doc: string;
  /** corpus id of the page */
  page: number;
  /** 1-based page number inside the document */
  pno: number;
  kind: "text" | "table" | "figure";
  /** [x1, y1, x2, y2] in page pixels */
  boxes: number[][];
  image: string;
  snippet: string;
  /** substring of snippet to highlight */
  highlight?: string;
  scores: EvidenceScores;
}

export interface AssistantMeta {
  confidence: Confidence;
  /** the chunks the answer cited */
  chunks: EvidenceChunk[];
  /** the full ranked list retrieval produced (top final_k pages), cited or not */
  retrieved?: EvidenceChunk[];
  config?: string;
  ttrMs: number;
  retrievalMs: number;
  cost?: number;
  model?: string;
  note?: string;
  clarify?: string[];
}

/** A retrieval step streamed before the answer, shown as a live step group. */
export interface ChatStep {
  name: string;
  detail?: string;
  state: "active" | "done" | "error";
  ms?: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  meta?: AssistantMeta;
  pending?: boolean;
  steps?: ChatStep[];
  startedAt?: number;
}

export type ChatEvent =
  | { type: "start"; conversationId: string }
  | { type: "step"; step: ChatStep }
  | { type: "token"; text: string }
  | { type: "evidence"; meta: AssistantMeta }
  | { type: "done" }
  | { type: "error"; message: string };

export interface IngestFile {
  name: string;
  size: number;
  type: string;
  chunks?: number;
}

export interface IngestStep {
  name: string;
  detail: string;
  state: "pending" | "active" | "done";
  ms?: number;
}

export interface IngestStatus {
  jobId: string;
  files: IngestFile[];
  steps: IngestStep[];
  /** 0..1 */
  progress: number;
  done: boolean;
  conversationId?: string;
}
