import type { AssistantMeta, EvidenceChunk } from "@/lib/types";

/**
 * Sample answers used by the mock API routes until the Modal backend exists.
 * Questions, answers, pages and boxes come from the ViDoRe V3 Python textbook subset
 * (raw dataset v3/python_textbook); scores are illustrative.
 */
const DOC = "Introduction to Python Programming";
const img = (page: number) => `/pages/python/${String(page).padStart(5, "0")}.jpg`;

const chunk = (c: Omit<EvidenceChunk, "doc" | "image">): EvidenceChunk => ({ ...c, doc: DOC, image: img(c.page) });

interface MockAnswer {
  match: RegExp;
  text: string;
  meta: Omit<AssistantMeta, "ttrMs">;
}

export const ANSWERS: MockAnswer[] = [
  {
    match: /image|pillow|pypi|package index/i,
    text: "Pillow is the package for image manipulation: it reads and writes JPG, PNG and other formats [1]. It is a third-party module, so it is installed from the Python Package Index (PyPI) rather than shipping with Python [1].",
    meta: {
      confidence: "high",
      retrievalMs: 214,
      chunks: [
        chunk({ id: "114-c05", page: 114, pno: 200, kind: "table", boxes: [[400, 1849, 1300, 1910]],
          snippet: "Third-party modules. The Python Package Index (PyPI), available at pypi.org, is the official third-party software library for Python. Pillow: image processing for JPG, PNG and other formats.",
          highlight: "Pillow: image processing for JPG, PNG and other formats",
          scores: { visualRank: 1, visualScore: 20.8, textRank: 1, textScore: 0.74, fusedRank: 1, rerank: 0.92 } }),
      ],
    },
  },
  {
    match: /dictionar|dict\b|keys?\b.*(modif|chang|mutab)|immutable/i,
    text: "No. A dictionary key cannot be modified after it is set, because keys must be immutable objects [1]. To change a key you remove the entry and insert a new one; values, by contrast, can be changed in place [1].",
    meta: {
      confidence: "high",
      retrievalMs: 198,
      chunks: [
        chunk({ id: "189-c03", page: 189, pno: 269, kind: "text", boxes: [[220, 1205, 1272, 1251]],
          snippet: "Dictionary keys must be immutable objects such as strings, numbers or tuples. Keys cannot be modified after they are set; remove the entry and add a new key instead.",
          highlight: "Keys cannot be modified after they are set",
          scores: { visualRank: 2, visualScore: 18.9, textRank: 1, textScore: 0.77, fusedRank: 1, rerank: 0.94 } }),
      ],
    },
  },
  {
    match: /escape|newline|\\n|\\t|\btab\b/i,
    text: "The newline escape \\n ends the current line and moves the cursor to the start of the next one [1]. The tab escape \\t inserts a horizontal tab character, which is used to indent paragraphs or line up text across lines [1].",
    meta: {
      confidence: "high",
      retrievalMs: 231,
      chunks: [
        chunk({ id: "397-c02", page: 397, pno: 83, kind: "table", boxes: [[183, 729, 1247, 785], [180, 794, 1498, 1241]],
          snippet: "Special characters. An escape sequence uses a backslash (\\) to represent a special character within a string. \\n newline: moves to the beginning of the next line. \\t tab: a horizontal tab, useful for indenting or aligning text.",
          highlight: "\\n newline: moves to the beginning of the next line. \\t tab: a horizontal tab",
          scores: { visualRank: 1, visualScore: 21.6, textRank: 3, textScore: 0.61, fusedRank: 1, rerank: 0.89 } }),
      ],
    },
  },
  {
    match: /pandas|tail\s*\(|head\s*\(|last n rows|first n rows/i,
    text: "Yes. tail(n) returns the last n rows of a DataFrame, and when no value is passed it returns the last 5 rows, mirroring head(n) which returns the first 5 by default [1].",
    meta: {
      confidence: "high",
      retrievalMs: 187,
      chunks: [
        chunk({ id: "300-c01", page: 300, pno: 369, kind: "table", boxes: [[191, 707, 1587, 1113]],
          snippet: "head(n): returns the first n rows. If a value is not passed, the first 5 rows will be shown. tail(n): returns the last n rows. If a value is not passed, the last 5 rows will be shown. Example: df.tail(3)",
          highlight: "tail(n): returns the last n rows. If a value is not passed, the last 5 rows will be shown",
          scores: { visualRank: 1, visualScore: 22.1, textRank: 1, textScore: 0.81, fusedRank: 1, rerank: 0.96 } }),
      ],
    },
  },
  {
    match: /what (does|do) (this|the) (document|book|file|workspace)|cover|about|summar|contain/i,
    text: "Introduction to Python Programming is a 415-page open textbook. It covers variables and expressions, control flow, functions, strings and objects, lists, tuples and dictionaries, modules and the Python Package Index, file handling, classes, exceptions, and a closing chapter on data science with pandas. Each chapter ends with exercises, and an answer key sits at the back.",
    meta: { confidence: "high", retrievalMs: 0, chunks: [] },
  },
];

export const FALLBACK: { text: string; meta: Omit<AssistantMeta, "ttrMs"> } = {
  text: "I couldn't find a passage in the indexed document that answers this with enough confidence. The closest pages are about modules and file handling, but neither states it. Do you mean one of these?",
  meta: { confidence: "low", retrievalMs: 203, chunks: [], clarify: ["Which package handles image manipulation?", "Can dictionary keys be modified after they are set?"] },
};

export function pickAnswer(question: string) {
  return ANSWERS.find((a) => a.match.test(question)) ?? FALLBACK;
}
