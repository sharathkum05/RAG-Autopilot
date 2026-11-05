"use client";

import { useId } from "react";
import { Badge } from "@/components/ui/badge";
import { AppShell } from "@/components/app-shell";
import { ChatView } from "@/components/chat/chat-view";
import type { ChatMessage } from "@/lib/types";

/** Questions taken from the dataset's own query set (ViDoRe V3, Python textbook). */
const SUGGESTIONS = [
  "Which Python package handles image manipulation?",
  "Can dictionary keys be modified after they are set?",
  "What is the difference between the \\n and \\t escape sequences?",
  "Does pandas tail() return the last 5 rows by default?",
];

const WELCOME: ChatMessage = {
  id: "welcome",
  role: "assistant",
  text:
    "This workspace is preloaded with Introduction to Python Programming, a 415-page open textbook from the ViDoRe V3 computer-science set, already parsed, chunked and indexed as text chunks and page images. Ask anything from it; every answer cites the chunk and page it came from. Runs on baseline model.",
};

export default function DatasetPage() {
  // useId is identical on server and client, so the badge hydrates cleanly. The backend assigns the real id on the first turn.
  const conversationId = `conv_ds_${useId().replace(/[^a-z0-9]/gi, "").slice(-4).toLowerCase()}`;
  return (
    <AppShell
      title="Dataset"
      subtitle="ViDoRe V3 · Introduction to Python Programming · preloaded"
      actions={
        <Badge variant="outline" className="font-mono font-normal text-muted-foreground">
          conversation <span className="ml-1 text-foreground">{conversationId}</span>
        </Badge>
      }
    >
      <ChatView
        source="dataset"
        conversationId={conversationId}
        welcome={WELCOME}
        suggestions={SUGGESTIONS}
        placeholder="Ask about the textbook"
        mode="fused"
        model="claude-opus-5-5"
      />
    </AppShell>
  );
}
