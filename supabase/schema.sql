-- Matt: metadata tables. Run once in the Supabase SQL editor (or: uv run python scripts/init_supabase.py).
-- Vectors live in Qdrant; page images live in the Supabase Storage bucket "pages" (public read).

create table if not exists documents (
  id            text primary key,                 -- e.g. doc_7c21ab or python_textbook
  workspace     text not null,                    -- conversation / dataset the document belongs to
  name          text not null,
  source        text not null default 'upload',   -- upload | vidore_v3
  file_type     text,
  pages         int  not null default 0,
  chunks        int  not null default 0,
  status        text not null default 'pending',  -- pending | indexing | indexed | failed
  error         text,
  created_at    timestamptz not null default now(),
  indexed_at    timestamptz
);

create table if not exists pages (
  id            text primary key,                 -- {document_id}:{page_no}
  document_id   text not null references documents(id) on delete cascade,
  page_no       int  not null,
  width         int, height int,
  image_url     text,                             -- public URL in Storage
  text          text,                             -- full page text (OCR or shipped markdown)
  n_chunks      int  not null default 0,
  n_visual_vectors int not null default 0,
  parser        text
);
create index if not exists pages_doc on pages(document_id, page_no);

create table if not exists chunks (
  id            text primary key,                 -- {document_id}:{page_no}:c{idx}
  document_id   text not null references documents(id) on delete cascade,
  page_id       text not null references pages(id) on delete cascade,
  page_no       int  not null,
  kind          text not null,                    -- text | table | figure
  heading_path  jsonb not null default '[]',
  bbox          jsonb,                            -- [x1,y1,x2,y2] normalised 0..1
  text          text not null,
  n_tokens      int  not null default 0,
  vector_id     text not null                     -- point id in Qdrant matt_chunks
);
create index if not exists chunks_doc on chunks(document_id, page_no);
create index if not exists chunks_text on chunks using gin (to_tsvector('english', text));

create table if not exists jobs (
  id            text primary key,
  workspace     text not null,
  status        text not null default 'running',  -- running | done | failed
  files         jsonb not null default '[]',
  steps         jsonb not null default '[]',      -- [{name, detail, state, ms}]
  progress      real not null default 0,
  error         text,
  created_at    timestamptz not null default now(),
  finished_at   timestamptz
);

create table if not exists conversations (
  id            text primary key,
  source        text not null,                    -- playground | dataset | compare (ViDoRe Pipeline tab)
  workspace     text not null,
  mode          text not null default 'fused',
  answer_model  text,
  created_at    timestamptz not null default now()
);

create table if not exists turns (
  id            bigserial primary key,
  conversation_id text not null references conversations(id) on delete cascade,
  question      text not null,
  answer        text,
  confidence    text,
  evidence      jsonb,                            -- chunks + scores shown to the user
  timings_ms    jsonb,                            -- per stage
  ttr_ms        int,
  tokens_in     int, tokens_out int, cost_usd numeric(10,5),
  trace_id      text,                             -- Langfuse
  status        text not null default 'ok',
  created_at    timestamptz not null default now()
);
create index if not exists turns_conv on turns(conversation_id, created_at);

create table if not exists eval_runs (
  id            text primary key,
  workspace     text not null,
  dataset       text not null,
  config        text not null,
  metrics       jsonb not null,
  per_query     jsonb,
  hardware      text,
  created_at    timestamptz not null default now()
);

-- Answer evals: retrieval + Claude answer per benchmark query, scored by deterministic checks (and an optional judge).
create table if not exists answer_eval_runs (
  id            text primary key,
  workspace     text not null,
  dataset       text not null,
  config        text not null,
  answer_model  text,
  judge_model   text,
  n_pages       int,
  rubric        jsonb,
  summary       jsonb not null,
  per_query     jsonb,
  hardware      text,
  created_at    timestamptz not null default now()
);
