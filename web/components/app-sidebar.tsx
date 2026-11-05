"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { Activity, BarChart3, ClipboardCheck, Database, FileSearch, FlaskConical, GitCompare, Settings } from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar";

const NAV = [
  {
    label: "Work",
    items: [
      { href: "/playground", title: "Playground", icon: FlaskConical, badge: null as string | null },
      { href: "/dataset", title: "Dataset", icon: Database, badge: "415 pages" },
      { href: "/pipeline", title: "ViDoRe Pipeline", icon: GitCompare, badge: null as string | null },
    ],
  },
  {
    label: "Quality",
    items: [
      { href: "/evals", title: "Retrieval Evals", icon: BarChart3, badge: null as string | null },
      { href: "/answer-evals", title: "Answer Evals", icon: ClipboardCheck, badge: null as string | null },
      { href: "/evidence", title: "Evidence", icon: FileSearch, badge: null as string | null },
      { href: "/observability", title: "Observability", icon: Activity, badge: null as string | null },
    ],
  },
];

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";
const STATIC_RUNTIME: [string, string][] = [["Vector store", "qdrant"], ["Visual index", "colqwen2.5"], ["Traces", "langfuse"]];

/** The query API line reads the live container (Modal GPU name, or the local device) and falls back to "modal" while loading. */
function useRuntimeRows(): [string, string][] {
  const [api, setApi] = useState("modal");
  useEffect(() => {
    let on = true;
    fetch(`${BASE}/api/system`, { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).then((s: { container?: { id: string; gpu: string } } | null) => {
      if (on && s?.container) setApi(`${s.container.id === "local" ? "local" : "modal"} · ${s.container.gpu.replace(/^NVIDIA /, "").toLowerCase()}`);
    }).catch(() => {});
    return () => { on = false; };
  }, []);
  return [["Query API", api], ...STATIC_RUNTIME];
}

export function AppSidebar() {
  const pathname = usePathname();
  const RUNTIME = useRuntimeRows();
  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" tooltip="Matt" render={<Link href="/playground" />}>
              <span className="flex size-8 items-center justify-center rounded-md bg-primary text-primary-foreground font-semibold">M</span>
              <span className="grid flex-1 text-left leading-tight">
                <span className="truncate font-semibold">Matt</span>
                <span className="truncate text-xs text-muted-foreground">multimodal RAG pipeline</span>
              </span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        {NAV.map((group) => (
          <SidebarGroup key={group.label}>
            <SidebarGroupLabel>{group.label}</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {group.items.map((item) => {
                  const active = pathname.startsWith(item.href);
                  return (
                    <SidebarMenuItem key={item.href}>
                      <SidebarMenuButton isActive={active} tooltip={item.title} render={<Link href={item.href} />}>
                        <item.icon />
                        <span>{item.title}</span>
                      </SidebarMenuButton>
                      {item.badge ? <SidebarMenuBadge className="font-mono text-[10px]">{item.badge}</SidebarMenuBadge> : null}
                    </SidebarMenuItem>
                  );
                })}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
      </SidebarContent>

      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton isActive={pathname.startsWith("/settings")} tooltip="Settings" render={<Link href="/settings" />}>
              <Settings />
              <span>Settings</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
        <div className="rounded-lg border border-sidebar-border p-3 text-xs group-data-[collapsible=icon]:hidden">
          <div className="mb-2 flex items-center gap-2 font-medium">
            <span className="size-2 rounded-full bg-success" aria-hidden="true" />
            Runtime
          </div>
          <dl className="grid gap-1.5">
            {RUNTIME.map(([k, v]) => (
              <div key={k} className="flex items-center justify-between gap-3">
                <dt className="text-muted-foreground">{k}</dt>
                <dd className="font-mono text-[11px]">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
