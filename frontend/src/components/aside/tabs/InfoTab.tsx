import { useServerInfo, useVersion } from "@/api/hooks/useServer";
import { BookOpen, Github, MessageCircle } from "lucide-react";

export function InfoTab() {
  const { data: serverInfo } = useServerInfo();
  const { data: version } = useVersion();

  return (
    <div className="p-3 space-y-4">
      <div className="space-y-2 text-xs">
        <Row label="Version" value={version?.app ?? serverInfo?.version?.app} />
        <Row label="Updated" value={version?.updated} />
        <Row label="Backend" value={serverInfo?.backend} />
        <Row label="Platform" value={serverInfo?.platform} />
        <Row label="GPU" value={serverInfo?.gpu} />
      </div>

      <div className="space-y-1.5 pt-2 border-t border-border">
        <ExternalLink href="https://vladmandic.github.io/sdnext-docs/" icon={BookOpen} label="Documentation" />
        <ExternalLink href="https://github.com/vladmandic/sdnext" icon={Github} label="GitHub" />
        <ExternalLink href="https://discord.gg/VjvR2tabEX" icon={MessageCircle} label="Discord" />
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value?: string | null }) {
  if (!value) return null;
  return (
    <div className="flex justify-between gap-2">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right truncate">{value}</span>
    </div>
  );
}

function ExternalLink({ href, icon: Icon, label }: { href: string; icon: React.ComponentType<{ className?: string }>; label: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground py-1 px-2 rounded hover:bg-muted/50 transition-colors"
    >
      <Icon className="h-3.5 w-3.5" />
      {label}
    </a>
  );
}
