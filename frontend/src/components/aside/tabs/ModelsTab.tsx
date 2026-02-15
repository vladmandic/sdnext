import { useState } from "react";
import { useLoadedModels } from "@/api/hooks/useServer";
import { cn } from "@/lib/utils";

import { CurrentSubTab } from "@/components/models/sub-tabs/CurrentSubTab";
import { ListSubTab } from "@/components/models/sub-tabs/ListSubTab";
import { MetadataSubTab } from "@/components/models/sub-tabs/MetadataSubTab";
import { LoaderSubTab } from "@/components/models/sub-tabs/LoaderSubTab";
import { MergeSubTab } from "@/components/models/sub-tabs/MergeSubTab";
import { ReplaceSubTab } from "@/components/models/sub-tabs/ReplaceSubTab";
import { CivitaiSubTab } from "@/components/models/sub-tabs/CivitaiSubTab";
import { HuggingfaceSubTab } from "@/components/models/sub-tabs/HuggingfaceSubTab";
import { ExtractLoraSubTab } from "@/components/models/sub-tabs/ExtractLoraSubTab";

const SUB_TABS = [
  "Current",
  "List",
  "Metadata",
  "Loader",
  "Merge",
  "Replace",
  "CivitAI",
  "Huggingface",
  "Extract LoRA",
] as const;

type SubTab = (typeof SUB_TABS)[number];

export function ModelsTab() {
  const [active, setActive] = useState<SubTab>("Current");
  const { data: loaded } = useLoadedModels();
  const loadedCount = loaded?.length ?? 0;

  return (
    <div>
      <div className="sticky top-0 z-10 bg-card p-2 space-y-2 border-b border-border">
        <p className="text-[11px] text-muted-foreground">
          {loadedCount} model{loadedCount !== 1 ? "s" : ""} loaded
        </p>
        <div className="flex items-center gap-1 flex-wrap">
          {SUB_TABS.map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => setActive(tab)}
              className={cn(
                "px-2 py-0.5 rounded-full text-[11px] font-medium transition-colors",
                active === tab
                  ? "bg-accent text-accent-foreground"
                  : "bg-muted text-muted-foreground hover:text-foreground",
              )}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>
      <div className="p-3">
        {active === "Current" && <CurrentSubTab />}
        {active === "List" && <ListSubTab />}
        {active === "Metadata" && <MetadataSubTab />}
        {active === "Loader" && <LoaderSubTab />}
        {active === "Merge" && <MergeSubTab />}
        {active === "Replace" && <ReplaceSubTab />}
        {active === "CivitAI" && <CivitaiSubTab />}
        {active === "Huggingface" && <HuggingfaceSubTab />}
        {active === "Extract LoRA" && <ExtractLoraSubTab />}
      </div>
    </div>
  );
}
