import { useEffect, useMemo, useState } from "react";
import { useBrowserFolders } from "@/api/hooks/useGallery";
import { useGalleryStore } from "@/stores/galleryStore";
import type { BrowserFolder } from "@/api/types/gallery";
import { FolderCard } from "./FolderCard";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2, FolderOpen } from "lucide-react";

interface FolderNode {
  folder: BrowserFolder;
  children: FolderNode[];
}

/** Build a tree from the flat folder list by detecting parent-child path relationships. */
function buildFolderTree(folders: BrowserFolder[]): FolderNode[] {
  // Sort by path length so parents come before children
  const sorted = [...folders].sort((a, b) => a.path.length - b.path.length);
  const nodes: FolderNode[] = [];
  const nodeMap = new Map<string, FolderNode>();

  for (const f of sorted) {
    const node: FolderNode = { folder: f, children: [] };
    const normPath = f.path.replace(/\/+$/, "");

    // Find a parent: the longest path that is a prefix of this one
    let parent: FolderNode | null = null;
    for (const [candidatePath, candidateNode] of nodeMap) {
      if (normPath.startsWith(candidatePath + "/") && normPath !== candidatePath) {
        if (!parent || candidatePath.length > parent.folder.path.replace(/\/+$/, "").length) {
          parent = candidateNode;
        }
      }
    }

    if (parent) {
      parent.children.push(node);
    } else {
      nodes.push(node);
    }
    nodeMap.set(normPath, node);
  }

  return nodes;
}

export function GalleryPanel() {
  const { data: folders, isLoading } = useBrowserFolders();
  const activeFolder = useGalleryStore((s) => s.activeFolder);
  const setActiveFolder = useGalleryStore((s) => s.setActiveFolder);
  const setFolders = useGalleryStore((s) => s.setFolders);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (folders) setFolders(folders);
  }, [folders, setFolders]);

  // Auto-expand nodes that have children on first load
  useEffect(() => {
    if (folders && expanded.size === 0) {
      const tree = buildFolderTree(folders);
      const toExpand = new Set<string>();
      for (const node of tree) {
        if (node.children.length > 0) toExpand.add(node.folder.path);
      }
      if (toExpand.size > 0) setExpanded(toExpand);
    }
  }, [folders, expanded.size]);

  const tree = useMemo(() => {
    if (!folders) return [];
    return buildFolderTree(folders);
  }, [folders]);

  const toggleExpand = (path: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  const handleSelect = (path: string) => {
    setActiveFolder(activeFolder === path ? null : path);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="px-3 py-2 border-b border-border flex items-center gap-2">
        <FolderOpen size={14} className="text-muted-foreground" />
        <span className="text-xs font-medium">Gallery</span>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-2 space-y-0.5">
          {isLoading && (
            <div className="flex items-center justify-center py-8 text-muted-foreground gap-2">
              <Loader2 size={14} className="animate-spin" />
              <span className="text-xs">Loading folders...</span>
            </div>
          )}
          {tree.map((node) => (
            <FolderTreeNode
              key={node.folder.path}
              node={node}
              indent={0}
              activeFolder={activeFolder}
              expanded={expanded}
              onSelect={handleSelect}
              onToggle={toggleExpand}
            />
          ))}
          {folders && folders.length === 0 && (
            <div className="text-xs text-muted-foreground text-center py-8">No output folders found</div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

function FolderTreeNode({ node, indent, activeFolder, expanded, onSelect, onToggle }: {
  node: FolderNode;
  indent: number;
  activeFolder: string | null;
  expanded: Set<string>;
  onSelect: (path: string) => void;
  onToggle: (path: string) => void;
}) {
  const isExpanded = expanded.has(node.folder.path);
  const hasChildren = node.children.length > 0;

  return (
    <>
      <FolderCard
        label={node.folder.label}
        path={node.folder.path}
        active={activeFolder === node.folder.path}
        indent={indent}
        expanded={isExpanded}
        hasChildren={hasChildren}
        onSelect={() => onSelect(node.folder.path)}
        onToggle={() => onToggle(node.folder.path)}
      />
      {hasChildren && isExpanded && node.children.map((child) => (
        <FolderTreeNode
          key={child.folder.path}
          node={child}
          indent={indent + 1}
          activeFolder={activeFolder}
          expanded={expanded}
          onSelect={onSelect}
          onToggle={onToggle}
        />
      ))}
    </>
  );
}
