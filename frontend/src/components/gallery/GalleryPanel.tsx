import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useBrowserFolders } from "@/api/hooks/useGallery";
import { api } from "@/api/client";
import { useGalleryStore } from "@/stores/galleryStore";
import type { BrowserFolder, BrowserSubdir } from "@/api/types/gallery";
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

/** Merge discovered subdirectories into a static folder tree (returns new array). */
function mergeDiscoveredSubdirs(
  tree: FolderNode[],
  discovered: Map<string, BrowserSubdir[]>,
): FolderNode[] {
  return tree.map((node) => {
    const normPath = node.folder.path.replace(/\/+$/, "");
    const subs = discovered.get(normPath);

    // Recurse into existing children first
    const mergedChildren = mergeDiscoveredSubdirs(node.children, discovered);

    if (!subs || subs.length === 0) {
      return { ...node, children: mergedChildren };
    }

    // Add discovered subdirs that aren't already represented as static children
    const existingPaths = new Set(mergedChildren.map((c) => c.folder.path.replace(/\/+$/, "")));
    const newChildren: FolderNode[] = [];
    for (const sub of subs) {
      const subNorm = sub.path.replace(/\/+$/, "");
      if (!existingPaths.has(subNorm)) {
        newChildren.push({
          folder: { path: sub.path, label: sub.label },
          children: [],
        });
      }
    }

    return { ...node, children: [...mergedChildren, ...newChildren] };
  });
}

export function GalleryPanel() {
  const { data: folders, isLoading } = useBrowserFolders();
  const activeFolder = useGalleryStore((s) => s.activeFolder);
  const setActiveFolder = useGalleryStore((s) => s.setActiveFolder);
  const setFolders = useGalleryStore((s) => s.setFolders);
  const [userExpanded, setUserExpanded] = useState<Set<string> | null>(null);

  // Dynamic subfolder state
  const [discoveredSubdirs, setDiscoveredSubdirs] = useState<Map<string, BrowserSubdir[]>>(new Map());
  const [loadingSubdirs, setLoadingSubdirs] = useState<Set<string>>(new Set());
  const [leafPaths, setLeafPaths] = useState<Set<string>>(new Set());

  // Ref to avoid stale closures in async callbacks
  const discoveredRef = useRef(discoveredSubdirs);
  discoveredRef.current = discoveredSubdirs;

  useEffect(() => {
    if (folders) setFolders(folders);
  }, [folders, setFolders]);

  // Build static tree then merge discovered subdirs
  const tree = useMemo(() => {
    if (!folders) return [];
    const staticTree = buildFolderTree(folders);
    if (discoveredSubdirs.size === 0) return staticTree;
    return mergeDiscoveredSubdirs(staticTree, discoveredSubdirs);
  }, [folders, discoveredSubdirs]);

  // Auto-expand parent nodes until user toggles; then respect user choice
  const expanded = useMemo(() => {
    if (userExpanded !== null) return userExpanded;
    if (!folders) return new Set<string>();
    const toExpand = new Set<string>();
    for (const node of tree) {
      if (node.children.length > 0) toExpand.add(node.folder.path);
    }
    return toExpand;
  }, [userExpanded, folders, tree]);

  const fetchSubdirs = useCallback(async (path: string) => {
    const normPath = path.replace(/\/+$/, "");

    // Already discovered or currently loading
    if (discoveredRef.current.has(normPath) || loadingSubdirs.has(normPath)) return;

    setLoadingSubdirs((prev) => new Set(prev).add(normPath));
    try {
      const subdirs = await api.get<BrowserSubdir[]>("/sdapi/v1/browser/subdirs", { folder: path });
      setDiscoveredSubdirs((prev) => {
        const next = new Map(prev);
        next.set(normPath, subdirs);
        return next;
      });
      if (subdirs.length === 0) {
        setLeafPaths((prev) => new Set(prev).add(normPath));
      }
    } catch {
      // On error, mark as leaf to suppress chevron
      setLeafPaths((prev) => new Set(prev).add(normPath));
    } finally {
      setLoadingSubdirs((prev) => {
        const next = new Set(prev);
        next.delete(normPath);
        return next;
      });
    }
  }, [loadingSubdirs]);

  const toggleExpand = useCallback((path: string) => {
    const normPath = path.replace(/\/+$/, "");
    const isCurrentlyExpanded = expanded.has(path);

    if (isCurrentlyExpanded) {
      // Collapse
      const next = new Set(expanded);
      next.delete(path);
      setUserExpanded(next);
    } else {
      // Expand — fetch subdirs if not yet discovered
      if (!discoveredRef.current.has(normPath)) {
        fetchSubdirs(path);
      }
      const next = new Set(expanded);
      next.add(path);
      setUserExpanded(next);
    }
  }, [expanded, fetchSubdirs]);

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
              loadingSubdirs={loadingSubdirs}
              leafPaths={leafPaths}
              discoveredSubdirs={discoveredSubdirs}
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

function FolderTreeNode({ node, indent, activeFolder, expanded, loadingSubdirs, leafPaths, discoveredSubdirs, onSelect, onToggle }: {
  node: FolderNode;
  indent: number;
  activeFolder: string | null;
  expanded: Set<string>;
  loadingSubdirs: Set<string>;
  leafPaths: Set<string>;
  discoveredSubdirs: Map<string, BrowserSubdir[]>;
  onSelect: (path: string) => void;
  onToggle: (path: string) => void;
}) {
  const normPath = node.folder.path.replace(/\/+$/, "");
  const isExpanded = expanded.has(node.folder.path);
  const isLoading = loadingSubdirs.has(normPath);
  const isLeaf = leafPaths.has(normPath);

  // A node has children if it has static children, discovered subdirs, or hasn't been explored yet
  const hasStaticChildren = node.children.length > 0;
  const hasChildren = hasStaticChildren || (!isLeaf && !discoveredSubdirs.has(normPath)) || (discoveredSubdirs.get(normPath)?.length ?? 0) > 0;

  return (
    <>
      <FolderCard
        label={node.folder.label}
        path={node.folder.path}
        active={activeFolder === node.folder.path}
        indent={indent}
        expanded={isExpanded}
        hasChildren={hasChildren}
        loading={isLoading}
        onSelect={() => onSelect(node.folder.path)}
        onToggle={() => onToggle(node.folder.path)}
      />
      {isExpanded && node.children.map((child) => (
        <FolderTreeNode
          key={child.folder.path}
          node={child}
          indent={indent + 1}
          activeFolder={activeFolder}
          expanded={expanded}
          loadingSubdirs={loadingSubdirs}
          leafPaths={leafPaths}
          discoveredSubdirs={discoveredSubdirs}
          onSelect={onSelect}
          onToggle={onToggle}
        />
      ))}
    </>
  );
}
