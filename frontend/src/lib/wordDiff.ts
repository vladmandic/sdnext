export interface DiffSegment {
  type: "equal" | "added" | "removed";
  text: string;
}

/** Tokenize on whitespace boundaries, preserving whitespace as separate tokens */
function tokenize(str: string): string[] {
  return str.match(/\S+|\s+/g) ?? [];
}

/** LCS-based word diff between two strings */
export function wordDiff(original: string, enhanced: string): DiffSegment[] {
  const a = tokenize(original);
  const b = tokenize(enhanced);
  const m = a.length;
  const n = b.length;

  // Build LCS table
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array<number>(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }

  // Backtrack to produce raw diff ops
  const raw: DiffSegment[] = [];
  let i = m;
  let j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) {
      raw.push({ type: "equal", text: a[i - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      raw.push({ type: "added", text: b[j - 1] });
      j--;
    } else {
      raw.push({ type: "removed", text: a[i - 1] });
      i--;
    }
  }
  raw.reverse();

  // Merge consecutive segments of the same type
  const merged: DiffSegment[] = [];
  for (const seg of raw) {
    const last = merged[merged.length - 1];
    if (last && last.type === seg.type) {
      last.text += seg.text;
    } else {
      merged.push({ ...seg });
    }
  }

  return merged;
}
