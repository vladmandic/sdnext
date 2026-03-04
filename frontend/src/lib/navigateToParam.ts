import { useUiStore } from "@/stores/uiStore";
import type { ImagesSubTab } from "@/stores/uiStore";
import type { AsideTab } from "@/lib/constants";

export interface NavigateTarget {
  view?: string;
  tab?: ImagesSubTab;
  aside?: AsideTab;
  section?: string;
  param?: string;
}

function waitForElement(selector: string, timeout = 500): Promise<Element | null> {
  const existing = document.querySelector(selector);
  if (existing) return Promise.resolve(existing);
  return new Promise((resolve) => {
    const timer = setTimeout(() => { observer.disconnect(); resolve(null); }, timeout);
    const observer = new MutationObserver(() => {
      const el = document.querySelector(selector);
      if (el) { clearTimeout(timer); observer.disconnect(); resolve(el); }
    });
    observer.observe(document.body, { childList: true, subtree: true });
  });
}

function highlight(el: Element) {
  el.removeAttribute("data-highlight");
  void (el as HTMLElement).offsetWidth;
  el.setAttribute("data-highlight", "");
  el.addEventListener("animationend", () => el.removeAttribute("data-highlight"), { once: true });
}

export async function navigateToParam(target: NavigateTarget) {
  const store = useUiStore.getState();

  // 1. Aside tab navigation (right panel)
  if (target.aside) {
    store.openAsideTab(target.aside);
    return;
  }

  // 2. Ensure left panel is open
  if (store.leftPanelCollapsed) store.toggleLeftPanel();

  // 3. Switch view/tab
  if (target.view) store.setSidebarView(target.view as Parameters<typeof store.setSidebarView>[0]);
  if (target.tab) {
    store.setSidebarView("images");
    store.setImagesSubTab(target.tab);
  }

  // 4. If no param specified (tab-only navigation), we're done
  if (!target.param) return;

  // 5. Expand section if specified
  if (target.section) {
    await waitForElement(`[data-section="${target.section}"]`, 300);
    document.dispatchEvent(new CustomEvent("param-section-expand", { detail: { section: target.section } }));
  }

  // 6. Wait for the param element to appear
  const el = await waitForElement(`[data-param="${target.param}"]`, 500);
  if (!el) return;

  // 7. Scroll into view
  el.scrollIntoView({ behavior: "smooth", block: "center" });

  // 8. Highlight
  requestAnimationFrame(() => highlight(el));
}
