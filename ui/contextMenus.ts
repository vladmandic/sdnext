import { gradioApp, getUICurrentTabContent } from './script';
import { log } from './logger';
import { authFetch } from './authWrap';
import { quickApplyStyle, quickSaveStyle } from './extraNetworks';

interface ContextMenuItem {
  id: string;
  name: string;
  func: () => void;
  primary: boolean;
}

const contextMenuInit = () => {
  let eventListenerApplied = false;
  const menuSpecs = new Map<string, ContextMenuItem[]>();

  const uid = () => Date.now().toString(36) + Math.random().toString(36).substring(2);

  function showContextMenu(event: MouseEvent, _element: Element, menuEntries: ContextMenuItem[]): void {
    const posx = event.clientX + document.body.scrollLeft + document.documentElement.scrollLeft;
    const posy = event.clientY + document.body.scrollTop + document.documentElement.scrollTop;
    const oldMenu = gradioApp().querySelector('#context-menu');
    if (oldMenu) oldMenu.remove();
    const contextMenu = document.createElement('nav');
    contextMenu.id = 'context-menu';
    contextMenu.style.top = `${posy}px`;
    contextMenu.style.left = `${posx}px`;
    const contextMenuList = document.createElement('ul');
    contextMenuList.className = 'context-menu-items';
    contextMenu.append(contextMenuList);
    menuEntries.forEach((entry) => {
      const contextMenuEntry = document.createElement('a');
      contextMenuEntry.innerHTML = entry.name;
      contextMenuEntry.addEventListener('click', () => entry.func());
      contextMenuList.append(contextMenuEntry);
    });
    gradioApp().appendChild(contextMenu);
    const menuWidth = contextMenu.offsetWidth + 4;
    const menuHeight = contextMenu.offsetHeight + 4;
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    if ((windowWidth - posx) < menuWidth) contextMenu.style.left = `${windowWidth - menuWidth}px`;
    if ((windowHeight - posy) < menuHeight) contextMenu.style.top = `${windowHeight - menuHeight}px`;
  }

  function appendContextMenuOption(targetElementSelector: string, entryName: string, entryFunction: () => void, primary = false): string {
    let currentItems = menuSpecs.get(targetElementSelector);
    if (!currentItems) {
      currentItems = [];
      menuSpecs.set(targetElementSelector, currentItems);
    }
    const newItem = {
      id: `${targetElementSelector}_${uid()}`,
      name: entryName,
      func: entryFunction,
      primary,
      // isNew: true,
    };
    currentItems.push(newItem);
    return newItem.id;
  }

  function removeContextMenuOption(id: string): void {
    menuSpecs.forEach((v, k) => {
      let index = -1;
      v.forEach((e, ei) => {
        if (e.id === id) { index = ei; }
      });
      if (index >= 0) v.splice(index, 1);
    });
  }

  window.appendContextMenuOption = appendContextMenuOption;
  window.removeContextMenuOption = removeContextMenuOption;

  async function addContextMenuEventListener(): Promise<void> {
    if (eventListenerApplied) return;
    log('initContextMenu');
    gradioApp().addEventListener('click', (e: Event) => {
      const mouseEvent = e as MouseEvent;
      if (!mouseEvent.isTrusted) return;
      const oldMenu = gradioApp().querySelector('#context-menu');
      if (oldMenu) oldMenu.remove();
      menuSpecs.forEach((v, k) => {
        const items = v.filter((item) => item.primary);
        const target = mouseEvent.target as Element | null;
        if (!target) return;
        const matched = target.closest(k);
        if (items.length > 0 && matched) {
          showContextMenu(mouseEvent, matched, items);
          mouseEvent.preventDefault();
        }
      });
    });
    gradioApp().addEventListener('contextmenu', (e: Event) => {
      const mouseEvent = e as MouseEvent;
      const oldMenu = gradioApp().querySelector('#context-menu');
      if (oldMenu) oldMenu.remove();
      menuSpecs.forEach((v, k) => {
        const items = v.filter((item) => !item.primary);
        const target = mouseEvent.target as Element | null;
        if (!target) return;
        const matched = target.closest(k);
        if (items.length > 0 && matched) {
          showContextMenu(mouseEvent, matched, items);
          mouseEvent.preventDefault();
        }
      });
    });
    eventListenerApplied = true;
  }
  return [appendContextMenuOption, removeContextMenuOption, addContextMenuEventListener];
};

const initContextResponse = contextMenuInit();
const appendContextMenuOption = initContextResponse[0];
const removeContextMenuOption = initContextResponse[1];
const addContextMenuEventListener = initContextResponse[2] as () => void;

let generateOnRepeatInterval: ReturnType<typeof setInterval> | null = null;

export const generateForever = (genbuttonid: string): void => {
  if (generateOnRepeatInterval) {
    log('generateForever: cancel');
    clearInterval(generateOnRepeatInterval);
    generateOnRepeatInterval = null;
  } else {
    const genbutton = gradioApp().querySelector(genbuttonid);
    if (!(genbutton instanceof HTMLElement)) return;
    const isBusy = () => {
      let busy = document.getElementById('progressbar')?.style.display === 'block';
      if (!busy) {
        // Also check in Modern UI
        const outerButton = genbutton.parentElement.closest('button');
        busy = outerButton?.classList.contains('generate') && outerButton?.classList.contains('active');
      }
      return busy;
    };
    log('generateForever: start');
    if (!isBusy()) genbutton.click();
    generateOnRepeatInterval = setInterval(() => {
      if (!isBusy()) genbutton.click();
    }, 500);
  }
};
window.generateForever = generateForever;

const reprocessClick = (tabId: string, state: string): void => {
  const btn = document.getElementById(`${tabId}_${state}`);
  window.submit_state = state;
  if (btn) btn.click();
};

const getStatus = async () => {
  const headers = new Headers();
  const body = JSON.stringify({ id_task: -1, id_live_preview: false });
  headers.set('Content-Type', 'application/json');
  const tab = getUICurrentTabContent()?.id.replace('tab_', '') || '';
  const el = gradioApp().querySelector(`#html_log_${tab} .performance p`);

  let res;
  let data;
  res = await fetch('./internal/progress', { method: 'POST', headers, body });
  if (res?.ok) {
    data = await res.json();
    log('progressInternal:', data);
    if (el) el.innerText += `\nProgress internal:\n${JSON.stringify(data, null, 2)}`;
  }
  res = await authFetch('./sdapi/v1/progress?skip_current_image=true', { method: 'GET', headers });
  if (res?.ok) {
    data = await res.json();
    log('progressAPI:', data);
    if (el) el.innerText += `\nProgress API:\n${JSON.stringify(data, null, 2)}`;
  }
};

export async function initContextMenu() {
  for (const tab of ['txt2img', 'img2img', 'control', 'video']) {
    appendContextMenuOption(`#${tab}_generate`, 'Get server status', getStatus);
    appendContextMenuOption(`#${tab}_generate`, 'Copy prompt to clipboard', () => navigator.clipboard.writeText(document.querySelector(`#${tab}_prompt > label > textarea`).value));
    appendContextMenuOption(`#${tab}_generate`, 'Generate forever', () => generateForever(`#${tab}_generate`));
    appendContextMenuOption(`#${tab}_generate`, 'Apply selected style', quickApplyStyle);
    appendContextMenuOption(`#${tab}_generate`, 'Quick save style', quickSaveStyle);
    appendContextMenuOption(`#${tab}_reprocess`, 'Decode full quality', () => reprocessClick(tab, 'reprocess_decode'), true);
    appendContextMenuOption(`#${tab}_reprocess`, 'Refine & HiRes pass', () => reprocessClick(tab, 'reprocess_refine'), true);
    appendContextMenuOption(`#${tab}_reprocess`, 'Detailer pass', () => reprocessClick(tab, 'reprocess_detail'), true);
  }
  // Right-click send-to-control button for prompt/params-only transfer.
  for (const tab of ['gallery', 'txt2img', 'img2img', 'extras']) {
    appendContextMenuOption(`#${tab}_tabitem #control_tab`, 'Transfer only prompt to Images tab', () => {
      document.querySelector(`#image_buttons_${tab} #control_tab_prompt`)?.click();
      document.getElementById('control_nav')?.click();
    });
    appendContextMenuOption(`#${tab}_tabitem #control_tab`, 'Transfer all parameters to Images tab', () => {
      document.querySelector(`#image_buttons_${tab} #control_tab_params`)?.click();
      document.getElementById('control_nav')?.click();
    });
  }
  addContextMenuEventListener();
}
