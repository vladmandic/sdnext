import { authFetch } from './authWrap';
import { log, debug, error } from './logger';
import { timer } from './timers';

let gradioObserver: MutationObserver | null = null;

export async function sleep(ms) {
  return new Promise((resolve) => { setTimeout(resolve, ms); });
}

export function gradioApp(): any {
  const elems = document.getElementsByTagName('gradio-app');
  const elem: any = elems.length === 0 ? document : elems[0];
  if (elem !== document) elem.getElementById = (id) => document.getElementById(id);
  return elem.shadowRoot ? elem.shadowRoot : elem;
}
window.gradioApp = gradioApp;

function logFn(func) { // not recommended: use log, debug or error explicitly
  return async function loggedFunction() {
    const t0 = performance.now();
    const returnValue = func(...arguments);
    const t1 = performance.now();
    log(func.name, `time=${Math.round(t1 - t0)}`);
    timer(func.name, t1 - t0);
    return returnValue;
  };
}

export function getUICurrentTab() {
  return gradioApp().querySelector('#tabs button.selected');
}

export function getUICurrentTabContent() {
  return gradioApp().querySelector('.tabitem[id^=tab_]:not([style*="display: none"])');
}

export const get_uiCurrentTabContent = getUICurrentTabContent;
export const get_uiCurrentTab = getUICurrentTab;
export const uiAfterUpdateCallbacks = [];
export const uiUpdateCallbacks = [];
export const uiLoadedCallbacks = [];
export const uiReadyCallbacks = [];
export const uiTabChangeCallbacks = [];
export const optionsChangedCallbacks = [];

let uiCurrentTab = null;
let uiAfterUpdateTimeout = null;

function registerCallback(queue, callback) {
  if (queue.includes(callback)) return;
  queue.push(callback);
}

export function onAfterUiUpdate(callback) {
  if (typeof callback !== 'function') {
    error(`onAfterUiUpdate was called without a valid value. Expected a function but got: ${callback}`);
    return;
  }
  registerCallback(uiAfterUpdateCallbacks, callback);
}
window.onAfterUiUpdate = onAfterUiUpdate;

export function onUiUpdate(callback) {
  if (typeof callback !== 'function') {
    error(`onUiUpdate was called without a valid value. Expected a function but got: ${callback}`);
    return;
  }
  registerCallback(uiUpdateCallbacks, callback);
}
window.onUiUpdate = onUiUpdate;

export function onUiLoaded(callback) {
  if (typeof callback !== 'function') {
    error(`onUiLoaded was called without a valid value. Expected a function but got: ${callback}`);
    return;
  }
  registerCallback(uiLoadedCallbacks, callback);
}
window.onUiLoaded = onUiLoaded;

export function onUiReady(callback) {
  if (typeof callback !== 'function') {
    error(`onUiReady was called without a valid value. Expected a function but got: ${callback}`);
    return;
  }
  registerCallback(uiReadyCallbacks, callback);
}
window.onUiReady = onUiReady;

export function onUiTabChange(callback) {
  if (typeof callback !== 'function') {
    error(`onUiTabChange was called without a valid value. Expected a function but got: ${callback}`);
    return;
  }
  registerCallback(uiTabChangeCallbacks, callback);
}
window.onUiTabChange = onUiTabChange;

export function onOptionsChanged(callback) {
  if (typeof callback !== 'function') {
    error(`onOptionsChanged was called without a valid value. Expected a function but got: ${callback}`);
    return;
  }
  registerCallback(optionsChangedCallbacks, callback);
}
window.onOptionsChanged = onOptionsChanged;

export function executeCallbacks(queue: any[], arg?: any) {
  // if (!uiLoaded) return
  for (const callback of queue) {
    if (!callback) continue;
    try {
      const t0 = performance.now();
      callback(arg);
      const t1 = performance.now();
      if (t1 - t0 > 250) log('callbackSlow', callback.name || callback, `time=${Math.round(t1 - t0)}`);
      timer(callback.name || 'anonymousCallback', t1 - t0);
    } catch (e) {
      error(`executeCallbacks: ${callback} ${e}`);
    }
  }
}

const anyPromptExists = () => gradioApp().querySelectorAll('.main-prompts').length > 0;

function scheduleAfterUiUpdateCallbacks() {
  clearTimeout(uiAfterUpdateTimeout);
  uiAfterUpdateTimeout = setTimeout(() => executeCallbacks(uiAfterUpdateCallbacks), 250);
}

let executedOnLoaded = false;
const ignoreElements = ['logMonitorData', 'logWarnings', 'logErrors', 'tooltip-container', 'logger'];
const ignoreElementsSet = new Set(ignoreElements);
const ignoreClasses = ['wrap'];

let mutationTimer = null;
let validMutations = [];

async function mutationCallback(mutations) {
  if (mutations.length <= 0) return;
  for (const mutation of mutations) {
    const { target } = mutation;
    if (target.nodeName === 'LABEL') continue;
    if (ignoreElementsSet.has(target.id)) continue;
    if (target.classList?.contains(ignoreClasses[0])) continue;
    validMutations.push(mutation);
  }
  if (validMutations.length < 1) return;

  if (mutationTimer) clearTimeout(mutationTimer);
  mutationTimer = setTimeout(async () => {
    if (!executedOnLoaded && anyPromptExists()) { // execute once
      executedOnLoaded = true;
      executeCallbacks(uiLoadedCallbacks);
    }
    if (executedOnLoaded) { // execute on each mutation
      executeCallbacks(uiUpdateCallbacks, mutations);
      scheduleAfterUiUpdateCallbacks();
    }
    const newTab = getUICurrentTab();
    if (newTab && (newTab !== uiCurrentTab)) {
      uiCurrentTab = newTab;
      executeCallbacks(uiTabChangeCallbacks);
    }
    validMutations = [];
    mutationTimer = null;
  }, 100);
}

document.addEventListener('DOMContentLoaded', () => {
  log('DOMContentLoaded');
  gradioObserver = new MutationObserver(mutationCallback);
  gradioObserver.observe(gradioApp(), { childList: true, subtree: true, attributes: false });
});

/**
 * Add a listener to the document for keydown events
 */
document.addEventListener('keydown', (e) => {
  let elem;
  if (e.key === 'Escape') elem = getUICurrentTabContent().querySelector('button[id$=_interrupt]');
  if (e.key === 'Enter' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id$=_generate]');
  if (e.key === 'i' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id$=_reprocess]');
  if (e.key === ' ' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id$=_extra_networks_btn]');
  if (e.key === 'n' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id$=_extra_networks_btn]');
  if (e.key === 's' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id^=save_]');
  if (e.key === 'Insert' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id^=save_]');
  if (e.key === 'd' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id^=delete_]');
  // if (e.key === 'm' && e.ctrlKey) elem = gradioApp().getElementById('setting_sd_model_checkpoint');
  if (elem) {
    e.preventDefault();
    log('hotkey', { key: e.key, meta: e.metaKey, ctrl: e.ctrlKey, alt: e.altKey }, elem?.id, elem.nodeName);
    if (elem.nodeName === 'BUTTON') elem.click();
    else elem.focus();
  }
});

function getSortableCellValue(cell, sortType) {
  const rawValue = cell?.dataset?.sortValue ?? cell?.textContent?.trim() ?? '';
  if (sortType === 'number') {
    const numericValue = Number.parseFloat(rawValue);
    return Number.isNaN(numericValue) ? Number.NEGATIVE_INFINITY : numericValue;
  }
  return rawValue.toLowerCase();
}

function sortTable(table, columnIndex, sortType, sortOrder) {
  const tbody = table.querySelector('tbody');
  if (!tbody) return;
  const rows = Array.from<any>(tbody.querySelectorAll('tr'));
  const direction = sortOrder === 'desc' ? -1 : 1;
  const sortedRows = rows
    .map((row, index) => ({ row, index }))
    .sort((a, b) => {
      const aCell = a.row.children[columnIndex];
      const bCell = b.row.children[columnIndex];
      const aValue = getSortableCellValue(aCell, sortType);
      const bValue = getSortableCellValue(bCell, sortType);
      if (aValue < bValue) return -1 * direction;
      if (aValue > bValue) return 1 * direction;
      return a.index - b.index;
    });
  tbody.replaceChildren(...sortedRows.map((item) => item.row));
}

function applySortIndicators(table, activeHeader, sortOrder) {
  const headers = table.querySelectorAll('th.sortable');
  for (const header of headers) {
    header.classList.remove('sorted-asc', 'sorted-desc');
    header.removeAttribute('aria-sort');
  }
  activeHeader.classList.add(sortOrder === 'desc' ? 'sorted-desc' : 'sorted-asc');
  activeHeader.setAttribute('aria-sort', sortOrder === 'desc' ? 'descending' : 'ascending');
}

function handleSortableTableClick(event) {
  const header = event.target.closest('th.sortable');
  if (!header) return;
  const table = header.closest('table[data-sortable="true"]');
  if (!table) return;
  const headers = Array.from<any>(table.querySelectorAll('th.sortable'));
  const columnIndex = headers.indexOf(header);
  if (columnIndex < 0) return;

  const currentSortKey = table.dataset.sortKey || table.dataset.defaultSortKey;
  const currentSortOrder = table.dataset.sortOrder || table.dataset.defaultSortOrder || 'asc';
  const isCurrentHeader = currentSortKey === header.dataset.sortKey;
  const nextOrder = isCurrentHeader && currentSortOrder === 'asc' ? 'desc' : 'asc';

  table.dataset.sortKey = header.dataset.sortKey;
  table.dataset.sortOrder = nextOrder;
  sortTable(table, columnIndex, header.dataset.sortType || 'text', nextOrder);
  applySortIndicators(table, header, nextOrder);
}

export async function initTableSorter() {
  const t0 = performance.now();
  const root = gradioApp();
  if (!root.dataset.tableSorterBound) {
    root.addEventListener('click', handleSortableTableClick);
    root.dataset.tableSorterBound = 'true';
  }
  const t1 = performance.now();
  log('initTableSorter', Math.round(t1 - t0));
  timer('initTableSorter', t1 - t0);
}

export async function deleteFile(filename) {
  if (!filename) return;
  // eslint-disable-next-line no-alert
  if (!confirm(`Are you sure you want to delete the object - This action cannot be undone? Object: ${filename}`)) return;
  const res = await authFetch(`${window.api}/delete-file?file=${encodeURIComponent(filename)}`);
  if (!res || res.status !== 200) {
    error('FileDelete', { file: filename, status: res?.status, statusText: res?.statusText });
    return;
  }
  const data = await res.json();
  log('FileDelete', data);
}
window.deleteFile = deleteFile;

/**
 * checks that a UI element is not in another hidden element or tab content
 */
export function uiElementIsVisible(el) {
  if (el === document) return true;
  const computedStyle = getComputedStyle(el);
  const isVisible = computedStyle.display !== 'none';
  if (!isVisible) return false;
  return uiElementIsVisible(el.parentNode);
}

export function uiElementInSight(el) {
  const clRect = el.getBoundingClientRect();
  const windowHeight = window.innerHeight;
  const isOnScreen = clRect.bottom > 0 && clRect.top < windowHeight;
  return isOnScreen;
}
