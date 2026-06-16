import { gradioApp } from './script';

export function uiOpenSubmenus() {
  const accordions = Array.from<HTMLElement>(gradioApp().querySelectorAll('.gradio-accordion'));
  const states: Record<string, boolean> = {};
  accordions.forEach((el) => {
    const labelEl = el.querySelector<HTMLSpanElement>('.label-wrap > span:not(.icon)');
    const name = labelEl instanceof HTMLElement ? labelEl.innerText.trim() : '';
    if (!name) return;
    const children = Array.from<ChildNode>(el.childNodes);
    const open = children.filter((c) => c instanceof HTMLElement && c.style.display === 'block');
    if (states[name] === undefined) states[name] = open.length > 0;
  });
  return states;
}

export async function getUIDefaults() {
  const btn = gradioApp().getElementById('ui_defaults_view');
  if (!btn) return;
  const intersectionObserver = new IntersectionObserver((entries) => {
    if (entries[0].intersectionRatio <= 0) { /* Pass */ }
    if (entries[0].intersectionRatio > 0) btn.click();
  });
  intersectionObserver.observe(btn); // monitor visibility of tab
}

window.uiOpenSubmenus = uiOpenSubmenus;
