import { gradioApp } from './script';
import { updateInput } from './ui';

type AccordionElement = HTMLElement & {
  visibleCheckbox: HTMLInputElement;
  onVisibleCheckboxChange: () => void;
  onChecked: (checked: boolean) => void;
};

export function inputAccordionChecked(id: string, checked: boolean): void {
  const accordion = gradioApp().getElementById(id);
  if (!(accordion instanceof HTMLElement)) return;
  const acc = accordion as AccordionElement;
  acc.visibleCheckbox.checked = checked;
  acc.onVisibleCheckboxChange();
}

function setupAccordion(accordion: Element): void {
  if (!(accordion instanceof HTMLElement)) return;
  const acc = accordion as AccordionElement;
  const labelWrap = accordion.querySelector('.label-wrap');
  const gradioCheckbox = gradioApp().querySelector(`#${accordion.id}-checkbox input`);
  const extra = gradioApp().querySelector(`#${accordion.id}-extra`);
  if (!(labelWrap instanceof HTMLElement) || !(gradioCheckbox instanceof HTMLInputElement)) return;
  const span = labelWrap.querySelector('span');
  if (!(span instanceof HTMLElement)) return;
  let linked = true;
  const isOpen = () => labelWrap.classList.contains('open');
  const observerAccordionOpen = new MutationObserver((mutations) => {
    mutations.forEach((mutationRecord) => {
      accordion.classList.toggle('input-accordion-open', isOpen());
      if (linked) {
        acc.visibleCheckbox.checked = isOpen();
        acc.onVisibleCheckboxChange();
      }
    });
  });
  observerAccordionOpen.observe(labelWrap, { attributes: true, attributeFilter: ['class'] });
  if (extra instanceof Node) labelWrap.insertBefore(extra, labelWrap.lastElementChild);
  acc.onChecked = (checked: boolean) => {
    if (isOpen() !== checked) labelWrap.click();
  };

  const visibleCheckbox = document.createElement('INPUT');
  visibleCheckbox.type = 'checkbox';
  visibleCheckbox.checked = isOpen();
  visibleCheckbox.id = `${accordion.id}-visible-checkbox`;
  visibleCheckbox.className = `${gradioCheckbox.className} input-accordion-checkbox`;
  span.insertBefore(visibleCheckbox, span.firstChild);
  acc.visibleCheckbox = visibleCheckbox as any;
  acc.onVisibleCheckboxChange = () => {
    if (linked && isOpen() !== visibleCheckbox.checked) labelWrap.click();
    gradioCheckbox.checked = visibleCheckbox.checked;
    updateInput(gradioCheckbox);
  };

  visibleCheckbox.addEventListener('click', (event) => {
    linked = false;
    event.stopPropagation();
  });
  visibleCheckbox.addEventListener('input', acc.onVisibleCheckboxChange);
}

window.inputAccordionChecked = inputAccordionChecked;

// onUiLoaded(() => {
//  for (const accordion of gradioApp().querySelectorAll('.input-accordion')) setupAccordion(accordion);
// });

export function initAccordions() {
  for (const accordion of gradioApp().querySelectorAll('.input-accordion')) setupAccordion(accordion);
}
