import { log } from './logger';
import { getENActiveTab } from './extraNetworks';
import { gradioApp } from './script';

type PromptDropTarget = EventTarget & {
  placeholder?: string;
};

export async function initDragDrop() {
  log('initDragDrop');
  window.addEventListener('drop', (e: DragEvent) => {
    const target = e.composedPath()[0] as PromptDropTarget;
    if (!target.placeholder) return;
    if (target.placeholder.indexOf('Prompt') === -1) return;
    const tabName = getENActiveTab();
    const promptTarget = `${tabName}_prompt_image`;
    const imgParent = gradioApp().getElementById(promptTarget);
    log('dropEvent', target, promptTarget, imgParent);
    if (!imgParent) return;
    const fileInput = imgParent.querySelector('input[type="file"]');
    if (!imgParent || !fileInput) return;
    if ((e.dataTransfer?.files?.length || 0) > 0) {
      e.stopPropagation();
      e.preventDefault();
      if (fileInput instanceof HTMLInputElement) fileInput.files = e.dataTransfer.files;
      fileInput.dispatchEvent(new Event('change'));
      log('dropEvent files', fileInput.files);
    }
  });
}
