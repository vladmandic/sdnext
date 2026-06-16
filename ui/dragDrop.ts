import { log } from './logger';
import { gradioApp, uiElementIsVisible, uiElementInSight } from './script';

type PromptDropTarget = EventTarget & {
  placeholder?: string;
  closest: (selector: string) => Element | null;
};

function isValidImageList(files: FileList | null | undefined): files is FileList {
  return files && files?.length === 1 && ['image/png', 'image/gif', 'image/jpeg'].includes(files[0].type);
}

function dropReplaceImage(imgWrap: Element, files: FileList): void {
  log('dropReplaceImage', imgWrap, files);
  if (!isValidImageList(files)) return;
  const tmpFile = files[0];
  imgWrap.querySelector('.modify-upload button + button, .touch-none + div button + button')?.click();
  const callback = () => {
    const fileInput = imgWrap.querySelector('input[type="file"]');
    if (fileInput instanceof HTMLInputElement) {
      if (files.length === 0) {
        const dt = new DataTransfer();
        dt.items.add(tmpFile);
        fileInput.files = dt.files;
      } else {
        fileInput.files = files;
      }
      fileInput.dispatchEvent(new Event('change'));
    }
  };

  if (imgWrap.closest('#pnginfo_image')) {
    const oldFetch = window.fetch;
    window.fetch = async (input: RequestInfo | URL, options?: RequestInit) => {
      const response = await oldFetch(input, options);
      if (input === 'api/predict/') {
        const content = await response.text();
        window.fetch = oldFetch;
        window.requestAnimationFrame(() => callback());
        return new Response(content, {
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
        });
      }
      return response;
    };
  } else {
    window.requestAnimationFrame(() => callback());
  }
}

window.document.addEventListener('dragover', (e) => {
  const target = e.composedPath()[0] as PromptDropTarget;
  const imgWrap = target.closest('[data-testid="image"]');
  if (!imgWrap && target.placeholder && target.placeholder.indexOf('Prompt') === -1) return;
  if ((e.dataTransfer?.files?.length || 0) > 0) {
    e.stopPropagation();
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  }
});

window.document.addEventListener('drop', (e) => {
  const target = e.composedPath()[0] as PromptDropTarget;
  log('dropEvent', e, target);
  if (!target.placeholder) return;
  if (target.placeholder.indexOf('Prompt') === -1) return;
  const imgWrap = target.closest('[data-testid="image"]');
  if (!imgWrap) return;
  if ((e.dataTransfer?.files?.length || 0) > 0) {
    e.stopPropagation();
    e.preventDefault();
    dropReplaceImage(imgWrap, e.dataTransfer.files);
  }
});

window.addEventListener('paste', (e) => {
  log('pasteEvent', e);
  const files = e.clipboardData?.files;
  if (!isValidImageList(files)) return;
  const visibleImageFields = [...gradioApp().querySelectorAll('[data-testid="image"]')]
    .filter((el) => uiElementIsVisible(el))
    .sort((a, b) => Number(uiElementInSight(b)) - Number(uiElementInSight(a)));
  if (!visibleImageFields.length) return;
  const firstFreeImageField = visibleImageFields.filter((el) => el.querySelector('input[type=file]'))?.[0];
  dropReplaceImage(firstFreeImageField || visibleImageFields[visibleImageFields.length - 1], files);
});
