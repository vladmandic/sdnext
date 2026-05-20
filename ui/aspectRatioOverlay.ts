import { gradioApp, onAfterUiUpdate } from './script';
import { get_tab_index } from './ui';

let currentWidth: number | null = null;
let currentHeight: number | null = null;
let arFrameTimeout: ReturnType<typeof setTimeout> | null = null;

function dimensionChange(e: Event, isWidth: boolean, isHeight: boolean): void {
  const { target } = e;
  if (!(target instanceof HTMLInputElement)) return;
  if (isWidth) currentWidth = Number(target.value);
  if (isHeight) currentHeight = Number(target.value);
  const tabImg2img = gradioApp().querySelector('#tab_img2img');
  if (!(tabImg2img instanceof HTMLElement)) return;
  const inImg2img = tabImg2img.style.display === 'block';
  if (!inImg2img) return;
  let targetElement: HTMLImageElement | null = null;
  const tabIndex = get_tab_index('mode_img2img');
  if (tabIndex === 0) targetElement = gradioApp().querySelector('#img2img_image div[data-testid=image] img'); // img2img
  else if (tabIndex === 1) targetElement = gradioApp().querySelector('#img2img_sketch div[data-testid=image] img'); // Sketch
  else if (tabIndex === 2) targetElement = gradioApp().querySelector('#img2maskimg div[data-testid=image] img'); // Inpaint
  else if (tabIndex === 3) targetElement = gradioApp().querySelector('#composite div[data-testid=image] img'); // Inpaint sketch

  if (targetElement && currentWidth && currentHeight) {
    let arPreviewRect = gradioApp().querySelector('#imageARPreview') as HTMLElement | null;
    if (!arPreviewRect) {
      arPreviewRect = document.createElement('div');
      arPreviewRect.id = 'imageARPreview';
      gradioApp().appendChild(arPreviewRect);
    }

    const viewportOffset = targetElement.getBoundingClientRect();
    const viewportscale = Math.min(targetElement.clientWidth / targetElement.naturalWidth, targetElement.clientHeight / targetElement.naturalHeight);
    const scaledx = targetElement.naturalWidth * viewportscale;
    const scaledy = targetElement.naturalHeight * viewportscale;
    const cleintRectTop = (viewportOffset.top + window.scrollY);
    const cleintRectLeft = (viewportOffset.left + window.scrollX);
    const cleintRectCentreY = cleintRectTop + (targetElement.clientHeight / 2);
    const cleintRectCentreX = cleintRectLeft + (targetElement.clientWidth / 2);
    const arscale = Math.min(scaledx / currentWidth, scaledy / currentHeight);
    const arscaledx = currentWidth * arscale;
    const arscaledy = currentHeight * arscale;
    const arRectTop = cleintRectCentreY - (arscaledy / 2);
    const arRectLeft = cleintRectCentreX - (arscaledx / 2);
    const arRectWidth = arscaledx;
    const arRectHeight = arscaledy;
    arPreviewRect.style.top = `${arRectTop}px`;
    arPreviewRect.style.left = `${arRectLeft}px`;
    arPreviewRect.style.width = `${arRectWidth}px`;
    arPreviewRect.style.height = `${arRectHeight}px`;

    if (arFrameTimeout) clearTimeout(arFrameTimeout);
    arFrameTimeout = setTimeout(() => { arPreviewRect.style.display = 'none'; }, 2000);
    arPreviewRect.style.display = 'block';
  }
}

export function aspectRatioCallback(): void {
  const arPreviewRect = gradioApp().querySelector('#imageARPreview');
  if (arPreviewRect instanceof HTMLElement) arPreviewRect.style.display = 'none';
  const tabImg2img = gradioApp().querySelector('#tab_img2img');
  if (tabImg2img instanceof HTMLElement) {
    const inImg2img = tabImg2img.style.display === 'block';
    if (inImg2img) {
      const inputs = gradioApp().querySelectorAll('input');
      inputs.forEach((e) => {
        if (!(e instanceof HTMLInputElement) || !(e.parentElement instanceof HTMLElement)) return;
        const isWidth = e.parentElement.id === 'img2img_width';
        const isHeight = e.parentElement.id === 'img2img_height';
        if ((isWidth || isHeight) && !e.classList.contains('scrollwatch')) {
          e.addEventListener('input', (evt) => { dimensionChange(evt, isWidth, isHeight); });
          e.classList.add('scrollwatch');
        }
        if (isWidth) currentWidth = Number(e.value);
        if (isHeight) currentHeight = Number(e.value);
      });
    }
  }
}

onAfterUiUpdate(aspectRatioCallback);
