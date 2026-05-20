// attaches listeners to the txt2img and img2img galleries to update displayed generation param text when the image changes

import { gradioApp } from './script';
import { getENActiveTab } from './extraNetworks';
import { log } from './logger';
import { timer } from './timers';

function attachGalleryListeners(tabName: string): Element | null {
  const gallery = gradioApp().querySelector(`#${tabName}_gallery`);
  if (!gallery) return null;
  gallery.addEventListener('click', () => {
    // log('galleryItemSelected:', tabName);
    const btn = gradioApp().getElementById(`${tabName}_generation_info_button`);
    if (btn) btn.click();
  });
  gallery.addEventListener('keydown', (e: KeyboardEvent) => {
    if (e.keyCode === 37 || e.keyCode === 39) {
      const btn = gradioApp().getElementById(`${tabName}_generation_info_button`);
      if (btn) btn.click(); // left or right arrow
    }
  });
  return gallery;
}

let txt2imgGallery: Element | null;
let img2imgGallery: Element | null;
let controlGallery: Element | null;
let modal: HTMLElement | null;

export async function initiGenerationParams() {
  const t0 = performance.now();
  if (!modal) modal = gradioApp().getElementById('lightboxModal');
  if (!modal) return;

  const modalObserver = new MutationObserver((mutations: MutationRecord[]) => {
    mutations.forEach((mutationRecord) => {
      const tabName = getENActiveTab();
      const mutationTarget = mutationRecord.target;
      if (mutationTarget instanceof HTMLElement && mutationTarget.style.display === 'none') {
        const btn = gradioApp().getElementById(`${tabName}_generation_info_button`);
        if (btn) btn.click();
      }
    });
  });

  if (!txt2imgGallery) txt2imgGallery = attachGalleryListeners('txt2img');
  if (!img2imgGallery) img2imgGallery = attachGalleryListeners('img2img');
  if (!controlGallery) controlGallery = attachGalleryListeners('control');
  modalObserver.observe(modal, { attributes: true, attributeFilter: ['style'] });
  const t1 = performance.now();
  log('initGenerationParams', Math.round(t1 - t0));
  timer('initGenerationParams', t1 - t0);
}
