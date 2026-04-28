// attaches listeners to the txt2img and img2img galleries to update displayed generation param text when the image changes

function attachGalleryListeners(tabName) {
  const gallery = gradioApp().querySelector(`#${tabName}_gallery`);
  if (!gallery) return null;
  gallery.addEventListener('click', () => {
    // log('galleryItemSelected:', tabName);
    const btn = gradioApp().getElementById(`${tabName}_generation_info_button`);
    if (btn) btn.click();
  });
  gallery?.addEventListener('keydown', (e) => {
    if (e.keyCode === 37 || e.keyCode === 39) gradioApp().getElementById(`${tabName}_generation_info_button`).click(); // left or right arrow
  });
  return gallery;
}

let txt2img_gallery;
let img2img_gallery;
let control_gallery;
let modal;

async function initiGenerationParams() {
  const t0 = performance.now();
  if (!modal) modal = gradioApp().getElementById('lightboxModal');
  if (!modal) return;

  const modalObserver = new MutationObserver((mutations) => {
    mutations.forEach((mutationRecord) => {
      const tabName = getENActiveTab();
      if (mutationRecord.target.style.display === 'none') {
        const btn = gradioApp().getElementById(`${tabName}_generation_info_button`);
        if (btn) btn.click();
      }
    });
  });

  if (!txt2img_gallery) txt2img_gallery = attachGalleryListeners('txt2img');
  if (!img2img_gallery) img2img_gallery = attachGalleryListeners('img2img');
  if (!control_gallery) control_gallery = attachGalleryListeners('control');
  modalObserver.observe(modal, { attributes: true, attributeFilter: ['style'] });
  const t1 = performance.now();
  log('initGenerationParams', Math.round(t1 - t0));
  timer('initGenerationParams', t1 - t0);
}
