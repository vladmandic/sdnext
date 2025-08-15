// A full size 'lightbox' preview modal shown when left clicking on gallery previews
let previewDrag = false;
let modalPreviewZone;
let previewInstance;

function cycleImageFit() {
  const root = document.documentElement;
  const current = getComputedStyle(root).getPropertyValue('--sd-image-fit').trim();
  let next = 'contain';
  if (current === 'contain') next = 'cover';
  else if (current === 'cover') next = 'fill';
  else if (current === 'fill') next = 'scale-down';
  else if (current === 'scale-down') next = 'none';
  root.style.setProperty('--sd-image-fit', next);
  log('cycleImageFit', current, next);
}

function closeModal(evt, force = false) {
  if (force) gradioApp().getElementById('lightboxModal').style.display = 'none';
  if (previewDrag) return;
  if (evt?.button !== 0) return;
  gradioApp().getElementById('lightboxModal').style.display = 'none';
}

function modalImageSwitch(offset) {
  const galleryButtons = all_gallery_buttons();
  if (galleryButtons.length > 1) {
    const currentButton = selected_gallery_button();
    let result = -1;
    galleryButtons.forEach((v, i) => {
      if (v === currentButton) result = i;
    });
    const negmod = (n, m) => ((n % m) + m) % m;
    if (result !== -1) {
      const nextButton = galleryButtons[negmod((result + offset), galleryButtons.length)];
      nextButton.click();
      const modalImage = gradioApp().getElementById('modalImage');
      const modal = gradioApp().getElementById('lightboxModal');
      modalImage.src = nextButton.children[0].src;
      if (modalImage.style.display === 'none') modal.style.setProperty('background-image', `url(${modalImage.src})`);
    }
  }
}

function modalSaveImage(event) {
  if (gradioApp().getElementById('tab_txt2img').style.display !== 'none') gradioApp().getElementById('save_txt2img').click();
  else if (gradioApp().getElementById('tab_img2img').style.display !== 'none') gradioApp().getElementById('save_img2img').click();
  else if (gradioApp().getElementById('tab_process').style.display !== 'none') gradioApp().getElementById('save_extras').click();
}

function modalKeyHandler(event) {
  switch (event.key) {
    case 's':
      modalSaveImage();
      break;
    case 'ArrowLeft':
      modalImageSwitch(-1);
      break;
    case 'ArrowRight':
      modalImageSwitch(1);
      break;
    case 'Escape':
      closeModal(null, true);
      break;
  }
  event.stopPropagation();
}

async function getExif(el) {
  let exif = '';
  try {
    exif = await window.exifr.parse(el, { userComment: true });
  } catch (e) {
    log('getExif', el, e);
    return exif;
  }
  // let html = `<b>Image</b> <a href="${el.src}" target="_blank">${el.src}</a> <b>Size</b> ${el.naturalWidth}x${el.naturalHeight}<br>`;
  let html = '';
  let params;
  if (exif.parameters) {
    params = exif.parameters;
  } else if (exif.userComment) {
    params = Array.from(exif.userComment)
      .map((c) => String.fromCharCode(c))
      .filter((c) => c !== '\x00')
      .join('')
      .replace('UNICODE', '');
  } else {
    params = '';
  }
  if (params.length > 0) html += `<b>Prompt</b> ${params || ''}<br>`;
  html = html.replace('Negative prompt:', '<br><b>Negative</b>');
  html = html.replace('Steps:', '<br><b>Params</b> Steps:');
  html = html.replaceAll('\n', '<br>');
  html = html.replaceAll('<br><br>', '<br>');
  return html;
}
window.getExif = getExif;

async function displayExif(el) {
  const modalExif = gradioApp().getElementById('modalExif');
  const html = await getExif(el);
  modalExif.innerHTML = html;
}

function showModal(event) {
  const source = event.target || event.srcElement;
  const modalImage = gradioApp().getElementById('modalImage');
  const lb = gradioApp().getElementById('lightboxModal');
  lb.ownerSVGElement = modalImage;
  modalImage.onload = () => {
    previewInstance.moveTo(0, 0);
    modalPreviewZone.focus();
    if (opts.viewer_show_metadata) displayExif(modalImage);
  };
  modalImage.src = source.src;
  if (modalImage.style.display === 'none') lb.style.setProperty('background-image', `url(${source.src})`);
  lb.style.display = 'flex';
  lb.onkeydown = modalKeyHandler;
  event.stopPropagation();
}

function modalDownloadImage() {
  const link = document.createElement('a');
  link.style.display = 'none';
  link.href = gradioApp().getElementById('modalImage').src;
  link.download = 'image';
  document.body.appendChild(link);
  link.click();
  setTimeout(() => {
    URL.revokeObjectURL(link.href);
    link.parentNode.removeChild(link);
  }, 0);
}

function modalZoomSet(modalImage, enable) {
  localStorage.setItem('modalZoom', enable ? 'yes' : 'no');
  if (modalImage) modalImage.classList.toggle('modalImageFullscreen', !!enable);
}

function setupImageForLightbox(image) {
  if (image.dataset.modded) return;
  image.dataset.modded = 'true';
  image.style.cursor = 'pointer';
  image.style.userSelect = 'none';
}

function modalZoomToggle(event) {
  const modalImage = gradioApp().getElementById('modalImage');
  modalZoomSet(modalImage, !modalImage.classList.contains('modalImageFullscreen'));
  event.stopPropagation();
}

function modalTileToggle(event) {
  const modalImage = gradioApp().getElementById('modalImage');
  const modal = gradioApp().getElementById('lightboxModal');
  const isTiling = modalImage.style.display === 'none';
  if (isTiling) {
    modalImage.style.display = 'block';
    modal.style.setProperty('background-image', 'none');
  } else {
    modalImage.style.display = 'none';
    modal.style.setProperty('background-image', `url(${modalImage.src})`);
  }
  event.stopPropagation();
}

function modalResetInstance(event) {
  const modalImage = document.getElementById('modalImage');
  previewInstance.dispose();
  previewInstance = panzoom(modalImage, { zoomSpeed: 0.05, minZoom: 0.1, maxZoom: 5.0, filterKey: (/* e, dx, dy, dz */) => true });
}

function modalToggleParams(event) {
  const modalExif = gradioApp().getElementById('modalExif');
  if (modalExif.style.display === 'none' || modalExif.style.display === '') {
    modalExif.style.display = 'block';
  } else {
    modalExif.style.display = 'none';
  }
  event.stopPropagation();
}

function galleryClickEventHandler(event) {
  if (event.button !== 0) return;
  if (event.target.nodeName === 'IMG' && !event.target.parentNode.classList.contains('thumbnail-item')) {
    const initialZoom = (localStorage.getItem('modalZoom') || true) === 'yes';
    modalZoomSet(gradioApp().getElementById('modalImage'), initialZoom);
    event.preventDefault();
    showModal(event);
  }
}

async function bindImageViewer() {
  // Each tab has its own gradio-gallery
  const galleryPreviews = gradioApp().querySelectorAll('.gradio-gallery > div.preview');
  for (const galleryPreview of galleryPreviews) {
    if (!galleryPreview.hasAttribute('data-listener')) galleryPreview.addEventListener('click', galleryClickEventHandler, true);
    galleryPreview.setAttribute('data-listener', true);
    galleryPreview.querySelectorAll('img').forEach(setupImageForLightbox);
  }
}

async function initImageViewer() {
  // main elements
  const modal = document.createElement('div');
  modal.id = 'lightboxModal';

  modalPreviewZone = document.createElement('div');
  modalPreviewZone.className = 'lightboxModalPreviewZone';

  const modalImage = document.createElement('img');
  modalImage.id = 'modalImage';
  modalPreviewZone.appendChild(modalImage);
  previewInstance = panzoom(modalImage, { zoomSpeed: 0.05, minZoom: 0.1, maxZoom: 5.0, filterKey: (/* e, dx, dy, dz */) => true });

  // toolbar
  const modalZoom = document.createElement('span');
  modalZoom.id = 'modal_zoom';
  modalZoom.className = 'cursor';
  modalZoom.innerHTML = '\uf531';
  modalZoom.title = 'Toggle zoomed view';
  modalZoom.addEventListener('click', modalZoomToggle, true);

  const modalReset = document.createElement('span');
  modalReset.id = 'modal_reset';
  modalReset.className = 'cursor';
  modalReset.innerHTML = '\uf532';
  modalReset.title = 'Reset zoomed view';
  modalReset.addEventListener('click', modalResetInstance, true);

  const modalTile = document.createElement('span');
  modalTile.id = 'modal_tile';
  modalTile.className = 'cursor';
  modalTile.innerHTML = '\udb81\udd70';
  modalTile.title = 'Preview tiling';
  modalTile.addEventListener('click', modalTileToggle, true);

  const modalSave = document.createElement('span');
  modalSave.id = 'modal_save';
  modalSave.className = 'cursor';
  modalSave.innerHTML = '\udb80\udd93';
  modalSave.title = 'Save Image';
  modalSave.addEventListener('click', modalSaveImage, true);

  const modalDownload = document.createElement('span');
  modalDownload.id = 'modal_download';
  modalDownload.className = 'cursor';
  modalDownload.innerHTML = '\udb85\udc62';
  modalDownload.title = 'Download Image';
  modalDownload.addEventListener('click', modalDownloadImage, true);

  const modalClose = document.createElement('span');
  modalClose.id = 'modal_close';
  modalClose.className = 'cursor';
  modalClose.innerHTML = '\udb80\udd57';
  modalClose.title = 'Close';
  modalClose.addEventListener('click', (evt) => closeModal(evt, true), true);

  const modalToggleParamsBtn = document.createElement('span');
  modalToggleParamsBtn.id = 'modal_toggle_params';
  modalToggleParamsBtn.className = 'cursor';
  modalToggleParamsBtn.innerHTML = '\uf05a';
  modalToggleParamsBtn.title = 'Toggle Parameters';
  modalToggleParamsBtn.addEventListener('click', modalToggleParams, true);

  // exif
  const modalExif = document.createElement('div');
  modalExif.id = 'modalExif';
  modalExif.style = 'position: absolute; bottom: 0px; width: 100%; background-color: rgba(0, 0, 0, 0.5); color: var(--neutral-300); padding: 1em; font-size: small; line-height: 1.2em; z-index: 1; display: none;';

  // handlers
  modalPreviewZone.addEventListener('mousedown', () => { previewDrag = false; });
  modalPreviewZone.addEventListener('touchstart', () => { previewDrag = false; }, { passive: true });
  modalPreviewZone.addEventListener('mousemove', () => { previewDrag = true; });
  modalPreviewZone.addEventListener('touchmove', () => { previewDrag = true; }, { passive: true });
  modalPreviewZone.addEventListener('scroll', () => { previewDrag = true; });
  modalPreviewZone.addEventListener('mouseup', (evt) => closeModal(evt));
  modalPreviewZone.addEventListener('touchend', (evt) => closeModal(evt));

  const modalPrev = document.createElement('a');
  modalPrev.className = 'modalPrev';
  modalPrev.innerHTML = '&#10094;';
  modalPrev.addEventListener('click', () => modalImageSwitch(-1), true);
  // modalPrev.addEventListener('keydown', modalKeyHandler, true);

  const modalNext = document.createElement('a');
  modalNext.className = 'modalNext';
  modalNext.innerHTML = '&#10095;';
  modalNext.addEventListener('click', () => modalImageSwitch(1), true);
  // modalNext.addEventListener('keydown', modalKeyHandler, true);

  const modalControls = document.createElement('div');
  modalControls.className = 'modalControls gradio-container';

  // build interface
  modal.appendChild(modalPrev);
  modal.appendChild(modalPreviewZone);
  modal.appendChild(modalNext);
  modal.append(modalControls);
  modalControls.appendChild(modalZoom);
  modalControls.appendChild(modalReset);
  modalControls.appendChild(modalTile);
  modalControls.appendChild(modalSave);
  modalControls.appendChild(modalDownload);
  modalControls.appendChild(modalToggleParamsBtn);
  modalControls.appendChild(modalClose);
  modal.append(modalExif);

  gradioApp().appendChild(modal);
  log('initImageViewer');
}

onAfterUiUpdate(bindImageViewer);
