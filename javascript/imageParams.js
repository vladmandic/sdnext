async function initDragDrop() {
  log('initDragDrop');
  window.addEventListener('drop', (e) => {
    const target = e.composedPath()[0];
    if (!target.placeholder) return;
    if (target.placeholder.indexOf('Prompt') === -1) return;
    const tabName = getENActiveTab();
    const promptTarget = `${tabName}_prompt_image`;
    const imgParent = gradioApp().getElementById(promptTarget);
    log('dropEvent', target, promptTarget, imgParent);
    const fileInput = imgParent.querySelector('input[type="file"]');
    if (!imgParent || !fileInput) return;
    if ((e.dataTransfer?.files?.length || 0) > 0) {
      e.stopPropagation();
      e.preventDefault();
      fileInput.files = e.dataTransfer.files;
      fileInput.dispatchEvent(new Event('change'));
      log('dropEvent files', fileInput.files);
    }
  });
}
