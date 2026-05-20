import { randomId } from './progressBar';

import { log } from './logger';
import { gradioApp } from './script';
import { restartReload, updateInput } from './ui';

export function extensions_apply(_extensionsDisabledList: unknown, _extensionsUpdateList: unknown, disableAll: unknown): [string, string, unknown] {
  const disable = [];
  const update = [];
  gradioApp().querySelectorAll('#extensions input[type="checkbox"]').forEach((x) => {
    if (!(x instanceof HTMLInputElement)) return;
    if (x.name.startsWith('enable_') && !x.checked) disable.push(x.name.substring(7));
    if (x.name.startsWith('update_') && x.checked) update.push(x.name.substring(7));
  });
  restartReload();
  log('Extensions apply:', { disable, update });
  return [JSON.stringify(disable), JSON.stringify(update), disableAll];
}

export function extensions_check(_info: unknown, _extensionsDisabledList: unknown, searchText: unknown, sortColumn: unknown): [string, string, unknown, unknown] {
  const disable = [];
  gradioApp().querySelectorAll('#extensions input[type="checkbox"]').forEach((x) => {
    if (!(x instanceof HTMLInputElement)) return;
    if (x.name.startsWith('enable_') && !x.checked) disable.push(x.name.substring(7));
  });
  const id = randomId();
  log('Extensions check:', { disable });
  return [id, JSON.stringify(disable), searchText, sortColumn];
}

export function install_extension(button: HTMLButtonElement | HTMLInputElement, url: string): void {
  button.disabled = true;
  button.value = 'Installing...';
  button.innerHTML = 'installing';
  const textarea = gradioApp().querySelector('#extension_to_install textarea');
  if (!(textarea instanceof HTMLTextAreaElement)) return;
  textarea.value = url;
  updateInput(textarea);
  log('Extension install:', { url });
  const installBtn = gradioApp().querySelector('#install_extension_button');
  if (installBtn instanceof HTMLElement) installBtn.click();
}

export function uninstall_extension(button: HTMLButtonElement | HTMLInputElement, url: string): void {
  button.disabled = true;
  button.value = 'Uninstalling...';
  button.innerHTML = 'uninstalling';
  const textarea = gradioApp().querySelector('#extension_to_install textarea');
  if (!(textarea instanceof HTMLTextAreaElement)) return;
  textarea.value = url;
  updateInput(textarea);
  log('Extension uninstall:', { url });
  const uninstallBtn = gradioApp().querySelector('#uninstall_extension_button');
  if (uninstallBtn instanceof HTMLElement) uninstallBtn.click();
}

export function update_extension(button: HTMLButtonElement | HTMLInputElement, url: string): void {
  button.value = 'Updating...';
  button.innerHTML = 'updating';
  const textarea = gradioApp().querySelector('#extension_to_install textarea');
  if (!(textarea instanceof HTMLTextAreaElement)) return;
  textarea.value = url;
  updateInput(textarea);
  log('Extension update:', { url });
  const updateBtn = gradioApp().querySelector('#update_extension_button');
  if (updateBtn instanceof HTMLInputElement) updateBtn.click();
}

window.extensions_apply = extensions_apply;
window.extensions_check = extensions_check;
window.uninstall_extension = uninstall_extension;
window.install_extension = install_extension;
window.update_extension = update_extension;
