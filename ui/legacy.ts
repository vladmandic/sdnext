import { log } from './logger';

export function addLegacyNotice() {
  log('legacyNotice');
  const notice = document.createElement('div');
  notice.id = 'legacy-notice';
  notice.className = 'legacy-standard';
  notice.textContent = 'Legacy';
  notice.title = 'Standard UI is a legacy interface that is no longer maintained and will be removed in the future. Please switch to ModernUI for best experience.';
  document.body.appendChild(notice);
}
