import { gradioApp } from './script';
import { log, error } from './logger';

let lastHeadImg = null;
let notificationButton = null;

export async function sendNotification() {
  try {
    if (!notificationButton) {
      notificationButton = gradioApp().getElementById('request_notifications');
      if (notificationButton) notificationButton.addEventListener('click', () => Notification.requestPermission(), true);
    }
    if (document.hasFocus()) return; // window is in focus so don't send notifications
    let galleryPreviews = gradioApp().querySelectorAll('div[id^="tab_"][style*="display: block"] div[id$="_results"] .thumbnail-item > img');
    if (!galleryPreviews || galleryPreviews.length === 0) galleryPreviews = gradioApp().querySelectorAll('.thumbnail-item > img');
    if (!galleryPreviews || galleryPreviews.length === 0) return;
    const headImg = galleryPreviews[0]?.src;
    if (!headImg || headImg === lastHeadImg || headImg.includes('logo-bg-')) return;
    const audioNotification = gradioApp().querySelector('#audio_notification audio');
    if (audioNotification instanceof HTMLAudioElement) audioNotification.play();
    lastHeadImg = headImg;
    const imgs = new Set(Array.from<any>(galleryPreviews).map((img) => (img instanceof HTMLImageElement ? img.src : ''))); // Multiple copies of the images are in the DOM when one is selected
    const notification = new Notification('SD.Next', {
      body: `Generated ${imgs.size > 1 ? imgs.size - window.opts.return_grid : 1} image${imgs.size > 1 ? 's' : ''}`,
      icon: headImg,
      image: headImg,
    } as any);
    notification.onclick = function onClick() {
      parent.focus();
      this.close();
    };
    log('sendNotifications');
  } catch (e) {
    error(`sendNotification: ${e}`);
  }
}
