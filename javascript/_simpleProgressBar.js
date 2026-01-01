class SimpleProgressBar {
  #container = document.createElement('div');
  #progress = document.createElement('div');
  #textDiv = document.createElement('div');
  #text = document.createElement('span');
  #visible = false;
  #hideTimeout = null;
  #interval = null;
  #max = 0;
  /** @type {Set} */
  #monitoredSet;

  constructor(monitoredSet) {
    this.#monitoredSet = monitoredSet;  // This is required because incrementing a variable with a class method turned out to not be an atomic operation
    this.#container.style.cssText = 'position:relative;overflow:hidden;border-radius:var(--sd-border-radius);width:100%;background-color:hsla(0,0%,36%,0.3);height:1.2rem;margin:0;padding:0;display:none;'
    this.#progress.style.cssText = 'position:absolute;left:0;height:100%;width:0;transition:width 200ms;'
    this.#progress.style.backgroundColor = 'hsla(110, 32%, 35%, 0.80)';  // alt: '#27911d'
    this.#textDiv.style.cssText = 'position:relative;margin:auto;width:max-content;height:100%;';
    this.#text.style.cssText = 'user-select:none;color:white;'

    this.#textDiv.append(this.#text);
    this.#container.append(this.#progress, this.#textDiv);
  }

  start(total) {
    this.clear();
    this.#max = total;
    this.#interval = setInterval(() => {
      this.#update(this.#monitoredSet.size, this.#max);
    }, 250);
  }

  attachTo(element) {
    if (element.hasChildNodes) {
      element.innerHTML = '';
    }
    element.appendChild(this.#container);
  }

  clear() {
    this.#stop();
    clearTimeout(this.#hideTimeout);
    this.#hideTimeout = null;
    this.#container.style.display = 'none';
    this.#visible = false;
    this.#progress.style.width = '0';
    this.#text.textContent = '';
  }

  #update(loaded, max) {
    if (this.#hideTimeout) {
      this.#hideTimeout = null;
    }

    this.#progress.style.width = `${Math.floor((loaded / max) * 100)}%`;
    this.#text.textContent = `${loaded}/${max}`;

    if (!this.#visible) {
      this.#container.style.display = 'block';
      this.#visible = true;
    }
    if (loaded >= max) {
      this.#stop()
      this.#hideTimeout = setTimeout(() => {
        this.clear();
      }, 1000);
    }
  }

  #stop() {
    clearInterval(this.#interval);
    this.#interval = null;
  }
}
