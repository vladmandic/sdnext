/* eslint max-classes-per-file: ["error", 2] */

class Bubble {
  constructor(min, start, end, label, scale) {
    this.min = min;
    this.start = start;
    this.end = end;
    this.scale = scale;
    this.offset = Math.round(this.scale * (this.start - this.min));
    this.width = Math.round(this.scale * (this.end - this.start));
    this.label = label;
    this.duration = Math.round(1000 * (this.end - this.start)) / 1000;
    this.title = `Job: ${this.label}\nDuration: ${this.duration}s\nStart: ${new Date(1000 * this.start).toLocaleString()}\nEnd: ${new Date(1000 * this.end).toLocaleString()}`;
  }

  getDateLabel() {
    return Math.round(1000 * (this.end - this.start)) / 1000;
  }
}

class Timesheet {
  constructor(container, data) {
    this.min = Math.floor(data[0].start);
    this.max = Math.round(data[data.length - 1].end + 0.5);
    this.data = data;
    this.container = container;
    const box = container.getBoundingClientRect();
    const width = box.width - 140;
    this.scale = width / (this.max - this.min);

    // draw sections
    let html = [];
    for (let c = 0; c <= this.max - this.min; c++) html.push(`<section style="width: ${this.scale}px;"></section>`);
    container.className = 'timesheet color-scheme-default';
    container.innerHTML = `<div class="scale"">${html.join('')}</div>`;

    // insert data
    html = [];
    for (let n = 0, m = this.data.length; n < m; n++) {
      const cur = this.data[n];
      const bubble = new Bubble(this.min, cur.start, cur.end, cur.label, this.scale);
      const line = [
        `<span title="${bubble.title}" style="margin-left: ${bubble.offset}px; width: ${bubble.width}px;" class="bubble" data-duration="${bubble.duration}"></span>`,
        `<span class="date">${bubble.duration}</span> `,
        `<span class="label">${bubble.label}</span>`,
      ].join('');
      html.push(`<li>${line}</li>`);
    }
    this.container.innerHTML += `<ul class="data">${html.join('')}</ul>`;
  }
}

window.Timesheet = Timesheet;
