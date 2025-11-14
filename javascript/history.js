const inferenceTypes = ['inference', 'vae', 'te'];
const ioTypes = ['load', 'save'];

function refreshHistory() {
  log('refreshHistory');
  fetch(`${window.api}/history`, { priority: 'low' }).then((res) => {
    const timeline = document.getElementById('history_timeline');
    const table = document.getElementById('history_table');
    timeline.innerHTML = '';
    res.json().then((data) => {
      if (!data || !data.length) {
        table.innerHTML = '<p>No history data available.</p>';
        return;
      }

      // build table
      let html = '<table><thead><tr><th>Time</th><th>ID</th><th>Job</th><th>Action</th><th>Duration</th><th>Outputs</th></tr></thead><tbody>';
      for (const entry of data) {
        const ts = new Date(1000 * entry.timestamp).toLocaleString();
        const duration = entry.duration ? `${(entry.duration).toFixed(3)}` : '';
        const outputs = entry.outputs.join(', ');
        html += `<tr><td>${ts}</td><td>${entry.id}</td><td>${entry.job}</td><td>${entry.op}</td><td>${duration}</td><td>${outputs}</td></tr>`;
      }
      html += '</tbody></table>';
      table.innerHTML = html;

      // crop data to last processing session
      let startIdx = -1;
      for (let i = data.length - 1; i >= 0; --i) {
        const e = data[i];
        if ((e.job === 'control' || e.job === 'text' || e.job === 'control' || e.job === 'image') && (e.op === 'begin')) {
          startIdx = i;
          break;
        }
      }
      if (startIdx >= 0) data = data.slice(startIdx);

      // build timeline
      const ts = [];
      for (const entry of data) {
        if (entry.op === 'begin') {
          const start = entry.timestamp;
          const end = data.find((e) => (e.id === entry.id && e.op === 'end')) || data[data.length - 1].timestamp;
          if (end.timestamp - start < 0.02) continue; // skip very short entries
          if (inferenceTypes.some((type) => entry.job.toLowerCase().startsWith(type))) entry.type = 'inference';
          else if (ioTypes.some((type) => entry.job.toLowerCase().startsWith(type))) entry.type = 'io';
          else entry.type = 'default';
          if (start && end.timestamp) ts.push({ start, end: end.timestamp, label: entry.job, type: entry.type });
        }
      }
      if (!ts.length) return;
      new Timesheet(timeline, ts); // eslint-disable-line no-undef, no-new
    });
  });
}
