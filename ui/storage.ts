import { Timesheet } from './timesheet';
import { log } from './logger';
import { authFetch } from './authWrap';

const types = ['Images', 'Videos', 'Models', 'Data', 'Cache', 'Code', 'Other'];

interface LocationEntry {
  name: string;
  type: 'Images' | 'Videos' | 'Models' | 'Data' | 'Cache' | 'Code' | 'Other';
  folders: string[];
  paths: string[];
  size: number;
  mtime: number;
  nfiles: number;
  nfolders: number;
  nsymlinks: number;
  nerrors: number;
  time: number;
}

interface TimelineEntry {
  start: number;
  end: number;
  label: string;
  type: 'inference' | 'io' | 'default';
}

function buildTable(type: string, data: LocationEntry[]) {
  // let html = `<h2>${type}</h2><table><thead><tr><th>Location</th><th>Size</th><th>MTime</th></tr></thead><tbody>`;
  const totalSize = data.reduce((acc, entry) => acc + entry.size, 0);
  const totalLoc = data.length;
  const totalFiles = data.reduce((acc, entry) => acc + entry.nfiles, 0);
  const totalFolders = data.reduce((acc, entry) => acc + entry.nfolders, 0);
  let title = `Locations: ${totalLoc}\nTotal Size: ${(totalSize / (1024 * 1024)).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} MB\nTotal Files: ${totalFiles}\nTotal Folders: ${totalFolders}\n`;
  let html = `<h2 title="${title}">${type}</h2><table><tbody>`;
  for (const entry of data) {
    if (entry.size === 0) continue;
    const size = (entry.size / (1024 * 1024)).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + ' MB';
    const mtime = entry.mtime > 0 ? new Date(entry.mtime * 1000).toLocaleString() : '';
    title = `Type: ${entry.type}\nName: ${entry.name}\nSize: ${size}\nLast modified: ${mtime}\n`;
    title += `Folders: ${entry.folders.join(', ')}\nResolved paths: ${entry.paths.join(', ')}\n`;
    title += `Subfolders: ${entry.nfolders}\nFiles: ${entry.nfiles}\nSymlinks: ${entry.nsymlinks}\nErrors: ${entry.nerrors}\n`;
    title += `Time to scan: ${entry.time.toFixed(3)} seconds`;
    const perc = Math.round((entry.size / totalSize) * 100);
    const color = `rgb(${perc}, 50, 80)`;
    const css = `background: linear-gradient(to right, ${color} ${perc}%, transparent ${perc}%);`;
    html += `<tr title="${title}"><td style="${css}">${entry.name}</td><td>${size}</td><td>${mtime}</td></tr>`;
  }
  html += '</tbody></table>';
  return html;
}

export async function refreshStorage(storageTypes: string[]) {
  log('refreshStorage', storageTypes);
  authFetch(`${window.api}/storage?types=${storageTypes.join(',')}`, { priority: 'low' }).then((res) => {
    if (!res) return;
    const timeline = document.getElementById('storage_timeline');
    const table = document.getElementById('storage_table');
    if (!timeline || !table) return;
    timeline.innerHTML = '';
    res.json().then((rawData) => {
      const data = rawData as LocationEntry[];
      if (!data || !data.length) {
        table.innerHTML = '<p>No storage data available.</p>';
        return;
      }
      table.innerHTML = '';
      if (storageTypes.includes('All')) storageTypes = types;
      for (const type of storageTypes) {
        const typeData = data.filter((entry) => entry.type === type);
        if (typeData.length > 0) table.innerHTML += buildTable(type, typeData);
      }

      /*
      // build timeline
      const ts: TimelineEntry[] = [];
      for (const entry of data) {
        if (entry.op === 'begin') {
          const start = entry.timestamp;
          const endEntry = data.find((e) => (e.id === entry.id && e.op === 'end'));
          const end = endEntry?.timestamp ?? data[data.length - 1].timestamp;
          if (end - start < 0.02) continue; // skip very short entries
          if (inferenceTypes.some((type) => entry.job.toLowerCase().startsWith(type))) entry.type = 'inference';
          else if (ioTypes.some((type) => entry.job.toLowerCase().startsWith(type))) entry.type = 'io';
          else entry.type = 'default';
          if (start && end) ts.push({ start, end, label: entry.job, type: entry.type });
        }
      }
      if (!ts.length) return;
      // eslint-disable-next-line no-new
      new Timesheet(timeline, ts);
      */
    });
  });
}

window.refreshStorage = refreshStorage;
