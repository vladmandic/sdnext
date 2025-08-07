let nvmlInterval = null; // eslint-disable-line prefer-const
let nvmlEl = null;
let nvmlTable = null;
const chartData = { mem: [], load: [] };

async function updateNVMLChart(mem, load) {
  const maxLen = 120;
  const colorRangeMap = $.range_map({ // eslint-disable-line no-undef
    '0:5': '#fffafa',
    '6:10': '#fff7ed',
    '11:20': '#fed7aa',
    '21:30': '#fdba74',
    '31:40': '#fb923c',
    '41:50': '#f97316',
    '51:60': '#ea580c',
    '61:70': '#c2410c',
    '71:80': '#9a3412',
    '81:90': '#7c2d12',
    '91:100': '#6c2e12',
  });
  const sparklineConfigLOAD = { type: 'bar', height: '100px', barWidth: '2px', barSpacing: '1px', chartRangeMin: 0, chartRangeMax: 100, barColor: '#89007D' };
  const sparklineConfigMEM = { type: 'bar', height: '100px', barWidth: '2px', barSpacing: '1px', chartRangeMin: 0, chartRangeMax: 100, colorMap: colorRangeMap, composite: true };
  if (chartData.load.length > maxLen) chartData.load.shift();
  chartData.load.push(load);
  if (chartData.mem.length > maxLen) chartData.mem.shift();
  chartData.mem.push(mem);
  $('#nvmlChart').sparkline(chartData.load, sparklineConfigLOAD); // eslint-disable-line no-undef
  $('#nvmlChart').sparkline(chartData.mem, sparklineConfigMEM); // eslint-disable-line no-undef
}

async function updateNVML() {
  try {
    const res = await fetch(`${window.api}/nvml`);
    if (!res.ok) {
      clearInterval(nvmlInterval);
      nvmlEl.style.display = 'none';
      return;
    }
    const data = await res.json();
    if (!data) {
      clearInterval(nvmlInterval);
      nvmlEl.style.display = 'none';
      return;
    }
    const nvmlTbody = nvmlTable.querySelector('tbody');
    for (const gpu of data) {
      const rows = `
        <tr><td>GPU</td><td>${gpu.name}</td></tr>
        <tr><td>Driver</td><td>${gpu.version.driver}</td></tr>
        <tr><td>VBIOS</td><td>${gpu.version.vbios}</td></tr>
        <tr><td>ROM</td><td>${gpu.version.rom}</td></tr>
        <tr><td>Driver</td><td>${gpu.version.driver}</td></tr>
        <tr><td>PCI</td><td>Gen.${gpu.pci.link} x${gpu.pci.width}</td></tr>
        <tr><td>Memory</td><td>${gpu.memory.used}Mb / ${gpu.memory.total}Mb</td></tr>
        <tr><td>Clock</td><td>${gpu.clock.gpu[0]}Mhz / ${gpu.clock.gpu[1]}Mhz</td></tr>
        <tr><td>Power</td><td>${gpu.power[0]}W / ${gpu.power[1]}W</td></tr>
        <tr><td>Load GPU</td><td>${gpu.load.gpu}%</td></tr>
        <tr><td>Load Memory</td><td>${gpu.load.memory}%</td></tr>
        <tr><td>Temperature</td><td>${gpu.load.temp}°C</td></tr>
        <tr><td>Fans</td><td>${gpu.load.fan}%</td></tr>
        <tr><td>State</td><td>${gpu.state}</td></tr>
      `;
      nvmlTbody.innerHTML = rows;
      updateNVMLChart(gpu.load.memory, gpu.load.gpu);
    }
    nvmlEl.style.display = 'block';
  } catch (e) {
    clearInterval(nvmlInterval);
    nvmlEl.style.display = 'none';
  }
}

async function initNVML() {
  nvmlEl = document.getElementById('nvml');
  if (!nvmlEl) {
    nvmlEl = document.createElement('div');
    nvmlEl.className = 'nvml';
    nvmlEl.id = 'nvml';
    nvmlTable = document.createElement('table');
    nvmlTable.className = 'nvml-table';
    nvmlTable.id = 'nvml-table';
    nvmlTable.innerHTML = `
      <thead><tr><th></th><th></th></tr></thead>
      <tbody></tbody>
    `;
    const nvmlChart = document.createElement('div');
    nvmlChart.id = 'nvmlChart';
    nvmlEl.appendChild(nvmlTable);
    nvmlEl.appendChild(nvmlChart);
    gradioApp().appendChild(nvmlEl);
    log('initNVML');
  }
}

async function startNVML() {
  nvmlEl = document.getElementById('nvml');
  if (nvmlInterval) {
    clearInterval(nvmlInterval);
    nvmlInterval = null;
    nvmlEl.style.display = 'none';
  } else {
    nvmlInterval = setInterval(updateNVML, 1000);
  }
}

async function disableNVML() {
  clearInterval(nvmlInterval);
  nvmlEl.style.display = 'none';
}
