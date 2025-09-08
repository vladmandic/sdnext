let gpuInterval = null; // eslint-disable-line prefer-const
const chartData = { mem: [], load: [] };

async function updateGPUChart(mem, load) {
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
  const sparklineConfigLOAD = { type: 'bar', height: '128px', barWidth: '3px', barSpacing: '1px', chartRangeMin: 0, chartRangeMax: 100, barColor: '#89007D' };
  const sparklineConfigMEM = { type: 'bar', height: '128px', barWidth: '3px', barSpacing: '1px', chartRangeMin: 0, chartRangeMax: 100, colorMap: colorRangeMap, composite: true };
  if (chartData.load.length > maxLen) chartData.load.shift();
  chartData.load.push(load);
  if (chartData.mem.length > maxLen) chartData.mem.shift();
  chartData.mem.push(mem);
  $('#gpuChart').sparkline(chartData.load, sparklineConfigLOAD); // eslint-disable-line no-undef
  $('#gpuChart').sparkline(chartData.mem, sparklineConfigMEM); // eslint-disable-line no-undef
}

async function updateGPU() {
  const gpuEl = document.getElementById('gpu');
  const gpuTable = document.getElementById('gpu-table');
  try {
    const res = await fetch(`${window.api}/gpu`);
    if (!res.ok) {
      clearInterval(gpuInterval);
      gpuEl.style.display = 'none';
      return;
    }
    const data = await res.json();
    if (!data) {
      clearInterval(gpuInterval);
      gpuEl.style.display = 'none';
      return;
    }
    const gpuTbody = gpuTable.querySelector('tbody');
    for (const gpu of data) {
      console.log(gpu);
      let rows = `<tr><td>GPU</td><td>${gpu.name}</td></tr>`;
      for (const item of Object.entries(gpu.data)) rows += `<tr><td>${item[0]}</td><td>${item[1]}</td></tr>`;
      gpuTbody.innerHTML = rows;
      if (gpu.chart && gpu.chart.length === 2) updateGPUChart(gpu.chart);
    }
    gpuEl.style.display = 'block';
  } catch (e) {
    error('updateGPU', e);
    clearInterval(gpuInterval);
    gpuEl.style.display = 'none';
  }
}

async function startGPU() {
  const gpuEl = document.getElementById('gpu');
  gpuEl.style.display = 'block';
  if (gpuInterval) clearInterval(gpuInterval);
  const interval = window.opts?.gpu_monitor || 3000;
  log('startGPU', interval);
  gpuInterval = setInterval(updateGPU, interval);
  updateGPU();
}

async function disableGPU() {
  clearInterval(gpuInterval);
  const gpuEl = document.getElementById('gpu');
  gpuEl.style.display = 'none';
}
