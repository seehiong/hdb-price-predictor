// pipeline_private/04_build_map.js
// Builds interactive standalone Leaflet map for Singapore Private Condominium Resale Pricing
// Incorporates real URA private residential resale transactions (recent caveats per project)
const fs = require('fs');
const path = require('path');

// How many recent transactions to embed per project. Matches the public map;
// see the note there on the size/benefit trade-off.
const MAX_TX_PER_PROJECT = 50;

const dataDirJson = path.join(__dirname, 'data', 'private_nonlanded_postal_data.json');
const rootJson = path.join(__dirname, '..', 'private_nonlanded_postal_data.json');
const localJson = path.join(__dirname, 'private_nonlanded_postal_data.json');
const condoJsonPath = fs.existsSync(dataDirJson) ? dataDirJson : (fs.existsSync(localJson) ? localJson : rootJson);
const condoData = JSON.parse(fs.readFileSync(condoJsonPath, 'utf8'));

const SECTOR_TO_DISTRICT = {
  '01': '01', '02': '01', '03': '01', '04': '01', '05': '01', '06': '01',
  '07': '02', '08': '02', '14': '03', '15': '03', '16': '03', '09': '04', '10': '04',
  '11': '05', '12': '05', '13': '05', '17': '06', '18': '07', '19': '07',
  '20': '08', '21': '08', '22': '09', '23': '09', '24': '10', '25': '10', '26': '10', '27': '10',
  '28': '11', '29': '11', '30': '11', '31': '12', '32': '12', '33': '12',
  '34': '13', '35': '13', '36': '13', '37': '13', '38': '14', '39': '14', '40': '14', '41': '14',
  '42': '15', '43': '15', '44': '15', '45': '15', '46': '16', '47': '16', '48': '16',
  '49': '17', '50': '17', '81': '17', '51': '18', '52': '18', '53': '19', '54': '19', '55': '19', '82': '19',
  '56': '20', '57': '20', '58': '21', '59': '21', '60': '22', '61': '22', '62': '22', '63': '22', '64': '22',
  '65': '23', '66': '23', '67': '23', '68': '23', '69': '24', '70': '24', '71': '24',
  '72': '25', '73': '25', '77': '26', '78': '26', '75': '27', '76': '27', '79': '28', '80': '28'
};

const DISTRICT_NAMES = {
  '01': 'Raffles Place, Marina Bay, Cecil', '02': 'Tanjong Pagar, Chinatown',
  '03': 'Queenstown, Tiong Bahru', '04': 'Telok Blangah, Harbourfront, Sentosa',
  '05': 'Pasir Panjang, Clementi, West Coast', '06': 'City Hall, Beach Road',
  '07': 'Bugis, Middle Road, Rochor', '08': 'Little India, Farrer Park',
  '09': 'Orchard, River Valley, Cairnhill', '10': 'Tanglin, Holland, Bukit Timah',
  '11': 'Newton, Novena, Thomson', '12': 'Toa Payoh, Balestier',
  '13': 'Macpherson, Potong Pasir', '14': 'Geylang, Paya Lebar, Eunos',
  '15': 'East Coast, Marine Parade, Katong', '16': 'Bedok, Upper East Coast, Siglap',
  '17': 'Changi, Loyang', '18': 'Tampines, Pasir Ris',
  '19': 'Serangoon, Hougang, Punggol, Sengkang', '20': 'Bishan, Ang Mo Kio',
  '21': 'Upper Bukit Timah, Clementi Park', '22': 'Jurong, Boon Lay',
  '23': 'Hillview, Dairy Farm, Bukit Panjang', '24': 'Lim Chu Kang, Tengah',
  '25': 'Woodlands, Admiralty', '26': 'Mandai, Upper Thomson, Springleaf',
  '27': 'Yishun, Sembawang', '28': 'Seletar, Yio Chu Kang'
};

const DISTRICT_BASE_PSF = {
  '01': 2650, '02': 2550, '03': 2100, '04': 2350, '05': 1850,
  '06': 2700, '07': 2400, '08': 1880, '09': 2850, '10': 2750,
  '11': 2600, '12': 1800, '13': 1850, '14': 1750, '15': 2150,
  '16': 1650, '17': 1420, '18': 1480, '19': 1580, '20': 1950,
  '21': 2050, '22': 1450, '23': 1520, '24': 1350, '25': 1380,
  '26': 1500, '27': 1400, '28': 1480
};

const projectSegments = {};
const encPath = path.join(__dirname, '..', 'models', 'condo_project_encoding.json');
if (fs.existsSync(encPath)) {
  try {
    const enc = JSON.parse(fs.readFileSync(encPath, 'utf8'));
    if (enc.project_segment) Object.assign(projectSegments, enc.project_segment);
  } catch (e) {}
}

const SECTOR_TO_SEGMENT = {
  // District 05 split: Pasir Panjang (11) & Buona Vista (13) are RCR; Clementi (12) is OCR
  '11': 'RCR',
  '12': 'OCR',
  '13': 'RCR',
  // District 20 split: Ang Mo Kio (56) is OCR; Bishan (57) is RCR
  '56': 'OCR',
  '57': 'RCR',
};

function getMarketSegment(dStr, projectName, sector) {
  const norm = (projectName || '').trim().toUpperCase();
  if (norm && projectSegments[norm]) {
    return projectSegments[norm];
  }
  if (sector && SECTOR_TO_SEGMENT[sector]) {
    return SECTOR_TO_SEGMENT[sector];
  }
  const d = parseInt(dStr, 10);
  if ([1, 2, 6, 7, 9, 10, 11].includes(d)) return 'CCR';
  if ([3, 4, 8, 12, 13, 14, 15, 20, 21].includes(d)) return 'RCR';
  return 'OCR';
}

function normalizeProject(str) {
  return str.toUpperCase().replace(/[^A-Z0-9]/g, '').replace(/CONDOMINIUM|CONDO|RESIDENCES|RESIDENCE|APARTMENTS|APARTMENT/g, '');
}

// Official BCA Management Corporation Strata Title (MCST) Registry from Data.gov.sg (Dataset d_f988c57e16e99ad3a649aa04572efd1c)
const mcstPath = path.join(__dirname, 'data', 'bca_mcst_registry.json');
let mcstRegistry = {};
let mcstByPostal = {};
if (fs.existsSync(mcstPath)) {
  try {
    mcstRegistry = JSON.parse(fs.readFileSync(mcstPath, 'utf8'));
    for (const item of Object.values(mcstRegistry)) {
      if (item.postal && !mcstByPostal[item.postal]) {
        mcstByPostal[item.postal] = item;
      }
    }
  } catch (e) {}
}

function formatTenure(raw) {
  if (!raw) return { label: '', leaseYear: '' };
  const s = raw.trim();
  if (/FREEHOLD/i.test(s)) return { label: 'Freehold', leaseYear: '' };
  const m = s.match(/(\d+)\s*yrs?(?:\s*lease)?(?:\s*(?:commencing\s+from|from)\s*(\d{4}))?/i);
  if (m) {
    const term = m[1];
    const year = m[2] || '';
    return { label: `${term}-yr Leasehold`, leaseYear: year };
  }
  return { label: s, leaseYear: '' };
}

// 1. Read real URA transactions from private_resi_transactions.csv
const csvPath = path.join(__dirname, 'data', 'private_resi_transactions.csv');
const projectTx = {};
const projectTxNorm = {};
const projectTenure = {};
const projectTenureNorm = {};

if (fs.existsSync(csvPath)) {
  console.log(`Reading URA private resale transactions from ${csvPath}...`);
  const lines = fs.readFileSync(csvPath, 'utf8').split('\n');
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const parts = line.split(',');
    if (parts.length < 10) continue;
    const rawProj = parts[0].trim().toUpperCase();
    if (!rawProj) continue;

    const rawDate = parts[7].trim();
    const ymDate = rawDate.length >= 7 ? rawDate.slice(0, 7) : rawDate;
    const price = Math.round(parseFloat(parts[9]) || 0);
    const psf = Math.round(parseFloat(parts[10]) || 0);
    const area = parseFloat(parts[4]) || 0;
    const floor = parts[5].trim();
    const pType = parts[11] ? parts[11].trim() : 'Condominium';
    const rawTenure = parts[12] ? parts[12].trim() : '';

    if (rawTenure && !projectTenure[rawProj]) {
      projectTenure[rawProj] = rawTenure;
      const normP = normalizeProject(rawProj);
      if (normP && !projectTenureNorm[normP]) projectTenureNorm[normP] = rawTenure;
    }

    const item = {
      d: ymDate,
      f: floor || '-',
      a: area,
      p: price,
      psf: psf,
      t: pType
    };

    if (!projectTx[rawProj]) projectTx[rawProj] = [];
    projectTx[rawProj].push(item);

    const normP = normalizeProject(rawProj);
    if (normP) {
      if (!projectTxNorm[normP]) projectTxNorm[normP] = [];
      projectTxNorm[normP].push(item);
    }
  }

  // Sort by date descending and cap the list per project
  for (const p in projectTx) {
    projectTx[p].sort((a, b) => b.d.localeCompare(a.d));
    projectTx[p] = projectTx[p].slice(0, MAX_TX_PER_PROJECT);
  }
  for (const np in projectTxNorm) {
    projectTxNorm[np].sort((a, b) => b.d.localeCompare(a.d));
    projectTxNorm[np] = projectTxNorm[np].slice(0, MAX_TX_PER_PROJECT);
  }
  console.log(`Indexed URA transactions for ${Object.keys(projectTx).length} distinct project names.`);
}

// 2. Aggregate by project name
const projectsMap = {};
for (const [postal, item] of Object.entries(condoData)) {
  const name = item.project_name ? item.project_name.trim() : '';
  if (!name || name === 'Unknown') continue;
  if (!item.latitude || !item.longitude) continue;

  const sector = postal.slice(0, 2);
  const district = SECTOR_TO_DISTRICT[sector] || '19';
  const segment = getMarketSegment(district, name, sector);

  if (!projectsMap[name]) {
    let propType = 'Condominium';
    if (item.property_type === 'EC' || name.toLowerCase().includes('executive') || item.property_type === 'Executive Condominium') {
      propType = 'Executive Condominium';
    } else if (item.property_type === 'Apartment') {
      propType = 'Apartment';
    }

    const txs = projectTx[name.toUpperCase()] || projectTxNorm[normalizeProject(name)] || [];
    const normName = name.toUpperCase();
    const normP = normalizeProject(name);
    const rawClean = name.toUpperCase().replace(/[^A-Z0-9]/g, '');
    const mcst = mcstRegistry[normP] || mcstRegistry[rawClean] || mcstByPostal[postal];
    const mcstNo = mcst ? mcst.mcno : '';
    const mcstYear = mcst ? mcst.year : '';
    const KNOWN_PROJECT_TENURE = {
      'CHARIS LODGE': 'Freehold',
      'THE ESTRELLA': 'Freehold',
      'JUBILEE RESIDENCE': 'Freehold',
      'ROYAL PALM MANSIONS': 'Freehold'
    };
    const rawTenure = projectTenure[normName] || projectTenureNorm[normP] || KNOWN_PROJECT_TENURE[normName] || '';
    const { label: tenureLabel, leaseYear } = formatTenure(rawTenure);

    projectsMap[name] = {
      name,
      postal,
      district,
      districtName: DISTRICT_NAMES[district] || '',
      segment,
      basePsf: DISTRICT_BASE_PSF[district] || 1650,
      propType,
      tenure: tenureLabel,
      leaseYear,
      mcstNo,
      mcstYear,
      units: item.dwelling_units || 0,
      lat: item.latitude,
      lon: item.longitude,
      postals: [postal],
      tx: txs
    };
  } else {
    projectsMap[name].postals.push(postal);
    projectsMap[name].units += (item.dwelling_units || 0);
  }
}

const projectsList = Object.values(projectsMap);
const matchedCount = projectsList.filter(p => p.tx && p.tx.length > 0).length;
console.log(`Aggregated ${projectsList.length} unique private residential projects (${matchedCount} with real URA resale caveats).`);

// Generate the HTML Leaflet Map
const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Singapore Private Condominium Resale Pricing Map</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css"/>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; }
    html, body {
      width: 100%; height: 100%; margin: 0; padding: 0;
      font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #0f172a;
    }
    #map {
      position: absolute; top: 0; bottom: 0; right: 0; left: 0;
      width: 100%; height: 100%;
    }
    /* Floating Header Control */
    .map-header-bar {
      position: absolute; top: 14px; left: 50%; transform: translateX(-50%);
      z-index: 1000; background: rgba(15, 23, 42, 0.88);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 9999px;
      padding: 6px 14px;
      display: flex; align-items: center; gap: 10px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.36);
      max-width: 95vw;
    }
    .map-header-title {
      font-size: 0.92rem; font-weight: 700; color: #f8fafc;
      display: flex; align-items: center; gap: 6px; white-space: nowrap;
    }
    .search-input {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 9999px;
      padding: 6px 14px; color: #f8fafc; font-size: 0.85rem;
      outline: none; width: 190px; transition: all 0.2s ease;
      font-family: inherit;
    }
    .search-input:focus {
      width: 250px; border-color: #38bdf8;
      background: rgba(255, 255, 255, 0.14);
    }
    .search-input::placeholder { color: #94a3b8; }
    /* Segment Filter Buttons */
    .filter-pills {
      display: flex; gap: 4px;
    }
    .filter-pill {
      background: transparent; border: 1px solid rgba(255, 255, 255, 0.14);
      color: #94a3b8; padding: 4px 10px; border-radius: 9999px;
      font-size: 0.78rem; font-weight: 600; cursor: pointer;
      transition: all 0.2s ease; font-family: inherit;
    }
    .filter-pill:hover { color: #f8fafc; border-color: rgba(255, 255, 255, 0.3); }
    .filter-pill.active {
      background: #2563eb; color: #ffffff; border-color: #3b82f6;
    }
    /* Floating Legend */
    .map-legend {
      position: absolute; bottom: 20px; right: 20px; z-index: 1000;
      background: rgba(15, 23, 42, 0.88); backdrop-filter: blur(14px);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px; padding: 12px 16px; color: #f8fafc;
      font-size: 0.78rem; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.36);
    }
    .legend-title { font-weight: 700; margin-bottom: 8px; color: #94a3b8; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.5px; }
    .legend-row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
    .legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    .dot-ccr { background: #ef4444; box-shadow: 0 0 8px rgba(239, 68, 68, 0.5); }
    .dot-rcr { background: #f59e0b; box-shadow: 0 0 8px rgba(245, 158, 11, 0.5); }
    .dot-ocr { background: #10b981; box-shadow: 0 0 8px rgba(16, 185, 129, 0.5); }
    /* Popup Styling */
    .leaflet-popup-content-wrapper {
      background: #0f172a; color: #f8fafc; border-radius: 14px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      box-shadow: 0 16px 36px rgba(0, 0, 0, 0.5);
      padding: 2px;
    }
    .leaflet-popup-tip { background: #0f172a; }
    .popup-box { padding: 12px; min-width: 320px; max-width: 440px; }
    .popup-title { font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
    .popup-badge-row { display: flex; gap: 6px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }
    .badge {
      font-size: 0.72rem; font-weight: 700; padding: 2px 7px; border-radius: 9999px;
      text-transform: uppercase; letter-spacing: 0.5px;
    }
    .badge-ccr { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-rcr { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-ocr { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-dist { background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.25); }
    .badge-tenure { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); text-transform: none; }
    .badge-type { background: rgba(148, 163, 184, 0.12); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.25); text-transform: none; }
    .popup-meta-box {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      padding: 8px 10px;
      margin-bottom: 12px;
      font-size: 0.77rem;
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .meta-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
    }
    .meta-row:not(:last-child) {
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      padding-bottom: 4px;
    }
    .meta-lbl {
      color: #94a3b8;
      font-weight: 500;
      white-space: nowrap;
    }
    .meta-val {
      color: #f1f5f9;
      font-weight: 600;
      text-align: right;
    }
    .table-scroll-wrap {
      max-height: 220px;
      overflow-y: auto;
      margin-bottom: 8px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
    }
    .table-scroll-wrap::-webkit-scrollbar { width: 5px; }
    .table-scroll-wrap::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.2); border-radius: 4px; }
    .price-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; margin-bottom: 0; }
    .price-table thead th { position: sticky; top: 0; background: #1e293b; z-index: 2; }
    .price-table th { background: rgba(255, 255, 255, 0.05); color: #94a3b8; font-weight: 600; text-align: left; padding: 6px 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.1); }
    .price-table td { padding: 5px 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); color: #cbd5e1; }
    .price-green { font-weight: 700; color: #34d399; }
    .psf-val { font-size: 1.15rem; font-weight: 800; color: #38bdf8; }
    .psf-sub { font-size: 0.76rem; color: #94a3b8; }
    /* Prominent Marker Styling on OpenStreetMap */
    path.condo-pin-marker {
      filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.75)) drop-shadow(0 0 1.5px rgba(0, 0, 0, 0.95));
      transition: r 0.15s ease, stroke-width 0.15s ease, filter 0.15s ease;
      cursor: pointer;
    }
    path.condo-pin-marker:hover {
      filter: drop-shadow(0 4px 10px rgba(0, 0, 0, 0.95)) drop-shadow(0 0 6px rgba(255, 255, 255, 0.95));
      stroke: #ffffff !important;
      stroke-width: 3.5px !important;
    }
    /* Modern high-contrast cluster bubbles for OSM */
    .marker-cluster-small, .marker-cluster-medium, .marker-cluster-large {
      background-color: rgba(15, 23, 42, 0.85) !important;
      border: 2px solid #ffffff !important;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.6) !important;
    }
    .marker-cluster div {
      background-color: transparent !important;
      color: #f8fafc !important;
      font-family: 'Outfit', sans-serif !important;
      font-weight: 800 !important;
      font-size: 0.82rem !important;
    }
    @media (max-width: 768px) {
      .map-header-bar { top: 10px; padding: 4px 8px; font-size: 0.8rem; flex-wrap: wrap; justify-content: center; }
      .search-input { width: 140px; }
      .search-input:focus { width: 180px; }
      .map-legend { display: none; }
    }
  </style>
</head>
<body>
  <!-- Floating Search & Filter Bar -->
  <div class="map-header-bar">
    <div class="map-header-title">🏙️ <span>Private Condos</span></div>
    <input type="text" id="projectSearch" class="search-input" placeholder="Search condo / postal...">
    <div class="filter-pills">
      <button class="filter-pill active" data-filter="ALL">All (2.4k)</button>
      <button class="filter-pill" data-filter="CCR">CCR</button>
      <button class="filter-pill" data-filter="RCR">RCR</button>
      <button class="filter-pill" data-filter="OCR">OCR</button>
    </div>
  </div>

  <!-- Legend -->
  <div class="map-legend">
    <div class="legend-title">Market Segment</div>
    <div class="legend-row"><span class="legend-dot dot-ccr"></span> Core Central Region (CCR)</div>
    <div class="legend-row"><span class="legend-dot dot-rcr"></span> Rest of Central (RCR)</div>
    <div class="legend-row"><span class="legend-dot dot-ocr"></span> Outside Central (OCR)</div>
  </div>

  <!-- Map Container -->
  <div id="map"></div>

  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
  <script>
    // Embedded Singapore Private Condominium Developments
    const CONDO_PROJECTS = ${JSON.stringify(projectsList)};

    // Initialize Map
    const map = L.map('map', {
      center: [1.3521, 103.8198],
      zoom: 12,
      minZoom: 10,
      maxZoom: 19
    });

    // Base Tile Layers: 100% free, zero API key required, no 403 blocks on file://
    const streetLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, METI, TomTom',
      maxZoom: 19
    }).addTo(map);

    const darkLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
      maxZoom: 19
    });

    const osmHotLayer = L.tileLayer('https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, Tiles style by Humanitarian OpenStreetMap Team',
      maxZoom: 19
    });

    L.control.layers({
      "🗺️ Street Map (Esri)": streetLayer,
      "🌙 Dark Canvas (Esri)": darkLayer,
      "🌐 OpenStreetMap (HOT)": osmHotLayer
    }, null, { position: 'bottomleft', collapsed: true }).addTo(map);

    // Marker Clusters
    let clusterGroup = L.markerClusterGroup({
      chunkedLoading: true,
      maxClusterRadius: 40,
      spiderfyOnMaxZoom: true,
      showCoverageOnHover: false
    });

    const markers = [];

    function getColorForSegment(segment) {
      if (segment === 'CCR') return '#ef4444';
      if (segment === 'RCR') return '#f59e0b';
      return '#10b981';
    }

    function formatMoney(num) {
      return 'S$ ' + num.toString().replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');
    }

    function createPopupContent(p) {
      const psf = p.basePsf;
      const hasTx = p.tx && p.tx.length > 0;

      let bodyContent = '';
      if (hasTx) {
        const latest = p.tx[0];
        const latestPsf = latest.psf || psf;
        const latestDate = latest.d;

        let rowsHtml = '';
        for (let i = 0; i < p.tx.length; i++) {
          const t = p.tx[i];
          rowsHtml += \`
            <tr>
              <td style="font-weight: 600; color: #94a3b8;">\${t.d}</td>
              <td>\${t.f}</td>
              <td>\${t.a ? t.a + ' sqm' : '-'}</td>
              <td class="price-green">\${formatMoney(t.p)}</td>
              <td style="color: #38bdf8;">\${t.psf ? 'S$ ' + t.psf.toLocaleString() : '-'}</td>
            </tr>
          \`;
        }

        bodyContent = \`
          <div style="margin-bottom: 8px; display: flex; justify-content: space-between; align-items: flex-end;">
            <div>
              <div class="psf-val">S$ \${latestPsf.toLocaleString()} psf</div>
              <div class="psf-sub">Latest Transacted PSF (\${latestDate})</div>
            </div>
            <div style="text-align: right; font-size: 0.74rem; color: #94a3b8;">
              District Ref: <span style="color: #cbd5e1;">S$ \${p.basePsf.toLocaleString()} psf</span>
            </div>
          </div>
          <div style="font-size: 0.8rem; font-weight: 700; margin-bottom: 6px; color: #f8fafc; display: flex; justify-content: space-between;">
            <span>Recent Resale Caveats (\${p.tx.length}):</span>
            <span style="font-size: 0.72rem; color: #94a3b8; font-weight: normal;">Max ${MAX_TX_PER_PROJECT}</span>
          </div>
          <div class="table-scroll-wrap">
            <table class="price-table">
              <thead>
                <tr>
                  <th>Month</th>
                  <th>Floor</th>
                  <th>Area</th>
                  <th>Price</th>
                  <th>PSF</th>
                </tr>
              </thead>
              <tbody>
                \${rowsHtml}
              </tbody>
            </table>
          </div>
          <div style="font-size: 0.71rem; color: #64748b; margin-top: 6px; line-height: 1.35; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 6px;">
            ℹ️ <em>Official URA private residential caveats (Singapore Property Transactions Dataset, CC BY 4.0).</em>
          </div>
        \`;
      } else {
        const p2BedMin = Math.round((psf * 0.95) * 807);
        const p2BedMax = Math.round((psf * 1.05) * 807);
        const p3BedMin = Math.round((psf * 0.95) * 1130);
        const p3BedMax = Math.round((psf * 1.05) * 1130);
        const p4BedMin = Math.round((psf * 0.95) * 1453);
        const p4BedMax = Math.round((psf * 1.05) * 1453);

        bodyContent = \`
          <div style="margin-bottom: 10px;">
            <div class="psf-val">S$ \${psf.toLocaleString()} psf</div>
            <div class="psf-sub">Official URA District Resale Benchmark</div>
          </div>
          <table class="price-table">
            <thead>
              <tr><th>Unit Type</th><th>Approx Size</th><th>Est. Resale Range</th></tr>
            </thead>
            <tbody>
              <tr><td>2-Bedroom</td><td>75 sqm (807 sqft)</td><td class="price-green">S$ \${(p2BedMin/1000000).toFixed(2)}M - S$ \${(p2BedMax/1000000).toFixed(2)}M</td></tr>
              <tr><td>3-Bedroom</td><td>105 sqm (1,130 sqft)</td><td class="price-green">S$ \${(p3BedMin/1000000).toFixed(2)}M - S$ \${(p3BedMax/1000000).toFixed(2)}M</td></tr>
              <tr><td>4-Bedroom</td><td>135 sqm (1,453 sqft)</td><td class="price-green">S$ \${(p4BedMin/1000000).toFixed(2)}M - S$ \${(p4BedMax/1000000).toFixed(2)}M</td></tr>
            </tbody>
          </table>
          <div style="font-size: 0.71rem; color: #64748b; margin-top: 8px; line-height: 1.35; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 6px;">
            ℹ️ <em>Showing official URA district benchmark valuation (no caveat recorded in recent 5-year window).</em>
          </div>
        \`;
      }

      const tenureBadge = p.tenure ? \`<span class="badge badge-tenure">📜 \${p.tenure}</span>\` : '';

      let strataStatus = '';
      if (p.mcstNo && p.mcstYear) {
        strataStatus = \`MCST #\${p.mcstNo} (Constituted \${p.mcstYear})\`;
      } else if (p.mcstNo) {
        strataStatus = \`MCST #\${p.mcstNo}\`;
      } else if (p.leaseYear) {
        strataStatus = \`Lease Commenced \${p.leaseYear}\`;
      } else {
        strataStatus = \`<span style="color: #64748b; font-weight: 500;">Single-Owner / Non-Strata</span>\`;
      }

      let unitsLabel = 'Private Residential';
      if (p.units > 0) {
        unitsLabel = p.units <= 20 ? \`\${p.units} Units (Boutique)\` : \`\${p.units.toLocaleString()} Units\`;
      }

      return \`
        <div class="popup-box">
          <div class="popup-title">\${p.name}</div>
          <div class="popup-badge-row">
            <span class="badge badge-\${p.segment.toLowerCase()}">\${p.segment}</span>
            <span class="badge badge-dist">District D\${p.district}</span>
            <span class="badge badge-type">\${p.propType}</span>
            \${tenureBadge}
          </div>
          <div class="popup-meta-box">
            <div class="meta-row">
              <span class="meta-lbl">📍 Postal</span>
              <span class="meta-val">\${p.postal}</span>
            </div>
            <div class="meta-row">
              <span class="meta-lbl">🏢 Scale</span>
              <span class="meta-val">\${unitsLabel}</span>
            </div>
            <div class="meta-row">
              <span class="meta-lbl">🏛️ BCA Strata</span>
              <span class="meta-val">\${strataStatus}</span>
            </div>
            <div class="meta-row">
              <span class="meta-lbl">🏙️ Area</span>
              <span class="meta-val">\${p.districtName}</span>
            </div>
          </div>
          \${bodyContent}
        </div>
      \`;
    }

    // Populate markers
    function populateMap(filterSegment = 'ALL') {
      clusterGroup.clearLayers();
      markers.length = 0;

      for (let i = 0; i < CONDO_PROJECTS.length; i++) {
        const p = CONDO_PROJECTS[i];
        if (filterSegment !== 'ALL' && p.segment !== filterSegment) {
          continue;
        }

        const color = getColorForSegment(p.segment);
        const marker = L.circleMarker([p.lat, p.lon], {
          className: 'condo-pin-marker',
          radius: 8.5,
          fillColor: color,
          color: '#ffffff',
          weight: 2.5,
          opacity: 1.0,
          fillOpacity: 0.98
        });

        marker.bindPopup(createPopupContent(p), { maxWidth: 460, minWidth: 320 });
        const tooltipPsf = (p.tx && p.tx.length > 0 && p.tx[0].psf) ? p.tx[0].psf : p.basePsf;
        marker.bindTooltip(\`<b>\${p.name}</b> (D\${p.district} • \${p.segment})<br>S$ \${tooltipPsf.toLocaleString()} psf\`, { sticky: true });

        marker.projectData = p;
        markers.push(marker);
        clusterGroup.addLayer(marker);
      }

      map.addLayer(clusterGroup);
    }

    populateMap('ALL');

    // Filter Buttons
    document.querySelectorAll('.filter-pill').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.filter-pill').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        const filter = e.target.getAttribute('data-filter');
        populateMap(filter);
      });
    });

    // Search Logic
    const searchInput = document.getElementById('projectSearch');
    searchInput.addEventListener('input', (e) => {
      const q = e.target.value.trim().toLowerCase();
      if (!q) return;

      const matched = markers.find(m => {
        const p = m.projectData;
        return p.name.toLowerCase().includes(q) || p.postal.includes(q);
      });

      if (matched) {
        map.setView([matched.projectData.lat, matched.projectData.lon], 16);
        matched.openPopup();
      }
    });
  </script>
</body>
</html>`;

const localMapPath = path.join(__dirname, 'condo_resale_price_map_clickable.html');
fs.writeFileSync(localMapPath, html, 'utf8');
console.log(`Successfully generated:\n  - Output: ${localMapPath} (${(fs.statSync(localMapPath).size / (1024 * 1024)).toFixed(2)} MB)`);
