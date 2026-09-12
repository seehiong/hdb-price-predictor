// pipeline_public/04_build_map.js
// Builds interactive standalone Leaflet map for Singapore HDB Resale Flats
// Incorporates real Data.gov.sg resale flat transactions (recent caveats per block)
const fs = require('fs');
const path = require('path');

// How many recent transactions to embed per block. Each one costs about 26
// bytes in the payload, so this is the main lever on map size: 20 gives a
// 6.0 MB page, 50 gives 7.8 MB. Beyond 50 there is little left to gain --
// only 6% of blocks have more than 50 transactions in the dataset.
const MAX_TX_PER_BLOCK = 50;
const readline = require('readline');

const TOWN_TO_REGION = {
  'CENTRAL AREA': 'Central',
  'BISHAN': 'Central',
  'BUKIT MERAH': 'Central',
  'GEYLANG': 'Central',
  'KALLANG/WHAMPOA': 'Central',
  'MARINE PARADE': 'Central',
  'QUEENSTOWN': 'Central',
  'TOA PAYOH': 'Central',
  'BUKIT TIMAH': 'Central',
  'BEDOK': 'East',
  'PASIR RIS': 'East',
  'TAMPINES': 'East',
  'SEMBAWANG': 'North',
  'WOODLANDS': 'North',
  'YISHUN': 'North',
  'ANG MO KIO': 'North-East',
  'HOUGANG': 'North-East',
  'PUNGGOL': 'North-East',
  'SENGKANG': 'North-East',
  'SERANGOON': 'North-East',
  'BUKIT BATOK': 'West',
  'BUKIT PANJANG': 'West',
  'CHOA CHU KANG': 'West',
  'CLEMENTI': 'West',
  'JURONG EAST': 'West',
  'JURONG WEST': 'West'
};

const REGION_COLORS = {
  'Central': '#e11d48',    // Vivid Crimson Rose
  'East': '#2563eb',       // Vivid Deep Cobalt Blue
  'North': '#059669',      // Deep Vivid Emerald (contrasts with OSM light park green)
  'North-East': '#7c3aed', // Electric Royal Violet
  'West': '#ea580c'        // Vivid Flame Orange
};

async function buildHdbMap() {
  console.log('=== Building Interactive HDB Resale Map ===');

  const propFile = path.join(__dirname, 'data', 'hdb_property_info.csv');
  const resaleFile = path.join(__dirname, 'data', 'resale_flat_prices_2017_present.csv');

  if (!fs.existsSync(propFile)) {
    console.error(`Error: Property info file not found at ${propFile}`);
    console.error('Run: python pipeline_public/01_fetch_data.py');
    process.exit(1);
  }

  // The map needs per-block coordinates, which the published Data.gov.sg
  // dataset does not carry -- it stops at 24 columns. The working copy is a
  // geocoded extension of it (34 columns, lon/lat at 26/27). Detect the plain
  // version up front rather than emitting an empty map with zero blocks on it.
  const propHeader = fs.readFileSync(propFile, 'utf8').split('\n')[0] || '';
  const propCols = propHeader.split(',').length;
  if (propCols < 28) {
    console.error(`Error: ${path.basename(propFile)} has ${propCols} columns; ` +
                  'the map needs a geocoded copy with longitude and latitude ' +
                  '(34 columns).');
    console.error('The Data.gov.sg dataset ships without coordinates, so this ' +
                  'file has to be geocoded before the map can be built.');
    console.error('Everything else -- training, the web app, the private map -- ' +
                  'works without it.');
    process.exit(1);
  }
  if (!fs.existsSync(resaleFile)) {
    console.error(`Error: Resale transactions file not found at ${resaleFile}`);
    process.exit(1);
  }

  // 1. Read HDB Property Information (Coordinates & Metadata)
  console.log(`Reading property information from ${propFile}...`);
  const propMap = new Map();
  const fileStream = fs.createReadStream(propFile, { encoding: 'utf8' });
  const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

  let isFirstLine = true;
  for await (const line of rl) {
    if (!line.trim()) continue;
    if (isFirstLine) {
      isFirstLine = false;
      continue;
    }

    // CSV line parser handling quotes
    const parts = [];
    let cur = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (c === '"') {
        inQuotes = !inQuotes;
      } else if (c === ',' && !inQuotes) {
        parts.push(cur);
        cur = '';
      } else {
        cur += c;
      }
    }
    parts.push(cur);

    const blk = parts[0]?.trim().toUpperCase();
    const street = parts[1]?.trim().toUpperCase();
    const maxFloor = parseInt(parts[2], 10) || 0;
    const year = parseInt(parts[3], 10) || 0;
    const units = parseInt(parts[11], 10) || 0;
    const postal = parts[24]?.trim() || '';
    const lon = parseFloat(parts[26]);
    const lat = parseFloat(parts[27]);
    const canonical = parts[33]?.trim().toUpperCase();

    if (blk && lat && lon) {
      const entry = {
        blk,
        street,
        postal,
        year,
        maxFloor,
        units,
        lat: Math.round(lat * 100000) / 100000,
        lon: Math.round(lon * 100000) / 100000
      };
      if (street) propMap.set(blk + '|' + street, entry);
      if (canonical && canonical !== street) {
        propMap.set(blk + '|' + canonical, { ...entry, street: canonical });
      }
    }
  }
  console.log(`Loaded ${propMap.size} block coordinate references.`);

  // 2. Read Resale Flat Transactions
  console.log(`Reading resale transactions from ${resaleFile}...`);
  const rStream = fs.createReadStream(resaleFile, { encoding: 'utf8' });
  const rRl = readline.createInterface({ input: rStream, crlfDelay: Infinity });

  const typeList = [];
  const typeMap = new Map();
  function getTypeIdx(t) {
    let idx = typeMap.get(t);
    if (idx === undefined) {
      idx = typeList.length;
      typeList.push(t);
      typeMap.set(t, idx);
    }
    return idx;
  }

  const modelList = [];
  const modelMap = new Map();
  function getModelIdx(m) {
    let idx = modelMap.get(m);
    if (idx === undefined) {
      idx = modelList.length;
      modelList.push(m);
      modelMap.set(m, idx);
    }
    return idx;
  }

  const blockMap = new Map();
  let rFirstLine = true;
  let txCount = 0;

  for await (const line of rRl) {
    if (!line.trim()) continue;
    if (rFirstLine) {
      rFirstLine = false;
      continue;
    }

    const parts = line.split(',');
    if (parts.length < 11) continue;

    const month = parts[0].trim();
    const town = parts[1].trim().toUpperCase();
    const flatType = parts[2].trim().toUpperCase();
    const blk = parts[3].trim().toUpperCase();
    const street = parts[4].trim().toUpperCase();
    const area = Math.round(parseFloat(parts[6]) || 0);
    const model = parts[7].trim();
    const lcd = parseInt(parts[8], 10) || 0;
    const price = Math.round(parseFloat(parts[10]) || 0);

    const key = blk + '|' + street;
    let blockObj = blockMap.get(key);
    if (!blockObj) {
      const prop = propMap.get(key);
      if (!prop) continue;

      const region = TOWN_TO_REGION[town] || 'Central';
      blockObj = {
        b: blk,
        s: street,
        t: town,
        r: region,
        p: prop.postal,
        y: prop.year || lcd,
        f: prop.maxFloor,
        u: prop.units,
        lat: prop.lat,
        lon: prop.lon,
        tx: []
      };
      blockMap.set(key, blockObj);
    }

    blockObj.tx.push([
      month,
      getTypeIdx(flatType),
      area,
      getModelIdx(model),
      price
    ]);
    txCount++;
  }

  console.log(`Processed ${txCount.toLocaleString()} total resale transactions.`);

  // 3. Process each block: sort transactions descending, cap the list, calculate summary metrics
  const allBlocks = [];
  for (const blockObj of blockMap.values()) {
    blockObj.tx.sort((a, b) => b[0].localeCompare(a[0]));
    blockObj.tx = blockObj.tx.slice(0, MAX_TX_PER_BLOCK);

    let sum = 0;
    for (let i = 0; i < blockObj.tx.length; i++) {
      sum += blockObj.tx[i][4];
    }
    blockObj.avg = blockObj.tx.length > 0 ? Math.round(sum / blockObj.tx.length) : 0;
    blockObj.latest = blockObj.tx.length > 0 ? blockObj.tx[0][4] : 0;
    blockObj.latestDate = blockObj.tx.length > 0 ? blockObj.tx[0][0] : '';
    blockObj.latestType = blockObj.tx.length > 0 ? blockObj.tx[0][1] : 0;

    allBlocks.push(blockObj);
  }

  // Sort blocks alphabetically by town, street, block
  allBlocks.sort((a, b) => {
    if (a.t !== b.t) return a.t.localeCompare(b.t);
    if (a.s !== b.s) return a.s.localeCompare(b.s);
    return a.b.localeCompare(b.b, undefined, { numeric: true });
  });

  console.log(`Compiled ${allBlocks.length.toLocaleString()} unique HDB blocks with coordinates.`);

  const payload = {
    types: typeList,
    models: modelList,
    blocks: allBlocks
  };

  const payloadJson = JSON.stringify(payload);
  console.log(`Embedded payload size: ${(Buffer.byteLength(payloadJson, 'utf8') / (1024 * 1024)).toFixed(2)} MB`);

  // 4. Generate Interactive Standalone HTML Leaflet Map
  console.log('Generating Leaflet HTML map...');
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Singapore HDB Resale Flats Interactive Map</title>
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
      z-index: 1000; background: rgba(15, 23, 42, 0.9);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 9999px;
      padding: 6px 14px;
      display: flex; align-items: center; gap: 10px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
      max-width: 95vw;
    }
    .map-header-title {
      font-size: 0.92rem; font-weight: 700; color: #f8fafc;
      display: flex; align-items: center; gap: 6px; white-space: nowrap;
    }
    .search-box-wrap {
      position: relative;
      display: flex; align-items: center;
    }
    .search-input {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 9999px;
      padding: 6px 14px; color: #f8fafc; font-size: 0.85rem;
      outline: none; width: 220px; transition: all 0.2s ease;
      font-family: inherit;
    }
    .search-input:focus {
      width: 280px; border-color: #38bdf8;
      background: rgba(255, 255, 255, 0.14);
      box-shadow: 0 0 12px rgba(56, 189, 248, 0.25);
    }
    .search-input::placeholder { color: #94a3b8; }
    .town-select {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.18);
      color: #f8fafc; border-radius: 9999px;
      padding: 5px 10px; font-size: 0.8rem; font-weight: 600;
      outline: none; cursor: pointer; font-family: inherit;
    }
    .town-select option {
      background: #1e293b; color: #f8fafc;
    }
    /* Region Filter Buttons */
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
    .legend-row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; cursor: pointer; }
    .legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    .dot-central { background: #e11d48; box-shadow: 0 0 8px rgba(225, 29, 72, 0.7); }
    .dot-east { background: #2563eb; box-shadow: 0 0 8px rgba(37, 99, 235, 0.7); }
    .dot-north { background: #059669; box-shadow: 0 0 8px rgba(5, 150, 105, 0.7); }
    .dot-northeast { background: #7c3aed; box-shadow: 0 0 8px rgba(124, 58, 237, 0.7); }
    .dot-west { background: #ea580c; box-shadow: 0 0 8px rgba(234, 88, 12, 0.7); }
    /* Popup Styling */
    .leaflet-popup-content-wrapper {
      background: #0f172a; color: #f8fafc; border-radius: 14px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      box-shadow: 0 16px 36px rgba(0, 0, 0, 0.55);
      padding: 2px;
    }
    .leaflet-popup-tip { background: #0f172a; }
    .popup-box { padding: 12px; min-width: 320px; max-width: 440px; }
    .popup-title { font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
    .popup-badge-row { display: flex; gap: 6px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }
    .badge {
      font-size: 0.72rem; font-weight: 700; padding: 2px 7px; border-radius: 9999px;
      letter-spacing: 0.4px;
    }
    .badge-central { background: rgba(225, 29, 72, 0.2); color: #fb7185; border: 1px solid rgba(225, 29, 72, 0.4); }
    .badge-east { background: rgba(37, 99, 235, 0.2); color: #60a5fa; border: 1px solid rgba(37, 99, 235, 0.4); }
    .badge-north { background: rgba(5, 150, 105, 0.2); color: #34d399; border: 1px solid rgba(5, 150, 105, 0.4); }
    .badge-northeast { background: rgba(124, 58, 237, 0.2); color: #c084fc; border: 1px solid rgba(124, 58, 237, 0.4); }
    .badge-west { background: rgba(234, 88, 12, 0.2); color: #fb923c; border: 1px solid rgba(234, 88, 12, 0.4); }
    .badge-meta { background: rgba(255, 255, 255, 0.08); color: #cbd5e1; border: 1px solid rgba(255, 255, 255, 0.15); }
    .popup-meta {
      font-size: 0.8rem; color: #94a3b8; margin-bottom: 8px; line-height: 1.4;
      display: grid; grid-template-columns: 1fr 1fr; gap: 4px;
      background: rgba(255, 255, 255, 0.03); padding: 8px; border-radius: 8px;
    }
    .popup-stat-bar {
      display: flex; justify-content: space-between; align-items: baseline;
      margin-bottom: 8px; padding: 6px 10px; background: rgba(56, 189, 248, 0.08);
      border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px;
    }
    .popup-stat-label { font-size: 0.75rem; color: #94a3b8; }
    .popup-stat-val { font-size: 0.98rem; font-weight: 700; color: #38bdf8; }
    .table-scroll-wrap {
      max-height: 200px;
      overflow-y: auto;
      margin-bottom: 6px;
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
    /* Prominent Marker Styling on OpenStreetMap */
    path.hdb-pin-marker {
      filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.75)) drop-shadow(0 0 1.5px rgba(0, 0, 0, 0.95));
      transition: r 0.15s ease, stroke-width 0.15s ease, filter 0.15s ease;
      cursor: pointer;
    }
    path.hdb-pin-marker:hover {
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
    /* Tooltip */
    .leaflet-tooltip {
      background: rgba(15, 23, 42, 0.92);
      border: 1px solid rgba(255, 255, 255, 0.16);
      color: #f8fafc; font-size: 0.8rem; border-radius: 8px;
      padding: 6px 10px; box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
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
  <!-- Floating Header Bar -->
  <div class="map-header-bar">
    <div class="map-header-title">🏢 <span>HDB Resale Flats</span></div>
    <div class="search-box-wrap">
      <input type="text" id="blockSearch" class="search-input" placeholder="Search block, street, postal...">
    </div>
    <select id="townSelect" class="town-select" title="Filter by Town">
      <option value="ALL">All Towns (26)</option>
    </select>
    <div class="filter-pills">
      <button class="filter-pill active" data-filter="ALL">All (${(allBlocks.length / 1000).toFixed(1)}k)</button>
      <button class="filter-pill" data-filter="Central">Central</button>
      <button class="filter-pill" data-filter="East">East</button>
      <button class="filter-pill" data-filter="North">North</button>
      <button class="filter-pill" data-filter="North-East">N-East</button>
      <button class="filter-pill" data-filter="West">West</button>
    </div>
  </div>

  <!-- Map Legend -->
  <div class="map-legend">
    <div class="legend-title">Region (HDB Planning)</div>
    <div class="legend-row" data-region="Central"><span class="legend-dot dot-central"></span> Central Region</div>
    <div class="legend-row" data-region="East"><span class="legend-dot dot-east"></span> East Region</div>
    <div class="legend-row" data-region="North"><span class="legend-dot dot-north"></span> North Region</div>
    <div class="legend-row" data-region="North-East"><span class="legend-dot dot-northeast"></span> North-East Region</div>
    <div class="legend-row" data-region="West"><span class="legend-dot dot-west"></span> West Region</div>
  </div>

  <!-- Map Container -->
  <div id="map"></div>

  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
  <script>
    // Embedded HDB Resale Data Payload
    const HDB_DATA = ${payloadJson};
    const TYPES = HDB_DATA.types;
    const MODELS = HDB_DATA.models;
    const BLOCKS = HDB_DATA.blocks;

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
    const clusterGroup = L.markerClusterGroup({
      chunkedLoading: true,
      maxClusterRadius: 40,
      spiderfyOnMaxZoom: true,
      showCoverageOnHover: false
    });

    const REGION_COLORS = {
      'Central': '#e11d48',
      'East': '#2563eb',
      'North': '#059669',
      'North-East': '#7c3aed',
      'West': '#ea580c'
    };

    function getColorForRegion(r) {
      return REGION_COLORS[r] || '#3b82f6';
    }

    function formatMoney(num) {
      if (!num) return 'N/A';
      return 'S$ ' + num.toString().replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');
    }

    function createPopupContent(b) {
      const regKey = b.r.toLowerCase().replace(/[^a-z]/g, '');
      const badgeClass = 'badge-' + regKey;

      let rowsHtml = '';
      if (b.tx && b.tx.length > 0) {
        for (let i = 0; i < b.tx.length; i++) {
          const t = b.tx[i];
          const m = t[0];
          const fType = TYPES[t[1]] || '-';
          const area = t[2] + ' sqm';
          const model = MODELS[t[3]] || '-';
          const price = formatMoney(t[4]);

          rowsHtml += \`<tr>
            <td>\${m}</td>
            <td>\${fType}</td>
            <td>\${area}</td>
            <td>\${model}</td>
            <td class="price-green">\${price}</td>
          </tr>\`;
        }
      } else {
        rowsHtml = '<tr><td colspan="5" style="text-align:center; color:#94a3b8; padding: 12px;">No resale transactions recorded since 2017</td></tr>';
      }

      return \`
        <div class="popup-box">
          <div class="popup-title">Blk \${b.b} \${b.s}</div>
          <div class="popup-badge-row">
            <span class="badge \${badgeClass}">\${b.t} (\${b.r})</span>
            <span class="badge badge-meta">Postal: \${b.p || 'N/A'}</span>
            <span class="badge badge-meta">Built: \${b.y || 'N/A'}</span>
            \${b.f ? \`<span class="badge badge-meta">Max Flr: \${b.f}</span>\` : ''}
            \${b.u ? \`<span class="badge badge-meta">Units: \${b.u}</span>\` : ''}
          </div>

          <div class="popup-stat-bar">
            <div>
              <div class="popup-stat-label">Latest Resale (\${b.latestDate || 'Recent'})</div>
              <div class="popup-stat-val">\${formatMoney(b.latest)}</div>
            </div>
            <div style="text-align: right;">
              <div class="popup-stat-label">Avg Recent Resale</div>
              <div class="popup-stat-val" style="color:#a78bfa;">\${formatMoney(b.avg)}</div>
            </div>
          </div>

          <div style="font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px; letter-spacing: 0.5px;">
            Recent Resale Transactions (Max ${MAX_TX_PER_BLOCK})
          </div>
          <div class="table-scroll-wrap">
            <table class="price-table">
              <thead>
                <tr>
                  <th>Month</th>
                  <th>Type</th>
                  <th>Area</th>
                  <th>Model</th>
                  <th>Resale Price</th>
                </tr>
              </thead>
              <tbody>
                \${rowsHtml}
              </tbody>
            </table>
          </div>
        </div>
      \`;
    }

    const markers = [];

    // Populate Town select dropdown options
    const uniqueTowns = [...new Set(BLOCKS.map(b => b.t))].sort();
    const townSelect = document.getElementById('townSelect');
    uniqueTowns.forEach(town => {
      const opt = document.createElement('option');
      opt.value = town;
      opt.textContent = town;
      townSelect.appendChild(opt);
    });

    function populateMap(regionFilter = 'ALL', townFilter = 'ALL') {
      clusterGroup.clearLayers();
      markers.length = 0;

      for (let i = 0; i < BLOCKS.length; i++) {
        const b = BLOCKS[i];
        if (regionFilter !== 'ALL' && b.r !== regionFilter) continue;
        if (townFilter !== 'ALL' && b.t !== townFilter) continue;

        const color = getColorForRegion(b.r);
        const marker = L.circleMarker([b.lat, b.lon], {
          className: 'hdb-pin-marker',
          radius: 8.5,
          fillColor: color,
          color: '#ffffff',
          weight: 2.5,
          opacity: 1.0,
          fillOpacity: 0.98
        });

        marker.bindPopup(() => createPopupContent(b), { maxWidth: 460, minWidth: 320 });

        const latestInfo = b.latest ? \`<br><span style="color:#34d399; font-weight:600;">Latest: \${formatMoney(b.latest)} (\${b.latestDate})</span>\` : '';
        marker.bindTooltip(\`<b>Blk \${b.b} \${b.s}</b> (\${b.p})<br>\${b.t} • \${b.r}\${latestInfo}\`, { sticky: true });

        marker.blockData = b;
        markers.push(marker);
        clusterGroup.addLayer(marker);
      }

      map.addLayer(clusterGroup);
    }

    populateMap('ALL', 'ALL');

    // Region Filter Buttons
    document.querySelectorAll('.filter-pill').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.filter-pill').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        const reg = e.target.getAttribute('data-filter');
        townSelect.value = 'ALL';
        populateMap(reg, 'ALL');
      });
    });

    // Town Select Dropdown
    townSelect.addEventListener('change', (e) => {
      const town = e.target.value;
      document.querySelectorAll('.filter-pill').forEach(b => b.classList.remove('active'));
      const allBtn = document.querySelector('.filter-pill[data-filter="ALL"]');
      if (allBtn) allBtn.classList.add('active');

      populateMap('ALL', town);

      // Auto center map on town if a specific town is selected
      if (town !== 'ALL') {
        const townBlocks = BLOCKS.filter(b => b.t === town);
        if (townBlocks.length > 0) {
          const lats = townBlocks.map(b => b.lat);
          const lons = townBlocks.map(b => b.lon);
          const minLat = Math.min(...lats), maxLat = Math.max(...lats);
          const minLon = Math.min(...lons), maxLon = Math.max(...lons);
          map.fitBounds([[minLat, minLon], [maxLat, maxLon]], { padding: [40, 40] });
        }
      }
    });

    // Legend Click Filter
    document.querySelectorAll('.legend-row').forEach(row => {
      row.addEventListener('click', (e) => {
        const reg = row.getAttribute('data-region');
        document.querySelectorAll('.filter-pill').forEach(b => {
          if (b.getAttribute('data-filter') === reg) {
            b.click();
          }
        });
      });
    });

    // Search Logic (Block, Street, Postal)
    const searchInput = document.getElementById('blockSearch');
    function performSearch() {
      const q = searchInput.value.trim().toLowerCase();
      if (!q) return;

      // Match block + street, street, or postal
      let matched = markers.find(m => {
        const b = m.blockData;
        const fullAddr = (b.b + ' ' + b.s).toLowerCase();
        return b.p === q || fullAddr.includes(q);
      });

      if (!matched) {
        // Try searching in all blocks even if filtered out
        const blockFound = BLOCKS.find(b => {
          const fullAddr = (b.b + ' ' + b.s).toLowerCase();
          return b.p === q || fullAddr.includes(q) || b.s.toLowerCase().includes(q);
        });

        if (blockFound) {
          // Reset filters to ALL to reveal marker
          document.querySelectorAll('.filter-pill').forEach(btn => btn.classList.remove('active'));
          const allBtn = document.querySelector('.filter-pill[data-filter="ALL"]');
          if (allBtn) allBtn.classList.add('active');
          townSelect.value = 'ALL';
          populateMap('ALL', 'ALL');

          matched = markers.find(m => m.blockData.b === blockFound.b && m.blockData.s === blockFound.s);
        }
      }

      if (matched) {
        map.setView([matched.blockData.lat, matched.blockData.lon], 17);
        setTimeout(() => {
          matched.openPopup();
        }, 300);
      }
    }

    searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        performSearch();
      }
    });

    searchInput.addEventListener('input', (e) => {
      const val = e.target.value.trim();
      if (val.length >= 5 && /^[0-9]+$/.test(val)) {
        performSearch();
      }
    });
  </script>
</body>
</html>`;

  // Write file to pipeline_public/
  const mapPath = path.join(__dirname, 'hdb_resale_price_map_clickable.html');
  fs.writeFileSync(mapPath, html, 'utf8');

  console.log(`\nSuccessfully generated:`);
  console.log(`  - Output: ${mapPath} (${(fs.statSync(mapPath).size / (1024 * 1024)).toFixed(2)} MB)`);
  console.log('=== HDB Map Generation Complete ===\n');
}

if (require.main === module) {
  buildHdbMap().catch(err => {
    console.error('Error generating HDB map:', err);
    process.exit(1);
  });
}

module.exports = { buildHdbMap };
