import * as THREE from 'three';
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry.js';
import { AUTH_TOKEN, readApiError } from './api.js';
import {
  VERT,
  CORE_FRAG,
  HAZE_FRAG,
  GLOW_FRAG,
  NEBULA_FRAG,
  TRAIL_VERT,
  TRAIL_BLEND_FRAG,
  TRAIL_PRESENT_FRAG,
} from './shaders.js';

if (AUTH_TOKEN) {
  const baseFetch = window.fetch.bind(window);
  window.fetch = (input, init) => {
    const options = init ? { ...init } : {};
    const headers = new Headers(options.headers || {});
    if (!headers.has('Authorization')) headers.set('Authorization', `Bearer ${AUTH_TOKEN}`);
    options.headers = headers;
    return baseFetch(input, options);
  };
}

// ── API access token + error envelope helpers ─────────────────────────────────

// ── OrbitControls (inline, r128 compatible) ───────────────────────────────────
class OrbitControls {
  constructor(camera, domElement) {
    this.camera = camera;
    this.domElement = domElement;
    this.target = new THREE.Vector3();
    this.enableDamping = true;
    this.dampingFactor = 0.05;
    this.rotateSpeed = 0.5;
    this.zoomSpeed = 1.2;
    this.panSpeed = 0.8;
    this.autoRotate = true;
    this.autoRotateSpeed = 0.05;

    this._spherical = new THREE.Spherical();
    this._sphericalDelta = new THREE.Spherical();
    this._scale = 1;
    this._panOffset = new THREE.Vector3();
    this._dragging = false;
    this._button = -1;
    this._lastPos = { x: 0, y: 0 };

    this._updateSpherical();
    this._bind();
  }

  _updateSpherical() {
    const offset = this.camera.position.clone().sub(this.target);
    this._spherical.setFromVector3(offset);
  }

  _bind() {
    const el = this.domElement;
    el.addEventListener('mousedown',  e => this._onMouseDown(e));
    el.addEventListener('mousemove',  e => this._onMouseMove(e));
    el.addEventListener('mouseup',    () => { this._dragging = false; });
    el.addEventListener('wheel',      e => this._onWheel(e), { passive: false });
    el.addEventListener('contextmenu', e => e.preventDefault());
  }

  _onMouseDown(e) {
    this._dragging = true;
    this._button = e.button;
    this._lastPos = { x: e.clientX, y: e.clientY };
  }

  _onMouseMove(e) {
    if (!this._dragging) return;
    const dx = e.clientX - this._lastPos.x;
    const dy = e.clientY - this._lastPos.y;
    this._lastPos = { x: e.clientX, y: e.clientY };

    if (this._button === 0) {
      this._sphericalDelta.theta -= (2 * Math.PI * dx / this.domElement.clientWidth) * this.rotateSpeed;
      this._sphericalDelta.phi   -= (2 * Math.PI * dy / this.domElement.clientHeight) * this.rotateSpeed;
    } else if (this._button === 2) {
      this._pan(dx, dy);
    }
  }

  _pan(dx, dy) {
    const el = this.domElement;
    const dist = this.camera.position.distanceTo(this.target);
    const fovRad = this.camera.fov * Math.PI / 180;
    const heightAtDist = 2 * dist * Math.tan(fovRad / 2);
    const px = (this.panSpeed * dx * heightAtDist) / el.clientHeight;
    const py = (this.panSpeed * dy * heightAtDist) / el.clientHeight;
    const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0);
    const up    = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1);
    this._panOffset.addScaledVector(right, -px);
    this._panOffset.addScaledVector(up, py);
  }

  _onWheel(e) {
    e.preventDefault();
    if (e.deltaY > 0) this._scale /= Math.pow(0.95, this.zoomSpeed);
    else              this._scale *= Math.pow(0.95, this.zoomSpeed);
  }

  update() {
    if (this.autoRotate && !this._dragging) {
      this._sphericalDelta.theta -= (2 * Math.PI / 60 / 60) * this.autoRotateSpeed;
    }

    this._spherical.theta += this._sphericalDelta.theta;
    this._spherical.phi   += this._sphericalDelta.phi;
    this._spherical.phi    = Math.max(0.05, Math.min(Math.PI - 0.05, this._spherical.phi));
    this._spherical.radius *= this._scale;
    this._spherical.radius  = Math.max(10, Math.min(2000, this._spherical.radius));

    this.target.add(this._panOffset);

    const offset = new THREE.Vector3().setFromSpherical(this._spherical);
    this.camera.position.copy(this.target).add(offset);
    this.camera.lookAt(this.target);

    if (this.enableDamping) {
      this._sphericalDelta.theta *= (1 - this.dampingFactor);
      this._sphericalDelta.phi   *= (1 - this.dampingFactor);
      this._panOffset.multiplyScalar(1 - this.dampingFactor);
    } else {
      this._sphericalDelta.set(0, 0, 0);
      this._panOffset.set(0, 0, 0);
    }
    this._scale = 1;
  }

  setAutoRotateSpeed(s) { this.autoRotateSpeed = s; }
}

// ── State ─────────────────────────────────────────────────────────────────────
let scene, camera, renderer, controls;
let sceneRenderTarget = null;
let trailRenderTargets = [null, null];
let trailReadIndex = 0;
let trailBlendScene = null;
let trailBlendCamera = null;
let trailBlendMaterial = null;
let trailPresentScene = null;
let trailPresentCamera = null;
let trailPresentMaterial = null;
let viewCubeScene = null;
let viewCubeCamera = null;
let viewCubeRenderer = null;
let viewCubeGroup = null;
let viewCubePickMesh = null;
let viewCubeCornerPickMeshes = [];
let particleSystem = null;
let hazeSystem = null;
let glowSystem = null;
let nebulaSystem = null;
let clusterHullsGroup = null;
let gridHelper = null;
let allPoints = [];
let currentCollection = 'meistro_brain';
let projectionMethod = 'pca';
let searchMode = 'keyword';
let crossCollectionTarget = '';
let selectedVectorName = '';
let pointLimit = 2000;
let pointSize = 6.0;
let sizeField = 'none';
let colorField = 'source';
let colorFieldCustom = '';
let coreOpacity = 0.85;
let hazeOpacity = 0.55;
let bloomStrength = 2.5;
let rotSpeed = 0.05;
let trailPersistence = 0.90;
let timelineReveal = 1.0;
let timelineAutoPlay = false;
let hueShift = 0;
let saturationMult = 1;
let lightnessMult = 1;
let activeTheme = 'deep-space';
let pointClusterIndices = [];
let activeClusterMap = null;
let pointIndexByID = new Map();
let collectionMetaByName = new Map();
let outlierFlags = null;
let outlierCount = 0;
let outliersOnly = false;
let basePointColors = null;
let basePointSizes = null;
let selectedPoint = null;
let selectedPointID = '';
let similarityModeActive = false;
let similarityRadius = 0.92;
let hoveredSimilarPointID = '';
let poppedSimilarPointID = '';
let scanNeighborIDs = new Set();
let scanResults = [];
let currentLoadCollectionLabel = '';
let currentLoadProgressID = '';
let currentProjectionMeta = null;
let uiEventSource = null;
let uiEventSourceCollection = '';
let cameraFlight = null;
let hoverDirty = true;
let lastHoverCheck = 0;
let lowFpsFrames = 0;
let qualityReduced = false;
let colorRefreshQueued = false;
let selectionHistory = [];
let selectionHistoryIndex = -1;
let bookmarks = [];
let lenses = [];
let pendingLens = null;
let suppressHistoryPush = false;
let lastHighlightEventID = '';
let liveIngestEnabled = false;
let liveIngestSource = null;
let liveIngestSocket = null;
let liveIngestReloadTimer = null;
let liveIngestPendingEvents = 0;
let liveIngestSyncInFlight = false;
let liveDropSystem = null;
const liveDropMax = 320;
const liveDropLife = 3.2;
const liveDrops = [];
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 3.0;
const mouse = new THREE.Vector2(-9999, -9999);
const viewCubeRaycaster = new THREE.Raycaster();
const viewCubeMouse = new THREE.Vector2();

const THEMES = {
  'deep-space': {
    css: { obsidian: '#050508', deep: '#080b12', panel: '#0a0d18cc', border: '#1a2a4a', neonC: '#00f5ff', neonM: '#ff00ff', neonG: '#39ff14', neonO: '#ff6600', text: '#c8d8f0', dim: '#4a6080' },
    scene: { background: 0x050508, fog: 0x050508, grid: 0x0a1428, stars: 0x334466 },
    palette: [0x00f5ff, 0xff00ff, 0x39ff14, 0xff6600, 0xffff00, 0x7b2fff, 0xff2d55, 0x00ff88, 0xff9500, 0x0099ff],
    outlier: 0xff3366,
  },
  bioluminescent: {
    css: { obsidian: '#020b0d', deep: '#041316', panel: '#07181dcc', border: '#15505c', neonC: '#38fff2', neonM: '#7cf7ff', neonG: '#8dff8b', neonO: '#58ffd2', text: '#cbfff8', dim: '#4d9f9b' },
    scene: { background: 0x020b0d, fog: 0x020b0d, grid: 0x0c2a30, stars: 0x2c6b66 },
    palette: [0x38fff2, 0x7cf7ff, 0x8dff8b, 0x58ffd2, 0xa4ffe6, 0x2ce6ff, 0x6affd4, 0x9fffe2, 0x43e8ff, 0x63ffad],
    outlier: 0xff8bd5,
  },
  amber: {
    css: { obsidian: '#120a04', deep: '#1a1207', panel: '#25170ccc', border: '#5b3a1e', neonC: '#ffbf47', neonM: '#ff7a2f', neonG: '#ffd35a', neonO: '#ff8f1f', text: '#f6ddb6', dim: '#9a7649' },
    scene: { background: 0x120a04, fog: 0x120a04, grid: 0x3a240f, stars: 0x7a552b },
    palette: [0xffbf47, 0xff7a2f, 0xffd35a, 0xffa84f, 0xffcc73, 0xd58b2a, 0xf7b863, 0xff9654, 0xe6b34f, 0xffd08a],
    outlier: 0xff5a36,
  },
  terminal: {
    css: { obsidian: '#010401', deep: '#041004', panel: '#071507cc', border: '#1b4f1b', neonC: '#39ff14', neonM: '#98ff74', neonG: '#61ff4b', neonO: '#2be62b', text: '#b8f8b8', dim: '#4f8755' },
    scene: { background: 0x010401, fog: 0x010401, grid: 0x123512, stars: 0x2d6b2d },
    palette: [0x39ff14, 0x98ff74, 0x61ff4b, 0x2be62b, 0x8cff5d, 0x6adf56, 0xb2ff7d, 0x52ff3b, 0x79ff55, 0x9cff8d],
    outlier: 0xff3b8a,
  },
};

let activePalette = THEMES['deep-space'].palette.slice();
let outlierColor = new THREE.Color(THEMES['deep-space'].outlier);

function getAdjustedPaletteColor(index, outColor = new THREE.Color()) {
  outColor.setHex(activePalette[index % activePalette.length]);
  const hsl = { h: 0, s: 0, l: 0 };
  outColor.getHSL(hsl);
  hsl.h = (hsl.h + hueShift / 360 + 1) % 1;
  hsl.s = Math.min(1, Math.max(0, hsl.s * saturationMult));
  hsl.l = Math.min(1, Math.max(0, hsl.l * lightnessMult));
  outColor.setHSL(hsl.h, hsl.s, hsl.l);
  return outColor;
}

function detectOutliers(positions, n) {
  if (!positions || n < 20) {
    return { flags: new Uint8Array(n), count: 0 };
  }

  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }

  const bins = Math.max(6, Math.min(18, Math.round(Math.cbrt(n) / 1.6)));
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const spanZ = maxZ - minZ || 1;
  const counts = new Map();
  const cells = new Int16Array(n * 3);

  const toBin = (value, min, span) => {
    const scaled = Math.floor(((value - min) / span) * bins);
    return Math.max(0, Math.min(bins - 1, scaled));
  };

  for (let i = 0; i < n; i++) {
    const bx = toBin(positions[i * 3], minX, spanX);
    const by = toBin(positions[i * 3 + 1], minY, spanY);
    const bz = toBin(positions[i * 3 + 2], minZ, spanZ);
    cells[i * 3] = bx;
    cells[i * 3 + 1] = by;
    cells[i * 3 + 2] = bz;
    const key = `${bx}|${by}|${bz}`;
    counts.set(key, (counts.get(key) || 0) + 1);
  }

  const densities = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const bx = cells[i * 3];
    const by = cells[i * 3 + 1];
    const bz = cells[i * 3 + 2];
    let density = 0;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const nx = bx + dx;
          const ny = by + dy;
          const nz = bz + dz;
          if (nx < 0 || ny < 0 || nz < 0 || nx >= bins || ny >= bins || nz >= bins) continue;
          density += counts.get(`${nx}|${ny}|${nz}`) || 0;
        }
      }
    }
    densities[i] = Math.max(0, density - 1);
  }

  const sorted = Array.from(densities).sort((a, b) => a - b);
  const targetCount = Math.max(1, Math.floor(n * 0.06));
  const cutoff = sorted[Math.min(sorted.length - 1, targetCount - 1)];
  const meanDensity = sorted.reduce((sum, value) => sum + value, 0) / sorted.length;

  const flags = new Uint8Array(n);
  let count = 0;
  const adaptiveCap = meanDensity * 0.55;
  for (let i = 0; i < n; i++) {
    if (densities[i] <= cutoff && densities[i] <= adaptiveCap) {
      flags[i] = 1;
      count++;
    }
  }

  if (count === 0 && n > 0) {
    let minIdx = 0;
    for (let i = 1; i < n; i++) {
      if (densities[i] < densities[minIdx]) minIdx = i;
    }
    flags[minIdx] = 1;
    count = 1;
  }

  return { flags, count };
}

function applyOutlierColors(colorArray) {
  if (!colorArray || !outlierFlags || outlierFlags.length === 0) return;
  for (let i = 0; i < outlierFlags.length; i++) {
    if (!outlierFlags[i]) continue;
    colorArray[i * 3] = outlierColor.r;
    colorArray[i * 3 + 1] = outlierColor.g;
    colorArray[i * 3 + 2] = outlierColor.b;
  }
}

function pointIDKey(id) {
  if (id === null || id === undefined) return '';
  if (typeof id === 'number' && Number.isFinite(id) && Number.isInteger(id)) {
    return String(id);
  }
  return String(id);
}

function hasPayloadValue(value) {
  if (value === undefined || value === null) return false;
  if (typeof value === 'string') return value.trim() !== '';
  return true;
}

function payloadLookup(payload, ...keys) {
  if (!payload || typeof payload !== 'object') return undefined;

  const variants = new Set();
  for (const key of keys) {
    variants.add(key);
    variants.add(String(key).toLowerCase());
    variants.add(String(key).toUpperCase());
  }

  for (const variant of variants) {
    const value = payload[variant];
    if (hasPayloadValue(value)) return value;
  }

  const queue = [payload];
  const seen = new Set();
  while (queue.length > 0) {
    const obj = queue.shift();
    if (!obj || typeof obj !== 'object') continue;
    if (seen.has(obj)) continue;
    seen.add(obj);

    for (const [k, v] of Object.entries(obj)) {
      if (variants.has(k) && hasPayloadValue(v)) return v;
      if (v && typeof v === 'object' && !Array.isArray(v)) {
        queue.push(v);
      }
    }
  }

  return undefined;
}

function parseNumericValue(value) {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value.trim());
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function parseTimestampValue(value) {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const direct = Number(value.trim());
    if (Number.isFinite(direct)) return direct;
    const t = Date.parse(value);
    if (Number.isFinite(t)) return t;
  }
  return null;
}

function extractSizeMetric(point) {
  if (!point || sizeField === 'none') return null;
  const payload = point.payload || {};
  if (sizeField === 'chunk_length') {
    const textValue = payloadLookup(payload, 'chunk_text', 'text', 'content', 'summary');
    return textValue === undefined ? null : String(textValue).length;
  }
  if (sizeField === 'text_length') {
    const textValue = payloadLookup(payload, 'text', 'content', 'chunk_text', 'summary');
    return textValue === undefined ? null : String(textValue).length;
  }
  return parseNumericValue(payloadLookup(payload, sizeField));
}

function createPointSizeArray(points, baseSize) {
  const n = points.length;
  const sizes = new Float32Array(n);
  if (sizeField === 'none') {
    sizes.fill(baseSize);
    return sizes;
  }

  const values = new Array(n).fill(null);
  let minValue = Infinity;
  let maxValue = -Infinity;
  for (let i = 0; i < n; i++) {
    const value = extractSizeMetric(points[i]);
    if (value === null) continue;
    values[i] = value;
    if (value < minValue) minValue = value;
    if (value > maxValue) maxValue = value;
  }

  const range = maxValue - minValue;
  for (let i = 0; i < n; i++) {
    const value = values[i];
    if (value === null || !Number.isFinite(value)) {
      sizes[i] = baseSize;
      continue;
    }
    const t = range < 1e-9 ? 0.5 : (value - minValue) / range;
    sizes[i] = baseSize * (0.6 + t * 1.8);
  }
  return sizes;
}

function createTimelineOrderArray(points) {
  const n = points.length;
  const order = new Float32Array(n);
  const stamps = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    const payload = points[i]?.payload || {};
    stamps[i] = parseTimestampValue(payloadLookup(payload, 'ingested_at', 'timestamp', 'created_at', 'date', 'time'));
  }

  const indices = Array.from({ length: n }, (_, idx) => idx);
  indices.sort((a, b) => {
    const ta = stamps[a];
    const tb = stamps[b];
    if (ta === null && tb === null) return a - b;
    if (ta === null) return 1;
    if (tb === null) return -1;
    if (ta === tb) return a - b;
    return ta - tb;
  });

  const denom = Math.max(1, n - 1);
  for (let rank = 0; rank < indices.length; rank++) {
    order[indices[rank]] = rank / denom;
  }
  return order;
}

function applyTheme(themeName) {
  const key = THEMES[themeName] ? themeName : 'deep-space';
  activeTheme = key;
  const theme = THEMES[key];
  const root = document.documentElement;
  root.style.setProperty('--obsidian', theme.css.obsidian);
  root.style.setProperty('--deep', theme.css.deep);
  root.style.setProperty('--panel', theme.css.panel);
  root.style.setProperty('--border', theme.css.border);
  root.style.setProperty('--neon-c', theme.css.neonC);
  root.style.setProperty('--neon-m', theme.css.neonM);
  root.style.setProperty('--neon-g', theme.css.neonG);
  root.style.setProperty('--neon-o', theme.css.neonO);
  root.style.setProperty('--text', theme.css.text);
  root.style.setProperty('--dim', theme.css.dim);

  activePalette = theme.palette.slice();
  outlierColor.setHex(theme.outlier);

  if (scene) {
    scene.background = new THREE.Color(theme.scene.background);
    if (scene.fog) scene.fog.color = new THREE.Color(theme.scene.fog);
  }
  if (gridHelper && gridHelper.material) {
    const mats = Array.isArray(gridHelper.material) ? gridHelper.material : [gridHelper.material];
    for (const mat of mats) {
      if (mat && mat.color) mat.color.setHex(theme.scene.grid);
    }
  }
  if (scene) {
    scene.traverse((obj) => {
      if (obj.isPoints && obj.geometry && obj.geometry.attributes && obj.geometry.attributes.position && obj !== particleSystem && obj !== hazeSystem && obj !== glowSystem && obj !== nebulaSystem) {
        if (obj.material && obj.material.color && obj.geometry.attributes.position.count === 6000) {
          obj.material.color.setHex(theme.scene.stars);
        }
      }
    });
  }

  refreshPointColors();
  if (particleSystem?.geometry?.attributes?.position && allPoints.length > 0) {
    rebuildClusterHulls(allPoints, particleSystem.geometry.attributes.position.array, activeClusterMap);
  }
}

function updateDensityFog(positions, n) {
  if (!scene || !scene.fog || !positions || n < 2) return;
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  const span = Math.max(1, (maxX - minX) * (maxY - minY) * (maxZ - minZ));
  const density = n / span;
  const targetFog = Math.min(0.0022, Math.max(0.00012, density * 5.0 + 0.00022));
  scene.fog.density = targetFog;
}

function rebuildClusterHulls(points, positions, clusterMap = activeClusterMap) {
  if (!scene) return;
  if (clusterHullsGroup) {
    scene.remove(clusterHullsGroup);
  }
  clusterHullsGroup = new THREE.Group();

  const buckets = new Map();
  for (let i = 0; i < points.length; i++) {
    const key = extractClusterKey(points[i]);
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(i);
  }

  const topKeys = Array.from(buckets.entries())
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 6)
    .map(([k]) => k);

  for (const key of topKeys) {
    const indices = buckets.get(key) || [];
    if (indices.length < 8) continue;
    const sampleStride = Math.max(1, Math.floor(indices.length / 120));
    const hullPoints = [];
    for (let i = 0; i < indices.length; i += sampleStride) {
      const idx = indices[i];
      hullPoints.push(new THREE.Vector3(
        positions[idx * 3],
        positions[idx * 3 + 1],
        positions[idx * 3 + 2]
      ));
    }
    if (hullPoints.length < 8) continue;

    let geometry;
    try {
      geometry = new ConvexGeometry(hullPoints);
    } catch {
      continue;
    }
    const clusterMeta = clusterMap?.get(key);
    const paletteIndex = clusterMeta ? clusterMeta.index % activePalette.length : Math.abs(key.length) % activePalette.length;
    const color = getAdjustedPaletteColor(paletteIndex, new THREE.Color());
    const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.08,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }));
    const wire = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.2 })
    );
    clusterHullsGroup.add(mesh);
    clusterHullsGroup.add(wire);
  }

  scene.add(clusterHullsGroup);
}

const LENSES_KEY = 'vectorview_lenses';

function encodeLens(lens) {
  const json = JSON.stringify(lens);
  return btoa(unescape(encodeURIComponent(json))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function decodeLens(token) {
  try {
    const padded = token.replace(/-/g, '+').replace(/_/g, '/');
    const json = decodeURIComponent(escape(atob(padded)));
    const parsed = JSON.parse(json);
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
}

function loadLenses() {
  try {
    const raw = localStorage.getItem(LENSES_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    lenses = Array.isArray(parsed) ? parsed.filter((entry) => entry && entry.name && entry.collection) : [];
  } catch {
    lenses = [];
  }
  renderLenses();
}

function persistLenses() {
  try {
    localStorage.setItem(LENSES_KEY, JSON.stringify(lenses));
  } catch {
  }
}

function captureLens(name) {
  const position = camera ? camera.position.toArray().map((value) => Number(value.toFixed(3))) : null;
  const target = controls ? controls.target.toArray().map((value) => Number(value.toFixed(3))) : null;
  return {
    version: 1,
    name,
    collection: currentCollection,
    projection: projectionMethod,
    limit: pointLimit,
    vector_name: selectedVectorName || '',
    theme: activeTheme,
    color_field: colorField || 'source',
    color_field_custom: colorFieldCustom || '',
    camera: { position, target },
    saved_at: Date.now(),
  };
}

function renderLenses() {
  const list = document.getElementById('lenses-list');
  const empty = document.getElementById('lenses-empty');
  if (!list || !empty) return;

  empty.style.display = lenses.length ? 'none' : 'block';
  list.innerHTML = '';

  for (const lens of lenses) {
    const row = document.createElement('div');
    row.className = 'bookmark-item';

    const label = document.createElement('span');
    label.className = 'bookmark-label';
    label.textContent = lens.name;
    label.title = `${lens.collection} · ${lens.projection} · ${lens.limit} pts`;
    label.addEventListener('click', () => applyLens(lens));

    const link = document.createElement('button');
    link.textContent = '⧉';
    link.title = 'Copy shareable link';
    link.addEventListener('click', (event) => {
      event.stopPropagation();
      copyLensLink(lens);
    });

    const remove = document.createElement('button');
    remove.textContent = '✕';
    remove.title = 'Delete lens';
    remove.addEventListener('click', (event) => {
      event.stopPropagation();
      lenses = lenses.filter((candidate) => candidate.name !== lens.name);
      persistLenses();
      renderLenses();
    });

    row.appendChild(label);
    row.appendChild(link);
    row.appendChild(remove);
    list.appendChild(row);
  }
}

function saveCurrentLens() {
  const input = document.getElementById('lens-name');
  const name = (input.value || '').trim() || `${currentCollection} · ${projectionMethod}`;
  const lens = captureLens(name);
  lenses = lenses.filter((entry) => entry.name !== name);
  lenses.unshift(lens);
  persistLenses();
  renderLenses();
  input.value = '';
  setScanStatus(`Lens saved: ${name}`, 'ok');
}

async function copyLensLink(lens) {
  const url = `${location.origin}${location.pathname}#lens=${encodeLens(lens)}`;
  try {
    await navigator.clipboard.writeText(url);
    setScanStatus('Lens link copied.', 'ok');
  } catch {
    setScanStatus(url, 'muted');
  }
}

function applyLens(lens) {
  if (!lens || !lens.collection) return;
  const limit = Math.max(500, Math.min(75000, Number(lens.limit) || pointLimit));
  const projection = ['pca', 'random', 'tsne', 'umap'].includes(lens.projection) ? lens.projection : 'pca';

  if (lens.collection !== currentCollection) {
    const select = document.getElementById('collection-select');
    if (select) select.value = lens.collection;
  }

  currentCollection = lens.collection;
  projectionMethod = projection;
  pointLimit = limit;
  selectedVectorName = lens.vector_name || '';
  lastHighlightEventID = '';

  const projectionSelect = document.getElementById('ctrl-projection');
  if (projectionSelect) projectionSelect.value = projection;
  const limitCtrl = document.getElementById('ctrl-point-limit');
  if (limitCtrl) limitCtrl.value = String(limit);
  const limitLabel = document.getElementById('val-point-limit');
  if (limitLabel) limitLabel.textContent = limit.toLocaleString();
  const vectorCtrl = document.getElementById('ctrl-vector-name');
  if (vectorCtrl) vectorCtrl.value = selectedVectorName;

  pendingLens = lens;
  updateCollectionMetaPanel(currentCollection);
  if (liveIngestEnabled) startLiveIngestConnection();
  loadPoints(currentCollection, pointLimit);
}

function applyPendingLens() {
  if (!pendingLens) return;
  const lens = pendingLens;
  pendingLens = null;

  if (lens.theme && THEMES[lens.theme]) {
    applyTheme(lens.theme);
    const themeSelect = document.getElementById('ctrl-theme');
    if (themeSelect) themeSelect.value = lens.theme;
  }

  if (lens.color_field) {
    const colorCtrl = document.getElementById('ctrl-color-field');
    if (colorCtrl) {
      const known = Array.from(colorCtrl.options).some((option) => option.value === lens.color_field);
      if (known) {
        colorCtrl.value = lens.color_field;
        colorFieldCustom = lens.color_field_custom || colorFieldCustom;
        recolorByField(lens.color_field);
      }
    }
  }

  if (lens.camera && Array.isArray(lens.camera.position) && Array.isArray(lens.camera.target)) {
    const target = new THREE.Vector3(...lens.camera.target);
    const position = new THREE.Vector3(...lens.camera.position);
    if ([target, position].every((v) => Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z))) {
      controls.target.copy(target);
      camera.position.copy(position);
      if (typeof controls._updateSpherical === 'function') controls._updateSpherical();
    }
  }
  setScanStatus(`Lens loaded: ${lens.name}`, 'ok');
}

function hydrateLensFromHash() {
  const match = /(?:^#|&)lens=([^&]+)/.exec(location.hash || '');
  if (!match) return null;
  return decodeLens(match[1]);
}

function downloadCanvas(canvas, filename) {
  const link = document.createElement('a');
  link.href = canvas.toDataURL('image/png');
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function drawExportLegend(ctx, width, height) {
  const entries = activeClusterMap
    ? Array.from(activeClusterMap.entries()).sort((a, b) => b[1].count - a[1].count)
    : [];
  const top = entries.slice(0, 12);

  const pad = Math.round(width * 0.014);
  const rowHeight = Math.round(width * 0.0165);
  const swatch = Math.round(rowHeight * 0.55);
  const captionHeight = Math.round(rowHeight * 1.4);
  const boxWidth = Math.round(width * 0.22);
  const boxHeight = pad * 2 + captionHeight + rowHeight * (top.length + (entries.length > top.length ? 1 : 0) + (outlierCount > 0 ? 1 : 0));

  const x = Math.round(width * 0.02);
  const y = height - boxHeight - Math.round(height * 0.03);

  ctx.save();
  ctx.fillStyle = 'rgba(4, 8, 18, 0.82)';
  ctx.strokeStyle = 'rgba(0, 245, 255, 0.35)';
  ctx.lineWidth = Math.max(1, Math.round(width * 0.0007));
  ctx.fillRect(x, y, boxWidth, boxHeight);
  ctx.strokeRect(x, y, boxWidth, boxHeight);

  ctx.fillStyle = '#8fa3bf';
  ctx.font = `${Math.round(rowHeight * 0.62)}px monospace`;
  ctx.textBaseline = 'middle';
  const caption = `${currentCollection.toUpperCase()} · ${projectionMethod.toUpperCase()} · ${allPoints.length} PTS`;
  ctx.fillText(caption, x + pad, y + pad + captionHeight * 0.5, boxWidth - pad * 2);

  const color = new THREE.Color();
  let rowY = y + pad + captionHeight;
  for (const [key, meta] of top) {
    getAdjustedPaletteColor(meta.index % activePalette.length, color);
    ctx.fillStyle = '#' + color.getHexString();
    ctx.beginPath();
    ctx.arc(x + pad + swatch * 0.5, rowY + rowHeight * 0.5, swatch * 0.4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#dfe9f7';
    ctx.font = `${Math.round(rowHeight * 0.55)}px monospace`;
    ctx.fillText(`${String(key)} (${meta.count})`, x + pad + swatch + pad, rowY + rowHeight * 0.5, boxWidth - swatch - pad * 3);
    rowY += rowHeight;
  }

  if (entries.length > top.length) {
    ctx.fillStyle = '#8fa3bf';
    ctx.font = `${Math.round(rowHeight * 0.5)}px monospace`;
    ctx.fillText(`…${entries.length - top.length} more`, x + pad, rowY + rowHeight * 0.5, boxWidth - pad * 2);
    rowY += rowHeight;
  }

  if (outlierCount > 0) {
    ctx.fillStyle = '#' + outlierColor.getHexString();
    ctx.beginPath();
    ctx.arc(x + pad + swatch * 0.5, rowY + rowHeight * 0.5, swatch * 0.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#dfe9f7';
    ctx.font = `${Math.round(rowHeight * 0.55)}px monospace`;
    ctx.fillText(`outliers (${outlierCount})`, x + pad + swatch + pad, rowY + rowHeight * 0.5, boxWidth - swatch - pad * 3);
  }
  ctx.restore();
}

function exportScreenshot() {
  if (!renderer) return;
  const source = renderer.domElement;
  const filename = `vectorview-${currentCollection}-${Date.now()}.png`;
  const legendToggle = document.getElementById('chk-export-legend');
  const includeLegend = !legendToggle || legendToggle.checked;

  if (!includeLegend) {
    downloadCanvas(source, filename);
    return;
  }

  const canvas = document.createElement('canvas');
  canvas.width = source.width;
  canvas.height = source.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(source, 0, 0);
  drawExportLegend(ctx, canvas.width, canvas.height);
  downloadCanvas(canvas, filename);
}

function scheduleColorRefresh() {
  if (colorRefreshQueued) return;
  colorRefreshQueued = true;
  requestAnimationFrame(() => {
    colorRefreshQueued = false;
    refreshPointColors();
  });
}

function refreshPointColors() {
  if (!particleSystem || pointClusterIndices.length === 0) return;
  const colorAttr = particleSystem.geometry.attributes.color;
  if (!colorAttr) return;

  const c = new THREE.Color();
  if (!basePointColors || basePointColors.length !== colorAttr.array.length) {
    basePointColors = new Float32Array(colorAttr.array.length);
  }
  for (let i = 0; i < pointClusterIndices.length; i++) {
    getAdjustedPaletteColor(pointClusterIndices[i], c);
    basePointColors[i * 3] = c.r;
    basePointColors[i * 3 + 1] = c.g;
    basePointColors[i * 3 + 2] = c.b;
  }

  applyOutlierColors(basePointColors);

  if (similarityModeActive) {
    applySignalScanHighlight();
  } else {
    colorAttr.array.set(basePointColors);
    colorAttr.needsUpdate = true;
  }

  if (activeClusterMap) buildLegend(activeClusterMap);
}

function recolorByField(field) {
  colorField = field;
  if (!particleSystem || !allPoints || allPoints.length === 0) return;

  const clusterMap = new Map();
  let clusterIdx = 0;
  for (const p of allPoints) {
    const key = extractClusterKey(p, colorField);
    if (!clusterMap.has(key)) {
      clusterMap.set(key, { index: clusterIdx++, count: 0 });
    }
    clusterMap.get(key).count++;
  }

  pointClusterIndices = new Array(allPoints.length);
  for (let i = 0; i < allPoints.length; i++) {
    const key = extractClusterKey(allPoints[i], colorField);
    pointClusterIndices[i] = clusterMap.get(key).index % activePalette.length;
  }

  activeClusterMap = clusterMap;
  refreshPointColors();

  if (particleSystem.geometry?.attributes?.position) {
    rebuildClusterHulls(allPoints, particleSystem.geometry.attributes.position.array, activeClusterMap);
  }
}

function setTimelineReveal(value) {
  timelineReveal = Math.max(0, Math.min(1, value));
  const percent = Math.round(timelineReveal * 100);
  const valEl = document.getElementById('val-time');
  if (valEl) valEl.textContent = `${percent}%`;
  const ctrl = document.getElementById('ctrl-time');
  if (ctrl && Number(ctrl.value) !== percent) ctrl.value = String(percent);

  const systems = [particleSystem, hazeSystem, glowSystem, nebulaSystem];
  for (const system of systems) {
    if (system?.material?.uniforms?.uReveal) {
      system.material.uniforms.uReveal.value = timelineReveal;
    }
  }

  const visible = Math.round((allPoints?.length || 0) * timelineReveal);
  const visibleEl = document.getElementById('stat-visible');
  if (visibleEl) visibleEl.textContent = Math.max(0, visible).toLocaleString();
}

function refreshPointSizes() {
  if (!particleSystem || !allPoints || allPoints.length === 0) return;
  const nextSizes = createPointSizeArray(allPoints, pointSize);
  if (outliersOnly && outlierFlags && outlierFlags.length === nextSizes.length) {
    for (let i = 0; i < nextSizes.length; i++) {
      if (!outlierFlags[i]) nextSizes[i] = 0;
    }
  }
  basePointSizes = nextSizes;
  const sizeAttr = particleSystem.geometry.attributes.size;
  if (!sizeAttr) return;

  if (similarityModeActive) {
    applySignalScanHighlight();
  } else {
    sizeAttr.array.set(nextSizes);
    sizeAttr.needsUpdate = true;
  }
}

function toggleOutliersOnly() {
  outliersOnly = !outliersOnly;
  const btn = document.getElementById('btn-outliers-only');
  if (btn) btn.textContent = outliersOnly ? 'SHOW ALL POINTS' : 'ISOLATE OUTLIERS';
  refreshPointSizes();
}

function renderOutliersList() {
  const list = document.getElementById('outliers-list');
  const countEl = document.getElementById('val-outlier-count');
  if (!list) return;
  list.innerHTML = '';
  if (countEl) countEl.textContent = `(${outlierCount})`;

  if (!outlierFlags || outlierCount === 0 || !allPoints || allPoints.length === 0) {
    list.innerHTML = '<div style="color:var(--dim);font-size:0.65rem;">No outliers detected in this sample.</div>';
    return;
  }

  let shown = 0;
  const maxShown = 40;
  for (let i = 0; i < allPoints.length && shown < maxShown; i++) {
    if (!outlierFlags[i]) continue;
    const point = allPoints[i];
    const payload = point.payload || {};
    const title = shortText(getSignalTitle(payload, point.id), 70);
    const snippet = shortText(getSignalSnippet(payload), 100);
    const provider = payloadLookup(payload, 'source_type', 'memory_kind', 'ai_provider') || 'signal';
    const speaker = payloadLookup(payload, 'speaker', 'project') || '';

    const item = document.createElement('div');
    item.className = 'similar-item';
    item.innerHTML = `
      <div class="similar-rank">${escapeHTML(String(provider))}${speaker ? ' · ' + escapeHTML(String(speaker)) : ''}</div>
      <div class="similar-title">${escapeHTML(title)}</div>
      <div class="similar-snippet">${escapeHTML(snippet)}</div>
    `;
    item.addEventListener('click', () => {
      selectPointByIndex(i);
      focusPointByIndex(i, true);
    });
    list.appendChild(item);
    shown++;
  }

  if (outlierCount > shown) {
    const more = document.createElement('div');
    more.style.cssText = 'color:var(--dim);font-size:0.6rem;text-align:center;';
    more.textContent = `…${outlierCount - shown} more not shown`;
    list.appendChild(more);
  }
}

function createViewCubeFaceMaterial(label, fillColor) {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = fillColor;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#0b1224';
  ctx.lineWidth = 10;
  ctx.strokeRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#f3f8ff';
  ctx.font = '700 42px "Orbitron", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  return new THREE.MeshBasicMaterial({ map: texture, transparent: true, opacity: 0.96 });
}

function resizeViewCubeRenderer() {
  if (!viewCubeRenderer || !viewCubeCamera) return;
  const host = document.getElementById('viewcube');
  if (!host) return;
  const rect = host.getBoundingClientRect();
  const size = Math.max(1, Math.floor(Math.min(rect.width, rect.height)));
  viewCubeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  viewCubeRenderer.setSize(size, size, false);
  viewCubeCamera.aspect = 1;
  viewCubeCamera.updateProjectionMatrix();
}

function createViewCubeCornerMarker(xSign, ySign, zSign) {
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.105, 16, 16),
    new THREE.MeshBasicMaterial({ color: 0xe7f2ff, transparent: true, opacity: 0.9 })
  );
  marker.position.set(0.72 * xSign, 0.72 * ySign, 0.72 * zSign);
  marker.userData.isoDirection = new THREE.Vector3(xSign, ySign, zSign).normalize();
  return marker;
}

function snapCameraToDirection(direction) {
  if (!camera || !controls) return;
  const distance = Math.max(camera.position.distanceTo(controls.target), 60);
  const nextPosition = controls.target.clone().add(direction.clone().multiplyScalar(distance));
  camera.position.copy(nextPosition);
  camera.up.set(0, 1, 0);
  camera.lookAt(controls.target);
  if (typeof controls._updateSpherical === 'function') controls._updateSpherical();
}

function onViewCubeClick(event) {
  if (!viewCubePickMesh || !viewCubeCamera) return;
  const rect = event.currentTarget.getBoundingClientRect();
  if (!rect.width || !rect.height) return;

  viewCubeMouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  viewCubeMouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  viewCubeRaycaster.setFromCamera(viewCubeMouse, viewCubeCamera);

  const pickTargets = [...viewCubeCornerPickMeshes, viewCubePickMesh].filter(Boolean);
  const hits = viewCubeRaycaster.intersectObjects(pickTargets, false);
  if (hits.length === 0) return;

  const isoHit = hits.find((hit) => hit.object?.userData?.isoDirection instanceof THREE.Vector3);
  if (isoHit) {
    snapCameraToDirection(isoHit.object.userData.isoDirection);
    return;
  }

  const cubeHit = hits.find((hit) => hit.object === viewCubePickMesh);
  if (!cubeHit) return;

  const materialIndex = cubeHit.face ? cubeHit.face.materialIndex : -1;
  const directions = [
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(-1, 0, 0),
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(0, -1, 0),
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(0, 0, -1),
  ];

  const direction = directions[materialIndex];
  if (direction) {
    snapCameraToDirection(direction);
  }
}

function initViewCube() {
  const host = document.getElementById('viewcube');
  if (!host) return;

  viewCubeScene = new THREE.Scene();
  viewCubeCamera = new THREE.PerspectiveCamera(35, 1, 0.1, 10);
  viewCubeCamera.position.set(0, 0, 3);
  viewCubeCamera.lookAt(0, 0, 0);

  const materials = [
    createViewCubeFaceMaterial('+X', '#1d6eff'),
    createViewCubeFaceMaterial('-X', '#1f3868'),
    createViewCubeFaceMaterial('+Z', '#ff4db5'),
    createViewCubeFaceMaterial('-Z', '#6d2f58'),
    createViewCubeFaceMaterial('+Y', '#34d26d'),
    createViewCubeFaceMaterial('-Y', '#205d38'),
  ];

  const cube = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.2, 1.2), materials);
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(cube.geometry),
    new THREE.LineBasicMaterial({ color: 0xc8d8f0, transparent: true, opacity: 0.5 })
  );

  viewCubeGroup = new THREE.Group();
  viewCubeGroup.add(cube);
  viewCubeGroup.add(edges);
  viewCubeScene.add(viewCubeGroup);
  viewCubePickMesh = cube;
  viewCubeCornerPickMeshes = [];

  const cornerSigns = [
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1],
    [-1, 1, 1],
    [-1, 1, -1],
    [-1, -1, 1],
    [-1, -1, -1],
  ];
  for (const [xSign, ySign, zSign] of cornerSigns) {
    const marker = createViewCubeCornerMarker(xSign, ySign, zSign);
    viewCubeGroup.add(marker);
    viewCubeCornerPickMeshes.push(marker);
  }

  viewCubeScene.add(new THREE.AmbientLight(0xffffff, 0.85 * Math.PI));
  const light = new THREE.DirectionalLight(0xffffff, 0.65 * Math.PI);
  light.position.set(2, 2, 3);
  viewCubeScene.add(light);

  viewCubeRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  viewCubeRenderer.setClearColor(0x000000, 0);
  host.appendChild(viewCubeRenderer.domElement);
  viewCubeRenderer.domElement.addEventListener('click', onViewCubeClick);
  resizeViewCubeRenderer();
}

function renderViewCube() {
  if (!viewCubeRenderer || !viewCubeScene || !viewCubeCamera || !viewCubeGroup || !camera) return;
  viewCubeGroup.quaternion.copy(camera.quaternion).invert();
  viewCubeRenderer.render(viewCubeScene, viewCubeCamera);
}

function initLiveDropSystem() {
  if (!scene) return;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(liveDropMax * 3), 3));
  geo.setDrawRange(0, 0);
  const mat = new THREE.PointsMaterial({
    color: 0x39ff14,
    size: 6,
    transparent: true,
    opacity: 0.8,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    sizeAttenuation: true,
  });
  liveDropSystem = new THREE.Points(geo, mat);
  scene.add(liveDropSystem);
}

function updateLiveDropSystem() {
  if (!liveDropSystem) return;
  const pos = liveDropSystem.geometry.attributes.position.array;
  const drawCount = Math.min(liveDrops.length, liveDropMax);
  for (let i = 0; i < drawCount; i++) {
    const drop = liveDrops[i];
    pos[i * 3] = drop.x;
    pos[i * 3 + 1] = drop.y;
    pos[i * 3 + 2] = drop.z;
  }
  liveDropSystem.geometry.setDrawRange(0, drawCount);
  liveDropSystem.geometry.attributes.position.needsUpdate = true;
  const alpha = drawCount > 0 ? Math.min(0.85, 0.35 + drawCount / liveDropMax) : 0;
  liveDropSystem.material.opacity = alpha;
}

function spawnLiveDropBurst(count = 1) {
  const center = controls?.target || new THREE.Vector3(0, 0, 0);
  for (let i = 0; i < count; i++) {
    if (liveDrops.length >= liveDropMax) {
      liveDrops.shift();
    }
    const angle = Math.random() * Math.PI * 2;
    const radius = 35 + Math.random() * 110;
    liveDrops.push({
      x: center.x + Math.cos(angle) * radius,
      y: center.y + 220 + Math.random() * 80,
      z: center.z + Math.sin(angle) * radius,
      vx: (Math.random() - 0.5) * 8,
      vy: -(35 + Math.random() * 28),
      vz: (Math.random() - 0.5) * 8,
      life: liveDropLife,
      ttl: liveDropLife,
    });
  }
  updateLiveDropSystem();
}

function stepLiveDrops(delta) {
  if (!liveDrops.length) return;
  const gravity = 26;
  for (let i = liveDrops.length - 1; i >= 0; i--) {
    const drop = liveDrops[i];
    drop.vy -= gravity * delta;
    drop.x += drop.vx * delta;
    drop.y += drop.vy * delta;
    drop.z += drop.vz * delta;
    drop.life -= delta;
    if (drop.life <= 0) {
      liveDrops.splice(i, 1);
    }
  }
  updateLiveDropSystem();
}

function stopLiveIngestConnection() {
  if (liveIngestSource) {
    liveIngestSource.close();
    liveIngestSource = null;
  }
  if (liveIngestSocket) {
    liveIngestSocket.close();
    liveIngestSocket = null;
  }
  if (liveIngestReloadTimer) {
    clearTimeout(liveIngestReloadTimer);
    liveIngestReloadTimer = null;
  }
  liveIngestPendingEvents = 0;
  liveIngestSyncInFlight = false;
}

async function syncLiveIngestPoints() {
  if (liveIngestSyncInFlight) return;
  if (!liveIngestEnabled) return;
  liveIngestSyncInFlight = true;
  try {
    const pending = liveIngestPendingEvents;
    liveIngestPendingEvents = 0;
    if (pending <= 0) return;

    if (projectionMethod !== 'random') {
      await loadPoints(currentCollection, pointLimit);
      setScanStatus(`Live ingest synced ${pending} new signals.`, 'ok');
      return;
    }

    const appendFrom = allPoints.length;
    const nextLimit = Math.min(pointLimit, appendFrom + pending);
    if (nextLimit <= appendFrom) {
      setScanStatus('Live ingest received new points. Increase Point Count to display them.', 'muted');
      return;
    }

    const vectorQuery = selectedVectorName ? `&vector_name=${encodeURIComponent(selectedVectorName)}` : '';
    const res = await fetch(`/api/points?collection=${encodeURIComponent(currentCollection)}&limit=${nextLimit}&projection=${encodeURIComponent(projectionMethod)}${vectorQuery}&append_from=${appendFrom}`);
    if (!res.ok) {
      const message = await readApiError(res, '');
      throw new Error(message || 'incremental live sync failed');
    }
    const data = await res.json();
    const incoming = Array.isArray(data.points) ? data.points : [];
    if (data.incremental) {
      buildParticles(allPoints.concat(incoming), data.projection || currentProjectionMeta || null);
      setScanStatus(`Live ingest appended ${incoming.length} new signals.`, 'ok');
      return;
    }

    await loadPoints(currentCollection, nextLimit);
    setScanStatus(`Live ingest synced ${pending} new signals.`, 'ok');
  } catch {
    await loadPoints(currentCollection, pointLimit);
    setScanStatus('Live ingest fallback sync complete.', 'muted');
  } finally {
    liveIngestSyncInFlight = false;
  }
}

function scheduleLiveIngestReload() {
  if (liveIngestReloadTimer) return;
  liveIngestReloadTimer = setTimeout(async () => {
    liveIngestReloadTimer = null;
    await syncLiveIngestPoints();
  }, 550);
}

function handleLiveStreamEvent(raw) {
  if (!raw || typeof raw !== 'object') return;
  const eventType = String(raw.type || '').trim();
  const collection = String(raw.collection || '').trim();
  if (collection && collection !== currentCollection) return;
  if (eventType !== 'point_ingested' && eventType !== 'telemetry' && eventType !== 'highlight') return;
  if (eventType === 'point_ingested') {
    liveIngestPendingEvents += 1;
    spawnLiveDropBurst(2);
    setScanStatus(`Live ingest: +${liveIngestPendingEvents} pending`, 'ok');
    scheduleLiveIngestReload();
    return;
  }
  spawnLiveDropBurst(1);
}

function startLiveIngestConnection() {
  stopLiveIngestConnection();
  const streamURL = `/api/stream?collection=${encodeURIComponent(currentCollection)}&replay=25&max_events_per_second=24`;
  liveIngestSource = new EventSource(streamURL);
  liveIngestSource.onmessage = () => {};
  liveIngestSource.addEventListener('point_ingested', (event) => {
    try {
      handleLiveStreamEvent(JSON.parse(event.data));
    } catch {
    }
  });
  liveIngestSource.addEventListener('telemetry', (event) => {
    try {
      handleLiveStreamEvent(JSON.parse(event.data));
    } catch {
    }
  });
  liveIngestSource.addEventListener('highlight', (event) => {
    try {
      handleLiveStreamEvent(JSON.parse(event.data));
    } catch {
    }
  });
  liveIngestSource.onerror = () => {
    setScanStatus('Live ingest reconnecting...', 'muted');
  };

  const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  try {
    liveIngestSocket = new WebSocket(`${wsProtocol}//${location.host}/api/ws`);
  } catch {
    liveIngestSocket = null;
  }
}

function setLiveIngestMode(enabled) {
  liveIngestEnabled = !!enabled;
  const btn = document.getElementById('btn-live-ingest');
  const val = document.getElementById('val-live-ingest');
  if (liveIngestEnabled) {
    btn.textContent = 'STOP LIVE MODE';
    btn.classList.add('secondary');
    val.textContent = 'ON';
    startLiveIngestConnection();
  } else {
    btn.textContent = 'START LIVE MODE';
    btn.classList.remove('secondary');
    val.textContent = 'OFF';
    stopLiveIngestConnection();
  }
}

// ── Init Three.js ─────────────────────────────────────────────────────────────
function initTrailPipeline() {
  if (!renderer) return;
  const width = Math.max(1, window.innerWidth);
  const height = Math.max(1, window.innerHeight);

  sceneRenderTarget = new THREE.WebGLRenderTarget(width, height, {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    format: THREE.RGBAFormat,
  });
  trailRenderTargets[0] = new THREE.WebGLRenderTarget(width, height, {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    format: THREE.RGBAFormat,
  });
  trailRenderTargets[1] = new THREE.WebGLRenderTarget(width, height, {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    format: THREE.RGBAFormat,
  });

  trailBlendScene = new THREE.Scene();
  trailBlendCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  trailBlendMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uCurrent: { value: null },
      uPrev: { value: null },
      uPersistence: { value: trailPersistence },
    },
    vertexShader: TRAIL_VERT,
    fragmentShader: TRAIL_BLEND_FRAG,
    depthWrite: false,
    depthTest: false,
  });
  trailBlendScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), trailBlendMaterial));

  trailPresentScene = new THREE.Scene();
  trailPresentCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  trailPresentMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uTrail: { value: null },
    },
    vertexShader: TRAIL_VERT,
    fragmentShader: TRAIL_PRESENT_FRAG,
    depthWrite: false,
    depthTest: false,
  });
  trailPresentScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), trailPresentMaterial));
}

function resizeTrailPipeline() {
  if (!sceneRenderTarget || !trailRenderTargets[0] || !trailRenderTargets[1]) return;
  const width = Math.max(1, window.innerWidth);
  const height = Math.max(1, window.innerHeight);
  sceneRenderTarget.setSize(width, height);
  trailRenderTargets[0].setSize(width, height);
  trailRenderTargets[1].setSize(width, height);
}

function renderSceneWithTrails() {
  if (!renderer || !scene || !camera || !sceneRenderTarget || !trailBlendMaterial || !trailPresentMaterial || trailPersistence <= 0.02) {
    renderer.setRenderTarget(null);
    renderer.render(scene, camera);
    return;
  }

  const read = trailRenderTargets[trailReadIndex];
  const write = trailRenderTargets[1 - trailReadIndex];

  renderer.setRenderTarget(sceneRenderTarget);
  renderer.clear();
  renderer.render(scene, camera);

  trailBlendMaterial.uniforms.uCurrent.value = sceneRenderTarget.texture;
  trailBlendMaterial.uniforms.uPrev.value = read.texture;
  trailBlendMaterial.uniforms.uPersistence.value = controls && controls.autoRotate ? trailPersistence : 0.72;

  renderer.setRenderTarget(write);
  renderer.clear();
  renderer.render(trailBlendScene, trailBlendCamera);

  trailPresentMaterial.uniforms.uTrail.value = write.texture;
  renderer.setRenderTarget(null);
  renderer.clear();
  renderer.render(trailPresentScene, trailPresentCamera);

  trailReadIndex = 1 - trailReadIndex;
}

function initScene() {
  const container = document.getElementById('canvas-container');

  scene = new THREE.Scene();
  scene.background = new THREE.Color(THEMES[activeTheme].scene.background);
  scene.fog = new THREE.FogExp2(THEMES[activeTheme].scene.fog, 0.0003);

  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 5000);
  camera.position.set(0, 0, 300);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ReinhardToneMapping;
  renderer.toneMappingExposure = 1.2;
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.autoRotateSpeed = rotSpeed;

  initTrailPipeline();
  initViewCube();
  initLiveDropSystem();

  // Star field background
  addStarField();

  // Grid helper — subtle
  gridHelper = new THREE.GridHelper(1000, 40, THEMES[activeTheme].scene.grid, THEMES[activeTheme].scene.grid);
  gridHelper.position.y = -150;
  scene.add(gridHelper);

  // Ambient light
  scene.add(new THREE.AmbientLight(0x111133, 2 * Math.PI));
  const pLight = new THREE.PointLight(0x00f5ff, 2 * Math.PI, 600);
  pLight.position.set(0, 100, 0);
  scene.add(pLight);

  applyTheme(activeTheme);

  window.addEventListener('resize', onResize);
  renderer.domElement.addEventListener('mousemove', onMouseMove);
  renderer.domElement.addEventListener('click', onClick);
}

function addStarField() {
  const geo = new THREE.BufferGeometry();
  const count = 6000;
  const pos = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    pos[i*3]   = (Math.random() - 0.5) * 4000;
    pos[i*3+1] = (Math.random() - 0.5) * 4000;
    pos[i*3+2] = (Math.random() - 0.5) * 4000;
  }
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({ size: 0.4, color: THEMES[activeTheme].scene.stars, sizeAttenuation: true });
  scene.add(new THREE.Points(geo, mat));
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  resizeTrailPipeline();
  resizeViewCubeRenderer();
}

function onMouseMove(e) {
  mouse.x =  (e.clientX / window.innerWidth)  * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
  hoverDirty = true;
}

function onClick(e) {
  if (!particleSystem) return;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObject(particleSystem);
  if (hits.length > 0) {
    const idx = hits[0].index;
    selectPointByIndex(idx);
  }
}

// ── Build particle cloud ──────────────────────────────────────────────────────
function extractClusterKey(point, field = colorField) {
  if (field === 'source') {
    const sourceValue = payloadLookup(point.payload, 'file_source', 'attachment_filename', 'source_title', 'sources', 'source_id', 'conversation_id', 'session_id', 'project', 'source_collection', 'source_file', 'source');
    let source = '';
    if (Array.isArray(sourceValue)) {
      source = sourceValue.length > 0 ? String(sourceValue[0]) : '';
    } else if (sourceValue !== undefined && sourceValue !== null) {
      source = String(sourceValue);
    }
    source = source.trim();
    if (!source) return 'unknown';

    // If generic title like "DeepDrill Research Thought" or "MindDrill Chat", differentiate by strategy or session
    if (source === 'DeepDrill Research Thought') {
      const alt = payloadLookup(point.payload, 'strategy', 'type', 'uncertainty_class', 'session_id');
      if (alt) return String(alt);
    }

    // Strip common file extensions
    source = source.replace(/\.(pdf|m4a|mp3|wav|json|md|txt|csv|html)$/i, '');
    const prefix = source.split('_')[0];
    const match = prefix.match(/^(\d+)/);
    if (match) return match[1];
    return source.replace(/_/g, ' ') || 'unknown';
  }

  const lookupKey = field === 'custom' ? (colorFieldCustom || '').trim() : field;
  if (!lookupKey) return 'unknown';
  const raw = payloadLookup(point.payload, lookupKey);
  if (raw === undefined || raw === null || raw === '') return 'unknown';
  if (Array.isArray(raw)) return raw.length > 0 ? String(raw[0]) : 'unknown';
  return String(raw);
}

function buildParticles(points, projectionMeta = null) {
  if (particleSystem) { scene.remove(particleSystem); particleSystem.material.dispose(); }
  if (hazeSystem)     { scene.remove(hazeSystem);     hazeSystem.material.dispose(); }
  if (glowSystem)     { scene.remove(glowSystem);     glowSystem.material.dispose(); }
  if (nebulaSystem)   { scene.remove(nebulaSystem);   nebulaSystem.material.dispose(); }
  if (clusterHullsGroup) { scene.remove(clusterHullsGroup); clusterHullsGroup = null; }
  if (particleSystem) { particleSystem.geometry.dispose(); }

  const n = points.length;
  const positions = new Float32Array(n * 3);
  const colors    = new Float32Array(n * 3);
  const sizes     = createPointSizeArray(points, pointSize);
  const orderNorm = createTimelineOrderArray(points);
  const scanFlags = new Float32Array(n);

  pointIndexByID = new Map();

  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    if (p.z < minZ) minZ = p.z; if (p.z > maxZ) maxZ = p.z;
  }
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const rangeZ = maxZ - minZ || 1;
  const scale = 200;

  const clusterMap = new Map();
  let clusterIdx = 0;
  for (const p of points) {
    const key = extractClusterKey(p);
    if (!clusterMap.has(key)) {
      clusterMap.set(key, { index: clusterIdx++, count: 0 });
    }
    clusterMap.get(key).count++;
  }

  pointClusterIndices = new Array(n);
  const c = new THREE.Color();
  for (let i = 0; i < n; i++) {
    const p = points[i];
    positions[i*3]   = ((p.x - minX) / rangeX - 0.5) * scale;
    positions[i*3+1] = ((p.y - minY) / rangeY - 0.5) * scale;
    positions[i*3+2] = ((p.z - minZ) / rangeZ - 0.5) * scale;

    const key = extractClusterKey(p);
    const ci  = clusterMap.get(key).index % activePalette.length;
    pointClusterIndices[i] = ci;
    getAdjustedPaletteColor(ci, c);
    colors[i*3] = c.r; colors[i*3+1] = c.g; colors[i*3+2] = c.b;

    scanFlags[i] = 0;
    pointIndexByID.set(pointIDKey(p.id), i);
  }

  const outlierResult = detectOutliers(positions, n);
  outlierFlags = outlierResult.flags;
  outlierCount = outlierResult.count;
  applyOutlierColors(colors);
  renderOutliersList();

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
  geo.setAttribute('size',     new THREE.BufferAttribute(sizes, 1));
  geo.setAttribute('orderNorm', new THREE.BufferAttribute(orderNorm, 1));
  geo.setAttribute('scanFlag', new THREE.BufferAttribute(scanFlags, 1));






  const createLayerMaterial = (fragmentShader, opacity, bloom, sizeMult) => new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uOpacity: { value: opacity },
      uBloom: { value: bloom },
      uSizeMult: { value: sizeMult },
      uReveal: { value: timelineReveal },
    },
    vertexShader: VERT,
    fragmentShader,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexColors: true,
  });

  particleSystem = new THREE.Points(geo, createLayerMaterial(CORE_FRAG, coreOpacity, bloomStrength, 1.0));
  hazeSystem = new THREE.Points(geo, createLayerMaterial(HAZE_FRAG, hazeOpacity, 1.8 * (bloomStrength / 2.5), 5.0));
  glowSystem = new THREE.Points(geo, createLayerMaterial(GLOW_FRAG, 0.18, 1.0 * (bloomStrength / 2.5), 14.0));
  nebulaSystem = new THREE.Points(geo, createLayerMaterial(NEBULA_FRAG, 0.06, 1.0 * (bloomStrength / 2.5), 30.0));

  scene.add(nebulaSystem);
  scene.add(glowSystem);
  scene.add(hazeSystem);
  scene.add(particleSystem);

  activeClusterMap = clusterMap;
  buildLegend(clusterMap);
  updateDistanceHeatmap(points, clusterMap, positions);
  updateDensityFog(positions, n);
  rebuildClusterHulls(points, positions, clusterMap);

  document.getElementById('stat-points').textContent  = n.toLocaleString();
  document.getElementById('stat-visible').textContent = n.toLocaleString();
  document.getElementById('stat-collection').textContent = currentCollection;
  setLoadStatus('READY', 'muted');

  allPoints = points;
  currentProjectionMeta = projectionMeta;
  updateProjectionAxes(currentProjectionMeta);
  basePointColors = new Float32Array(colors);
  basePointSizes = new Float32Array(sizes);
  selectedPoint = null;
  selectedPointID = '';
  lastHoverIdx = -1;
  resetSignalScannerUI();
  applySignalScanHighlight();
  setTimelineReveal(timelineReveal);
  applyPendingLens();
  autoFitCamera();
}

function autoFitCamera() {
  if (!camera || !controls || !particleSystem) return;

  particleSystem.geometry.computeBoundingBox();
  const box = particleSystem.geometry.boundingBox;
  if (!box) return;

  const center = new THREE.Vector3();
  box.getCenter(center);

  const size = new THREE.Vector3();
  box.getSize(size);
  const radius = Math.max(size.length() * 0.5, 1);

  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(0, 0, radius * 2.5));
  camera.near = Math.max(0.1, radius / 100);
  camera.far = Math.max(5000, radius * 20);
  camera.updateProjectionMatrix();

  if (typeof controls._updateSpherical === 'function') {
    controls._updateSpherical();
  }
}

function buildLegend(clusterMap) {
  const el = document.getElementById('legend');
  el.innerHTML = '';

  const sorted = Array.from(clusterMap.entries())
    .sort((a, b) => b[1].count - a[1].count);

  const top = sorted.slice(0, 10);
  const colorObj = new THREE.Color();
  for (const [key, meta] of top) {
    getAdjustedPaletteColor(meta.index % activePalette.length, colorObj);
    const color = '#' + colorObj.getHexString();
    el.innerHTML += `<div class="legend-item">
      <div class="legend-dot" style="background:${color};box-shadow:0 0 6px ${color}"></div>
      <span style="color:var(--text)">${escapeHTML(String(key))} <span style="color:var(--dim)">(${meta.count})</span></span>
    </div>`;
  }

  const remaining = sorted.length - top.length;
  if (remaining > 0) {
    el.innerHTML += `<div style="color:var(--dim);font-size:0.6rem">…${remaining} more</div>`;
  }

  if (outlierCount > 0) {
    const outlierHex = '#' + outlierColor.getHexString();
    el.innerHTML += `<div class="legend-item">
      <div class="legend-dot" style="background:${outlierHex};box-shadow:0 0 6px ${outlierHex}"></div>
      <span style="color:var(--text)">outliers <span style="color:var(--dim)">(${outlierCount})</span></span>
    </div>`;
  }
}

function updateDistanceHeatmap(points, clusterMap, positions) {
  const canvas = document.getElementById('distance-heatmap');
  const labels = document.getElementById('distance-labels');
  if (!canvas || !labels) return;

  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#060b16';
  ctx.fillRect(0, 0, width, height);

  const topClusters = Array.from(clusterMap.entries())
    .sort((a, b) => b[1].count - a[1].count)
    .slice(0, 6)
    .map(([key]) => key);

  if (topClusters.length < 2 || points.length < 2) {
    labels.textContent = 'Need at least two clusters for distance matrix.';
    return;
  }

  const sums = new Map();
  const counts = new Map();
  for (const key of topClusters) {
    sums.set(key, { x: 0, y: 0, z: 0 });
    counts.set(key, 0);
  }

  for (let i = 0; i < points.length; i++) {
    const key = extractClusterKey(points[i]);
    if (!sums.has(key)) continue;
    const acc = sums.get(key);
    acc.x += positions[i * 3];
    acc.y += positions[i * 3 + 1];
    acc.z += positions[i * 3 + 2];
    counts.set(key, counts.get(key) + 1);
  }

  const activeKeys = topClusters.filter((key) => counts.get(key) > 0);
  if (activeKeys.length < 2) {
    labels.textContent = 'Need at least two visible clusters for matrix.';
    return;
  }

  const centroids = new Map();
  for (const key of activeKeys) {
    const c = counts.get(key);
    const s = sums.get(key);
    centroids.set(key, { x: s.x / c, y: s.y / c, z: s.z / c });
  }

  const matrix = [];
  let maxDist = 0;
  for (let row = 0; row < activeKeys.length; row++) {
    const rowVals = [];
    for (let col = 0; col < activeKeys.length; col++) {
      const a = centroids.get(activeKeys[row]);
      const b = centroids.get(activeKeys[col]);
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dz = a.z - b.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      rowVals.push(dist);
      if (dist > maxDist) maxDist = dist;
    }
    matrix.push(rowVals);
  }

  const n = activeKeys.length;
  const cell = Math.floor(Math.min(width, height) / n);
  const drawW = cell * n;
  const drawH = cell * n;

  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const normalized = maxDist <= 1e-9 ? 0 : matrix[row][col] / maxDist;
      const hue = 205 - normalized * 205;
      const light = 24 + normalized * 50;
      ctx.fillStyle = `hsl(${hue}, 92%, ${light}%)`;
      ctx.fillRect(col * cell, row * cell, cell, cell);
    }
  }

  ctx.strokeStyle = '#1a2a4a';
  ctx.lineWidth = 1;
  for (let i = 0; i <= n; i++) {
    ctx.beginPath();
    ctx.moveTo(0, i * cell + 0.5);
    ctx.lineTo(drawW, i * cell + 0.5);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(i * cell + 0.5, 0);
    ctx.lineTo(i * cell + 0.5, drawH);
    ctx.stroke();
  }

  labels.innerHTML = activeKeys.map((key, idx) => `${idx + 1}. ${escapeHTML(key)}`).join(' · ');
}

function updateCollectionMetaPanel(collectionName) {
  const meta = collectionMetaByName.get(collectionName) || null;
  document.getElementById('meta-name').textContent = collectionName || '—';
  document.getElementById('meta-points').textContent = meta && Number.isFinite(meta.points_count) ? Number(meta.points_count).toLocaleString() : '—';
  document.getElementById('meta-dim').textContent = meta && meta.vector_size ? String(meta.vector_size) : '—';
  document.getElementById('meta-distance').textContent = meta && meta.distance_metric ? String(meta.distance_metric) : '—';
  if (!meta) {
    document.getElementById('meta-projection').textContent = '—';
    return;
  }
  const note = meta.projection_note ? ` · ${meta.projection_note}` : '';
  const vectorLabel = selectedVectorName ? ` · vector ${selectedVectorName}` : '';
  document.getElementById('meta-projection').textContent = `${meta.projection_ready === false ? 'unsupported' : 'ready'}${note}${vectorLabel}`;
}

function updateProjectionAxes(meta) {
  const el = document.getElementById('projection-axes');
  if (!el) return;

  if (!meta || !Array.isArray(meta.axes) || meta.axes.length === 0) {
    el.textContent = 'Load a projection to see axis variance summary.';
    return;
  }

  const labels = meta.axes.slice(0, 3).map((axis, idx) => {
    const component = axis && axis.component ? axis.component : `PC${idx + 1}`;
    const variance = Number(axis?.variance_explained);
    const percent = Number.isFinite(variance) ? variance : 0;
    return `${component} ${percent.toFixed(1)}%`;
  });

  const suffix = meta.method === 'pca-go-fallback' ? ' · inline Go PCA (Python worker unavailable)' : '';
  el.textContent = labels.join(' · ') + suffix;
}

// ── Inspector ─────────────────────────────────────────────────────────────────
function escapeHTML(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function shortText(value, maxLen) {
  if (value === null || value === undefined) return '';
  let s;
  if (Array.isArray(value)) {
    s = value.map(v => typeof v === 'object' ? JSON.stringify(v) : String(v)).join(', ');
  } else if (typeof value === 'object') {
    s = JSON.stringify(value);
  } else {
    s = String(value);
  }
  return s.length > maxLen ? s.slice(0, maxLen) + '…' : s;
}

function getSignalTitle(payload, fallbackID) {
  const title = payloadLookup(payload, 'source_title', 'title', 'attachment_filename', 'question', 'thought_id', 'memory_id', 'conversation_id', 'project', 'source_id', 'file_source', 'source_file');
  return title !== undefined ? title : `Signal ${fallbackID}`;
}

function getSignalSnippet(payload) {
  const snippet = payloadLookup(payload, 'evidence_summary', 'source_quote', 'text', 'content', 'chunk_text', 'summary', 'claims', 'questions');
  return snippet !== undefined ? snippet : 'No preview text available.';
}

function setScanStatus(message, kind = 'muted') {
  const el = document.getElementById('scan-status');
  el.textContent = message || '';
  if (kind === 'error') {
    el.style.color = 'var(--neon-o)';
  } else if (kind === 'ok') {
    el.style.color = 'var(--neon-g)';
  } else {
    el.style.color = 'var(--dim)';
  }
}

function setLoadStatus(text, kind = 'muted') {
  const el = document.getElementById('stat-load');
  if (!el) return;
  el.textContent = text || 'READY';
  if (kind === 'error') {
    el.style.color = 'var(--neon-o)';
  } else if (kind === 'ok') {
    el.style.color = 'var(--neon-g)';
  } else {
    el.style.color = 'var(--text)';
  }
}

function isQdrantUnreachableError(message) {
  const m = String(message || '').toLowerCase();
  return m.includes('failed to fetch') ||
    m.includes('networkerror') ||
    m.includes('network error') ||
    m.includes('connection refused') ||
    m.includes('timed out') ||
    m.includes('timeout') ||
    m.includes('qdrant') ||
    m.includes('502') ||
    m.includes('503') ||
    m.includes('504');
}

function showErrorOverlay(contextLabel, rawError) {
  const overlay = document.getElementById('error-overlay');
  const detail = document.getElementById('error-detail');
  detail.textContent = `${contextLabel}: ${String(rawError || 'unknown error')}`;
  overlay.classList.remove('hidden');
}

function hideErrorOverlay() {
  document.getElementById('error-overlay').classList.add('hidden');
}

function maybeShowQdrantError(contextLabel, err) {
  const raw = err && err.message ? err.message : err;
  if (isQdrantUnreachableError(raw)) {
    showErrorOverlay(contextLabel, raw);
  }
}

function updateLoadingProgressText(progress, collectionLabel) {
  const loadingSub = document.getElementById('loading-sub');
  const total = Number(progress?.limit) || 0;
  const loaded = Number(progress?.loaded) || 0;
  const percent = Number(progress?.percent) || 0;
  const upperCollection = String(collectionLabel || '').toUpperCase();

  if (progress?.status === 'scrolling') {
    const countText = total > 0 ? `${Math.min(loaded, total).toLocaleString()} / ${total.toLocaleString()}` : `${loaded.toLocaleString()}`;
    loadingSub.textContent = `PULLING ${countText} VECTORS FROM ${upperCollection}… ${Math.min(99, percent)}%`;
    setLoadStatus(`FETCH ${Math.min(99, percent)}%`, 'muted');
    return;
  }
  if (progress?.status === 'projecting') {
    loadingSub.textContent = `PROJECTING ${Math.max(loaded, total).toLocaleString()} VECTORS…`;
    setLoadStatus('PROJECTING', 'ok');
    return;
  }
  if (progress?.status === 'done') {
    loadingSub.textContent = 'RENDERING PARTICLE CLOUD…';
    setLoadStatus('RENDERING', 'ok');
    return;
  }
  if (progress?.status === 'failed') {
    setLoadStatus('ERROR', 'error');
  }
}

function endLoadProgress() {
  currentLoadProgressID = '';
}

function beginLoadProgress(progressID, collectionLabel) {
  currentLoadProgressID = progressID || '';
  currentLoadCollectionLabel = collectionLabel || '';
  startUIEventStream();
}

async function retryLoadFromError() {
  hideErrorOverlay();
  await loadCollections();
  await loadPoints(currentCollection, pointLimit);
}

function showInspector(pt, mode = 'selected') {
  if (!pt) return;
  document.getElementById('inspector-empty').style.display = 'none';
  const el = document.getElementById('inspector-content');
  el.classList.add('active');

  const payload = pt.payload || {};
  const modeLabel = mode === 'hover' ? 'HOVER' : 'SELECTED';
  let html = `<div class="insp-field"><div class="insp-key">Signal</div><div class="insp-val highlight">${modeLabel}</div></div>`;
  html += `<div class="insp-field"><div class="insp-key">ID</div><div class="insp-val highlight">${escapeHTML(pt.id)}</div></div>`;
  html += `<div class="insp-field"><div class="insp-key">PCA Coords</div><div class="insp-val">${pt.x.toFixed(3)}, ${pt.y.toFixed(3)}, ${pt.z.toFixed(3)}</div></div>`;

  const priority = [
    'entity_type', 'source_title', 'attachment_filename', 'attachment_category',
    'speaker', 'memory_kind', 'project', 'source_type', 'conversation_id', 'session_id',
    'thought_id', 'memory_id', 'question', 'type', 'strategy', 'uncertainty_class',
    'importance', 'evidence_strength', 'provenance_score', 'confidence',
    'source_quote', 'evidence_summary', 'summary', 'text', 'content', 'chunk_text', 'document',
    'claims', 'concepts', 'questions', 'tags', 'sources', 'tone', 'author',
    'timestamp', 'date_basis', 'approximate_date', 'awakening_phase',
    'ai_provider', 'ai_model', 'embedding_model', 'source_id', 'source_collection',
    'source_file', 'chunk_index'
  ];
  const longTextKeys = new Set(['text', 'content', 'chunk_text', 'document', 'source_quote', 'summary', 'evidence_summary', 'question', 'page_segment']);
  const shown = new Set();
  for (const k of priority) {
    const value = payloadLookup(payload, k);
    if (value === undefined) continue;
    const canonicalKey = k.toLowerCase();
    if (shown.has(canonicalKey)) continue;
    const long = longTextKeys.has(canonicalKey) && typeof value === 'string' && value.length > 200;
    const v = shortText(value, long ? 1200 : 200);
    html += `<div class="insp-field"><div class="insp-key">${escapeHTML(k)}</div><div class="insp-val${long ? ' long' : ''}">${escapeHTML(v)}</div></div>`;
    shown.add(canonicalKey);
  }
  for (const [k, v] of Object.entries(payload)) {
    if (shown.has(k.toLowerCase())) continue;
    const sv = shortText(v, 120);
    html += `<div class="insp-field"><div class="insp-key">${escapeHTML(k)}</div><div class="insp-val">${escapeHTML(sv)}</div></div>`;
  }
  el.innerHTML = html;
}

function cancelCameraFlight() {
  cameraFlight = null;
}

function flyCameraTo(targetPoint, distance) {
  if (!controls || !camera) return;
  const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const direction = camera.position.clone().sub(controls.target);
  if (direction.lengthSq() < 1e-6) direction.set(0, 0, 1);
  direction.normalize();

  if (reduceMotion) {
    controls.target.copy(targetPoint);
    camera.position.copy(targetPoint.clone().add(direction.multiplyScalar(distance)));
    if (typeof controls._updateSpherical === 'function') controls._updateSpherical();
    cameraFlight = null;
    return;
  }

  cameraFlight = {
    start: performance.now(),
    duration: 650,
    startTarget: controls.target.clone(),
    endTarget: targetPoint.clone(),
    startPos: camera.position.clone(),
    endPos: targetPoint.clone().add(direction.multiplyScalar(distance)),
  };
}

function stepCameraFlight(now) {
  if (!cameraFlight) return;
  const t = Math.min(1, (now - cameraFlight.start) / cameraFlight.duration);
  const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  controls.target.lerpVectors(cameraFlight.startTarget, cameraFlight.endTarget, eased);
  camera.position.lerpVectors(cameraFlight.startPos, cameraFlight.endPos, eased);
  if (typeof controls._updateSpherical === 'function') controls._updateSpherical();
  if (t >= 1) cameraFlight = null;
}

function flyToIndices(indices) {
  if (!particleSystem || !controls || !camera) return;
  const positions = particleSystem.geometry.attributes.position.array;
  const centroid = new THREE.Vector3();
  let counted = 0;
  for (const index of indices.slice(0, 48)) {
    if (!Number.isInteger(index) || index < 0 || index * 3 + 2 >= positions.length) continue;
    centroid.x += positions[index * 3];
    centroid.y += positions[index * 3 + 1];
    centroid.z += positions[index * 3 + 2];
    counted += 1;
  }
  if (!counted) return;
  centroid.multiplyScalar(1 / counted);
  const currentDistance = camera.position.distanceTo(controls.target) || 120;
  const distance = Math.max(70, Math.min(currentDistance, 240));
  flyCameraTo(centroid, distance);
}

const BOOKMARKS_KEY = 'vectorview_bookmarks';
const HISTORY_LIMIT = 50;

function pointLabel(point) {
  if (!point) return 'unknown';
  const payload = point.payload || {};
  for (const key of ['title', 'source_title', 'text', 'content', 'chunk_text', 'summary', 'question']) {
    const value = payload[key];
    if (typeof value === 'string' && value.trim()) return shortText(value.trim(), 48);
  }
  return String(pointIDKey(point.id));
}

function loadBookmarks() {
  try {
    const raw = localStorage.getItem(BOOKMARKS_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    bookmarks = Array.isArray(parsed)
      ? parsed.filter((entry) => entry && entry.id && entry.collection)
      : [];
  } catch {
    bookmarks = [];
  }
  renderBookmarks();
}

function persistBookmarks() {
  try {
    localStorage.setItem(BOOKMARKS_KEY, JSON.stringify(bookmarks));
  } catch {
  }
}

function bookmarkKey(entry) {
  return `${entry.collection}::${entry.id}`;
}

function isBookmarked(point, collection) {
  if (!point) return false;
  const key = `${collection}::${pointIDKey(point.id)}`;
  return bookmarks.some((entry) => bookmarkKey(entry) === key);
}

function toggleBookmark() {
  if (!selectedPoint) return;
  const id = pointIDKey(selectedPoint.id);
  const key = `${currentCollection}::${id}`;
  const existing = bookmarks.findIndex((entry) => bookmarkKey(entry) === key);
  if (existing >= 0) {
    bookmarks.splice(existing, 1);
    setScanStatus('Bookmark removed.', 'muted');
  } else {
    bookmarks.unshift({
      id,
      collection: currentCollection,
      label: pointLabel(selectedPoint),
      saved_at: Date.now(),
    });
    setScanStatus('Bookmarked.', 'ok');
  }
  persistBookmarks();
  renderBookmarks();
  refreshBookmarkButton();
}

function renderBookmarks() {
  const list = document.getElementById('bookmarks-list');
  const empty = document.getElementById('bookmarks-empty');
  const count = document.getElementById('val-bookmark-count');
  if (!list || !empty || !count) return;

  count.textContent = `(${bookmarks.length})`;
  empty.style.display = bookmarks.length ? 'none' : 'block';
  list.innerHTML = '';

  for (const entry of bookmarks) {
    const row = document.createElement('div');
    row.className = 'bookmark-item';

    const label = document.createElement('span');
    label.className = 'bookmark-label';
    label.textContent = entry.label || entry.id;
    label.title = `${entry.collection} · ${entry.id}`;
    label.addEventListener('click', () => jumpToBookmark(entry));

    const remove = document.createElement('button');
    remove.textContent = '✕';
    remove.title = 'Remove bookmark';
    remove.addEventListener('click', (event) => {
      event.stopPropagation();
      bookmarks = bookmarks.filter((candidate) => bookmarkKey(candidate) !== bookmarkKey(entry));
      persistBookmarks();
      renderBookmarks();
    });

    row.appendChild(label);
    row.appendChild(remove);
    list.appendChild(row);
  }
}

function jumpToBookmark(entry) {
  if (!entry || !entry.id) return;
  if (entry.collection !== currentCollection) {
    setScanStatus(`Bookmark belongs to ${entry.collection}. Load that collection first.`, 'error');
    return;
  }
  const index = pointIndexByID.get(pointIDKey(entry.id));
  if (!Number.isInteger(index)) {
    setScanStatus('Bookmark is outside the visible sample.', 'error');
    return;
  }
  selectPointByIndex(index);
  flyToIndices([index]);
}

function refreshBookmarkButton() {
  const button = document.getElementById('btn-bookmark');
  if (!button) return;
  button.disabled = !selectedPoint;
  button.textContent = isBookmarked(selectedPoint, currentCollection) ? 'UNBOOKMARK' : 'BOOKMARK';
}

function historyEntryFor(point) {
  if (!point) return null;
  return { id: pointIDKey(point.id), collection: currentCollection, label: pointLabel(point) };
}

function recordSelection(point) {
  if (!point || suppressHistoryPush) return;
  const entry = historyEntryFor(point);
  if (!entry || !entry.id) return;
  const current = selectionHistory[selectionHistoryIndex];
  if (current && current.collection === entry.collection && current.id === entry.id) return;
  selectionHistory = selectionHistory.slice(0, selectionHistoryIndex + 1);
  selectionHistory.push(entry);
  if (selectionHistory.length > HISTORY_LIMIT) selectionHistory.shift();
  selectionHistoryIndex = selectionHistory.length - 1;
  refreshHistoryButtons();
}

function refreshHistoryButtons() {
  const back = document.getElementById('btn-history-back');
  const forward = document.getElementById('btn-history-forward');
  if (back) back.disabled = selectionHistoryIndex <= 0;
  if (forward) forward.disabled = selectionHistoryIndex < 0 || selectionHistoryIndex >= selectionHistory.length - 1;
}

function stepSelectionHistory(direction) {
  const next = selectionHistoryIndex + direction;
  if (next < 0 || next >= selectionHistory.length) return;
  const entry = selectionHistory[next];
  if (entry.collection !== currentCollection) {
    setScanStatus(`History entry belongs to ${entry.collection}.`, 'error');
    return;
  }
  const index = pointIndexByID.get(pointIDKey(entry.id));
  if (!Number.isInteger(index)) {
    setScanStatus('History entry is outside the visible sample.', 'error');
    return;
  }
  selectionHistoryIndex = next;
  suppressHistoryPush = true;
  selectPointByIndex(index);
  suppressHistoryPush = false;
  flyToIndices([index]);
  refreshHistoryButtons();
}

function focusPointByIndex(index, closeInspect = false) {
  if (!particleSystem || !controls || !camera || index < 0) return;
  const pos = particleSystem.geometry.attributes.position.array;
  const target = new THREE.Vector3(pos[index * 3], pos[index * 3 + 1], pos[index * 3 + 2]);
  const currentDistance = camera.position.distanceTo(controls.target);
  const distance = closeInspect ? Math.max(28, currentDistance * 0.38) : Math.max(60, currentDistance);
  const direction = camera.position.clone().sub(controls.target).normalize();
  controls.target.copy(target);
  camera.position.copy(target.clone().add(direction.multiplyScalar(distance)));
  if (typeof controls._updateSpherical === 'function') controls._updateSpherical();
}

function updateSignalButtons() {
  const hasSelection = !!selectedPoint;
  document.getElementById('btn-find-similar').disabled = !hasSelection;
  document.getElementById('btn-radius-scan').disabled = !hasSelection;
  document.getElementById('btn-clear-scan').disabled = !similarityModeActive;
}

function applySignalScanHighlight() {
  if (!particleSystem || !basePointColors || !basePointSizes) return;
  const colorAttr = particleSystem.geometry.attributes.color;
  const sizeAttr = particleSystem.geometry.attributes.size;
  const scanAttr = particleSystem.geometry.attributes.scanFlag;
  if (!colorAttr || !sizeAttr || !scanAttr) return;

  // Point highlighting mode uses scanFlag: 0=normal, 1=dimmed, 2=similar, 3=pulse, 4=popout.
  colorAttr.array.set(basePointColors);
  sizeAttr.array.set(basePointSizes);
  scanAttr.array.fill(0);

  if (similarityModeActive) {
    for (let i = 0; i < allPoints.length; i++) {
      const id = pointIDKey(allPoints[i].id);
      if (id === poppedSimilarPointID) {
        scanAttr.array[i] = 4;
      } else if (id === hoveredSimilarPointID || id === selectedPointID) {
        scanAttr.array[i] = 3;
      } else if (scanNeighborIDs.has(id)) {
        scanAttr.array[i] = 2;
      } else {
        scanAttr.array[i] = 1;
      }
    }
  }

  colorAttr.needsUpdate = true;
  sizeAttr.needsUpdate = true;
  scanAttr.needsUpdate = true;
}

function resetSignalScannerUI() {
  similarityModeActive = false;
  hoveredSimilarPointID = '';
  poppedSimilarPointID = '';
  scanNeighborIDs = new Set();
  scanResults = [];
  setScanStatus('');
  document.getElementById('similar-list').innerHTML = '';
  document.getElementById('similar-empty').style.display = 'block';
  document.getElementById('similar-empty').textContent = 'Select a point and run Find Similar.';
  updateSignalButtons();
}

function clearSignalScan() {
  similarityModeActive = false;
  hoveredSimilarPointID = '';
  poppedSimilarPointID = '';
  scanNeighborIDs = new Set();
  scanResults = [];
  applySignalScanHighlight();
  renderSimilarSignals([]);
  setScanStatus('Scan cleared.', 'muted');
  updateSignalButtons();
}

function updateSimilarItemState(item, idKey) {
  item.classList.toggle('focused', idKey !== '' && idKey === hoveredSimilarPointID);
  item.classList.toggle('inspecting', idKey !== '' && idKey === poppedSimilarPointID);
}

function syncSimilarItemStates() {
  const list = document.getElementById('similar-list');
  for (const child of list.children) {
    const childID = child.dataset.signalId || '';
    updateSimilarItemState(child, childID);
  }
}

function renderSimilarSignals(results) {
  const list = document.getElementById('similar-list');
  const empty = document.getElementById('similar-empty');
  list.innerHTML = '';

  if (!results || results.length === 0) {
    hoveredSimilarPointID = '';
    poppedSimilarPointID = '';
    empty.style.display = 'block';
    empty.textContent = similarityModeActive ? 'No neighbors returned.' : 'Select a point and run Find Similar.';
    applySignalScanHighlight();
    return;
  }

  empty.style.display = 'none';
  for (let i = 0; i < results.length; i++) {
    const row = results[i];
    const idKey = pointIDKey(row.id);
    const idx = pointIndexByID.get(idKey);
    const payload = row.payload || {};
    const title = shortText(getSignalTitle(payload, row.id), 80);
    const snippet = shortText(getSignalSnippet(payload), 120);
    const source = payloadLookup(payload, 'source_title', 'attachment_filename', 'project', 'conversation_id', 'source_type', 'file_source', 'source_id', 'source_file', 'source', 'date') || 'unknown source';
    const offscreen = idx === undefined;

    const item = document.createElement('div');
    item.className = `similar-item${offscreen ? ' offscreen' : ''}`;
    item.dataset.signalId = idKey;
    item.innerHTML = `
      <div class="similar-rank">${i + 1}. SIGNAL ${escapeHTML(row.id)}</div>
      <div class="similar-title">${escapeHTML(title)}</div>
      <div class="similar-meta">score: ${(Number(row.score) || 0).toFixed(3)} · ${escapeHTML(shortText(source, 38))}</div>
      <div class="similar-snippet">${escapeHTML(snippet)}</div>
      ${offscreen ? '<div class="similar-meta">not in current visible sample</div>' : ''}
    `;
    if (!offscreen) {
      item.addEventListener('mouseenter', () => {
        hoveredSimilarPointID = idKey;
        updateSimilarItemState(item, idKey);
        applySignalScanHighlight();
      });
      item.addEventListener('mouseleave', () => {
        hoveredSimilarPointID = '';
        updateSimilarItemState(item, idKey);
        applySignalScanHighlight();
      });
      item.addEventListener('click', () => {
        poppedSimilarPointID = idKey;
        hoveredSimilarPointID = idKey;
        syncSimilarItemStates();
        selectPointByIndex(idx, { fromSimilarPane: true });
        focusPointByIndex(idx, true);
      });
    }
    updateSimilarItemState(item, idKey);
    list.appendChild(item);
  }
}

function selectPointByIndex(index, options = {}) {
  if (index < 0 || index >= allPoints.length) return;
  const fromSimilarPane = options.fromSimilarPane === true;
  selectedPoint = allPoints[index];
  selectedPointID = pointIDKey(selectedPoint.id);
  if (!fromSimilarPane) {
    hoveredSimilarPointID = '';
    poppedSimilarPointID = '';
    syncSimilarItemStates();
  }
  showInspector(selectedPoint, 'selected');
  hydrateFullPayload(index);
  recordSelection(selectedPoint);
  refreshBookmarkButton();
  updateSignalButtons();
  if (similarityModeActive) {
    applySignalScanHighlight();
  }
}

async function hydrateFullPayload(index) {
  const point = allPoints[index];
  if (!point || !point.payload_truncated) return;
  const id = pointIDKey(point.id);
  try {
    const res = await fetch(`/api/collections/${encodeURIComponent(currentCollection)}/points/${encodeURIComponent(id)}`);
    if (!res.ok) return;
    const data = await res.json();
    if (!data || !data.payload) return;
    if (pointIDKey(point.id) !== id) return;
    point.payload = data.payload;
    point.payload_truncated = false;
    if (selectedPoint === point) {
      showInspector(selectedPoint, 'selected');
    }
  } catch {
  }
}

function clearInspectorSelection() {
  selectedPoint = null;
  selectedPointID = '';
  hoveredSimilarPointID = '';
  poppedSimilarPointID = '';
  document.getElementById('inspector-content').classList.remove('active');
  document.getElementById('inspector-content').innerHTML = '';
  document.getElementById('inspector-empty').style.display = 'block';
  if (similarityModeActive) {
    syncSimilarItemStates();
    applySignalScanHighlight();
  }
  refreshBookmarkButton();
  updateSignalButtons();
}

function handleGlobalShortcuts(event) {
  if (event.key === 'Escape') {
    clearInspectorSelection();
    return;
  }

  if (event.altKey && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) {
    event.preventDefault();
    stepSelectionHistory(event.key === 'ArrowLeft' ? -1 : 1);
    return;
  }

  const target = event.target;
  const tag = target && target.tagName ? target.tagName.toLowerCase() : '';
  const isTyping = tag === 'input' || tag === 'textarea' || (target && target.isContentEditable);
  if (isTyping) return;

  if (event.key === 'r' || event.key === 'R') {
    event.preventDefault();
    loadPoints(currentCollection, pointLimit);
    document.getElementById('search-status').textContent = '';
    document.getElementById('search-input').value = '';
    return;
  }

  if (event.code === 'Space' || event.key === ' ') {
    event.preventDefault();
    controls.autoRotate = !controls.autoRotate;
  }
}

function applySimilarityResponse(neighbors) {
  scanResults = Array.isArray(neighbors) ? neighbors : [];
  scanNeighborIDs = new Set(scanResults.map((n) => pointIDKey(n.id)));
  similarityModeActive = true;
  hoveredSimilarPointID = '';
  poppedSimilarPointID = '';
  applySignalScanHighlight();
  renderSimilarSignals(scanResults);

  let visibleHits = 0;
  for (const id of scanNeighborIDs) {
    if (pointIndexByID.has(id)) visibleHits++;
  }
  const offscreen = scanResults.length - visibleHits;
  const note = offscreen > 0 ? ` · ${offscreen} outside visible sample` : '';
  const neighborIndices = scanResults
    .map((neighbor) => pointIndexByID.get(pointIDKey(neighbor.id)))
    .filter((index) => Number.isInteger(index));
  flyToIndices(neighborIndices);
  return { visibleHits, note };
}

async function runFindSimilar() {
  if (!selectedPoint || !selectedPointID) {
    setScanStatus('Select a signal first.', 'error');
    return;
  }

  setScanStatus('Querying nearest neighbors...', 'muted');
  try {
    const vectorQuery = selectedVectorName ? `&vector_name=${encodeURIComponent(selectedVectorName)}` : '';
    const res = await fetch(`/api/collections/${encodeURIComponent(currentCollection)}/points/${encodeURIComponent(selectedPointID)}/similar?limit=12${vectorQuery}`);
    if (!res.ok) {
      const message = await readApiError(res, '');
      throw new Error(message || 'similar search failed');
    }

    const data = await res.json();
    const summary = applySimilarityResponse(data.neighbors);
    setScanStatus(`${scanResults.length} similar signals · ${summary.visibleHits} highlighted${summary.note}`, 'ok');
  } catch (err) {
    maybeShowQdrantError('Similarity scan failed', err);
    similarityModeActive = false;
    hoveredSimilarPointID = '';
    poppedSimilarPointID = '';
    scanNeighborIDs = new Set();
    scanResults = [];
    applySignalScanHighlight();
    renderSimilarSignals([]);
    setScanStatus(`Scan failed: ${err.message}`, 'error');
  }

  updateSignalButtons();
}

async function runSimilarityRadius() {
  if (!selectedPoint || !selectedPointID) {
    setScanStatus('Select a signal first.', 'error');
    return;
  }

  setScanStatus(`Scanning cosine radius ≥ ${similarityRadius.toFixed(2)}...`, 'muted');
  try {
    const vectorQuery = selectedVectorName ? `&vector_name=${encodeURIComponent(selectedVectorName)}` : '';
    const res = await fetch(`/api/collections/${encodeURIComponent(currentCollection)}/points/${encodeURIComponent(selectedPointID)}/similar-radius?radius=${encodeURIComponent(similarityRadius.toFixed(2))}&limit=400${vectorQuery}`);
    if (!res.ok) {
      const message = await readApiError(res, '');
      throw new Error(message || 'radius scan failed');
    }

    const data = await res.json();
    const summary = applySimilarityResponse(data.neighbors);
    setScanStatus(`${scanResults.length} signals within cosine ≥ ${similarityRadius.toFixed(2)} · ${summary.visibleHits} highlighted${summary.note}`, 'ok');
  } catch (err) {
    maybeShowQdrantError('Radius scan failed', err);
    similarityModeActive = false;
    hoveredSimilarPointID = '';
    poppedSimilarPointID = '';
    scanNeighborIDs = new Set();
    scanResults = [];
    applySignalScanHighlight();
    renderSimilarSignals([]);
    setScanStatus(`Radius scan failed: ${err.message}`, 'error');
  }

  updateSignalButtons();
}

function applyExternalHighlight(event) {
  const rawIDs = Array.isArray(event?.ids) ? event.ids : [];
  const ids = [];
  const seen = new Set();
  for (const rawID of rawIDs) {
    const key = pointIDKey(rawID);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    ids.push(key);
  }
  if (ids.length === 0) {
    clearSignalScan();
    setScanStatus('External highlight cleared.', 'muted');
    return;
  }

  const focusKey = pointIDKey(event?.focus_id) || ids[0];
  const focusIndex = pointIndexByID.get(focusKey);
  if (Number.isInteger(focusIndex) && focusIndex >= 0 && focusIndex < allPoints.length) {
    selectPointByIndex(focusIndex);
    flyToIndices([focusIndex]);
  }

  similarityModeActive = true;
  hoveredSimilarPointID = '';
  poppedSimilarPointID = '';
  scanNeighborIDs = new Set(ids);
  scanResults = ids.map((idKey) => {
    const idx = pointIndexByID.get(idKey);
    if (Number.isInteger(idx) && idx >= 0 && idx < allPoints.length) {
      const point = allPoints[idx];
      return { id: point.id, score: 1, payload: point.payload || {} };
    }
    return { id: idKey, score: 1, payload: {} };
  });

  applySignalScanHighlight();
  renderSimilarSignals(scanResults);

  let visibleHits = 0;
  for (const idKey of ids) {
    if (pointIndexByID.has(idKey)) visibleHits++;
  }
  const offscreen = ids.length - visibleHits;
  const note = offscreen > 0 ? ` · ${offscreen} outside visible sample` : '';
  setScanStatus(`External highlight: ${ids.length} signals · ${visibleHits} highlighted${note}`, 'ok');
  updateSignalButtons();
}

async function refreshExternalHighlightOnce() {
  if (!currentCollection) return;
  const sinceQuery = lastHighlightEventID ? `&since=${encodeURIComponent(lastHighlightEventID)}` : '';
  try {
    const res = await fetch(`/api/highlight?collection=${encodeURIComponent(currentCollection)}${sinceQuery}`);
    if (res.status === 204) return;
    if (!res.ok) {
      const message = await readApiError(res, '');
      throw new Error(message || 'highlight refresh failed');
    }
    const event = await res.json();
    const eventID = String(event?.event_id || '').trim();
    if (!eventID || eventID === lastHighlightEventID) return;
    lastHighlightEventID = eventID;
    applyExternalHighlight(event);
  } catch {
  }
}

function startUIEventStream() {
  if (uiEventSource && uiEventSourceCollection === currentCollection) return;
  if (uiEventSource) {
    uiEventSource.close();
    uiEventSource = null;
  }
  uiEventSourceCollection = currentCollection;
  const source = new EventSource(`/api/stream?collection=${encodeURIComponent(currentCollection)}&max_events_per_second=30`);
  uiEventSource = source;

  source.addEventListener('stream_ready', () => {
    if (source === uiEventSource) refreshExternalHighlightOnce();
  });

  source.addEventListener('load_progress', (event) => {
    if (source !== uiEventSource) return;
    let envelope;
    try { envelope = JSON.parse(event.data); } catch { return; }
    const payload = envelope && envelope.payload ? envelope.payload : null;
    if (!payload) return;
    if (currentLoadProgressID && payload.progress_id && payload.progress_id !== currentLoadProgressID) return;
    updateLoadingProgressText(payload, currentLoadCollectionLabel);
    if (payload.done) endLoadProgress();
  });

  source.addEventListener('highlight', (event) => {
    if (source !== uiEventSource) return;
    let envelope;
    try { envelope = JSON.parse(event.data); } catch { return; }
    const payload = envelope && envelope.payload ? envelope.payload : null;
    if (!payload) return;
    const eventID = String(payload.event_id || '').trim();
    if (eventID && eventID === lastHighlightEventID) return;
    if (eventID) lastHighlightEventID = eventID;
    applyExternalHighlight(payload);
  });

  source.onerror = () => {
    setScanStatus('Event stream reconnecting...', 'muted');
  };
}

// ── Raycasting hover ──────────────────────────────────────────────────────────
let lastHoverIdx = -1;
function checkHover() {
  hoverDirty = false;
  if (!particleSystem || allPoints.length === 0) return;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObject(particleSystem);
  const idx = hits.length > 0 ? hits[0].index : -1;
  if (idx !== lastHoverIdx) {
    lastHoverIdx = idx;
    if (idx >= 0) {
      renderer.domElement.style.cursor = 'pointer';
      if (!selectedPoint) {
        showInspector(allPoints[idx], 'hover');
      }
    } else {
      renderer.domElement.style.cursor = 'default';
    }
  }
}

// ── Render loop ───────────────────────────────────────────────────────────────
let lastTime = performance.now();
let frameCount = 0;
let fps = 0;

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const delta = (now - lastTime) / 1000;
  lastTime = now;
  frameCount++;
  if (frameCount % 30 === 0) {
    fps = Math.round(30 / (delta * 30 + 0.0001));
    document.getElementById('stat-fps').textContent = fps;
    if (fps > 0 && fps < 28) {
      lowFpsFrames = Math.min(lowFpsFrames + 1, 10);
    } else {
      lowFpsFrames = Math.max(0, lowFpsFrames - 1);
    }
    if (lowFpsFrames >= 4 && !qualityReduced) {
      qualityReduced = true;
      if (nebulaSystem) nebulaSystem.visible = false;
    } else if (lowFpsFrames === 0 && qualityReduced) {
      qualityReduced = false;
      if (nebulaSystem) nebulaSystem.visible = true;
    }
  }

  if (timelineAutoPlay && allPoints.length > 0) {
    const nextReveal = timelineReveal + delta * 0.05;
    if (nextReveal >= 1) {
      timelineAutoPlay = false;
      setTimelineReveal(1);
    } else {
      setTimelineReveal(nextReveal);
    }
  }

  stepCameraFlight(now);
  controls.setAutoRotateSpeed(rotSpeed);
  controls.update();
  stepLiveDrops(delta);

  const autoRotating = controls.autoRotate && !controls._dragging;
  if (hoverDirty || (autoRotating && now - lastHoverCheck > 100)) {
    lastHoverCheck = now;
    checkHover();
  }

  if (particleSystem) {
    particleSystem.material.uniforms.uTime.value = now * 0.001;
  }
  if (hazeSystem) {
    hazeSystem.material.uniforms.uTime.value = now * 0.001;
  }
  if (glowSystem) {
    glowSystem.material.uniforms.uTime.value = now * 0.001;
  }
  if (nebulaSystem) {
    nebulaSystem.material.uniforms.uTime.value = now * 0.001;
  }

  renderSceneWithTrails();
  renderViewCube();
}

// ── Data loading ──────────────────────────────────────────────────────────────
async function loadCollections() {
  try {
    const res = await fetch('/api/collections');
    if (!res.ok) {
      const message = await readApiError(res, '');
      throw new Error(message || 'failed to load collections');
    }
    const cols = await res.json();
    const sel = document.getElementById('collection-select');
    const crossSel = document.getElementById('cross-target-select');
    const list = Array.isArray(cols) ? cols : [];
    sel.innerHTML = '';
    collectionMetaByName = new Map(list.map((c) => [c.name, c]));

    if (crossSel) {
      crossSel.innerHTML = '<option value="">SAME COLLECTION</option>';
      for (const c of list) {
        const opt = document.createElement('option');
        opt.value = c.name;
        opt.textContent = c.name;
        crossSel.appendChild(opt);
      }
      if (crossCollectionTarget && list.some((c) => c.name === crossCollectionTarget)) {
        crossSel.value = crossCollectionTarget;
      } else {
        crossCollectionTarget = '';
        crossSel.value = '';
      }
    }

    if (list.length === 0) {
      const empty = document.createElement('option');
      empty.value = '';
      empty.textContent = 'No collections available';
      empty.disabled = true;
      empty.selected = true;
      sel.appendChild(empty);
      updateCollectionMetaPanel('');
      return;
    }

    let selectedName = '';
    let firstReadyName = '';
    for (const c of list) {
      const ready = c.projection_ready !== false;
      if (ready && !firstReadyName) firstReadyName = c.name;

      const opt = document.createElement('option');
      opt.value = c.name;
      opt.disabled = !ready;

      const pointLabel = c.points_count?.toLocaleString() ?? '?';
      const dimLabel = c.vector_size ? ` · dim ${c.vector_size}` : '';
      const stateLabel = ready ? '' : ` · ${c.projection_note || 'unsupported'}`;
      opt.textContent = `${c.name} (${pointLabel} pts${dimLabel}${stateLabel})`;

      if (c.name === currentCollection && ready) {
        opt.selected = true;
        selectedName = c.name;
      }
      sel.appendChild(opt);
    }

    if (!selectedName) {
      selectedName = firstReadyName || list[0].name;
      sel.value = selectedName;
    }

    currentCollection = selectedName || currentCollection;
    updateCollectionMetaPanel(currentCollection);
    hideErrorOverlay();
  } catch (e) {
    console.warn('Could not load collections:', e);
    maybeShowQdrantError('Loading collections failed', e);
  }
}

async function loadPoints(collection, limit = pointLimit) {
  document.getElementById('loading').classList.remove('hidden');

  const canIncremental = projectionMethod === 'random' &&
    collection === currentCollection &&
    Array.isArray(allPoints) &&
    allPoints.length > 0 &&
    limit > allPoints.length;
  const appendFrom = canIncremental ? allPoints.length : 0;

  document.getElementById('loading-sub').textContent = appendFrom > 0
    ? `APPENDING ${Math.max(0, limit - appendFrom)} VECTORS TO ${collection.toUpperCase()}…`
    : `PULLING ${limit} VECTORS FROM ${collection.toUpperCase()}… 0%`;
  setLoadStatus(appendFrom > 0 ? 'APPENDING' : 'FETCH 0%', 'muted');

  const progressID = `lp-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  if (appendFrom > 0) {
    endLoadProgress();
  } else {
    beginLoadProgress(progressID, collection);
  }
  try {
    const vectorQuery = selectedVectorName ? `&vector_name=${encodeURIComponent(selectedVectorName)}` : '';
    const appendQuery = appendFrom > 0 ? `&append_from=${appendFrom}` : '';
    const res = await fetch(`/api/points?collection=${encodeURIComponent(collection)}&limit=${limit}&projection=${encodeURIComponent(projectionMethod)}&progress_id=${encodeURIComponent(progressID)}${vectorQuery}${appendQuery}`);
    if (!res.ok) {
      const message = await readApiError(res, '');
      throw new Error(message || 'failed to load points');
    }
    const data = await res.json();
    endLoadProgress();
    document.getElementById('loading-sub').textContent = 'RENDERING PARTICLE CLOUD…';
    setLoadStatus('RENDERING', 'ok');

    const incoming = Array.isArray(data.points) ? data.points : [];
    if (appendFrom > 0 && data.incremental) {
      buildParticles(allPoints.concat(incoming), data.projection || currentProjectionMeta || null);
    } else {
      buildParticles(incoming, data.projection || null);
    }

    lastHighlightEventID = '';
    refreshExternalHighlightOnce();
    hideErrorOverlay();
  } catch(e) {
    endLoadProgress();
    maybeShowQdrantError('Loading points failed', e);
    document.getElementById('loading-sub').textContent = 'ERROR: ' + e.message;
    setLoadStatus('ERROR', 'error');
    return;
  }
  setTimeout(() => document.getElementById('loading').classList.add('hidden'), 500);
}

async function doSearch() {
  const q = document.getElementById('search-input').value.trim();
  if (!q) {
    await loadPoints(currentCollection, pointLimit);
    document.getElementById('search-status').textContent = '';
    return;
  }

  const endpoint = searchMode === 'semantic' ? '/api/semantic-search' : '/api/search';
  const limit = searchMode === 'semantic' ? 120 : 500;
  document.getElementById('search-status').textContent = searchMode === 'semantic' ? 'SEMANTIC SCAN…' : 'SCANNING…';
  try {
    const vectorQuery = selectedVectorName ? `&vector_name=${encodeURIComponent(selectedVectorName)}` : '';
    const crossTarget = searchMode === 'semantic' ? (crossCollectionTarget || '') : '';
    const targetQuery = crossTarget ? `&target_collection=${encodeURIComponent(crossTarget)}` : '';
    const res = await fetch(`${endpoint}?collection=${encodeURIComponent(currentCollection)}&q=${encodeURIComponent(q)}&limit=${limit}&projection=${encodeURIComponent(projectionMethod)}${vectorQuery}${targetQuery}`);
    if (!res.ok) {
      const message = await readApiError(res, '');
      throw new Error(message || 'search failed');
    }
    const data = await res.json();
    buildParticles(data.points || [], data.projection || null);
    flyToIndices(allPoints.map((_, index) => index));
    const modeLabel = searchMode === 'semantic' ? 'SEM' : 'KEY';
    const crossNote = searchMode === 'semantic' && crossCollectionTarget ? ` (${currentCollection} → ${crossCollectionTarget})` : '';
    document.getElementById('search-status').textContent = `${(data.points || []).length} ${modeLabel} HITS${crossNote}`;
  } catch(e) {
    maybeShowQdrantError('Search failed', e);
    document.getElementById('search-status').textContent = 'ERROR';
  }
}

// ── Controls wiring ───────────────────────────────────────────────────────────
function wireControls() {
  if (renderer && renderer.domElement) {
    renderer.domElement.addEventListener('pointerdown', cancelCameraFlight);
    renderer.domElement.addEventListener('wheel', cancelCameraFlight, { passive: true });
  }

  const pointLimitCtrl = document.getElementById('ctrl-point-limit');
  pointLimitCtrl.addEventListener('input', e => {
    pointLimit = parseInt(e.target.value, 10);
    document.getElementById('val-point-limit').textContent = pointLimit.toLocaleString();
  });
  pointLimitCtrl.addEventListener('change', () => {
    loadPoints(currentCollection, pointLimit);
    document.getElementById('search-status').textContent = '';
    document.getElementById('search-input').value = '';
  });

  document.getElementById('ctrl-projection').addEventListener('change', e => {
    projectionMethod = e.target.value;
    loadPoints(currentCollection, pointLimit);
    document.getElementById('search-status').textContent = '';
    document.getElementById('search-input').value = '';
  });

  const vectorNameCtrl = document.getElementById('ctrl-vector-name');
  vectorNameCtrl.addEventListener('change', e => {
    selectedVectorName = e.target.value.trim();
    updateCollectionMetaPanel(currentCollection);
    loadPoints(currentCollection, pointLimit);
    document.getElementById('search-status').textContent = '';
    document.getElementById('search-input').value = '';
  });

  document.getElementById('ctrl-size').addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    pointSize = v;
    document.getElementById('val-size').textContent = v.toFixed(1);
    refreshPointSizes();
  });

  const outliersOnlyBtn = document.getElementById('btn-outliers-only');
  if (outliersOnlyBtn) outliersOnlyBtn.addEventListener('click', toggleOutliersOnly);

  const colorFieldCtrl = document.getElementById('ctrl-color-field');
  const colorFieldCustomRow = document.getElementById('row-color-field-custom');
  const colorFieldCustomCtrl = document.getElementById('ctrl-color-field-custom');
  colorFieldCtrl.addEventListener('change', e => {
    const value = e.target.value;
    colorFieldCustomRow.style.display = value === 'custom' ? '' : 'none';
    if (value === 'custom' && !colorFieldCustom.trim()) return;
    recolorByField(value);
  });
  colorFieldCustomCtrl.addEventListener('change', e => {
    colorFieldCustom = e.target.value.trim();
    if (colorFieldCtrl.value === 'custom') recolorByField('custom');
  });

  const sizeFieldCtrl = document.getElementById('ctrl-size-field');
  const applySizeField = (value) => {
    sizeField = value;
    refreshPointSizes();
  };
  sizeFieldCtrl.addEventListener('input', e => {
    applySizeField(e.target.value);
  });
  sizeFieldCtrl.addEventListener('change', e => {
    applySizeField(e.target.value);
  });

  document.getElementById('ctrl-core-opacity').addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    coreOpacity = v;
    document.getElementById('val-core-opacity').textContent = v.toFixed(2);
    if (particleSystem) particleSystem.material.uniforms.uOpacity.value = v;
  });

  document.getElementById('ctrl-bloom').addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    bloomStrength = v;
    document.getElementById('val-bloom').textContent = v.toFixed(1);
    const bloomScale = v / 2.5;
    if (particleSystem) particleSystem.material.uniforms.uBloom.value = v;
    if (hazeSystem)     hazeSystem.material.uniforms.uBloom.value = 1.8 * bloomScale;
    if (glowSystem)     glowSystem.material.uniforms.uBloom.value = 1.0 * bloomScale;
    if (nebulaSystem)   nebulaSystem.material.uniforms.uBloom.value = 1.0 * bloomScale;
  });

  document.getElementById('ctrl-haze-opacity').addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    hazeOpacity = v;
    document.getElementById('val-haze-opacity').textContent = v.toFixed(2);
    if (hazeSystem) hazeSystem.material.uniforms.uOpacity.value = v;
  });

  document.getElementById('ctrl-speed').addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    document.getElementById('val-speed').textContent = v.toFixed(2);
    rotSpeed = v;
  });

  document.getElementById('ctrl-trail').addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    trailPersistence = v;
    document.getElementById('val-trail').textContent = v.toFixed(2);
  });

  document.getElementById('ctrl-time').addEventListener('input', e => {
    const v = parseFloat(e.target.value) / 100;
    timelineAutoPlay = false;
    document.getElementById('ctrl-time-auto').value = '0';
    document.getElementById('val-time-auto').textContent = 'OFF';
    setTimelineReveal(v);
  });

  document.getElementById('ctrl-time-auto').addEventListener('input', e => {
    timelineAutoPlay = e.target.value === '1';
    document.getElementById('val-time-auto').textContent = timelineAutoPlay ? 'ON' : 'OFF';
    if (timelineAutoPlay && timelineReveal >= 0.999) {
      setTimelineReveal(0);
    }
  });

  document.getElementById('ctrl-theme').addEventListener('change', e => {
    applyTheme(e.target.value);
  });

  document.getElementById('ctrl-hue').addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    hueShift = v;
    document.getElementById('val-hue').textContent = `${Math.round(v)}°`;
    scheduleColorRefresh();
  });

  document.getElementById('ctrl-saturation').addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    saturationMult = v;
    document.getElementById('val-saturation').textContent = v.toFixed(2);
    scheduleColorRefresh();
  });

  document.getElementById('ctrl-lightness').addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    lightnessMult = v;
    document.getElementById('val-lightness').textContent = v.toFixed(2);
    scheduleColorRefresh();
  });

  document.getElementById('collection-select').addEventListener('change', e => {
    currentCollection = e.target.value;
    lastHighlightEventID = '';
    updateCollectionMetaPanel(currentCollection);
    if (liveIngestEnabled) {
      startLiveIngestConnection();
    }
    loadPoints(currentCollection, pointLimit);
    document.getElementById('search-status').textContent = '';
    document.getElementById('search-input').value = '';
  });

  document.getElementById('btn-reload').addEventListener('click', () => {
    loadPoints(currentCollection, pointLimit);
  });

  document.getElementById('btn-live-ingest').addEventListener('click', () => {
    setLiveIngestMode(!liveIngestEnabled);
  });

  document.getElementById('btn-snapshot').addEventListener('click', () => {
    exportScreenshot();
  });

  document.getElementById('btn-find-similar').addEventListener('click', runFindSimilar);
  document.getElementById('btn-radius-scan').addEventListener('click', runSimilarityRadius);
  document.getElementById('btn-clear-scan').addEventListener('click', clearSignalScan);
  document.getElementById('btn-lens-save').addEventListener('click', saveCurrentLens);
  document.getElementById('lens-name').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') saveCurrentLens();
  });
  document.getElementById('btn-bookmark').addEventListener('click', toggleBookmark);
  document.getElementById('btn-history-back').addEventListener('click', () => stepSelectionHistory(-1));
  document.getElementById('btn-history-forward').addEventListener('click', () => stepSelectionHistory(1));

  document.getElementById('ctrl-radius').addEventListener('input', e => {
    similarityRadius = parseFloat(e.target.value);
    document.getElementById('radius-val').textContent = similarityRadius.toFixed(2);
  });

  document.getElementById('search-mode').addEventListener('change', e => {
    searchMode = e.target.value;
    const crossSel = document.getElementById('cross-target-select');
    crossSel.style.opacity = searchMode === 'semantic' ? '1' : '0.6';
    document.getElementById('search-status').textContent = '';
  });

  document.getElementById('cross-target-select').addEventListener('change', e => {
    crossCollectionTarget = e.target.value;
    document.getElementById('search-status').textContent = '';
  });

  document.getElementById('btn-search').addEventListener('click', doSearch);
  document.getElementById('search-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') doSearch();
  });

  document.getElementById('error-retry').addEventListener('click', () => {
    retryLoadFromError();
  });

  window.addEventListener('keydown', handleGlobalShortcuts);

  const crossSel = document.getElementById('cross-target-select');
  crossSel.style.opacity = searchMode === 'semantic' ? '1' : '0.6';

  updateSignalButtons();
  setLiveIngestMode(false);
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
async function main() {
  initScene();
  animate();
  wireControls();
  loadBookmarks();
  loadLenses();
  startUIEventStream();
  await loadCollections();

  const lensFromUrl = hydrateLensFromHash();
  if (lensFromUrl) {
    applyLens(lensFromUrl);
    return;
  }
  await loadPoints(currentCollection, pointLimit);
}

main();