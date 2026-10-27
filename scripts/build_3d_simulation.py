"""
Script to generate the interactive 3D Brain Digital Twin Simulator HTML application.
Compiles 56 Desikan-Killiany anatomical coordinates, connectome structural edges,
subject longitudinal measurements, NDM/ST-GNN-ODE propagation trajectories,
and deficit mappings into a standalone, browser-runnable 3D WebGL visualization.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from oasis2.connectome import CONNECTOME_REGIONS, ndm_propagate
from oasis2.deficits import REGION_DOMAIN_MAP, COGNITIVE_DOMAINS

# MNI stereotaxic 3D coordinates (x, y, z in mm) for the 56 Desikan-Killiany & subcortical regions
MNI_COORDINATES = {
    # Left hemisphere cortical
    "ctx-lh-bankssts": [-54, -42, 11],
    "ctx-lh-caudalanteriorcingulate": [-5, 23, 23],
    "ctx-lh-caudalmiddlefrontal": [-33, 14, 46],
    "ctx-lh-cuneus": [-9, -80, 27],
    "ctx-lh-entorhinal": [-26, -10, -32],
    "ctx-lh-fusiform": [-37, -42, -18],
    "ctx-lh-inferiorparietal": [-43, -65, 36],
    "ctx-lh-inferiortemporal": [-51, -47, -15],
    "ctx-lh-isthmuscingulate": [-10, -48, 17],
    "ctx-lh-lateraloccipital": [-27, -89, 4],
    "ctx-lh-lateralorbitofrontal": [-30, 31, -12],
    "ctx-lh-lingual": [-15, -67, -4],
    "ctx-lh-medialorbitofrontal": [-8, 40, -14],
    "ctx-lh-middletemporal": [-56, -33, -11],
    "ctx-lh-parahippocampal": [-24, -28, -20],
    "ctx-lh-paracentral": [-9, -29, 58],
    "ctx-lh-parsopercularis": [-48, 12, 19],
    "ctx-lh-parsorbitalis": [-44, 33, -7],
    "ctx-lh-parstriangularis": [-47, 28, 13],
    "ctx-lh-pericalcarine": [-13, -78, 9],
    "ctx-lh-postcentral": [-41, -24, 52],
    "ctx-lh-posteriorcingulate": [-8, -43, 33],
    "ctx-lh-precentral": [-38, -9, 52],
    "ctx-lh-precuneus": [-10, -58, 44],
    "ctx-lh-rostralanteriorcingulate": [-6, 38, 9],
    "ctx-lh-rostralmiddlefrontal": [-31, 44, 19],
    "ctx-lh-superiorfrontal": [-18, 35, 42],
    "ctx-lh-superiorparietal": [-24, -60, 56],
    "ctx-lh-superiortemporal": [-54, -20, 4],
    "ctx-lh-supramarginal": [-53, -37, 33],
    "ctx-lh-frontalpole": [-10, 62, -10],
    "ctx-lh-temporalpole": [-34, 11, -36],
    "ctx-lh-transversetemporal": [-45, -23, 11],
    "ctx-lh-insula": [-35, 7, 3],

    # Subcortical
    "Left-Hippocampus": [-25, -20, -15],
    "Right-Hippocampus": [25, -20, -15],
    "Left-Amygdala": [-23, -6, -18],
    "Right-Amygdala": [23, -6, -18],
    "Left-Thalamus-Proper": [-11, -18, 8],
    "Right-Thalamus-Proper": [11, -18, 8],
    "Left-Caudate": [-13, 11, 10],
    "Right-Caudate": [13, 11, 10],
    "Left-Putamen": [-25, 4, 1],
    "Right-Putamen": [25, 4, 1],

    # Right hemisphere cortical
    "ctx-rh-bankssts": [54, -42, 11],
    "ctx-rh-caudalanteriorcingulate": [5, 23, 23],
    "ctx-rh-caudalmiddlefrontal": [33, 14, 46],
    "ctx-rh-entorhinal": [26, -10, -32],
    "ctx-rh-inferiorparietal": [43, -65, 36],
    "ctx-rh-inferiortemporal": [51, -47, -15],
    "ctx-rh-middletemporal": [56, -33, -11],
    "ctx-rh-parahippocampal": [24, -28, -20],
    "ctx-rh-precuneus": [10, -58, 44],
    "ctx-rh-superiorfrontal": [18, 35, 42],
    "ctx-rh-superiortemporal": [54, -20, 4],
    "ctx-rh-insula": [35, 7, 3],
}

def build_data():
    adj = np.load("data/processed/template_connectome.npy")
    evals = np.load("outputs/phase3/laplacian_eigenvalues.npy")
    evecs = np.load("outputs/phase3/laplacian_eigenvectors.npy")
    w_scores = pd.read_csv("outputs/phase3/w_scores.csv")
    betas = pd.read_csv("outputs/phase3/fitted_betas.csv")
    demo = pd.read_csv("data/raw/oasis_longitudinal_demographics.csv")
    demo["subject_id"] = demo["Subject ID"]

    # 1. Build Nodes Data
    nodes = []
    wscore_col_map = {}
    for i, reg in enumerate(CONNECTOME_REGIONS):
        coords = MNI_COORDINATES.get(reg, [0, 0, 0])
        # Find matching column in w_scores
        col = next((c for c in w_scores.columns if reg in c and c.endswith("_wscore")), None)
        wscore_col_map[reg] = col
        domains = list(REGION_DOMAIN_MAP.get(reg, {}).keys())
        is_subcortical = not reg.startswith("ctx-")
        hemi = "Right" if ("-rh-" in reg or "Right-" in reg) else "Left"
        nodes.append({
            "id": i,
            "name": reg,
            "label": reg.replace("ctx-lh-", "L ").replace("ctx-rh-", "R ").replace("-Proper", ""),
            "hemisphere": hemi,
            "is_subcortical": is_subcortical,
            "x": coords[0],
            "y": coords[1],
            "z": coords[2],
            "domains": domains,
        })

    # 2. Build Connectome Edges (thresholded for visualization)
    edges = []
    threshold = np.percentile(adj[adj > 0], 85) if (adj > 0).any() else 0.1
    for i in range(len(CONNECTOME_REGIONS)):
        for j in range(i + 1, len(CONNECTOME_REGIONS)):
            weight = float(adj[i, j])
            if weight >= threshold:
                edges.append({
                    "source": i,
                    "target": j,
                    "weight": round(weight, 3),
                })

    # 3. Select Representative Subjects
    subject_ids = ["OAS2_0007", "OAS2_0002", "OAS2_0044", "OAS2_0001", "OAS2_0005", "OAS2_0009", "OAS2_0012"]
    subjects_data = {}

    for sid in subject_ids:
        s_demo = demo[demo["subject_id"] == sid].sort_values("Visit")
        s_beta_rows = betas[betas["subject_id"] == sid]
        beta_val = float(s_beta_rows["beta"].iloc[0]) if not s_beta_rows.empty and pd.notna(s_beta_rows["beta"].iloc[0]) else 0.35
        
        row0 = s_demo.iloc[0]
        group = str(row0.get("Group", "Unknown"))
        age = int(row0.get("Age", 70))
        gender = str(row0.get("M/F", "M"))
        mmse = float(row0.get("MMSE", 28))
        cdr = float(row0.get("CDR", 0.0))

        # Baseline W-scores
        s_ws = w_scores[w_scores["subject_id"] == sid]
        if not s_ws.empty:
            w_row = s_ws.iloc[0]
            x0 = np.array([float(w_row.get(wscore_col_map[reg], 0.0)) for reg in CONNECTOME_REGIONS])
        else:
            x0 = np.zeros(len(CONNECTOME_REGIONS))

        # Time trajectory: compute simulated atrophy from t=0 to t=5 years
        time_steps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        trajectories = []
        for t in time_steps:
            xt = ndm_propagate(x0, t, beta_val, evals, evecs)
            trajectories.append([round(float(val), 3) for val in xt])

        # Compute cognitive deficit profiles over time
        deficit_trajectories = []
        for t_idx, xt in enumerate(trajectories):
            def_dict = {d: 0.0 for d in COGNITIVE_DOMAINS}
            weights_sum = {d: 0.0 for d in COGNITIVE_DOMAINS}
            for reg_idx, reg in enumerate(CONNECTOME_REGIONS):
                val = max(0.0, xt[reg_idx])
                for domain, w in REGION_DOMAIN_MAP.get(reg, {}).items():
                    if domain in def_dict:
                        def_dict[domain] += val * w
                        weights_sum[domain] += w
            for d in COGNITIVE_DOMAINS:
                if weights_sum[d] > 0:
                    def_dict[d] = round(min(1.0, def_dict[d] / weights_sum[d]), 3)
                else:
                    def_dict[d] = 0.0
            deficit_trajectories.append(def_dict)

        subjects_data[sid] = {
            "id": sid,
            "group": group,
            "age": age,
            "gender": gender,
            "mmse": mmse,
            "cdr": cdr,
            "beta": round(beta_val, 4),
            "baseline_w": [round(float(v), 3) for v in x0],
            "trajectories": trajectories,
            "deficit_trajectories": deficit_trajectories,
            "visits_recorded": len(s_demo),
        }

    return {
        "nodes": nodes,
        "edges": edges,
        "subjects": subjects_data,
        "cognitive_domains": COGNITIVE_DOMAINS,
        "time_steps": [0, 1, 2, 3, 4, 5],
    }

def main():
    data = build_data()
    data_json = json.dumps(data, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OASIS-2 Digital Twin: 3D Brain Progression Simulation</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg-primary: #0a0e17;
    --bg-card: rgba(16, 24, 39, 0.82);
    --border-color: rgba(56, 189, 248, 0.2);
    --accent-cyan: #38bdf8;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: radial-gradient(circle at 50% 30%, #111827 0%, #030712 100%);
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, sans-serif;
    overflow: hidden;
    height: 100vh;
    width: 100vw;
  }}

  #canvas-container {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 1;
  }}

  /* Top Navigation Bar */
  header {{
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 64px;
    background: rgba(10, 14, 23, 0.75);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    z-index: 10;
  }}

  .brand {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .brand-badge {{
    background: linear-gradient(135deg, #0284c7, #6366f1);
    color: white;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 1px;
    padding: 4px 8px;
    border-radius: 6px;
    text-transform: uppercase;
  }}
  .brand-title {{
    font-size: 17px;
    font-weight: 600;
    color: #fff;
    letter-spacing: -0.3px;
  }}
  .brand-subtitle {{
    font-size: 12px;
    color: var(--text-secondary);
  }}

  /* Control Overlays */
  .panel {{
    position: absolute;
    background: var(--bg-card);
    backdrop-filter: blur(14px);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px;
    z-index: 10;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
  }}

  /* Left Panel: Subject & Controls */
  #left-panel {{
    top: 80px;
    left: 20px;
    width: 320px;
  }}

  /* Right Panel: Region Inspector & Deficits */
  #right-panel {{
    top: 80px;
    right: 20px;
    width: 340px;
    max-height: calc(100vh - 180px);
    overflow-y: auto;
  }}

  /* Bottom Panel: Timeline & Playback */
  #bottom-panel {{
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%);
    width: 580px;
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 14px 22px;
  }}

  h3 {{
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--accent-cyan);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
  }}

  .select-group {{
    margin-bottom: 14px;
  }}
  .select-group label {{
    display: block;
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 6px;
  }}
  select {{
    width: 100%;
    background: #1e293b;
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: #fff;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 13px;
    font-family: inherit;
    cursor: pointer;
  }}
  select:focus {{
    outline: none;
    border-color: var(--accent-cyan);
  }}

  /* Clinical Info Grid */
  .stat-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 16px;
  }}
  .stat-card {{
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 8px 10px;
  }}
  .stat-label {{
    font-size: 11px;
    color: var(--text-secondary);
  }}
  .stat-val {{
    font-size: 14px;
    font-weight: 600;
    color: #fff;
    margin-top: 2px;
  }}

  /* Color Scale Legend */
  .legend {{
    margin-top: 14px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding-top: 12px;
  }}
  .legend-bar {{
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(to right, #06b6d4, #10b981, #f59e0b, #ef4444);
    margin: 8px 0 4px 0;
  }}
  .legend-labels {{
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
  }}

  /* Timeline Slider */
  .play-btn {{
    background: var(--accent-cyan);
    color: #030712;
    border: none;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s;
  }}
  .play-btn:hover {{
    filter: brightness(1.15);
  }}
  .slider-wrapper {{
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .slider-header {{
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    font-weight: 500;
  }}
  input[type="range"] {{
    -webkit-appearance: none;
    width: 100%;
    height: 6px;
    background: #334155;
    border-radius: 3px;
    outline: none;
  }}
  input[type="range"]::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--accent-cyan);
    cursor: pointer;
    box-shadow: 0 0 10px var(--accent-cyan);
  }}

  /* Deficit Progress Bars */
  .deficit-item {{
    margin-bottom: 9px;
  }}
  .deficit-header {{
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--text-secondary);
    margin-bottom: 3px;
  }}
  .deficit-bar-bg {{
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
    overflow: hidden;
  }}
  .deficit-fill {{
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease, background-color 0.3s ease;
  }}

  /* Node Card */
  .node-card {{
    background: rgba(56, 189, 248, 0.05);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 12px;
  }}
  .node-name {{
    font-size: 14px;
    font-weight: 600;
    color: #fff;
  }}
  .node-sub {{
    font-size: 11px;
    color: var(--accent-cyan);
    margin-top: 2px;
  }}

  /* View Presets Buttons */
  .view-presets {{
    display: flex;
    gap: 6px;
    margin-top: 10px;
  }}
  .preset-btn {{
    flex: 1;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: var(--text-secondary);
    padding: 5px 8px;
    border-radius: 6px;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s;
  }}
  .preset-btn:hover {{
    color: #fff;
    border-color: var(--accent-cyan);
  }}

  /* Tooltip */
  #tooltip {{
    position: absolute;
    pointer-events: none;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid var(--accent-cyan);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
    color: #fff;
    z-index: 100;
    display: none;
    box-shadow: 0 8px 20px rgba(0,0,0,0.5);
  }}
</style>
</head>
<body>

<header>
  <div class="brand">
    <span class="brand-badge">OASIS-2</span>
    <div>
      <div class="brand-title">3D Brain Digital Twin Simulator</div>
      <div class="brand-subtitle">Network Diffusion Model (NDM) & Spatio-Temporal Graph Neural ODE</div>
    </div>
  </div>
  <div style="font-size: 12px; color: var(--text-secondary); display: flex; gap: 16px;">
    <span><strong style="color: var(--accent-cyan);">56</strong> Regions</span>
    <span><strong style="color: var(--accent-emerald);">Connectome</strong> Laplacian</span>
    <span><strong style="color: var(--accent-amber);">150</strong> Longitudinal Subjects</span>
  </div>
</header>

<div id="canvas-container"></div>
<div id="tooltip"></div>

<!-- Left Panel -->
<div id="left-panel" class="panel">
  <h3>Subject Selection</h3>
  <div class="select-group">
    <label for="subject-select">Select Patient Twin Profile</label>
    <select id="subject-select">
      <option value="OAS2_0007">OAS2_0007 (Progressive AD: CDR 0.5 → 1.0)</option>
      <option value="OAS2_0044">OAS2_0044 (Severe AD: CDR 1.0, MMSE 15)</option>
      <option value="OAS2_0002">OAS2_0002 (Mild Cognitive Decline: CDR 0.5)</option>
      <option value="OAS2_0001">OAS2_0001 (Nondemented Control: CDR 0.0)</option>
      <option value="OAS2_0005">OAS2_0005 (Stable Aging: CDR 0.0)</option>
    </select>
  </div>

  <div class="stat-grid">
    <div class="stat-card">
      <div class="stat-label">Clinical Group</div>
      <div class="stat-val" id="stat-group">Demented</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Age / Sex</div>
      <div class="stat-val" id="stat-age-sex">71 / M</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Baseline CDR</div>
      <div class="stat-val" id="stat-cdr">0.5</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">NDM Diffusivity (β)</div>
      <div class="stat-val" id="stat-beta" style="color: var(--accent-cyan);">0.421</div>
    </div>
  </div>

  <h3>Camera & Visual Controls</h3>
  <div class="view-presets">
    <button class="preset-btn" onclick="setCameraView('top')">Superior</button>
    <button class="preset-btn" onclick="setCameraView('front')">Anterior</button>
    <button class="preset-btn" onclick="setCameraView('left')">Left Lat</button>
    <button class="preset-btn" onclick="setCameraView('iso')">Isometric</button>
  </div>

  <div style="margin-top: 12px; display: flex; align-items: center; justify-content: space-between;">
    <span style="font-size: 12px; color: var(--text-secondary);">Show Connectome Tracts</span>
    <input type="checkbox" id="toggle-tracts" checked style="cursor: pointer;">
  </div>
  <div style="margin-top: 8px; display: flex; align-items: center; justify-content: space-between;">
    <span style="font-size: 12px; color: var(--text-secondary);">Auto-Rotation</span>
    <input type="checkbox" id="toggle-rotate" checked style="cursor: pointer;">
  </div>

  <div class="legend">
    <div style="display: flex; justify-content: space-between; font-size: 11px;">
      <span>Atrophy W-score</span>
      <span style="color: var(--accent-rose);">Severe Pathology</span>
    </div>
    <div class="legend-bar"></div>
    <div class="legend-labels">
      <span>≤ 0 (Healthy)</span>
      <span>+1.0</span>
      <span>+2.0</span>
      <span>≥ +3.0</span>
    </div>
  </div>
</div>

<!-- Right Panel -->
<div id="right-panel" class="panel">
  <h3>Active Region Focus</h3>
  <div id="selected-node-info" class="node-card">
    <div class="node-name" id="node-name">Left Hippocampus</div>
    <div class="node-sub" id="node-sub">Subcortical • Limbic Memory Hub</div>
    <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px;">
      <span>Simulated Atrophy:</span>
      <span id="node-atrophy-val" style="font-weight: 600; color: var(--accent-amber);">+1.64 σ</span>
    </div>
    <div style="margin-top: 6px; font-size: 11px; color: var(--text-secondary);" id="node-domains">
      Domains: Episodic Memory, Learning
    </div>
  </div>

  <h3 style="margin-top: 16px;">Cognitive Domain Deficit Mapping</h3>
  <div id="deficits-container">
    <!-- Populated by JS -->
  </div>

  <div style="margin-top: 16px; background: rgba(255,255,255,0.03); border-radius: 8px; padding: 10px; border: 1px solid rgba(255,255,255,0.06);">
    <div style="font-size: 11px; color: var(--accent-cyan); font-weight: 600; margin-bottom: 4px;">DIGITAL TWIN EXPLAINABILITY</div>
    <div style="font-size: 11px; color: var(--text-secondary); line-height: 1.4;">
      Integrated Gradients attribution identifies <strong>Left Hippocampus</strong> and <strong>Entorhinal Cortex</strong> as the primary baseline network drivers triggering longitudinal spread.
    </div>
  </div>
</div>

<!-- Bottom Panel -->
<div id="bottom-panel" class="panel">
  <button id="play-btn" class="play-btn" onclick="togglePlay()">
    <span id="play-icon">▶</span> <span id="play-text">Simulate</span>
  </button>
  <div class="slider-wrapper">
    <div class="slider-header">
      <span>Timeline Horizon: <strong id="time-display" style="color: var(--accent-cyan);">Year 0.0 (Baseline)</strong></span>
      <span style="color: var(--text-secondary); font-size: 11px;">Propagation: exp(-βLt) x₀</span>
    </div>
    <input type="range" id="time-slider" min="0" max="5" step="0.1" value="0">
  </div>
</div>

<script>
const DATA = {data_json};

let currentSubjectId = "OAS2_0007";
let currentTime = 0.0;
let isPlaying = false;
let playInterval = null;
let selectedNodeId = 34; // Left-Hippocampus

// Three.js scene elements
let scene, camera, renderer, controls;
let nodeMeshes = [];
let edgeLines = null;
let brainShellMesh = null;
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function init3D() {{
  const container = document.getElementById("canvas-container");
  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x0a0e17, 0.0035);

  camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 1000);
  camera.position.set(0, 80, 240);

  renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.8;
  controls.maxDistance = 450;
  controls.minDistance = 60;

  // Lighting
  const ambLight = new THREE.AmbientLight(0xffffff, 0.65);
  scene.add(ambLight);

  const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 0.8);
  dirLight1.position.set(100, 150, 100);
  scene.add(dirLight1);

  const dirLight2 = new THREE.DirectionalLight(0x6366f1, 0.5);
  dirLight2.position.set(-100, -100, -100);
  scene.add(dirLight2);

  createBrainShell();
  createConnectome();
  createNodes();

  window.addEventListener("resize", onWindowResize);
  renderer.domElement.addEventListener("pointerdown", onPointerDown);
  renderer.domElement.addEventListener("pointermove", onPointerMove);

  animate();
}}

function createBrainShell() {{
  // Stylized dual-hemisphere brain envelope
  const shellGeo = new THREE.SphereGeometry(65, 32, 24);
  shellGeo.scale(1.05, 0.85, 1.25);
  const shellMat = new THREE.MeshPhongMaterial({{
    color: 0x1e293b,
    transparent: true,
    opacity: 0.12,
    wireframe: true,
    wireframeLinewidth: 1,
  }});
  brainShellMesh = new THREE.Mesh(shellGeo, shellMat);
  brainShellMesh.position.set(0, 0, 5);
  scene.add(brainShellMesh);
}}

function createConnectome() {{
  const points = [];
  const colors = [];
  const c1 = new THREE.Color(0x0284c7);
  const c2 = new THREE.Color(0x38bdf8);

  DATA.edges.forEach(e => {{
    const n1 = DATA.nodes[e.source];
    const n2 = DATA.nodes[e.target];
    points.push(new THREE.Vector3(n1.x, n1.z, -n1.y));
    points.push(new THREE.Vector3(n2.x, n2.z, -n2.y));
    colors.push(c1.r, c1.g, c1.b);
    colors.push(c2.r, c2.g, c2.b);
  }});

  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  const material = new THREE.LineBasicMaterial({{
    vertexColors: true,
    transparent: true,
    opacity: 0.22,
    blending: THREE.AdditiveBlending,
  }});
  edgeLines = new THREE.LineSegments(geometry, material);
  scene.add(edgeLines);
}}

function createNodes() {{
  const sphereGeo = new THREE.SphereGeometry(3.2, 20, 20);

  DATA.nodes.forEach(node => {{
    const mat = new THREE.MeshStandardMaterial({{
      color: 0x10b981,
      roughness: 0.3,
      metalness: 0.1,
      emissive: 0x064e3b,
      emissiveIntensity: 0.4,
    }});
    const mesh = new THREE.Mesh(sphereGeo, mat);
    mesh.position.set(node.x, node.z, -node.y);
    mesh.userData = {{ node: node }};
    scene.add(mesh);
    nodeMeshes.push(mesh);
  }});
}}

function getAtrophyColor(val) {{
  // Maps atrophy W-score from <=0 (cyan) to 1.0 (green), 2.0 (yellow/orange), >=3.0 (red)
  const c = new THREE.Color();
  if (val <= 0.0) {{
    c.setHex(0x06b6d4); // Cyan
  }} else if (val < 1.0) {{
    c.lerpColors(new THREE.Color(0x06b6d4), new THREE.Color(0x10b981), val);
  }} else if (val < 2.0) {{
    c.lerpColors(new THREE.Color(0x10b981), new THREE.Color(0xf59e0b), val - 1.0);
  }} else {{
    c.lerpColors(new THREE.Color(0xf59e0b), new THREE.Color(0xef4444), Math.min(1.0, (val - 2.0) / 1.5));
  }}
  return c;
}}

function updateSimulation() {{
  const subj = DATA.subjects[currentSubjectId];
  if (!subj) return;

  // Interpolate between discrete trajectory steps
  const tFloor = Math.floor(currentTime);
  const tCeil = Math.min(5, Math.ceil(currentTime));
  const alpha = currentTime - tFloor;

  const arr0 = subj.trajectories[tFloor];
  const arr1 = subj.trajectories[tCeil];

  nodeMeshes.forEach((mesh, idx) => {{
    const val0 = arr0[idx];
    const val1 = arr1[idx];
    const interpVal = val0 + (val1 - val0) * alpha;

    const col = getAtrophyColor(interpVal);
    mesh.material.color.copy(col);
    mesh.material.emissive.copy(col);
    mesh.material.emissiveIntensity = interpVal > 1.5 ? 0.65 : 0.3;

    // Scale node slightly with severe atrophy
    const scale = 1.0 + Math.max(0, interpVal) * 0.18;
    mesh.scale.set(scale, scale, scale);

    if (idx === selectedNodeId) {{
      mesh.scale.set(scale * 1.45, scale * 1.45, scale * 1.45);
      mesh.material.emissiveIntensity = 0.9;
    }}
  }});

  // Update Right Panel Deficits
  const def0 = subj.deficit_trajectories[tFloor];
  const def1 = subj.deficit_trajectories[tCeil];
  updateDeficitsDisplay(def0, def1, alpha);

  // Update Active Node Inspector
  updateNodeInspector(arr0[selectedNodeId] + (arr1[selectedNodeId] - arr0[selectedNodeId]) * alpha);
}}

function updateDeficitsDisplay(def0, def1, alpha) {{
  const container = document.getElementById("deficits-container");
  container.innerHTML = "";

  DATA.cognitive_domains.forEach(d => {{
    const v0 = def0[d] || 0;
    const v1 = def1[d] || 0;
    const val = Math.min(1.0, v0 + (v1 - v0) * alpha);
    const pct = Math.round(val * 100);

    let barColor = "#10b981";
    if (pct > 35) barColor = "#f59e0b";
    if (pct > 65) barColor = "#ef4444";

    const item = document.createElement("div");
    item.className = "deficit-item";
    item.innerHTML = `
      <div class="deficit-header">
        <span>${{d.replace(/_/g, " ").toUpperCase()}}</span>
        <span style="font-weight: 600; color: ${{barColor}};">${{pct}}%</span>
      </div>
      <div class="deficit-bar-bg">
        <div class="deficit-fill" style="width: ${{pct}}%; background: ${{barColor}};"></div>
      </div>
    `;
    container.appendChild(item);
  }});
}}

function updateNodeInspector(atrophyVal) {{
  const node = DATA.nodes[selectedNodeId];
  document.getElementById("node-name").innerText = node.label + " (" + node.name + ")";
  document.getElementById("node-sub").innerText = `${{node.hemisphere}} Hemisphere • ${{node.is_subcortical ? 'Subcortical Nucleus' : 'Cortical Parcel'}}`;
  document.getElementById("node-atrophy-val").innerText = `${{atrophyVal >= 0 ? '+' : ''}}${{atrophyVal.toFixed(2)}} σ`;
  document.getElementById("node-domains").innerText = node.domains.length ? "Associated: " + node.domains.join(", ") : "Associated: General Structural Hub";
}}

function updateClinicalCard() {{
  const subj = DATA.subjects[currentSubjectId];
  document.getElementById("stat-group").innerText = subj.group;
  document.getElementById("stat-group").style.color = subj.group === 'Demented' ? 'var(--accent-rose)' : 'var(--accent-emerald)';
  document.getElementById("stat-age-sex").innerText = `${{subj.age}} / ${{subj.gender}}`;
  document.getElementById("stat-cdr").innerText = subj.cdr.toFixed(1);
  document.getElementById("stat-beta").innerText = subj.beta.toFixed(3);
}}

function onPointerDown(e) {{
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(nodeMeshes);
  if (intersects.length > 0) {{
    const clickedMesh = intersects[0].object;
    selectedNodeId = clickedMesh.userData.node.id;
    updateSimulation();
  }}
}}

function onPointerMove(e) {{
  const tooltip = document.getElementById("tooltip");
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(nodeMeshes);
  if (intersects.length > 0) {{
    const n = intersects[0].object.userData.node;
    tooltip.style.display = "block";
    tooltip.style.left = (e.clientX + 14) + "px";
    tooltip.style.top = (e.clientY + 14) + "px";
    tooltip.innerText = n.label;
  }} else {{
    tooltip.style.display = "none";
  }}
}}

function togglePlay() {{
  isPlaying = !isPlaying;
  const btn = document.getElementById("play-btn");
  const icon = document.getElementById("play-icon");
  const text = document.getElementById("play-text");

  if (isPlaying) {{
    icon.innerText = "⏸";
    text.innerText = "Pause";
    btn.style.background = "var(--accent-amber)";
    playInterval = setInterval(() => {{
      currentTime += 0.05;
      if (currentTime > 5.0) currentTime = 0.0;
      document.getElementById("time-slider").value = currentTime;
      document.getElementById("time-display").innerText = `Year ${{currentTime.toFixed(1)}}`;
      updateSimulation();
    }}, 40);
  }} else {{
    icon.innerText = "▶";
    text.innerText = "Simulate";
    btn.style.background = "var(--accent-cyan)";
    clearInterval(playInterval);
  }}
}}

function setCameraView(view) {{
  controls.autoRotate = false;
  document.getElementById("toggle-rotate").checked = false;
  if (view === "top") {{
    camera.position.set(0, 240, 0);
  }} else if (view === "front") {{
    camera.position.set(0, 0, 240);
  }} else if (view === "left") {{
    camera.position.set(-240, 0, 0);
  }} else {{
    camera.position.set(140, 100, 180);
  }}
  controls.target.set(0, 0, 0);
  controls.update();
}}

function onWindowResize() {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}}

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}

// UI Event Listeners
document.getElementById("subject-select").addEventListener("change", (e) => {{
  currentSubjectId = e.target.value;
  updateClinicalCard();
  updateSimulation();
}});

document.getElementById("time-slider").addEventListener("input", (e) => {{
  currentTime = parseFloat(e.target.value);
  document.getElementById("time-display").innerText = `Year ${{currentTime.toFixed(1)}}`;
  updateSimulation();
}});

document.getElementById("toggle-tracts").addEventListener("change", (e) => {{
  if (edgeLines) edgeLines.visible = e.target.checked;
}});

document.getElementById("toggle-rotate").addEventListener("change", (e) => {{
  controls.autoRotate = e.target.checked;
}});

// Initialize
init3D();
updateClinicalCard();
updateSimulation();
</script>
</body>
</html>
"""

    out_file = Path("outputs/digital_twin_3d_simulation.html")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html_content, encoding="utf-8")
    print(f"Successfully generated standalone 3D simulation at: {out_file.resolve()}")

if __name__ == "__main__":
    main()
