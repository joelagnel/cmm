#!/usr/bin/env python3
"""
DAG Visualization Generator
Reads extracted reasoning nodes from ChromaDB and generates:
  1. An interactive HTML file (self-contained, no dependencies)
  2. A static PNG via graphviz (if installed)
  3. A Mermaid diagram for GitHub README embedding

Usage:
  python scripts/visualize_dag.py --store data/memory_store --project supply-chain --output output/
  python scripts/visualize_dag.py --store data/memory_store --project supply-chain --output output/ --format html
  python scripts/visualize_dag.py --store data/memory_store --project supply-chain --output output/ --format all
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Configuration ────────────────────────────────────────────────

TYPE_COLORS = {
    "context_load": {"fill": "#64748b", "bg": "#1e293b", "label": "CONTEXT", "icon": "\U0001f4d6"},
    "hypothesis":   {"fill": "#3b82f6", "bg": "#1e3a5f", "label": "HYPOTHESIS", "icon": "\U0001f4ad"},
    "investigation":{"fill": "#8b5cf6", "bg": "#2d1f5e", "label": "INVESTIGATE", "icon": "\U0001f50d"},
    "discovery":    {"fill": "#10b981", "bg": "#1a3a2a", "label": "DISCOVERY", "icon": "\U0001f4a1"},
    "pivot":        {"fill": "#f59e0b", "bg": "#3d2e0a", "label": "PIVOT", "icon": "\U0001f504"},
    "dead_end":     {"fill": "#ef4444", "bg": "#3d1a1a", "label": "DEAD END", "icon": "\U0001f6ab"},
    "solution":     {"fill": "#22c55e", "bg": "#1a3d1a", "label": "SOLUTION", "icon": "\u2705"},
}

EDGE_COLORS = {
    "led_to": "#94a3b8",
    "caused": "#f87171",
    "triggered_pivot": "#f59e0b",
    "informed": "#60a5fa",
    "enabled": "#34d399",
    "revealed": "#a78bfa",
    "refined": "#94a3b8",
    "contradicted": "#ef4444",
}


# ─── Data Loading ─────────────────────────────────────────────────

def load_from_chromadb(store_path, project_id):
    """Load nodes from ChromaDB store."""
    try:
        import chromadb
    except ImportError:
        print("ERROR: chromadb not installed. Run: pip install chromadb")
        sys.exit(1)

    client = chromadb.PersistentClient(path=store_path)

    # Use the reasoning_nodes collection specifically
    try:
        col = client.get_collection("reasoning_nodes")
    except Exception:
        print(f"ERROR: No 'reasoning_nodes' collection found in {store_path}")
        sys.exit(1)

    results = col.get(include=["documents", "metadatas"], where={"project_id": project_id})

    nodes = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        nodes.append({
            "node_id": meta.get("node_id", f"n{len(nodes):03d}"),
            "node_type": meta.get("node_type", "investigation"),
            "summary": doc,
            "evidence": meta.get("evidence", ""),
            "confidence": float(meta.get("confidence", 0.5)),
            "is_pivot": bool(meta.get("is_pivot", False)),
            "message_range": [
                int(meta.get("msg_start", 0)),
                int(meta.get("msg_end", 0))
            ],
            "session_id": meta.get("session_id", "unknown"),
        })

    # Sort by message_range start for chronological order
    nodes.sort(key=lambda n: n["message_range"][0])

    # Build edges from sequential ordering + type transitions
    edges = []
    pivot_nodes = [n["node_id"] for n in nodes if n["is_pivot"]]

    for i in range(1, len(nodes)):
        prev = nodes[i - 1]
        curr = nodes[i]

        if curr["node_type"] == "pivot":
            rel = "triggered_pivot"
        elif prev["node_type"] == "dead_end":
            rel = "triggered_pivot" if curr["node_type"] == "pivot" else "led_to"
        elif curr["node_type"] == "discovery":
            rel = "revealed"
        elif prev["node_type"] == "hypothesis":
            rel = "informed"
        elif curr["node_type"] == "solution":
            rel = "enabled"
        else:
            rel = "led_to"

        edges.append({
            "source_id": prev["node_id"],
            "target_id": curr["node_id"],
            "relationship": rel,
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "pivot_nodes": pivot_nodes,
        "project_id": project_id,
        "generated_at": datetime.now().isoformat(),
    }


def load_from_json(json_path):
    """Load a pre-built DAG from JSON file."""
    with open(json_path) as f:
        return json.load(f)


# ─── HTML Generator ───────────────────────────────────────────────

def generate_html(dag, output_path):
    """Generate a self-contained interactive HTML visualization."""

    nodes_json = json.dumps(dag["nodes"], indent=2)
    edges_json = json.dumps(dag["edges"], indent=2)
    pivots_json = json.dumps(dag["pivot_nodes"])
    project = dag.get("project_id", "unknown")
    generated = dag.get("generated_at", datetime.now().isoformat())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reasoning DAG — {project}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'JetBrains Mono', monospace;
    background: #0f172a; color: #e2e8f0;
    overflow: hidden; width: 100vw; height: 100vh;
  }}

  #header {{
    position: fixed; top: 0; left: 0; right: 0; z-index: 10;
    padding: 16px 24px;
    background: linear-gradient(180deg, rgba(15,23,42,0.97) 60%, transparent);
  }}
  #header .tag {{ font-size: 11px; font-weight: 600; color: #f59e0b; letter-spacing: 0.15em; text-transform: uppercase; }}
  #header h1 {{ font-size: 18px; font-weight: 700; margin: 4px 0; }}
  #header .meta {{ font-size: 12px; color: #64748b; }}

  #legend {{
    position: fixed; top: 16px; right: 24px; z-index: 10;
    display: flex; gap: 6px; flex-wrap: wrap; max-width: 500px;
  }}
  .legend-item {{
    display: flex; align-items: center; gap: 4px;
    padding: 3px 8px; border-radius: 4px;
    background: rgba(255,255,255,0.05);
    font-size: 9px; font-weight: 600; letter-spacing: 0.05em;
  }}

  #canvas {{ width: 100%; height: 100%; }}

  .node {{ cursor: pointer; transition: opacity 0.15s; }}
  .node:hover {{ opacity: 0.9; }}
  .edge {{ transition: opacity 0.15s; }}
  .edge:hover {{ opacity: 1 !important; }}
  .edge-label {{ pointer-events: none; }}

  #detail {{
    position: fixed; bottom: 16px; left: 16px; max-width: 700px;
    background: rgba(30,41,59,0.97); border-radius: 12px;
    padding: 20px; backdrop-filter: blur(16px);
    box-shadow: 0 24px 48px rgba(0,0,0,0.4);
    display: none; z-index: 20;
  }}
  #detail.visible {{ display: block; }}
  #detail .type-badge {{ font-size: 13px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }}
  #detail .summary {{ font-size: 13px; line-height: 1.6; margin: 12px 0; }}
  #detail .evidence {{ font-size: 11px; color: #94a3b8; padding: 8px 12px; background: rgba(0,0,0,0.2); border-radius: 6px; line-height: 1.5; }}
  #detail .evidence-label {{ color: #64748b; font-weight: 600; }}
  #detail .meta-row {{ display: flex; gap: 16px; margin-top: 10px; font-size: 10px; color: #475569; }}
  #detail .close {{ position: absolute; top: 12px; right: 16px; background: none; border: none; color: #64748b; cursor: pointer; font-size: 18px; }}
  .pivot-badge {{ font-size: 9px; padding: 2px 8px; border-radius: 4px; background: rgba(245,158,11,0.13); color: #f59e0b; font-weight: 700; border: 1px solid rgba(245,158,11,0.27); margin-left: 8px; }}

  #instructions {{
    position: fixed; bottom: 16px; right: 16px; font-size: 10px;
    color: #334155; text-align: right; line-height: 1.6; z-index: 10;
  }}

  @keyframes fadeInNode {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  @keyframes drawEdge {{
    from {{ stroke-dashoffset: 500; }}
    to {{ stroke-dashoffset: 0; }}
  }}
</style>
</head>
<body>

<div id="header">
  <div class="tag">Reasoning DAG</div>
  <h1>Session Extraction: {project}</h1>
  <div class="meta" id="meta">Generated: {generated}</div>
</div>

<div id="legend"></div>

<svg id="canvas"></svg>

<div id="detail">
  <button class="close" onclick="closeDetail()">&times;</button>
  <div id="detail-content"></div>
</div>

<div id="instructions">scroll to zoom &middot; drag to pan &middot; click node for details</div>

<script>
const NODES = {nodes_json};
const EDGES = {edges_json};
const PIVOTS = {pivots_json};

const TYPE_CFG = {{
  context_load: {{ fill: "#64748b", label: "CONTEXT", icon: "\\ud83d\\udcd6" }},
  hypothesis:   {{ fill: "#3b82f6", label: "HYPOTHESIS", icon: "\\ud83d\\udcad" }},
  investigation:{{ fill: "#8b5cf6", label: "INVESTIGATE", icon: "\\ud83d\\udd0d" }},
  discovery:    {{ fill: "#10b981", label: "DISCOVERY", icon: "\\ud83d\\udca1" }},
  pivot:        {{ fill: "#f59e0b", label: "PIVOT", icon: "\\ud83d\\udd04" }},
  dead_end:     {{ fill: "#ef4444", label: "DEAD END", icon: "\\ud83d\\udeab" }},
  solution:     {{ fill: "#22c55e", label: "SOLUTION", icon: "\\u2705" }},
}};

const EDGE_CLR = {{
  led_to: "#94a3b8", caused: "#f87171", triggered_pivot: "#f59e0b",
  informed: "#60a5fa", enabled: "#34d399", revealed: "#a78bfa",
  refined: "#94a3b8", contradicted: "#ef4444",
}};

// Stats
const metaEl = document.getElementById('meta');
metaEl.textContent =
  NODES.length + ' nodes \\u00b7 ' + EDGES.length + ' edges \\u00b7 ' + PIVOTS.length + ' pivots';

// Legend
const legend = document.getElementById('legend');
Object.entries(TYPE_CFG).forEach(function(entry) {{
  var t = entry[0], c = entry[1];
  var d = document.createElement('div');
  d.className = 'legend-item';
  d.innerHTML = '<span>' + c.icon + '</span><span style="color:' + c.fill + '">' + c.label + '</span>';
  legend.appendChild(d);
}});

// Layout
var W = 240, H = 120, PX = 80, PY = 60;
var levels = {{}};
var visited = {{}};
var roots = {{}};
NODES.forEach(function(n) {{ roots[n.node_id] = true; }});
EDGES.forEach(function(e) {{ delete roots[e.target_id]; }});

function assignLevel(id, lv) {{
  if (visited[id] && (levels[id] || 0) >= lv) return;
  visited[id] = true;
  levels[id] = Math.max(levels[id] || 0, lv);
  EDGES.filter(function(e) {{ return e.source_id === id; }}).forEach(function(e) {{
    assignLevel(e.target_id, lv + 1);
  }});
}}
Object.keys(roots).forEach(function(r) {{ assignLevel(r, 0); }});
NODES.forEach(function(n) {{ if (!(n.node_id in levels)) levels[n.node_id] = 0; }});

var byLevel = {{}};
NODES.forEach(function(n) {{
  var l = levels[n.node_id];
  if (!byLevel[l]) byLevel[l] = [];
  byLevel[l].push(n.node_id);
}});

var pos = {{}};
Object.entries(byLevel).forEach(function(entry) {{
  var lv = entry[0], ids = entry[1];
  var tw = ids.length * W + (ids.length - 1) * PX;
  var sx = -tw / 2 + W / 2;
  ids.forEach(function(id, i) {{
    pos[id] = {{ x: sx + i * (W + PX), y: Number(lv) * (H + PY) }};
  }});
}});

// SVG setup
var svg = document.getElementById('canvas');
var ns = 'http://www.w3.org/2000/svg';
var vb = calcViewBox();
function calcViewBox() {{
  var xs = Object.values(pos).map(function(p) {{ return p.x; }});
  var ys = Object.values(pos).map(function(p) {{ return p.y; }});
  return {{
    x: Math.min.apply(null, xs) - W - 40,
    y: Math.min.apply(null, ys) - 80,
    w: Math.max.apply(null, xs) - Math.min.apply(null, xs) + W * 2 + 80,
    h: Math.max.apply(null, ys) - Math.min.apply(null, ys) + H + 160
  }};
}}
function setVB() {{
  svg.setAttribute('viewBox', vb.x + ' ' + vb.y + ' ' + vb.w + ' ' + vb.h);
}}
setVB();

// Defs (arrowheads)
var defs = document.createElementNS(ns, 'defs');
Object.entries(EDGE_CLR).forEach(function(entry) {{
  var rel = entry[0], clr = entry[1];
  var m = document.createElementNS(ns, 'marker');
  m.setAttribute('id', 'a-' + rel);
  m.setAttribute('viewBox', '0 0 10 7');
  m.setAttribute('refX', '10'); m.setAttribute('refY', '3.5');
  m.setAttribute('markerWidth', '8'); m.setAttribute('markerHeight', '6');
  m.setAttribute('orient', 'auto-start-reverse');
  var p = document.createElementNS(ns, 'path');
  p.setAttribute('d', 'M 0 0 L 10 3.5 L 0 7 z');
  p.setAttribute('fill', clr);
  m.appendChild(p); defs.appendChild(m);
}});
svg.appendChild(defs);

// Draw edges with animation
EDGES.forEach(function(e, i) {{
  var f = pos[e.source_id], t = pos[e.target_id];
  if (!f || !t) return;
  var clr = EDGE_CLR[e.relationship] || '#475569';
  var my = (f.y + H / 2 + t.y - 10) / 2;

  var g = document.createElementNS(ns, 'g');
  g.classList.add('edge');

  var path = document.createElementNS(ns, 'path');
  path.setAttribute('d', 'M ' + f.x + ' ' + (f.y + H / 2) +
    ' C ' + f.x + ' ' + my + ', ' + t.x + ' ' + my + ', ' + t.x + ' ' + (t.y - 10));
  path.setAttribute('fill', 'none');
  path.setAttribute('stroke', clr);
  path.setAttribute('stroke-width', '1.5');
  path.setAttribute('opacity', '0.4');
  path.setAttribute('marker-end', 'url(#a-' + e.relationship + ')');
  path.setAttribute('stroke-dasharray', '500');
  path.setAttribute('stroke-dashoffset', '500');
  path.style.animation = 'drawEdge 0.6s ease-out ' + (i * 0.03) + 's forwards';
  if (e.relationship === 'informed') path.setAttribute('stroke-dasharray', '6,4');

  g.addEventListener('mouseenter', function() {{
    path.setAttribute('opacity', '1');
    path.setAttribute('stroke-width', '3');
    var lbl = g.querySelector('.edge-label');
    if (!lbl) {{
      lbl = document.createElementNS(ns, 'text');
      lbl.classList.add('edge-label');
      lbl.setAttribute('x', (f.x + t.x) / 2);
      lbl.setAttribute('y', my - 8);
      lbl.setAttribute('text-anchor', 'middle');
      lbl.setAttribute('font-size', '10');
      lbl.setAttribute('fill', clr);
      lbl.setAttribute('font-weight', '600');
      lbl.setAttribute('font-family', 'JetBrains Mono, monospace');
      lbl.textContent = e.relationship;
      g.appendChild(lbl);
    }}
  }});
  g.addEventListener('mouseleave', function() {{
    path.setAttribute('opacity', '0.4');
    path.setAttribute('stroke-width', '1.5');
    var lbl = g.querySelector('.edge-label');
    if (lbl) lbl.remove();
  }});

  g.appendChild(path);
  svg.appendChild(g);
}});

// Draw nodes with animation
NODES.forEach(function(n, idx) {{
  var p = pos[n.node_id];
  if (!p) return;
  var cfg = TYPE_CFG[n.node_type] || TYPE_CFG.context_load;
  var isPivot = PIVOTS.indexOf(n.node_id) !== -1;

  var g = document.createElementNS(ns, 'g');
  g.classList.add('node');
  g.style.opacity = '0';
  g.style.animation = 'fadeInNode 0.4s ease-out ' + (idx * 0.04) + 's forwards';
  g.addEventListener('click', function() {{ showDetail(n); }});

  // Pivot outline
  if (isPivot) {{
    var outline = document.createElementNS(ns, 'rect');
    outline.setAttribute('x', p.x - W / 2 - 4);
    outline.setAttribute('y', p.y - 4);
    outline.setAttribute('width', W + 8);
    outline.setAttribute('height', H + 8);
    outline.setAttribute('rx', '12');
    outline.setAttribute('fill', 'none');
    outline.setAttribute('stroke', '#f59e0b');
    outline.setAttribute('stroke-width', '2');
    outline.setAttribute('stroke-dasharray', '8,4');
    outline.setAttribute('opacity', '0.6');
    g.appendChild(outline);
  }}

  // Main rect
  var rect = document.createElementNS(ns, 'rect');
  rect.setAttribute('x', p.x - W / 2);
  rect.setAttribute('y', p.y);
  rect.setAttribute('width', W);
  rect.setAttribute('height', H);
  rect.setAttribute('rx', '8');
  rect.setAttribute('fill', '#1e293b');
  rect.setAttribute('stroke', cfg.fill + '66');
  rect.setAttribute('stroke-width', '1');
  g.appendChild(rect);

  // Type badge bg
  var badgeBg = document.createElementNS(ns, 'rect');
  badgeBg.setAttribute('x', p.x - W / 2 + 8);
  badgeBg.setAttribute('y', p.y + 8);
  badgeBg.setAttribute('width', cfg.label.length * 7.5 + 28);
  badgeBg.setAttribute('height', '20');
  badgeBg.setAttribute('rx', '4');
  badgeBg.setAttribute('fill', cfg.fill + '22');
  g.appendChild(badgeBg);

  // Type badge text
  var badge = document.createElementNS(ns, 'text');
  badge.setAttribute('x', p.x - W / 2 + 14);
  badge.setAttribute('y', p.y + 22);
  badge.setAttribute('font-size', '9');
  badge.setAttribute('font-weight', '700');
  badge.setAttribute('fill', cfg.fill);
  badge.setAttribute('font-family', 'JetBrains Mono, monospace');
  badge.setAttribute('letter-spacing', '0.08em');
  badge.textContent = cfg.icon + ' ' + cfg.label;
  g.appendChild(badge);

  // Confidence
  var conf = document.createElementNS(ns, 'text');
  conf.setAttribute('x', p.x + W / 2 - 12);
  conf.setAttribute('y', p.y + 22);
  conf.setAttribute('font-size', '9');
  conf.setAttribute('fill', '#94a3b8');
  conf.setAttribute('text-anchor', 'end');
  conf.setAttribute('font-family', 'JetBrains Mono, monospace');
  conf.textContent = Math.round(n.confidence * 100) + '%';
  g.appendChild(conf);

  // Summary (foreignObject for text wrapping)
  var fo = document.createElementNS(ns, 'foreignObject');
  fo.setAttribute('x', p.x - W / 2 + 8);
  fo.setAttribute('y', p.y + 34);
  fo.setAttribute('width', W - 16);
  fo.setAttribute('height', H - 50);
  var div = document.createElement('div');
  div.style.cssText = 'font-size:10px;color:#cbd5e1;line-height:1.4;overflow:hidden;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;font-family:JetBrains Mono,monospace;';
  div.textContent = n.summary;
  fo.appendChild(div);
  g.appendChild(fo);

  // Node ID
  var nid = document.createElementNS(ns, 'text');
  nid.setAttribute('x', p.x + W / 2 - 12);
  nid.setAttribute('y', p.y + H - 8);
  nid.setAttribute('font-size', '8');
  nid.setAttribute('fill', '#475569');
  nid.setAttribute('text-anchor', 'end');
  nid.setAttribute('font-family', 'JetBrains Mono, monospace');
  nid.textContent = n.node_id;
  g.appendChild(nid);

  svg.appendChild(g);
}});

// Pan & zoom
var dragging = false, ds = null;
svg.addEventListener('mousedown', function(e) {{
  if (e.target.closest('.node')) return;
  dragging = true;
  ds = {{ x: e.clientX, y: e.clientY, vb: {{ x: vb.x, y: vb.y, w: vb.w, h: vb.h }} }};
  svg.style.cursor = 'grabbing';
}});
svg.addEventListener('mousemove', function(e) {{
  if (!dragging) return;
  var dx = (e.clientX - ds.x) * (vb.w / svg.clientWidth);
  var dy = (e.clientY - ds.y) * (vb.h / svg.clientHeight);
  vb.x = ds.vb.x - dx;
  vb.y = ds.vb.y - dy;
  setVB();
}});
svg.addEventListener('mouseup', function() {{ dragging = false; svg.style.cursor = 'grab'; }});
svg.addEventListener('mouseleave', function() {{ dragging = false; svg.style.cursor = 'grab'; }});
svg.addEventListener('wheel', function(e) {{
  e.preventDefault();
  var s = e.deltaY > 0 ? 1.1 : 0.9;
  vb.x += vb.w * (1 - s) / 2;
  vb.y += vb.h * (1 - s) / 2;
  vb.w *= s; vb.h *= s;
  setVB();
}}, {{ passive: false }});
svg.style.cursor = 'grab';

// Detail panel
function showDetail(n) {{
  var cfg = TYPE_CFG[n.node_type] || TYPE_CFG.context_load;
  var isPivot = PIVOTS.indexOf(n.node_id) !== -1;
  document.getElementById('detail-content').innerHTML =
    '<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">' +
    '<span style="font-size:18px">' + cfg.icon + '</span>' +
    '<span class="type-badge" style="color:' + cfg.fill + '">' + cfg.label + '</span>' +
    '<span style="font-size:11px;color:#64748b">(' + n.node_id + ')</span>' +
    (isPivot ? '<span class="pivot-badge">PIVOT</span>' : '') +
    '</div>' +
    '<div class="summary">' + n.summary + '</div>' +
    '<div class="evidence"><span class="evidence-label">Evidence: </span>' + (n.evidence || 'N/A') + '</div>' +
    '<div class="meta-row">' +
    '<span>Messages: ' + n.message_range[0] + '\\u2013' + n.message_range[1] + '</span>' +
    '<span>Confidence: ' + Math.round(n.confidence * 100) + '%</span>' +
    '</div>';
  document.getElementById('detail').classList.add('visible');
}}

function closeDetail() {{
  document.getElementById('detail').classList.remove('visible');
}}
</script>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"  Generated: {output_path}")
    return str(output_path)


# ─── Mermaid Generator ────────────────────────────────────────────

def generate_mermaid(dag, output_path):
    """Generate a Mermaid diagram file."""
    lines = ["graph TD"]

    type_shapes = {
        "context_load": ("([", "])"),
        "hypothesis": ("{{", "}}"),
        "investigation": ("[", "]"),
        "discovery": ("([", "])"),
        "pivot": ("[[", "]]"),
        "dead_end": ("((", "))"),
        "solution": ("[/", "/]"),
    }

    for node in dag["nodes"]:
        shape = type_shapes.get(node["node_type"], ("[", "]"))
        label = node["summary"][:60].replace('"', "'")
        icon = TYPE_COLORS.get(node["node_type"], {}).get("icon", "")
        lines.append(f'    {node["node_id"]}{shape[0]}"{icon} {label}..."{shape[1]}')

    lines.append("")
    for edge in dag["edges"]:
        label = edge["relationship"]
        lines.append(f'    {edge["source_id"]} -->|{label}| {edge["target_id"]}')

    lines.append("")
    lines.append("    %% Style pivot nodes")
    for pid in dag["pivot_nodes"]:
        lines.append(f"    style {pid} stroke:#f59e0b,stroke-width:3px,stroke-dasharray:5 5")

    lines.append("")
    for node_type, cfg in TYPE_COLORS.items():
        ids = [n["node_id"] for n in dag["nodes"] if n["node_type"] == node_type]
        if ids:
            lines.append(f"    style {','.join(ids)} fill:{cfg['bg']},stroke:{cfg['fill']},color:#e2e8f0")

    content = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)
    print(f"  Generated: {output_path}")


# ─── JSON Export ──────────────────────────────────────────────────

def export_json(dag, output_path):
    """Export the DAG as a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dag, f, indent=2, default=str)
    print(f"  Generated: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate DAG visualizations from cognitive memory store"
    )
    parser.add_argument(
        "--store", default=None,
        help="Path to ChromaDB memory store (default: data/memory_store)"
    )
    parser.add_argument("--project", "-p", required=True, help="Project ID to visualize")
    parser.add_argument("--output", "-o", default="output/", help="Output directory")
    parser.add_argument(
        "--format", "-f", default="html",
        choices=["html", "mermaid", "json", "all"],
        help="Output format (default: html)"
    )
    parser.add_argument("--json-input", help="Load from a pre-built DAG JSON instead of ChromaDB")

    args = parser.parse_args()

    store_path = args.store or os.environ.get(
        "CMM_STORE_PATH",
        str(Path(__file__).parent.parent / "data" / "memory_store")
    )

    # Load data
    print(f"\nLoading DAG data for project: {args.project}")
    if args.json_input:
        dag = load_from_json(args.json_input)
    else:
        dag = load_from_chromadb(store_path, args.project)

    print(f"  {len(dag['nodes'])} nodes, {len(dag['edges'])} edges, {len(dag['pivot_nodes'])} pivots\n")

    if not dag["nodes"]:
        print(f"No nodes found for project '{args.project}'. Check project ID and store path.")
        sys.exit(1)

    # Generate outputs
    out_dir = Path(args.output)
    fmt = args.format
    if fmt in ("html", "all"):
        generate_html(dag, out_dir / f"dag_{args.project}.html")
    if fmt in ("mermaid", "all"):
        generate_mermaid(dag, out_dir / f"dag_{args.project}.mermaid")
    if fmt in ("json", "all"):
        export_json(dag, out_dir / f"dag_{args.project}.json")

    print(f"\nDone! Files in: {out_dir}/")


if __name__ == "__main__":
    main()
