#!/usr/bin/env python3
"""Read-only browser for aggregated preference labels.

  python scripts/pref_bench/browse_labels.py \\
      --config /path/to/pref_bench/config.yaml --port 8767

  # extra supervision files, repeatable:
  python scripts/pref_bench/browse_labels.py \\
      --config /path/to/pref_bench/config.yaml \\
      --extra holdout=/path/to/other/supervision.jsonl

Reads ``{out_root}/supervision.jsonl`` from the config. Optional packs can also
be listed under ``browse_packs`` in that config. The page never writes labels.
A and B are the stored candidates, not the left/right order used while annotating.
"""

from __future__ import annotations

import json
import mimetypes
import re
from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

PAGE = r"""<!doctype html>
<meta charset="utf-8"/>
<title>Labeled preferences</title>
<style>
  :root { color-scheme: dark; --bg:#0e1116; --panel:#171c24; --line:#2a3342; --text:#e7ecf3; --muted:#8b97a8; --ok:#3dd68c; --warn:#f0b429; }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--text); font-family:"Segoe UI",sans-serif; }
  body { display:flex; flex-direction:column; }
  header { display:flex; gap:8px; align-items:center; padding:8px 12px; background:#12161d; border-bottom:1px solid var(--line); flex-wrap:wrap; }
  select, input, button { background:#0e1116; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:6px 8px; font-size:13px; }
  button { cursor:pointer; }
  button:hover, select:hover { border-color:#7aa2f7; }
  .muted { color:var(--muted); font-size:12px; }
  #count { margin-left:auto; }
  #app { flex:1; min-height:0; display:flex; flex-direction:column; }
  .prompt { margin:6px 10px 0; padding:6px 10px; background:var(--panel); border:1px solid var(--line); border-radius:6px; font-size:13px; }
  .prompt b.A { color:var(--ok); } .prompt b.B { color:#7aa2f7; } .prompt b.T { color:var(--warn); }
  .grid { flex:1; min-height:0; display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr 1fr; gap:6px; padding:6px 10px 10px; }
  figure { margin:0; min-height:0; background:#000; border:1px solid var(--line); border-radius:6px; display:flex; flex-direction:column; overflow:hidden; }
  figure.win { border-color:var(--ok); box-shadow:inset 0 0 0 2px var(--ok); }
  figure.tie { border-color:var(--warn); }
  figcaption { display:flex; justify-content:space-between; gap:8px; padding:3px 8px; font-size:12px; color:var(--muted); background:var(--panel); }
  .imgwrap { flex:1; min-height:0; position:relative; }
  img { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }
</style>
<header>
  <strong>Labeled preferences</strong>
  <select id="pack" title="Supervision pack"></select>
  <select id="split" title="Split"></select>
  <select id="verdict" title="Label">
    <option value="all">All labels</option>
    <option value="decisive">Decisive only</option>
    <option value="A">A is better</option>
    <option value="B">B is better</option>
    <option value="T">Tie or close</option>
  </select>
  <input id="q" placeholder="Scene or candidate" />
  <button id="prev" type="button">Previous</button>
  <button id="next" type="button">Next</button>
  <button id="rand" type="button">Random</button>
  <span class="muted" id="count">—</span>
</header>
<div id="app"></div>
<script>
let items = [];
let view = [];
let pos = 0;
const pack = document.getElementById('pack');
const split = document.getElementById('split');
const verdict = document.getElementById('verdict');
const q = document.getElementById('q');

function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function verdictOf(it){
  if (it.q >= 0.6) return 'A';
  if (it.q <= 0.4) return 'B';
  return 'T';
}
function apply(){
  const s = split.value, v = verdict.value, text = q.value.trim().toLowerCase();
  view = items.filter(it => {
    if (s !== 'all' && it.split !== s) return false;
    const got = verdictOf(it);
    if (v === 'decisive' && got === 'T') return false;
    if ((v === 'A' || v === 'B' || v === 'T') && got !== v) return false;
    if (text) {
      const blob = `${it.scene_id} ${it.candidate_a} ${it.candidate_b} ${it.pair_id}`.toLowerCase();
      if (!blob.includes(text)) return false;
    }
    return true;
  });
  pos = 0;
  render();
}
function cell(idx, side, title, sub, cls){
  const src = `/img/${idx}/${side}?pack=${encodeURIComponent(pack.value)}`;
  return `<figure class="${cls}"><div class="imgwrap"><img src="${src}" alt="${esc(title)}"/></div><figcaption><span>${esc(title)}</span><span>${esc(sub)}</span></figcaption></figure>`;
}
function render(){
  const app = document.getElementById('app');
  const n = view.length;
  document.getElementById('count').textContent = n ? `${pos+1} / ${n}` : '0';
  if (!n) { app.innerHTML = '<div class="prompt">No labeled pairs match these filters.</div>'; return; }
  const it = view[pos];
  const got = verdictOf(it);
  const word = got === 'A' ? 'A is better' : got === 'B' ? 'B is better' : 'Tie or close';
  const aCls = got === 'A' ? 'win' : got === 'T' ? 'tie' : '';
  const bCls = got === 'B' ? 'win' : got === 'T' ? 'tie' : '';
  const aNote = got === 'A' ? 'Selected' : '';
  const bNote = got === 'B' ? 'Selected' : '';
  app.innerHTML = `
    <div class="prompt"><b class="${got}">${word}</b>
      · q=${it.q.toFixed(2)} · y=(${it.y_A}, ${it.y_T}, ${it.y_B})
      · ${esc(it.dataset)} / ${esc(it.split)} · ${esc(it.scene_id)}
      · ${esc(it.pair_id)}
      <span class="muted"> · A and B are the stored candidates, not the left/right order shown while annotating.</span></div>
    <div class="grid">
      ${cell(it.i,'ir','Infrared (IR)', it.scene_id, '')}
      ${cell(it.i,'vi','Visible (VI)', it.split, '')}
      ${cell(it.i,'a','Candidate A · '+it.candidate_a, aNote, aCls)}
      ${cell(it.i,'b','Candidate B · '+it.candidate_b, bNote, bCls)}
    </div>`;
}
async function loadPack(){
  const res = await fetch('/api/items?pack=' + encodeURIComponent(pack.value));
  items = await res.json();
  const splits = ['all', ...new Set(items.map(it => it.split).filter(Boolean))];
  const cur = split.value || 'test';
  split.innerHTML = splits.map(s => `<option value="${esc(s)}">${s==='all'?'All splits':esc(s)}</option>`).join('');
  split.value = splits.includes(cur) ? cur : 'all';
  apply();
}
pack.onchange = loadPack;
split.onchange = apply;
verdict.onchange = apply;
q.oninput = apply;
document.getElementById('prev').onclick = () => { if (view.length) { pos = (pos - 1 + view.length) % view.length; render(); } };
document.getElementById('next').onclick = () => { if (view.length) { pos = (pos + 1) % view.length; render(); } };
document.getElementById('rand').onclick = () => { if (view.length) { pos = Math.floor(Math.random()*view.length); render(); } };
window.onkeydown = (e) => {
  if (e.target === q) return;
  if (e.key === 'ArrowLeft') document.getElementById('prev').click();
  if (e.key === 'ArrowRight') document.getElementById('next').click();
  if (e.key === 'r' || e.key === 'R') document.getElementById('rand').click();
};
fetch('/api/packs').then(r => r.json()).then(rows => {
  pack.innerHTML = rows.map(r => `<option value="${esc(r.id)}">${esc(r.label)} · ${r.n}</option>`).join('');
  verdict.value = 'decisive';
  loadPack();
});
</script>
"""

_ID_RE = re.compile(r'[^\w\-]+')


def load_config(path: Path) -> dict:
    with path.open(encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not data.get('out_root'):
        raise SystemExit(f'{path} must set out_root')
    return data


def _pack_id(text: str, used: set[str]) -> str:
    slug = _ID_RE.sub('-', text.strip()).strip('-').lower() or 'pack'
    base = slug
    n = 2
    while slug in used:
        slug = f'{base}-{n}'
        n += 1
    used.add(slug)
    return slug


def load_supervision(path: Path) -> list[dict]:
    rows = []
    for index, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ('y_A', 'y_T', 'y_B', 'ir', 'vi', 'path_a', 'path_b'):
            if key not in row:
                raise SystemExit(f'{path} line {index + 1} is missing {key}')
        q = float(row['y_A']) + 0.5 * float(row['y_T'])
        rows.append({
            'i': index,
            'pair_id': row.get('pair_id', ''),
            'scene_id': row.get('scene_id', ''),
            'dataset': row.get('dataset', ''),
            'split': row.get('split', ''),
            'candidate_a': row.get('candidate_a', ''),
            'candidate_b': row.get('candidate_b', ''),
            'y_A': float(row['y_A']),
            'y_T': float(row['y_T']),
            'y_B': float(row['y_B']),
            'q': q,
            '_ir': row['ir'],
            '_vi': row['vi'],
            '_a': row['path_a'],
            '_b': row['path_b'],
        })
    return rows


def discover_packs(cfg: dict, extras: list[str]) -> dict[str, dict]:
    """Primary pack is ``{out_root}/supervision.jsonl``. Extras are ``label=path``."""
    used: set[str] = set()
    loaded: dict[str, dict] = {}
    primary = Path(cfg['out_root']) / 'supervision.jsonl'
    if not primary.is_file():
        raise SystemExit(
            f'missing {primary}. Run aggregate_preferences.py before browsing labels.'
        )
    loaded['main'] = {
        'label': Path(cfg['out_root']).name or 'main',
        'rows': load_supervision(primary),
    }
    used.add('main')

    declared = []
    for item in cfg.get('browse_packs') or []:
        if not isinstance(item, dict) or not item.get('path'):
            raise SystemExit('each browse_packs entry needs a path')
        declared.append((str(item.get('label') or item.get('id') or 'extra'), str(item['path'])))
    for spec in extras:
        if '=' not in spec:
            raise SystemExit(f'--extra must be label=path, got {spec!r}')
        label, path = spec.split('=', 1)
        declared.append((label.strip() or 'extra', path.strip()))

    for label, raw_path in declared:
        path = Path(raw_path)
        if not path.is_file():
            raise SystemExit(f'missing extra supervision file: {path}')
        pack_id = _pack_id(label, used)
        loaded[pack_id] = {'label': label, 'rows': load_supervision(path)}
    return loaded


def serve(loaded: dict[str, dict], port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *log_args) -> None:
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == '/':
                self._send(200, PAGE.encode(), 'text/html; charset=utf-8')
                return
            if parsed.path == '/api/packs':
                payload = [
                    {'id': key, 'label': value['label'], 'n': len(value['rows'])}
                    for key, value in loaded.items()
                ]
                self._send(200, json.dumps(payload).encode(), 'application/json')
                return
            if parsed.path == '/api/items':
                pack_id = parse_qs(parsed.query).get('pack', ['main'])[0]
                rows = loaded.get(pack_id, {}).get('rows', [])
                public = [{k: v for k, v in row.items() if not k.startswith('_')} for row in rows]
                self._send(200, json.dumps(public).encode(), 'application/json')
                return
            parts = parsed.path.strip('/').split('/')
            if len(parts) == 3 and parts[0] == 'img':
                pack_id = parse_qs(parsed.query).get('pack', ['main'])[0]
                try:
                    index = int(parts[1])
                except ValueError:
                    self._send(404, b'missing', 'text/plain; charset=utf-8')
                    return
                side = parts[2]
                rows = loaded.get(pack_id, {}).get('rows', [])
                chosen = rows[index] if 0 <= index < len(rows) else None
                key = {'ir': '_ir', 'vi': '_vi', 'a': '_a', 'b': '_b'}.get(side)
                if chosen is None or key is None:
                    self._send(404, b'missing', 'text/plain; charset=utf-8')
                    return
                image = Path(chosen[key])
                if not image.is_file():
                    self._send(404, b'missing file', 'text/plain; charset=utf-8')
                    return
                kind = mimetypes.guess_type(image.name)[0] or 'application/octet-stream'
                self._send(200, image.read_bytes(), kind)
                return
            self._send(404, b'not found', 'text/plain; charset=utf-8')

    counts = ', '.join(f'{key}={len(value["rows"])}' for key, value in loaded.items())
    print(f'[browse] packs: {counts}')
    print(f'[browse] http://127.0.0.1:{port}/')
    print('[browse] read-only. Keys: Left/Right = previous/next, R = random.')
    ThreadingHTTPServer(('0.0.0.0', port), Handler).serve_forever()


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument(
        '--extra', action='append', default=[],
        help='extra pack as label=/path/to/supervision.jsonl (repeatable)',
    )
    args = parser.parse_args()
    loaded = discover_packs(load_config(args.config), args.extra)
    serve(loaded, args.port)


if __name__ == '__main__':
    main()
