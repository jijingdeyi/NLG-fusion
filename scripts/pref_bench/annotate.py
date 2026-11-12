#!/usr/bin/env python3
"""Preference pairwise annotation UI.

  python scripts/pref_bench/annotate.py --config /path/to/pref_bench/config.yaml --port 8766

Open the page, choose a database, then new annotation, a recommended
re-annotation queue, or a read-only view of labeled results. Re-annotation
is a routine check (preference cycles, otherwise a small random sample) or
pairs where the trained scorer strongly disagrees with the recorded label.
Earlier labels are kept. Back or Escape returns to that menu. Each browser
stores its own identity, so several people can share one server.

Writes append-only: {out_root}/labels_{annotator}.jsonl
Expertise is stored on the annotator (annotators.jsonl), not on each click.
By default skips task_ids already labeled by that person (resume-safe).
Pass --allow-reannotate to enable multi-round labels for the same pair.
Each card randomly swaps left/right display (§3.3.1); responses are mapped
to fixed A/B labels via left_candidate/right_candidate.
"""

from __future__ import annotations

import json
import random
import re
import signal
import sys
import threading
import time
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pref.schema import response_to_label

EXPERTISE = ('professional', 'non-professional')
ANNOTATOR_RE = re.compile(r'^[\w\-]{1,32}$', re.UNICODE)

# A recorded choice is worth another look when the trained scorer assigns the
# opposite candidate at least this probability. Labels are never overwritten.
MODEL_DISAGREE = 0.6

PROMPT = (
    'Considering infrared salient targets, visible texture, structural clarity, and artifacts, '
    'choose the fusion result with better overall quality.'
)


def load_config(path: Path) -> dict:
    with path.open(encoding='utf-8') as f:
        return yaml.safe_load(f)


def spread_by_scene(pairs: list[dict]) -> list[dict]:
    """Order pairs so the same scene does not appear twice in a row."""
    groups: dict[str, list[dict]] = {}
    for pair in pairs:
        groups.setdefault(str(pair.get('scene_id') or pair.get('task_id') or ''), []).append(pair)
    buckets = list(groups.values())
    for bucket in buckets:
        random.shuffle(bucket)
    ordered: list[dict] = []
    previous: list[dict] | None = None
    while buckets:
        random.shuffle(buckets)
        if previous is not None and len(buckets) > 1 and buckets[0] is previous:
            buckets.append(buckets.pop(0))
        bucket = buckets[0]
        ordered.append(bucket.pop())
        if bucket:
            previous = bucket
        else:
            buckets.pop(0)
            previous = None
    return ordered


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def valid_annotator_id(annotator_id: str) -> str:
    aid = (annotator_id or '').strip()
    if not ANNOTATOR_RE.fullmatch(aid):
        raise ValueError('Username must be 1–32 characters: letters, digits, underscore, or hyphen')
    return aid


def valid_expertise(expertise: str) -> str:
    if expertise not in EXPERTISE:
        raise ValueError('expertise must be professional or non-professional')
    return expertise


def _valid_client_id(client_id: str) -> str:
    cid = (client_id or '').strip()
    if not re.fullmatch(r'[\w\-]{8,64}', cid):
        raise ValueError('invalid client_id')
    return cid


class AnnotateState:
    """One server, many annotators. Progress is tracked per annotator id."""

    def __init__(
        self,
        cfg: dict,
        split: str | None,
        allow_reannotate: bool = False,
        queue_task_ids: set[str] | None = None,
        default_annotator: str | None = None,
        extras: list[str] | None = None,
        db_id: str = 'main',
        db_label: str | None = None,
    ):
        self.cfg = cfg
        self.out = Path(cfg['out_root']).resolve()
        self.db_id = db_id
        self.db_label = db_label or self.out.name or db_id
        self.split = split
        self.allow_reannotate = allow_reannotate
        self.queue_task_ids = queue_task_ids
        self.default_annotator = default_annotator
        self.extras = list(extras or [])
        self.relabel_baselines: dict[str, dict[str, int]] = {}
        self.model_baselines: dict[str, dict[str, int]] = {}
        self.label_restrict: dict[str, set[str]] = {}
        self.label_note: dict[str, str] = {}
        self.score_lock = threading.Lock()
        self.labeled_rows: dict[str, list[dict]] = {}
        self.pairs_path = self.out / 'pairs.jsonl'
        self.annotators_path = self.out / 'annotators.jsonl'
        self.annotations_path = self.out / 'annotations.jsonl'
        self.lock = threading.Lock()
        self.users: dict[str, dict] = {}
        self.user_mode: dict[str, str] = {}
        self.recheck_ids: set[str] = set()
        self.recommend_meta: dict = {}
        self.n_new = 0
        self.clients: dict[str, float] = {}
        self.leaving: set[str] = set()
        self.seen_browser = False
        self.httpd = None
        self._stopping = False
        # (annotator_id, task_id) → display-side assignment for the open card
        self.pending_display: dict[tuple[str, str], dict] = {}
        self.reload_pairs()

    def reload_pairs(self) -> None:
        pairs = load_jsonl(self.pairs_path)
        if self.split:
            pairs = [p for p in pairs if p.get('split') == self.split]
        if self.queue_task_ids is not None:
            pairs = [p for p in pairs if p['task_id'] in self.queue_task_ids]
        self.pairs = pairs
        self.by_id = {p['task_id']: p for p in pairs}
        self.users.clear()

    def labels_path(self, annotator_id: str) -> Path:
        return self.out / f'labels_{annotator_id}.jsonl'

    def recheck_pass_path(self, annotator_id: str) -> Path:
        return self.out / 'fit' / f'recheck_pass_{annotator_id}.json'

    def relabel_pass_path(self, annotator_id: str) -> Path:
        return self.out / 'fit' / f'relabel_pass_{annotator_id}.json'

    def model_pass_path(self, annotator_id: str) -> Path:
        return self.out / 'fit' / f'model_pass_{annotator_id}.json'

    def _round_counts_locked(self, annotator_id: str) -> Counter:
        rows = load_jsonl(self.labels_path(annotator_id))
        return Counter(r['task_id'] for r in rows if 'task_id' in r)

    def _resume_recheck_baseline(self, annotator_id: str, pair_ids: list[str]) -> dict[str, int]:
        """Keep the same pass across refresh; only start a new pass when the queue set changes."""
        path = self.recheck_pass_path(annotator_id)
        want = sorted(set(pair_ids))
        prev = None
        if path.is_file():
            try:
                prev = json.loads(path.read_text(encoding='utf-8'))
            except Exception:
                prev = None
        if prev and sorted(prev.get('pair_ids') or []) == want and isinstance(prev.get('baseline_rounds'), dict):
            return {str(k): int(v) for k, v in prev['baseline_rounds'].items()}
        baseline = dict(self._round_counts_locked(annotator_id))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({'pair_ids': want, 'baseline_rounds': baseline}, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        return baseline

    def heartbeat(self, client_id: str) -> None:
        cid = _valid_client_id(client_id)
        with self.lock:
            self.clients[cid] = time.monotonic()
            self.leaving.discard(cid)
            self.seen_browser = True

    def browser_left(self, client_id: str) -> None:
        """Page closed or refreshed. Wait briefly so a reload can heartbeat back."""
        cid = _valid_client_id(client_id)
        with self.lock:
            if cid not in self.clients and cid not in self.leaving:
                self.clients[cid] = time.monotonic()
            self.leaving.add(cid)
            self.seen_browser = True

        def _confirm() -> None:
            should_stop = False
            with self.lock:
                if cid not in self.leaving:
                    return
                self.leaving.discard(cid)
                self.clients.pop(cid, None)
                should_stop = not self.clients and not self.leaving
            if should_stop:
                self.stop_server('browser closed')

        threading.Timer(4.0, _confirm).start()

    def sweep_clients(self) -> None:
        """Forget tabs that stopped heartbeating (sleep / lock / crash).

        Do NOT stop the server here: screen lock and laptop sleep also pause
        heartbeats, which previously looked like "browser closed".
        Shutdown only happens after an explicit /api/bye from pagehide.
        """
        with self.lock:
            if not self.seen_browser:
                return
            now = time.monotonic()
            stale = [cid for cid, seen in self.clients.items() if now - seen > 60.0]
            for cid in stale:
                self.clients.pop(cid, None)
                self.leaving.discard(cid)

    def stop_server(self, reason: str) -> None:
        with self.lock:
            if self._stopping:
                return
            self._stopping = True
            httpd = self.httpd
        print(f'\n[annotate] {reason}, stopping')
        if httpd is not None:
            threading.Thread(target=httpd.shutdown, daemon=True).start()

    def list_annotators(self) -> list[dict]:
        by_id: dict[str, dict] = {}
        for row in load_jsonl(self.annotators_path):
            aid = row.get('annotator_id')
            if not aid:
                continue
            by_id[aid] = {
                'annotator_id': aid,
                'expertise': row.get('expertise') or 'professional',
            }
        for fp in sorted(self.out.glob('labels_*.jsonl')):
            aid = fp.stem[len('labels_'):]
            by_id.setdefault(aid, {'annotator_id': aid, 'expertise': 'professional'})
        out = []
        for aid, row in sorted(by_id.items()):
            n = sum(1 for line in load_jsonl(self.labels_path(aid)) if line.get('task_id'))
            out.append({**row, 'n_labels': n})
        return out

    def upsert_annotator(
        self,
        annotator_id: str,
        expertise: str,
        mode: str = 'label',
        only_task_ids: list | None = None,
        note: str = '',
    ) -> dict:
        aid = valid_annotator_id(annotator_id)
        exp = valid_expertise(expertise)
        if mode not in ('label', 'recheck', 'relabel', 'model'):
            raise ValueError('mode must be label, recheck, relabel, or model')
        resume_model = False
        if mode == 'model':
            with self.lock:
                resume_model = bool(self._pass_remaining(self.model_pass_path(aid), aid))
        prepared = None
        if mode == 'model' and not resume_model:
            prepared = self._disagreement_queue()
        with self.lock:
            rows = load_jsonl(self.annotators_path)
            found = False
            for row in rows:
                if row.get('annotator_id') == aid:
                    row['expertise'] = exp
                    row.pop('note', None)
                    found = True
                    break
            if not found:
                rows.append({'annotator_id': aid, 'expertise': exp})
            self.annotators_path.parent.mkdir(parents=True, exist_ok=True)
            with self.annotators_path.open('w', encoding='utf-8') as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            recommendation = None
            if mode == 'recheck':
                recommendation = self._enter_recheck_locked(aid)
                self.user_mode[aid] = 'recheck'
            elif mode == 'relabel':
                recommendation = self._enter_relabel_locked(aid)
                self.user_mode[aid] = 'relabel'
            elif mode == 'model':
                recommendation = self._enter_model_locked(aid, prepared)
                self.user_mode[aid] = 'model'
            else:
                if only_task_ids:
                    self.label_restrict[aid] = {str(item) for item in only_task_ids}
                else:
                    self.label_restrict.pop(aid, None)
                self.label_note[aid] = note or ''
                self.user_mode[aid] = 'normal'
            self._load_user(aid)
            out = {'annotator_id': aid, 'expertise': exp, 'progress': self.progress(aid)}
            if recommendation is not None:
                out['recommendation'] = recommendation
            return out

    def _enter_recheck_locked(self, annotator_id: str) -> dict:
        """Resume an unfinished recheck pass across refresh; else build a new queue."""
        path = self.recheck_pass_path(annotator_id)
        rounds = self._round_counts_locked(annotator_id)
        prev = None
        if path.is_file():
            try:
                prev = json.loads(path.read_text(encoding='utf-8'))
            except Exception:
                prev = None
        if prev and prev.get('pair_ids') and isinstance(prev.get('baseline_rounds'), dict):
            baseline = {str(k): int(v) for k, v in prev['baseline_rounds'].items()}
            pair_ids = list(prev['pair_ids'])
            remaining = [
                pid for pid in pair_ids
                if int(rounds.get(pid, 0)) <= int(baseline.get(pid, 0))
            ]
            if remaining:
                return self._bind_recheck_pass(pair_ids, remaining, resumed=True)
            # The saved pass is finished. Recommend from the current labels
            # instead of replaying that queue.
            info = self._recommend_locked()
            self._resume_recheck_baseline(annotator_id, list(info['pair_ids']))
            return info

        # Adopt an existing queue written before this process (e.g. mid-pass refresh
        # before pass files existed): treat labels newer than the queue file as
        # already done for this pass.
        adopted = self._try_adopt_queue_locked(annotator_id, rounds)
        if adopted is not None:
            return adopted

        info = self._recommend_locked()
        self._resume_recheck_baseline(annotator_id, list(info['pair_ids']))
        return info

    def _enter_relabel_locked(self, annotator_id: str) -> dict:
        """One additional judgment for each pair this annotator has already labeled."""
        rounds = self._round_counts_locked(annotator_id)
        labeled = [p['task_id'] for p in self.pairs if int(rounds.get(p['task_id'], 0)) > 0]
        if not labeled:
            self.relabel_baselines[annotator_id] = {}
            info = {
                'pair_ids': [],
                'n': 0,
                'n_remaining': 0,
                'note': 'This annotator has no labeled pairs to re-annotate.',
            }
            self.recommend_meta = info
            return info
        path = self.relabel_pass_path(annotator_id)
        baseline = None
        if path.is_file():
            try:
                prev = json.loads(path.read_text(encoding='utf-8'))
            except Exception:
                prev = None
            if prev and sorted(prev.get('pair_ids') or []) == sorted(labeled):
                saved = prev.get('baseline_rounds')
                if isinstance(saved, dict):
                    baseline = {str(k): int(v) for k, v in saved.items()}

        def remaining(base: dict[str, int]) -> list[str]:
            return [
                pid for pid in labeled
                if int(rounds.get(pid, 0)) <= int(base.get(pid, 0))
            ]

        resumed = baseline is not None and bool(remaining(baseline))
        if not resumed:
            baseline = {pid: int(rounds.get(pid, 0)) for pid in labeled}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(
                {'pair_ids': labeled, 'baseline_rounds': baseline},
                ensure_ascii=False, indent=2,
            ), encoding='utf-8')
        assert baseline is not None
        self.relabel_baselines[annotator_id] = baseline
        left = remaining(baseline)
        note = (
            f'Resuming re-annotation ({len(left)} / {len(labeled)} left)'
            if resumed else
            f'Re-annotation · {len(labeled)} labeled pairs'
        )
        info = {'pair_ids': labeled, 'n': len(labeled), 'n_remaining': len(left), 'note': note}
        self.recommend_meta = info
        return info

    def _bind_recheck_pass(
        self,
        pair_ids: list[str],
        remaining: list[str],
        *,
        resumed: bool,
    ) -> dict:
        self.recheck_ids = set(pair_ids)
        note = (
            f'Resuming the previous review queue ({len(remaining)} / {len(pair_ids)} left; '
            f'refresh will not bring back items already labeled)'
            if resumed
            else f'Review queue · {len(pair_ids)} items'
        )
        info = {
            'pair_ids': pair_ids,
            'n': len(pair_ids),
            'n_remaining': len(remaining),
            'reasons': ['resume'] if resumed else ['adopt'],
            'cycle_scenes': [],
            'note': note,
        }
        self.recommend_meta = info
        return info

    def _try_adopt_queue_locked(self, annotator_id: str, rounds: Counter) -> dict | None:
        qpath = self.out / 'fit' / 'reannotation_queue.json'
        if not qpath.is_file():
            return None
        try:
            q = json.loads(qpath.read_text(encoding='utf-8'))
        except Exception:
            return None
        pair_ids = list(q.get('reannotation_pair_ids') or [])
        if not pair_ids:
            return None
        cutoff = qpath.stat().st_mtime
        id_set = set(pair_ids)
        after: Counter = Counter()
        # Also credit same-calendar-day re-labels on this queue: the queue file
        # may have been rewritten mid-pass (mtime jumps), which would otherwise
        # make refresh look like "start over".
        day_hits: Counter = Counter()
        today = datetime.now().astimezone().date()
        for row in load_jsonl(self.labels_path(annotator_id)):
            tid = row.get('task_id')
            if tid not in id_set:
                continue
            ts = row.get('ts')
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00')).astimezone()
            except Exception:
                continue
            if dt.timestamp() > cutoff:
                after[tid] += 1
            if dt.date() == today and int(rounds.get(tid, 0)) >= 2:
                day_hits[tid] = 1
        credit = {pid: max(int(after.get(pid, 0)), int(day_hits.get(pid, 0))) for pid in pair_ids}
        baseline = {
            pid: int(rounds.get(pid, 0)) - int(credit.get(pid, 0))
            for pid in pair_ids
        }
        remaining = [
            pid for pid in pair_ids
            if int(rounds.get(pid, 0)) <= int(baseline.get(pid, 0))
        ]
        if not remaining:
            return None
        path = self.recheck_pass_path(annotator_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({'pair_ids': pair_ids, 'baseline_rounds': baseline}, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        note_extra = q.get('note') or ''
        info = self._bind_recheck_pass(pair_ids, remaining, resumed=True)
        if note_extra:
            info['note'] = f'{info["note"]}; {note_extra}'
        return info

    def _recommend_locked(self) -> dict:
        from pref.io import export_supervision_jsonl
        from pref.qc import recommend_review

        eta = float(self.cfg.get('eta', 0.5))
        # Refresh y/R so cycle / low-R recommendations track the latest labels.
        export_supervision_jsonl(self.out, eta=eta, cfg=self.cfg)
        rows = load_jsonl(self.out / 'supervision.jsonl')
        info = recommend_review(rows, self.pairs, seed=int(self.cfg.get('seed', 443)))
        self.recheck_ids = set(info['pair_ids'])
        self.recommend_meta = info
        dest = self.out / 'fit' / 'reannotation_queue.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({
            'reannotation_pair_ids': info['pair_ids'],
            'reasons': info['reasons'],
            'note': info['note'],
            'cycle_scenes': info['cycle_scenes'],
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        return info

    def _pass_remaining(self, path: Path, annotator_id: str) -> list[str]:
        if not path.is_file():
            return []
        try:
            prev = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return []
        if float(prev.get('thresh') or 0) != MODEL_DISAGREE:
            return []
        pair_ids = [str(pid) for pid in (prev.get('pair_ids') or [])]
        baseline = prev.get('baseline_rounds')
        if not pair_ids or not isinstance(baseline, dict):
            return []
        rounds = self._round_counts_locked(annotator_id)
        return [
            pid for pid in pair_ids
            if int(rounds.get(pid, 0)) <= int(baseline.get(str(pid), 0))
        ]

    def _enter_model_locked(self, annotator_id: str, prepared: tuple[list[str], str] | None) -> dict:
        """One extra judgment on model-disagreement pairs. Does not overwrite labels."""
        path = self.model_pass_path(annotator_id)
        rounds = self._round_counts_locked(annotator_id)
        remaining = self._pass_remaining(path, annotator_id)
        if remaining:
            prev = json.loads(path.read_text(encoding='utf-8'))
            pair_ids = [str(pid) for pid in prev['pair_ids']]
            baseline = {str(k): int(v) for k, v in prev['baseline_rounds'].items()}
            note = prev.get('note') or 'Resuming model-disagreement review'
            note = f'Resuming model-disagreement review ({len(remaining)} / {len(pair_ids)} left). {note}'
        else:
            pair_ids, note = prepared or ([], 'No strong model disagreements.')
            pair_ids = [pid for pid in pair_ids if pid in self.by_id]
            baseline = {pid: int(rounds.get(pid, 0)) for pid in pair_ids}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                'pair_ids': pair_ids,
                'baseline_rounds': baseline,
                'note': note,
                'thresh': MODEL_DISAGREE,
            }, ensure_ascii=False, indent=2), encoding='utf-8')
            remaining = list(pair_ids)
        self.model_baselines[annotator_id] = baseline
        info = {
            'pair_ids': pair_ids,
            'n': len(pair_ids),
            'n_remaining': len(remaining),
            'reasons': ['model'],
            'note': note,
        }
        self.recommend_meta = info
        return info

    def extend_defaults(self) -> dict:
        cfg = self._extend_cfg()
        datasets = []
        for name, spec in (cfg.get('datasets') or {}).items():
            if isinstance(spec, dict) and spec.get('root'):
                datasets.append({'name': str(name), 'root': str(spec['root'])})
        gen = cfg.get('gen_eval') or {}
        return {
            'datasets': datasets,
            'has_candidates': bool(cfg.get('candidates')),
            'pairs_per_scene': int(cfg.get('pairs_per_scene') or gen.get('pairs_per_scene') or 8),
        }

    def _extend_cfg(self) -> dict:
        local = self.out / 'config.yaml'
        if local.is_file():
            loaded = yaml.safe_load(local.read_text(encoding='utf-8')) or {}
            if isinstance(loaded, dict):
                loaded['out_root'] = str(self.out)
                return loaded
        cfg = dict(self.cfg)
        cfg['out_root'] = str(self.out)
        return cfg

    def extend_data(self, kind: str, fields: dict) -> dict:
        import importlib.util

        path = Path(__file__).resolve().parent / 'build.py'
        spec = importlib.util.spec_from_file_location('pref_bench_build', path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Cannot load {path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = self._extend_cfg()
        config_path = self.out / 'config.yaml'
        if kind == 'scenes':
            fusion = fields.get('fusion') or None
            info = module.extend_scenes(
                cfg,
                str(fields.get('dataset') or ''),
                Path(str(fields.get('source') or '')),
                Path(fusion) if fusion else None,
                config_path=config_path if config_path.is_file() else None,
                confirm=bool(fields.get('confirm')),
            )
        elif kind == 'method':
            info = module.extend_method(
                cfg,
                str(fields.get('method') or ''),
                Path(str(fields.get('results') or '')),
                config_path=config_path if config_path.is_file() else None,
                confirm=bool(fields.get('confirm')),
            )
        else:
            raise ValueError('kind must be scenes or method')
        with self.lock:
            self.reload_pairs()
        return info

    def _scorer_ckpt(self) -> Path:
        candidates = []
        configured = self.cfg.get('scorer_ckpt')
        if configured:
            candidates.append(Path(configured))
        candidates.append(Path('/data/ykx/pref_bench/fit/scorer_stage1_best.pt'))
        candidates.append(Path(__file__).resolve().parents[2] / 'checkpoints' / 'scorer_stage1_best.pt')
        for path in candidates:
            if path.is_file():
                return path
        raise ValueError('Trained scorer checkpoint was not found')

    def _disagreement_queue(self) -> tuple[list[str], str]:
        """Decisive labels the trained scorer contradicts at probability above MODEL_DISAGREE."""
        import torch
        from pref.dataset import load_triple
        from pref.model.resnet_scorer import ResNet18ScalarScorer
        from pref.qc import conflict_pairs

        with self.score_lock:
            self.refresh_supervision()
            rows = load_jsonl(self.out / 'supervision.jsonl')
            if not rows:
                return [], 'This database has no labeled pairs to review.'
            ckpt_path = self._scorer_ckpt()
            cache_path = self.out / 'fit' / 'fusion_scores.json'
            cache: dict[str, float] = {}
            ckpt_mtime = ckpt_path.stat().st_mtime
            if cache_path.is_file():
                try:
                    saved = json.loads(cache_path.read_text(encoding='utf-8'))
                except Exception:
                    saved = {}
                if saved.get('ckpt') == str(ckpt_path) and saved.get('mtime') == ckpt_mtime:
                    cache = {str(k): float(v) for k, v in (saved.get('scores') or {}).items()}
            needed: dict[str, tuple[str, str]] = {}
            for row in rows:
                ir, vi = row.get('ir'), row.get('vi')
                if not ir or not vi:
                    continue
                for key in ('path_a', 'path_b'):
                    path = row.get(key)
                    if path and path not in cache and path not in needed:
                        needed[path] = (ir, vi)
            if needed:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                model = ResNet18ScalarScorer(pretrained=False).to(device)
                model.load_state_dict(ckpt['model'], strict=True)
                model.eval()
                size = int(ckpt.get('size') or 224)
                items = [(path, ir, vi) for path, (ir, vi) in needed.items()]
                print(f'[annotate] scoring {len(items)} fusions on {device} for {self.db_label}', flush=True)
                with torch.no_grad():
                    for start in range(0, len(items), 16):
                        chunk = items[start:start + 16]
                        try:
                            batch = torch.stack([
                                load_triple(ir, vi, path, size=size) for path, ir, vi in chunk
                            ]).to(device)
                            values = model.score(batch).detach().cpu().tolist()
                            scored = list(zip(chunk, values))
                        except Exception as exc:
                            print(f'[annotate] batch failed ({exc}); scoring one by one', flush=True)
                            scored = []
                            for path, ir, vi in chunk:
                                try:
                                    one = load_triple(ir, vi, path, size=size).unsqueeze(0).to(device)
                                    value = float(model.score(one).detach().cpu())
                                    scored.append(((path, ir, vi), value))
                                except Exception as one_exc:
                                    print(f'[annotate] skip {path}: {one_exc}', flush=True)
                        for (path, _, _), value in scored:
                            cache[path] = float(value)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({
                    'ckpt': str(ckpt_path),
                    'mtime': ckpt_mtime,
                    'scores': cache,
                }), encoding='utf-8')
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            probs: dict[str, tuple[float, float, float]] = {}
            margins: dict[str, float] = {}
            for row in rows:
                sa = cache.get(row.get('path_a'))
                sb = cache.get(row.get('path_b'))
                if sa is None or sb is None:
                    continue
                prob_a = float(torch.sigmoid(torch.tensor(sa - sb)))
                pid = row['pair_id']
                probs[pid] = (prob_a, 0.0, 1.0 - prob_a)
                margins[pid] = max(prob_a, 1.0 - prob_a)
            flagged = conflict_pairs(rows, probs, conf_thresh=MODEL_DISAGREE, human_margin=0.2)
            flagged.sort(key=lambda pid: margins.get(pid, 0.0), reverse=True)
            chosen = [pid for pid in flagged if pid in self.by_id]
            thresh = f'{MODEL_DISAGREE:.1f}'
            if not chosen:
                note = (
                    'No decisive label has the trained scorer assigning the other '
                    f'candidate a probability above {thresh}. Earlier labels were not changed.'
                )
            else:
                note = (
                    f'Model disagreements: {len(chosen)} decisive labels where the trained '
                    f'scorer gives the other candidate probability above {thresh}. '
                    'Stronger disagreements come first. Earlier labels are kept.'
                )
            return chosen, note

    def refresh_supervision(self) -> dict:
        from pref.io import export_supervision_jsonl

        eta = float(self.cfg.get('eta', 0.5))
        path = export_supervision_jsonl(self.out, eta=eta, cfg=self.cfg)
        return {'path': str(path), 'eta': eta}

    def _browse_sources(self) -> list[tuple[str, str, Path]]:
        sources = [(self.db_id, self.db_label, self.out / 'supervision.jsonl')]
        used = {self.db_id}
        declared: list[tuple[str, str]] = []
        for item in self.cfg.get('browse_packs') or []:
            if not isinstance(item, dict) or not item.get('path'):
                continue
            declared.append((str(item.get('label') or item.get('id') or 'extra'), str(item['path'])))
        for spec in self.extras:
            if '=' not in spec:
                raise ValueError(f'--extra must be label=path, got {spec!r}')
            label, raw = spec.split('=', 1)
            declared.append((label.strip() or 'extra', raw.strip()))
        for label, raw in declared:
            slug = re.sub(r'[^\w\-]+', '-', label.strip()).strip('-').lower() or 'pack'
            base = slug
            n = 2
            while slug in used:
                slug = f'{base}-{n}'
                n += 1
            used.add(slug)
            sources.append((slug, label, Path(raw)))
        return sources

    def labeled_catalog(self) -> list[dict]:
        """Refresh this database, then list every supervision file the page can show."""
        with self.lock:
            self.refresh_supervision()
            catalog = []
            self.labeled_rows = {}
            for pack_id, label, path in self._browse_sources():
                if not path.is_file():
                    continue
                rows = []
                for index, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    q = float(row['y_A']) + 0.5 * float(row['y_T'])
                    pair_id = row.get('pair_id', '')
                    protocol = 'e1' if pair_id.startswith('e1::') else 'e2' if pair_id.startswith('e2::') else ''
                    rows.append({
                        'i': index,
                        'pair_id': pair_id,
                        'scene_id': row.get('scene_id', ''),
                        'dataset': row.get('dataset', ''),
                        'split': row.get('split', ''),
                        'protocol': protocol,
                        'candidate_a': row.get('candidate_a', ''),
                        'candidate_b': row.get('candidate_b', ''),
                        'y_A': float(row['y_A']),
                        'y_T': float(row['y_T']),
                        'y_B': float(row['y_B']),
                        'q': q,
                        'counts': row.get('per_annotator_counts') or {},
                        '_ir': row['ir'],
                        '_vi': row['vi'],
                        '_a': row['path_a'],
                        '_b': row['path_b'],
                    })
                self.labeled_rows[pack_id] = rows
                catalog.append({'id': pack_id, 'label': label, 'n': len(rows)})
            return catalog

    def method_labels(self) -> dict[str, str]:
        """Real method names for internal slots such as method_a."""
        labels = {}
        for spec in self.cfg.get('candidates') or []:
            name = str(spec.get('name') or '')
            real = str(spec.get('method') or '')
            if name and real and real != name:
                labels[name] = real
        return labels

    def labeled_items(self, pack_id: str) -> list[dict]:
        rows = self.labeled_rows.get(pack_id, [])
        return [{k: v for k, v in row.items() if not k.startswith('_')} for row in rows]

    def labeled_image(self, pack_id: str, index: int, side: str) -> Path | None:
        rows = self.labeled_rows.get(pack_id, [])
        if not 0 <= index < len(rows):
            return None
        key = {'ir': '_ir', 'vi': '_vi', 'a': '_a', 'b': '_b'}.get(side)
        if key is None:
            return None
        path = Path(rows[index][key])
        return path if path.is_file() else None

    def _load_user(self, annotator_id: str) -> dict:
        rows = load_jsonl(self.labels_path(annotator_id))
        done = {r['task_id'] for r in rows if 'task_id' in r}
        round_counts = Counter(r['task_id'] for r in rows if 'task_id' in r)
        mode = self.user_mode.get(annotator_id, 'reannotate' if self.allow_reannotate else 'normal')
        if mode in ('relabel', 'model'):
            baseline = (
                self.model_baselines if mode == 'model' else self.relabel_baselines
            ).get(annotator_id, {})
            pool = [p for p in self.pairs if p['task_id'] in baseline]
            queue = spread_by_scene([
                p for p in pool
                if int(round_counts.get(p['task_id'], 0)) <= int(baseline.get(p['task_id'], 0))
            ])
        elif mode == 'recheck':
            pool = [
                p for p in self.pairs
                if p['task_id'] in self.recheck_ids or p.get('pair_id') in self.recheck_ids
            ]
            baseline = self._resume_recheck_baseline(
                annotator_id,
                [p['task_id'] for p in pool],
            )
            # Same pass across page refresh: skip pairs already labeled once this pass.
            queue = spread_by_scene([
                p for p in pool
                if int(round_counts.get(p['task_id'], 0)) <= int(baseline.get(p['task_id'], 0))
            ])
        elif mode == 'reannotate' or self.allow_reannotate:
            mode = 'reannotate'
            queue = spread_by_scene(list(self.pairs))
            pool = self.pairs
        else:
            restrict = self.label_restrict.get(annotator_id)
            pool = [p for p in self.pairs if restrict is None or p['task_id'] in restrict]
            queue = spread_by_scene([p for p in pool if p['task_id'] not in done])
        restricted = mode == 'normal' and annotator_id in self.label_restrict
        user = {
            'done': done,
            'round_counts': round_counts,
            'queue': queue,
            'pool': pool,
            'target': len(pool) if mode != 'normal' or restricted else len(self.pairs),
            'mode': mode,
            'restricted': restricted,
            'cursor': 0,
        }
        self.users[annotator_id] = user
        return user

    def _user(self, annotator_id: str) -> dict:
        aid = valid_annotator_id(annotator_id)
        return self.users.get(aid) or self._load_user(aid)

    def progress(self, annotator_id: str) -> dict:
        aid = valid_annotator_id(annotator_id)
        user = self._user(aid)
        mode = user.get('mode', 'normal')
        if mode == 'normal' and not user.get('restricted'):
            total = len(self.pairs)
            n_done = sum(1 for p in self.pairs if p['task_id'] in user['done'])
            remaining = total - n_done
        elif mode == 'normal':
            total = int(user.get('target') or 0)
            remaining = len(user['queue'])
            n_done = max(0, total - remaining)
        else:
            total = int(user.get('target') or 0)
            remaining = len(user['queue'])
            n_done = max(0, total - remaining)
        expertise = 'professional'
        for row in self.list_annotators():
            if row['annotator_id'] == aid:
                expertise = row['expertise']
                break
        return {
            'annotator': aid,
            'expertise': expertise,
            'mode': mode,
            'total': total,
            'done': n_done,
            'remaining': remaining,
            'labels_path': str(self.labels_path(aid)),
            'supervision_path': str(self.out / 'supervision.jsonl'),
            'database': self.db_label,
            'recommend_note': (
                self.label_note.get(aid, '') if mode == 'normal'
                else self.recommend_meta.get('note') if mode in ('recheck', 'relabel', 'model')
                else ''
            ),
        }

    def current(self, annotator_id: str) -> dict | None:
        user = self._user(annotator_id)
        if user['cursor'] >= len(user['queue']):
            return None
        return user['queue'][user['cursor']]

    def _pair_ab(self, task: dict) -> tuple[str, str]:
        cand_a = task.get('candidate_a')
        cand_b = task.get('candidate_b')
        if cand_a and cand_b:
            return cand_a, cand_b
        left, right = task['left'], task['right']
        return (left, right) if left <= right else (right, left)

    def assign_display(self, annotator_id: str, task: dict, *, force_new: bool = False) -> dict:
        """Randomize which candidate is shown on the left (score.md §3.3.1).

        Method names stay server-side; the UI only sees A/B slots. Mapping back to
        fixed candidate_a/b uses response_to_label at submit time.
        """
        aid = valid_annotator_id(annotator_id)
        key = (aid, task['task_id'])
        if not force_new and key in self.pending_display:
            return self.pending_display[key]
        swap = random.random() < 0.5
        if swap:
            left, right = task['right'], task['left']
            left_path, right_path = task['right_path'], task['left_path']
        else:
            left, right = task['left'], task['right']
            left_path, right_path = task['left_path'], task['right_path']
        assign = {
            'left_candidate': left,
            'right_candidate': right,
            'left_path': left_path,
            'right_path': right_path,
            'display_swapped': swap,
        }
        self.pending_display[key] = assign
        return assign

    def public_task(self, task: dict, annotator_id: str) -> dict:
        """Hide method identity from the annotator UI."""
        assign = self.assign_display(annotator_id, task)
        aid_q = quote(annotator_id, safe='')
        tid_q = quote(task['task_id'], safe='')
        db_q = quote(self.db_id, safe='')
        return {
            'task_id': task['task_id'],
            'scene_id': task['scene_id'],
            'dataset': task['dataset'],
            'split': task['split'],
            'display_swapped': bool(assign['display_swapped']),
            'ir_url': f'/img?side=ir&task_id={tid_q}&annotator_id={aid_q}&database={db_q}',
            'vi_url': f'/img?side=vi&task_id={tid_q}&annotator_id={aid_q}&database={db_q}',
            'left_url': f'/img?side=left&task_id={tid_q}&annotator_id={aid_q}&database={db_q}',
            'right_url': f'/img?side=right&task_id={tid_q}&annotator_id={aid_q}&database={db_q}',
            'prompt': PROMPT,
            'progress': self.progress(annotator_id),
        }

    def resolve_image(self, task_id: str, side: str, annotator_id: str | None = None) -> Path | None:
        task = self.by_id.get(task_id)
        if task is None:
            return None
        if side in ('left', 'right') and annotator_id:
            assign = self.pending_display.get((annotator_id, task_id))
            if assign is not None:
                path = Path(assign[f'{side}_path']).resolve()
            else:
                # Fallback if the card assignment expired; keep pairs.jsonl order.
                path = Path(task[f'{side}_path']).resolve()
        else:
            key = {'ir': 'ir', 'vi': 'vi', 'left': 'left_path', 'right': 'right_path'}.get(side)
            if key is None:
                return None
            path = Path(task[key]).resolve()
        try:
            path.relative_to(self.out)
        except ValueError:
            return None
        return path if path.is_file() else None

    def submit(self, annotator_id: str, task_id: str, choice: str) -> dict:
        if choice not in ('left', 'right', 'tie'):
            raise ValueError('choice must be left|right|tie')
        aid = valid_annotator_id(annotator_id)
        with self.lock:
            task = self.by_id.get(task_id)
            if task is None:
                raise KeyError(f'unknown task_id {task_id}')
            user = self._user(aid)
            if user.get('mode') == 'normal' and task_id in user['done']:
                return {'ok': True, 'duplicate': True, 'progress': self.progress(aid)}

            assign = self.pending_display.get((aid, task_id))
            if assign is None:
                # Same fallback as resolve_image: pairs.jsonl order.
                # Never invent a new random layout at submit time.
                assign = {
                    'left_candidate': task['left'],
                    'right_candidate': task['right'],
                    'left_path': task['left_path'],
                    'right_path': task['right_path'],
                    'display_swapped': False,
                }
            left = assign['left_candidate']
            right = assign['right_candidate']
            cand_a, cand_b = self._pair_ab(task)
            label = response_to_label(choice, left, right, cand_a, cand_b)

            round_id = int(user['round_counts'].get(task_id, 0)) + 1
            row = {
                'task_id': task_id,
                'pair_id': task.get('pair_id', task_id),
                'annotation_id': f'{aid}:{task_id}:r{round_id}',
                'annotator': aid,
                'annotator_id': aid,
                'round_id': round_id,
                'choice': choice,
                'response': choice,
                'label': label,
                'left_candidate': left,
                'right_candidate': right,
                'display_swapped': bool(assign['display_swapped']),
                'candidate_a': cand_a,
                'candidate_b': cand_b,
                'ts': datetime.now(timezone.utc).isoformat(),
                'scene_id': task['scene_id'],
                'dataset': task['dataset'],
                'split': task['split'],
                'left': left,
                'right': right,
                'is_gold': False,
            }
            path = self.labels_path(aid)
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(row, ensure_ascii=False) + '\n'
            with path.open('a', encoding='utf-8') as f:
                f.write(line)
            ann = {
                'annotation_id': row['annotation_id'],
                'pair_id': row['pair_id'],
                'task_id': task_id,
                'annotator_id': aid,
                'round_id': round_id,
                'left_candidate': left,
                'right_candidate': right,
                'display_swapped': bool(assign['display_swapped']),
                'response': choice,
                'label': label,
                'ts': row['ts'],
                'scene_id': task['scene_id'],
                'dataset': task.get('dataset'),
                'split': task.get('split'),
                'is_gold': False,
            }
            with self.annotations_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(ann, ensure_ascii=False) + '\n')

            self.pending_display.pop((aid, task_id), None)
            user['done'].add(task_id)
            user['round_counts'][task_id] = round_id
            if user.get('mode') == 'normal':
                user['queue'] = [p for p in user['queue'] if p['task_id'] not in user['done']]
            else:
                user['queue'] = [p for p in user['queue'] if p['task_id'] != task_id]
            user['cursor'] = 0
            self.n_new += 1
            return {
                'ok': True,
                'duplicate': False,
                'round_id': round_id,
                'label': label,
                'display_swapped': bool(assign['display_swapped']),
                'progress': self.progress(aid),
            }

    def correct_label(self, annotator_id: str, task_id: str, label: str) -> dict:
        """Append one more vote on this pair. Earlier votes stay in the total."""
        if label not in ('A', 'Tie', 'B'):
            raise ValueError('label must be A, Tie, or B')
        aid = valid_annotator_id(annotator_id)
        with self.lock:
            task = self.by_id.get(task_id)
            if task is None:
                raise KeyError(f'unknown task_id {task_id}')
            pair_id = task.get('pair_id', task_id)
            cand_a, cand_b = self._pair_ab(task)
            max_round = 0
            if self.annotations_path.is_file():
                for row in load_jsonl(self.annotations_path):
                    row_aid = str(row.get('annotator_id') or row.get('annotator') or '')
                    row_pid = str(row.get('pair_id') or row.get('task_id') or '')
                    if row_aid != aid or row_pid != pair_id:
                        continue
                    max_round = max(max_round, int(row.get('round_id') or 0))
            round_id = max_round + 1
            choice = {'A': 'left', 'B': 'right', 'Tie': 'tie'}[label]
            now = datetime.now(timezone.utc).isoformat()
            ann_id = f'{aid}:{task_id}:r{round_id}'
            row = {
                'task_id': task_id,
                'pair_id': pair_id,
                'annotation_id': ann_id,
                'annotator': aid,
                'annotator_id': aid,
                'round_id': round_id,
                'choice': choice,
                'response': choice,
                'label': label,
                'left_candidate': cand_a,
                'right_candidate': cand_b,
                'display_swapped': False,
                'candidate_a': cand_a,
                'candidate_b': cand_b,
                'ts': now,
                'scene_id': task['scene_id'],
                'dataset': task.get('dataset'),
                'split': task.get('split'),
                'left': cand_a,
                'right': cand_b,
                'is_gold': False,
            }
            path = self.labels_path(aid)
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(row, ensure_ascii=False) + '\n'
            with path.open('a', encoding='utf-8') as f:
                f.write(line)
            ann = {
                'annotation_id': ann_id,
                'pair_id': pair_id,
                'task_id': task_id,
                'annotator_id': aid,
                'round_id': round_id,
                'left_candidate': cand_a,
                'right_candidate': cand_b,
                'display_swapped': False,
                'response': choice,
                'label': label,
                'ts': now,
                'scene_id': task['scene_id'],
                'dataset': task.get('dataset'),
                'split': task.get('split'),
                'is_gold': False,
            }
            with self.annotations_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(ann, ensure_ascii=False) + '\n')
            user = self.users.get(aid)
            if user is not None:
                user['done'].add(task_id)
                user['round_counts'][task_id] = round_id
            self.n_new += 1
            self.refresh_supervision()
            note = f'Added another {label} vote by {aid}. Earlier votes stay in the total.'
            return {'ok': True, 'label': label, 'pair_id': pair_id, 'note': note}

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Pref Bench Annotate</title>
<style>
  :root {
    --bg: #0e1116;
    --panel: #171b22;
    --line: #2a3140;
    --text: #e8ecf1;
    --muted: #9aa3b2;
    --accent: #3d8bfd;
    --ok: #3dd68c;
    --warn: #f0b429;
    --hdr: 40px;
    --act: 58px;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; overflow: hidden; }
  body {
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background: var(--bg); color: var(--text);
    display: flex; flex-direction: column;
  }
  header {
    height: var(--hdr); flex: 0 0 var(--hdr);
    display: flex; gap: 10px; align-items: center;
    padding: 0 12px; background: #12161d; border-bottom: 1px solid var(--line);
  }
  header strong { font-size: 16px; white-space: nowrap; }
  .muted { color: var(--muted); font-size: 12px; }
  header .muted { font-size: 15px; }
  .bar {
    flex: 1; min-width: 80px; height: 6px; background: #222833; border-radius: 3px; overflow: hidden;
  }
  .bar > i { display: block; height: 100%; background: var(--accent); width: 0%; }
  #app {
    flex: 1; min-height: 0;
    display: flex; flex-direction: column;
    overflow: auto;
  }
  .prompt {
    flex: 0 0 auto;
    margin: 4px 10px 0; padding: 4px 10px;
    background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
    font-size: 12px; line-height: 1.3;
  }
  .prompt b.A { color: var(--ok); }
  .prompt b.B { color: #7aa2f7; }
  .prompt b.T { color: var(--warn); }
  #view-host { flex: 1; min-height: 0; display: flex; flex-direction: column; }
  #view-host .grid4 { flex: 1; }
  .grid4 {
    flex: 1 1 auto; min-height: 0;
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 6px;
    padding: 6px 10px;
  }
  figure {
    margin: 0; min-height: 0; min-width: 0;
    background: #000; border: 1px solid var(--line); border-radius: 6px;
    overflow: hidden; display: flex; flex-direction: column;
  }
  figure.choice { cursor: pointer; }
  figure.choice:hover { outline: 2px solid var(--accent); outline-offset: 0; }
  #view-host figure { border-width: 4px; }
  #view-host figcaption { font-size: 13px; font-weight: 700; }
  figure.win { border-color: #3dff9a; }
  figure.win figcaption { background: #3dff9a; color: #062016; }
  figure.tie { border-color: var(--warn); }
  figure.tie figcaption { background: var(--warn); color: #111; }
  figure.lose { border-color: var(--line); }
  figure.lose figcaption { color: var(--muted); }
  .viewbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; padding: 8px 10px 0; }
  .viewbar select, .viewbar input, .viewbar button { min-width: 0; padding: 6px 8px; font-size: 13px; }
  .viewnav {
    display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
    padding: 8px 10px 0;
  }
  .viewnav button {
    font-size: 18px; font-weight: 700; padding: 12px 18px; min-width: 108px;
  }
  .viewnav button.jump { min-width: 76px; padding: 12px 14px; }
  .viewnav .muted { font-size: 15px; }
  .vfix {
    margin: 8px 10px 0; padding: 10px 12px;
    background: var(--panel); border: 1px solid var(--accent); border-radius: 8px;
    display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
  }
  .vfix[hidden] { display: none; }
  .vfix p { margin: 0; flex: 1 1 280px; font-size: 15px; line-height: 1.4; }
  .vfix select, .vfix button { font-size: 16px; }
  figcaption {
    flex: 0 0 auto;
    padding: 2px 8px; font-size: 11px; color: var(--muted);
    display: flex; justify-content: space-between; background: var(--panel);
  }
  .imgwrap {
    flex: 1; min-height: 0; min-width: 0;
    position: relative;
    background: #000;
  }
  .imgwrap img {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    object-fit: contain;
    display: block;
  }
  .actions {
    flex: 0 0 var(--act);
    z-index: 30;
    display: flex; flex-wrap: nowrap; gap: 8px; justify-content: center; align-items: center;
    padding: 0 10px; border-top: 1px solid var(--line); background: #12161d;
  }
  button {
    border: 1px solid var(--line); background: var(--panel); color: var(--text);
    padding: 10px 16px; border-radius: 6px; font-size: 14px; cursor: pointer; min-width: 120px;
  }
  button:hover { border-color: var(--accent); }
  button.primary { background: #1a2a44; border-color: var(--accent); }
  button.tie {
    color: #111; background: var(--warn); border-color: #d4a017; font-weight: 700;
    min-width: 168px;
  }
  button.tie:hover { filter: brightness(1.08); }
  button:disabled { opacity: .45; cursor: default; }
  .hint { font-size: 11px; color: var(--muted); margin-left: 8px; white-space: nowrap; }
  .done-box {
    margin: auto; width: min(720px, calc(100% - 32px)); text-align: center; padding: 36px 32px;
    border: 1px solid var(--line); border-radius: 12px; background: var(--panel);
  }
  .done-box h1 { margin: 0 0 14px; font-size: 32px; color: var(--ok); }
  .done-box .muted, .done-box p { font-size: 18px; line-height: 1.5; }
  .done-box button { font-size: 18px; padding: 14px 28px; margin-top: 18px; min-width: 160px; }
  .note { font-size: 12px; color: var(--muted); padding: 6px 12px 0; line-height: 1.4; }
  .gate {
    margin: auto; width: min(720px, calc(100% - 32px));
    padding: 28px 32px; border: 1px solid var(--line); border-radius: 12px; background: var(--panel);
  }
  .gate h1 { margin: 0 0 10px; font-size: 32px; }
  .gate .muted, .gate p { font-size: 18px; line-height: 1.5; }
  .gate label { display: block; margin: 16px 0 6px; font-size: 16px; color: var(--muted); }
  .gate select, .gate input[type=text] {
    width: 100%; padding: 12px 14px; border-radius: 8px;
    border: 1px solid var(--line); background: #0e1116; color: var(--text); font-size: 18px;
  }
  .radios { display: flex; gap: 10px; }
  .radios label {
    flex: 1; margin: 0; padding: 12px 10px; text-align: center; color: var(--text);
    border: 1px solid var(--line); border-radius: 8px; cursor: pointer; font-size: 17px;
  }
  .radios input { margin-right: 6px; }
  .gate .go { width: 100%; margin-top: 14px; font-size: 18px; padding: 14px 16px; }
  #switch { margin-left: auto; min-width: 88px; padding: 6px 14px; font-size: 14px; }
  @media (max-width: 700px) {
    .hint { display: none; }
    button { min-width: 0; flex: 1; padding: 10px 6px; }
  }
</style>
</head>
<body>
<header>
  <strong>IVIF Preference Annotation</strong>
  <span class="muted" id="meta">Select an annotator</span>
  <div class="bar" title="progress"><i id="pbar"></i></div>
  <span class="muted" id="count">—</span>
  <button type="button" id="switch" hidden>Back</button>
</header>
<div id="app"></div>
<script>
const app = document.getElementById('app');
const DATABASE = __DATABASE__;
const STORE = 'prefbench.session';
const VIEW_STORE = 'prefbench.view';
const DEFAULT_ID = __DEFAULT_ANNOTATOR__;
let task = null;
let busy = false;
let session = null;
let roster = [];
let databases = [];
let currentDb = 'main';

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function expLabel(e) { return e === 'non-professional' ? 'Non-professional' : 'Professional'; }

function showBack() {
  const btn = document.getElementById('switch');
  btn.hidden = false;
  btn.textContent = 'Back';
}
function setProgress(p) {
  const pct = p.total ? Math.round(100 * p.done / p.total) : 0;
  document.getElementById('pbar').style.width = pct + '%';
  document.getElementById('count').textContent = `${p.done} / ${p.total}`;
  const modeName = {
    normal: 'New annotation',
    relabel: 'Re-annotation',
    recheck: 'Routine check',
    model: 'Model disagreements',
    reannotate: 'Re-annotation',
  }[p.mode] || '';
  document.getElementById('meta').textContent =
    `${p.database || ''} · ${p.annotator} · ${expLabel(p.expertise)} · ${modeName} · ${p.remaining} left`;
  showBack();
}

function renderDone(p) {
  setProgress(p);
  const extra = p.recommend_note ? `<p class="muted">${esc(p.recommend_note)}</p>` : '';
  const title = p.total ? 'This round is complete' : 'Nothing in this queue';
  app.innerHTML = `<div class="done-box">
    <h1>${title}</h1>
    <p class="muted">${esc(p.annotator)} (${expLabel(p.expertise)}) has finished the current queue.</p>
    <p class="muted">Raw labels are in ${esc(p.labels_path)}. Closing this program recomputes and saves supervision.jsonl.</p>
    ${extra}
    <button type="button" id="back-done">Back</button>
  </div>`;
  document.getElementById('back-done').onclick = goBack;
}

function cell(url, title, sub, choice) {
  const cls = choice ? ' choice' : '';
  const click = choice ? ` onclick="submit('${choice}')"` : '';
  return `<figure class="${cls.trim()}"${click}>
    <div class="imgwrap"><img src="${url}" alt="${esc(title)}"/></div>
    <figcaption><span>${esc(title)}</span><span>${esc(sub)}</span></figcaption>
  </figure>`;
}

function renderTask(t) {
  task = t;
  busy = false;
  setProgress(t.progress);
  const note = t.progress.recommend_note
    ? `<div class="note">${esc(t.progress.recommend_note)}</div>` : '';
  app.innerHTML = `
    ${note}
    <div class="prompt">${esc(t.prompt)}</div>
    <div class="grid4">
      ${cell(t.ir_url, 'Infrared (IR)', t.dataset + ' · ' + t.scene_id)}
      ${cell(t.vi_url, 'Visible (VI)', t.split)}
      ${cell(t.left_url, 'Result A', 'Click or press 1', 'left')}
      ${cell(t.right_url, 'Result B', 'Click or press 3', 'right')}
    </div>
    <div class="actions">
      <button type="button" class="primary" id="btn-left">A is better (1)</button>
      <button type="button" class="tie" id="btn-tie">Tie / about the same (2 / Space)</button>
      <button type="button" class="primary" id="btn-right">B is better (3)</button>
      <button type="button" id="back-task">Back</button>
      <span class="hint">Clicking an image selects that side. For a tie, use the yellow button or press 2 / Space. Escape returns to the menu.</span>
    </div>
  `;
  document.getElementById('btn-left').onclick = () => submit('left');
  document.getElementById('btn-tie').onclick = () => submit('tie');
  document.getElementById('btn-right').onclick = () => submit('right');
  document.getElementById('back-task').onclick = goBack;
}

let draft = null;

function identityFromGate() {
  const annotator_id = document.getElementById('newid').value.trim();
  const picked = document.querySelector('input[name=exp]:checked');
  if (!annotator_id) throw new Error('Enter a username');
  return { annotator_id, expertise: picked.value, database: currentDb };
}

function openPrepare() {
  try { draft = identityFromGate(); }
  catch (ex) { renderGate(ex.message || String(ex)); return; }
  renderPrepare('', false);
}

async function beginSession(onlyIds, note) {
  const r = await fetch('/api/session', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      annotator_id: draft.annotator_id,
      expertise: draft.expertise,
      mode: 'label',
      database: draft.database,
      only_task_ids: onlyIds || null,
      note: note || '',
    }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || 'Could not save the user');
  session = {
    annotator_id: data.annotator_id,
    expertise: data.expertise,
    mode: 'label',
    database: draft.database,
  };
  localStorage.setItem(STORE, JSON.stringify(session));
  await refreshRoster();
  await loadNext();
}

async function renderPrepare(message, readyToAdd) {
  showBack();
  document.getElementById('meta').textContent = 'New annotation';
  document.getElementById('count').textContent = '—';
  document.getElementById('pbar').style.width = '0%';
  app.innerHTML = `<form class="gate" id="prep"><h1>New annotation</h1><p class="muted">Loading this database…</p></form>`;
  let defaults = { datasets: [], has_candidates: false };
  try {
    const res = await fetch('/api/extend-defaults?database=' + encodeURIComponent(draft.database));
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Could not read this database');
    defaults = data;
  } catch (ex) {
    app.innerHTML = `<form class="gate"><h1>New annotation</h1><p class="muted" style="color:#f0b429">${esc(ex.message || ex)}</p><button type="button" id="prep-back">Back</button></form>`;
    document.getElementById('prep-back').onclick = goBack;
    return;
  }
  const options = defaults.datasets.map(d => `<option value="${esc(d.name)}">${esc(d.name)}</option>`).join('');
  app.innerHTML = `
    <form class="gate" id="prep">
      <h1>New annotation</h1>
      <p class="muted">Use the folders below only for new scenes or a new method. Comparisons already labeled are kept. If the images are already prepared, go straight to labeling.</p>
      <button class="primary go" type="button" id="prep-continue">Label prepared pairs</button>
      <h2 style="font-size:22px;margin:22px 0 8px">Add images</h2>
      <label>What are you adding?</label>
      <div class="radios">
        <label><input type="radio" name="addkind" value="scenes" checked/> New scenes</label>
        <label><input type="radio" name="addkind" value="method"/> New method</label>
      </div>
      <div id="add-scenes">
        <label for="add-dataset">Dataset</label>
        <select id="add-dataset">${options}<option value="">Other</option></select>
        <input id="add-dataset-name" type="text" maxlength="64" placeholder="Dataset name, e.g. FMB"/>
        <label for="add-source">Folder with ir/ and vi/</label>
        <input id="add-source" type="text" placeholder="/path/to/dataset"/>
        <label for="add-fusion">Fusion folder, if it is not already set up</label>
        <input id="add-fusion" type="text" placeholder="Optional: one subfolder per method"/>
        <p class="muted">Infrared and visible images are enough. Reference fusions are built from them, so you do not need several finished projects. A fusion folder is optional: one subfolder per method you do have, files named like the infrared image. Methods you do not have are skipped.</p>
      </div>
      <div id="add-method" hidden>
        <label for="add-method-name">Method name</label>
        <input id="add-method-name" type="text" maxlength="64" placeholder="e.g. emma"/>
        <label for="add-results">Results folder</label>
        <input id="add-results" type="text" placeholder="/path/to/method_results"/>
        <p class="muted">One method is enough. Name each file like the infrared image. Where a scene has no other fusion, it is compared with reference fusions built from the infrared and visible images. When one filename belongs to more than one dataset, put the files in a subfolder with that dataset's name.</p>
      </div>
      ${message ? `<p class="muted" style="color:#f0b429">${esc(message)}</p>` : ''}
      <button class="go" type="button" id="prep-add">${readyToAdd ? 'Add these images' : 'Check these folders'}</button>
    </form>`;
  const known = Object.fromEntries((defaults.datasets || []).map(d => [d.name, d.root]));
  const dataset = document.getElementById('add-dataset');
  const datasetName = document.getElementById('add-dataset-name');
  const source = document.getElementById('add-source');
  function syncDataset() {
    const selected = dataset.value;
    datasetName.hidden = selected !== '';
    if (selected && known[selected]) {
      datasetName.value = selected;
      source.value = known[selected];
    }
  }
  function syncKind() {
    const kind = document.querySelector('input[name=addkind]:checked').value;
    document.getElementById('add-scenes').hidden = kind !== 'scenes';
    document.getElementById('add-method').hidden = kind !== 'method';
  }
  dataset.onchange = syncDataset;
  document.querySelectorAll('input[name=addkind]').forEach(el => { el.onchange = syncKind; });
  syncDataset();
  syncKind();
  document.getElementById('prep-continue').onclick = () => {
    beginSession(null, '').catch(ex => renderPrepare(ex.message || String(ex), false));
  };
  document.getElementById('prep-add').onclick = async () => {
    const button = document.getElementById('prep-add');
    const kind = document.querySelector('input[name=addkind]:checked').value;
    const body = { database: draft.database, kind, confirm: !!readyToAdd };
    if (kind === 'scenes') {
      body.dataset = dataset.value || datasetName.value.trim();
      body.source = source.value.trim();
      body.fusion = document.getElementById('add-fusion').value.trim();
    } else {
      body.method = document.getElementById('add-method-name').value.trim();
      body.results = document.getElementById('add-results').value.trim();
    }
    button.disabled = true;
    button.textContent = 'Preparing images…';
    try {
      const r = await fetch('/api/extend', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Could not prepare the images');
      if (data.needs_confirm) {
        renderPrepare(data.note || 'Click Add these images to continue.', true);
        return;
      }
      if (!data.added_pairs) {
        renderPrepare(data.note || 'No new comparisons were added.', false);
        return;
      }
      await beginSession(data.task_ids, data.note || '');
    } catch (ex) {
      renderPrepare(ex.message || String(ex), false);
    }
  };
}

function renderGate(err) {
  task = null;
  viewing = false;
  document.getElementById('switch').hidden = true;
  document.getElementById('meta').textContent = 'Choose a database and a task';
  document.getElementById('count').textContent = '—';
  document.getElementById('pbar').style.width = '0%';
  const opts = roster.map(a => {
    const sel = session && session.annotator_id === a.annotator_id ? ' selected' : '';
    return `<option value="${esc(a.annotator_id)}"${sel}>${esc(a.annotator_id)} · ${expLabel(a.expertise)} · ${a.n_labels} labels</option>`;
  }).join('');
  const dbOpts = databases.map(d => {
    const sel = d.id === currentDb ? ' selected' : '';
    return `<option value="${esc(d.id)}"${sel}>${esc(d.label)} · ${d.n_pairs} pairs</option>`;
  }).join('');
  const cur = (session && session.annotator_id) || DEFAULT_ID || '';
  const exp = (session && session.expertise) || 'professional';
  app.innerHTML = `
    <form class="gate" id="gate">
      <h1>Choose a task</h1>
      <p class="muted">Pick the database first. New annotation can keep labeling prepared pairs, or add scenes and methods from folders. View labeled results does not write. Back or Escape returns here.</p>
      <label for="database">Database</label>
      <select id="database">${dbOpts}</select>
      <label for="who">Existing users</label>
      <select id="who">
        <option value="">— New user —</option>
        ${opts}
      </select>
      <label for="newid">Username (new, or confirm after selecting)</label>
      <input id="newid" type="text" maxlength="32" placeholder="e.g. ann0" value="${esc(cur)}" autocomplete="off"/>
      <label>Expertise</label>
      <div class="radios">
        <label><input type="radio" name="exp" value="professional" ${exp !== 'non-professional' ? 'checked' : ''}/> Professional</label>
        <label><input type="radio" name="exp" value="non-professional" ${exp === 'non-professional' ? 'checked' : ''}/> Non-professional</label>
      </div>
      <p class="muted">Routine check reviews preference-cycle edges first. If there is no cycle, it takes a small set of the least reliable pairs, or a random sample when reliability does not vary. Model disagreements are decisive labels where the trained scorer gives the other candidate probability above 0.6. Earlier labels are kept.</p>
      ${err ? `<p class="muted" style="color:#f0b429">${esc(err)}</p>` : ''}
      <button class="primary go" type="button" id="new-btn">New annotation</button>
      <button class="go" type="submit" data-mode="recheck">Re-annotate: routine check</button>
      <button class="go" type="submit" data-mode="model">Re-annotate: model disagreements</button>
      <button class="go" type="button" id="view-btn">View labeled results</button>
    </form>`;
  const who = document.getElementById('who');
  const box = document.getElementById('newid');
  const database = document.getElementById('database');
  if (cur) {
    const hit = roster.find(a => a.annotator_id === cur);
    if (hit) who.value = cur;
  }
  who.onchange = () => {
    const row = roster.find(a => a.annotator_id === who.value);
    if (!row) { box.value = ''; return; }
    box.value = row.annotator_id;
    const radio = document.querySelector(`input[name=exp][value="${row.expertise}"]`);
    if (radio) radio.checked = true;
  };
  database.onchange = async () => {
    currentDb = database.value;
    await refreshRoster();
    renderGate();
  };
  document.getElementById('view-btn').onclick = () => startView();
  document.getElementById('new-btn').onclick = () => openPrepare();
  document.getElementById('gate').onsubmit = async (e) => {
    e.preventDefault();
    const annotator_id = box.value.trim();
    const expertise = document.querySelector('input[name=exp]:checked').value;
    const mode = (e.submitter && e.submitter.dataset.mode) || 'label';
    const submitter = e.submitter;
    if (submitter) submitter.disabled = true;
    if (mode === 'model' && submitter) submitter.textContent = 'Scoring with the trained model…';
    try {
      const r = await fetch('/api/session', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ annotator_id, expertise, mode, database: currentDb }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Could not save the user');
      session = { annotator_id: data.annotator_id, expertise: data.expertise, mode, database: currentDb };
      localStorage.setItem(STORE, JSON.stringify(session));
      if (data.recommendation && data.recommendation.note) {
        document.getElementById('meta').textContent =
          `${data.recommendation.note} · ${data.recommendation.n} items`;
      }
      await refreshRoster();
      await loadNext();
    } catch (ex) {
      renderGate(ex.message || String(ex));
    }
  };
}

let viewing = false;
let viewItems = [];
let viewList = [];
let viewPos = 0;
let viewUsesProtocol = false;
let methodLabels = {};
let correctOpen = false;
let correctRoster = [];
let correctNote = '';

function verdictOf(it) {
  if (it.q >= 0.6) return 'A';
  if (it.q <= 0.4) return 'B';
  return 'T';
}
function readViewPlace() {
  try { return JSON.parse(localStorage.getItem(VIEW_STORE) || 'null'); } catch (e) { return null; }
}
function writeViewPlace() {
  if (!viewing) return;
  const pack = document.getElementById('vpack');
  if (!pack) return;
  const it = viewList[viewPos];
  localStorage.setItem(VIEW_STORE, JSON.stringify({
    pack: pack.value,
    split: document.getElementById('vsplit').value,
    dataset: document.getElementById('vdataset').value,
    method: document.getElementById('vmethod').value,
    verdict: document.getElementById('vverdict').value,
    query: document.getElementById('vq').value,
    pair_id: it ? it.pair_id : '',
  }));
}
function methodName(name) {
  const real = methodLabels[name];
  if (!real || real === name) return name;
  return `${real} (${name})`;
}
function setChoice(select, entries, current, allLabel) {
  const wanted = current || 'all';
  select.innerHTML = [`<option value="all">${allLabel}</option>`]
    .concat(entries.map(([value, label]) => `<option value="${esc(value)}">${esc(label)}</option>`))
    .join('');
  const values = [...select.options].map(o => o.value);
  select.value = values.includes(wanted) ? wanted : 'all';
}
function refreshViewChoices(place) {
  const dataset = document.getElementById('vdataset');
  const method = document.getElementById('vmethod');
  const datasets = [...new Set(viewItems.map(it => it.dataset).filter(Boolean))].sort();
  setChoice(
    dataset,
    datasets.map(name => [name, name]),
    (place && place.dataset) || dataset.value,
    'All datasets',
  );
  const chosen = dataset.value;
  const pool = viewItems.filter(it => chosen === 'all' || it.dataset === chosen);
  const names = [...new Set(pool.flatMap(it => [it.candidate_a, it.candidate_b]).filter(Boolean))];
  const rank = name => (name.startsWith('prior_') || name.startsWith('deg_')) ? 1 : 0;
  names.sort((a, b) => rank(a) - rank(b) || methodName(a).localeCompare(methodName(b)));
  setChoice(
    method,
    names.map(name => [name, methodName(name)]),
    (place && place.method) || method.value,
    'All methods',
  );
}
function applyView(keepPairId) {
  const s = document.getElementById('vsplit').value;
  const dataset = document.getElementById('vdataset').value;
  const method = document.getElementById('vmethod').value;
  const v = document.getElementById('vverdict').value;
  const text = document.getElementById('vq').value.trim().toLowerCase();
  viewList = viewItems.filter(it => {
    if (viewUsesProtocol) {
      if (s !== 'all' && it.protocol !== s) return false;
    } else if (s !== 'all' && it.split !== s) return false;
    if (dataset !== 'all' && it.dataset !== dataset) return false;
    if (method !== 'all' && it.candidate_a !== method && it.candidate_b !== method) return false;
    const got = verdictOf(it);
    if (v === 'decisive' && got === 'T') return false;
    if ((v === 'A' || v === 'B' || v === 'T') && got !== v) return false;
    if (text && !`${it.scene_id} ${it.candidate_a} ${it.candidate_b} ${it.pair_id}`.toLowerCase().includes(text)) return false;
    return true;
  });
  viewPos = 0;
  if (keepPairId) {
    const idx = viewList.findIndex(it => it.pair_id === keepPairId);
    if (idx >= 0) viewPos = idx;
  }
  paintView();
}
function stepView(delta) {
  const n = viewList.length;
  if (!n) return;
  const next = Math.max(0, Math.min(n - 1, viewPos + delta));
  if (next === viewPos) return;
  viewPos = next;
  paintView();
}
function paintView() {
  const n = viewList.length;
  document.getElementById('count').textContent = n ? `${viewPos + 1} / ${n}` : '0';
  for (const [id, delta] of [['vback10', -10], ['vback5', -5], ['vprev', -1], ['vnext', 1], ['vfwd5', 5], ['vfwd10', 10]]) {
    const el = document.getElementById(id);
    if (el) el.disabled = !n || viewPos + delta < 0 || viewPos + delta > n - 1;
  }
  const host = document.getElementById('view-host');
  if (!host) return;
  if (!n) { host.innerHTML = '<div class="prompt">No labeled pairs match these filters.</div>'; return; }
  const it = viewList[viewPos];
  const got = verdictOf(it);
  const word = got === 'A' ? 'A is better' : got === 'B' ? 'B is better' : 'Tie or close';
  const subset = it.protocol === 'e1' ? 'Seen methods (E1)' : it.protocol === 'e2' ? 'Unseen methods (E2)' : it.split;
  const pack = document.getElementById('vpack').value;
  const cell = (side, title, sub, cls) => `<figure class="${cls}"><div class="imgwrap"><img src="/labeled-img/${it.i}/${side}?pack=${encodeURIComponent(pack)}" alt="${esc(title)}"/></div><figcaption><span>${esc(title)}</span><span>${esc(sub)}</span></figcaption></figure>`;
  const aCls = got === 'A' ? 'win' : got === 'B' ? 'lose' : 'tie';
  const bCls = got === 'B' ? 'win' : got === 'A' ? 'lose' : 'tie';
  const aMark = got === 'A' ? 'Selected' : got === 'T' ? 'Tie' : '';
  const bMark = got === 'B' ? 'Selected' : got === 'T' ? 'Tie' : '';
  host.innerHTML = `
    <div class="prompt"><b class="${got}">${word}</b>
      · q=${it.q.toFixed(2)} · y=(${it.y_A}, ${it.y_T}, ${it.y_B})
      · ${esc(it.dataset)} / ${esc(subset)} · ${esc(it.scene_id)}
      · ${esc(it.pair_id)}
      <span class="muted"> · A and B are the stored candidates, not the left/right order used while annotating. Re-annotate this pair adds one vote. Earlier votes stay and are counted together.</span></div>
    <div class="grid4">
      ${cell('ir', 'Infrared (IR)', it.scene_id, '')}
      ${cell('vi', 'Visible (VI)', subset, '')}
      ${cell('a', 'Candidate A · ' + methodName(it.candidate_a), aMark, aCls)}
      ${cell('b', 'Candidate B · ' + methodName(it.candidate_b), bMark, bCls)}
    </div>`;
  writeViewPlace();
  renderCorrect();
}
function voteSummary(it) {
  const counts = it.counts || {};
  const parts = Object.entries(counts).map(([aid, c]) => {
    const bits = [];
    if (c.A) bits.push(`A × ${c.A}`);
    if (c.Tie) bits.push(`Tie × ${c.Tie}`);
    if (c.B) bits.push(`B × ${c.B}`);
    return `${aid}: ${bits.join(', ') || 'none'}`;
  });
  return parts.join('; ') || 'No stored votes';
}
function renderCorrect() {
  const box = document.getElementById('vfix');
  if (!box) return;
  box.hidden = !correctOpen;
  if (!correctOpen) { box.innerHTML = ''; return; }
  const it = viewList[viewPos];
  if (!it) { box.innerHTML = '<p>No pair is open.</p>'; return; }
  const ids = new Set([
    ...correctRoster.map(r => r.annotator_id),
    ...Object.keys(it.counts || {}),
  ]);
  const previous = document.getElementById('vfix-user');
  const wanted = previous ? previous.value : '';
  const only = Object.keys(it.counts || {});
  const chosen = ids.has(wanted) ? wanted : (only.length === 1 ? only[0] : (session && ids.has(session.annotator_id) ? session.annotator_id : [...ids][0] || ''));
  const options = [...ids].sort().map(id => `<option value="${esc(id)}"${id === chosen ? ' selected' : ''}>${esc(id)}</option>`).join('');
  const effect = 'Saving adds one vote on this pair. Earlier votes stay in the total.';
  box.innerHTML = `
    <p>${esc(voteSummary(it))}. ${effect}${correctNote ? ' ' + esc(correctNote) : ''}</p>
    <select id="vfix-user">${options}</select>
    <button type="button" id="vfix-a">A is better</button>
    <button type="button" id="vfix-t">Tie</button>
    <button type="button" id="vfix-b">B is better</button>
    <button type="button" id="vfix-cancel">Cancel</button>`;
  document.getElementById('vfix-user').onchange = () => { correctNote = ''; renderCorrect(); };
  document.getElementById('vfix-a').onclick = () => saveCorrect('A');
  document.getElementById('vfix-t').onclick = () => saveCorrect('Tie');
  document.getElementById('vfix-b').onclick = () => saveCorrect('B');
  document.getElementById('vfix-cancel').onclick = () => { correctOpen = false; correctNote = ''; renderCorrect(); };
}
async function saveCorrect(label) {
  const it = viewList[viewPos];
  const who = document.getElementById('vfix-user');
  if (!it || !who || busy) return;
  busy = true;
  ['vfix-a', 'vfix-t', 'vfix-b'].forEach(id => { const el = document.getElementById(id); if (el) el.disabled = true; });
  try {
    const pack = document.getElementById('vpack').value;
    const res = await fetch('/api/correct', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        database: pack,
        annotator_id: who.value,
        task_id: it.pair_id,
        label,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Could not save the correction');
    correctNote = data.note || 'Saved.';
    const place = readViewPlace() || {};
    place.pack = pack;
    place.pair_id = it.pair_id;
    place.split = document.getElementById('vsplit').value;
    place.dataset = document.getElementById('vdataset').value;
    place.method = document.getElementById('vmethod').value;
    place.verdict = document.getElementById('vverdict').value;
    place.query = document.getElementById('vq').value;
    await loadViewPack(place);
    if (!viewList.some(row => row.pair_id === it.pair_id)) {
      correctNote = `${data.note || 'Saved.'} The pair no longer matches the current filters.`;
      renderCorrect();
    }
    busy = false;
  } catch (ex) {
    alert(ex.message || ex);
    busy = false;
    renderCorrect();
  }
}
async function loadViewPack(place) {
  const pack = document.getElementById('vpack').value;
  const res = await fetch('/api/labeled?pack=' + encodeURIComponent(pack));
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Could not load labels');
  viewItems = data.items || [];
  methodLabels = data.method_labels || {};
  viewUsesProtocol = viewItems.some(it => it.protocol === 'e1' || it.protocol === 'e2');
  const split = document.getElementById('vsplit');
  const cur = (place && place.split) || split.value || 'all';
  if (viewUsesProtocol) {
    const choices = [['all', 'All comparisons'], ['e1', 'Seen methods (E1)'], ['e2', 'Unseen methods (E2)']];
    split.innerHTML = choices.map(([value, label]) => `<option value="${value}">${label}</option>`).join('');
  } else {
    const splits = ['all', ...new Set(viewItems.map(it => it.split).filter(Boolean))];
    split.innerHTML = splits.map(s => `<option value="${esc(s)}">${s === 'all' ? 'All splits' : esc(s)}</option>`).join('');
  }
  const values = [...split.options].map(o => o.value);
  split.value = values.includes(cur) ? cur : 'all';
  refreshViewChoices(place);
  applyView(place && place.pair_id);
}
async function startView() {
  const place = readViewPlace();
  viewing = true;
  task = null;
  showBack();
  document.getElementById('meta').textContent = 'Viewing labeled results';
  document.getElementById('pbar').style.width = '0%';
  const res = await fetch('/api/labeled-packs');
  const packs = await res.json();
  if (!res.ok) { renderGate(packs.error || 'Could not load labels'); return; }
  app.innerHTML = `
    <div class="viewbar">
      <select id="vpack"></select>
      <select id="vdataset"></select>
      <select id="vsplit"></select>
      <select id="vmethod"></select>
      <select id="vverdict">
        <option value="all">All labels</option>
        <option value="decisive" selected>Decisive only</option>
        <option value="A">A is better</option>
        <option value="B">B is better</option>
        <option value="T">Tie or close</option>
      </select>
      <input id="vq" placeholder="Scene or candidate"/>
      <button type="button" id="vback">Back</button>
    </div>
    <div class="viewnav">
      <button type="button" class="jump" id="vback10">−10</button>
      <button type="button" class="jump" id="vback5">−5</button>
      <button type="button" id="vprev">Previous</button>
      <button type="button" id="vnext">Next</button>
      <button type="button" class="jump" id="vfwd5">+5</button>
      <button type="button" class="jump" id="vfwd10">+10</button>
      <button type="button" id="vrand">Random</button>
      <button type="button" id="vcorrect">Re-annotate this pair</button>
      <span class="muted">Arrows ±1 · Shift+arrows ±5 · Page Up/Down ±10</span>
    </div>
    <div id="vfix" class="vfix" hidden></div>
    <div id="view-host"></div>`;
  const pack = document.getElementById('vpack');
  pack.innerHTML = (packs.packs || []).map(p => `<option value="${esc(p.id)}">${esc(p.label)} · ${p.n}</option>`).join('');
  if (place && [...pack.options].some(o => o.value === place.pack)) pack.value = place.pack;
  const verdict = document.getElementById('vverdict');
  if (place && [...verdict.options].some(o => o.value === place.verdict)) verdict.value = place.verdict;
  if (place && place.query) document.getElementById('vq').value = place.query;
  pack.onchange = () => loadViewPack().catch(ex => alert(ex.message || ex));
  document.getElementById('vdataset').onchange = () => { refreshViewChoices(); applyView(); };
  document.getElementById('vsplit').onchange = () => applyView();
  document.getElementById('vmethod').onchange = () => applyView();
  document.getElementById('vverdict').onchange = () => applyView();
  document.getElementById('vq').oninput = () => applyView();
  document.getElementById('vback').onclick = goBack;
  document.getElementById('vback10').onclick = () => stepView(-10);
  document.getElementById('vback5').onclick = () => stepView(-5);
  document.getElementById('vprev').onclick = () => stepView(-1);
  document.getElementById('vnext').onclick = () => stepView(1);
  document.getElementById('vfwd5').onclick = () => stepView(5);
  document.getElementById('vfwd10').onclick = () => stepView(10);
  document.getElementById('vrand').onclick = () => { if (viewList.length) { viewPos = Math.floor(Math.random() * viewList.length); paintView(); } };
  document.getElementById('vcorrect').onclick = async () => {
    correctOpen = !correctOpen;
    correctNote = '';
    if (correctOpen) {
      const pack = document.getElementById('vpack').value;
      const res = await fetch('/api/annotators?database=' + encodeURIComponent(pack));
      const data = await res.json();
      correctRoster = data.annotators || [];
    }
    renderCorrect();
  };
  await loadViewPack(place);
}

async function refreshRoster() {
  const r = await fetch('/api/annotators?database=' + encodeURIComponent(currentDb));
  const data = await r.json();
  roster = data.annotators || [];
}

async function loadNext() {
  if (!session) { renderGate(); return; }
  const q = encodeURIComponent(session.annotator_id);
  const db = encodeURIComponent(session.database || currentDb);
  const r = await fetch('/api/next?annotator_id=' + q + '&database=' + db);
  const data = await r.json();
  if (!r.ok) { renderGate(data.error || 'Select the user again'); return; }
  if (!data.task) renderDone(data.progress);
  else renderTask(data.task);
}

async function submit(choice) {
  if (!task || busy || !session) return;
  busy = true;
  ['btn-left','btn-tie','btn-right'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = true;
  });
  try {
    const r = await fetch('/api/label', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        annotator_id: session.annotator_id,
        task_id: task.task_id,
        choice,
        database: session.database || currentDb,
      }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || 'submit failed');
    await loadNext();
  } catch (e) {
    alert(e.message || e);
    busy = false;
    ['btn-left','btn-tie','btn-right'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.disabled = false;
    });
  }
}

function goBack() {
  task = null;
  viewing = false;
  busy = false;
  correctOpen = false;
  correctNote = '';
  localStorage.removeItem(VIEW_STORE);
  document.getElementById('switch').hidden = true;
  renderGate();
}
document.getElementById('switch').onclick = goBack;
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && (task || viewing)) {
    e.preventDefault();
    goBack();
    return;
  }
  if (viewing) {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;
    if (e.key === 'ArrowLeft') { e.preventDefault(); stepView(e.shiftKey ? -5 : -1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); stepView(e.shiftKey ? 5 : 1); }
    else if (e.key === 'PageUp') { e.preventDefault(); stepView(-10); }
    else if (e.key === 'PageDown') { e.preventDefault(); stepView(10); }
    else if (e.key === 'r' || e.key === 'R') document.getElementById('vrand')?.click();
    return;
  }
  if (!task || busy) return;
  if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;
  if (e.repeat) return;
  const k = e.key;
  if (k === '1' || k === 'ArrowLeft') { e.preventDefault(); submit('left'); }
  else if (k === '2' || k === ' ' || k === 'Spacebar' || k === 's' || k === 'S' || k === 't' || k === 'T') {
    e.preventDefault(); submit('tie');
  }
  else if (k === '3' || k === 'ArrowRight') { e.preventDefault(); submit('right'); }
});

const clientId = sessionStorage.getItem('prefbench.client') ||
  (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2));
sessionStorage.setItem('prefbench.client', clientId);
function ping() {
  fetch('/api/heartbeat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ client_id: clientId }),
    keepalive: true,
  }).catch(() => {});
}
ping();
setInterval(ping, 3000);
window.addEventListener('pagehide', () => {
  const body = new Blob([JSON.stringify({ client_id: clientId })], {type: 'application/json'});
  navigator.sendBeacon('/api/bye', body);
});

(async function boot() {
  try { session = JSON.parse(localStorage.getItem(STORE) || 'null'); } catch (e) { session = null; }
  try {
    const res = await fetch('/api/databases');
    const data = await res.json();
    databases = data.databases || [];
  } catch (e) {
    databases = [];
  }
  if (session && session.database && databases.some(d => d.id === session.database)) {
    currentDb = session.database;
  } else if (databases[0]) {
    currentDb = databases[0].id;
  }
  await refreshRoster();
  if (readViewPlace()) await startView();
  else renderGate();
})();
</script>
</body>
</html>
"""


def _html(default_annotator: str | None, database: str) -> str:
    return (
        HTML.replace('__DEFAULT_ANNOTATOR__', json.dumps(default_annotator or ''))
        .replace('__DATABASE__', json.dumps(database))
    )


class Hub:
    """One annotator process, several databases. The primary owns the HTTP server."""

    def __init__(self, states: dict[str, AnnotateState], primary_id: str):
        self.states = states
        self.primary_id = primary_id

    @property
    def primary(self) -> AnnotateState:
        return self.states[self.primary_id]

    def get(self, db_id: str | None) -> AnnotateState:
        key = (db_id or '').strip() or self.primary_id
        state = self.states.get(key)
        if state is None:
            raise ValueError(f'Unknown database: {key}')
        return state

    def catalog(self) -> list[dict]:
        return [
            {'id': key, 'label': state.db_label, 'n_pairs': len(state.pairs)}
            for key, state in self.states.items()
        ]


def make_handler(hub: Hub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if self.path.startswith('/img'):
                return
            super().log_message(fmt, *args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode('utf-8'),
                       'application/json; charset=utf-8')

        def _read_json(self) -> dict:
            n = int(self.headers.get('Content-Length', '0'))
            raw = self.rfile.read(n) if n else b'{}'
            return json.loads(raw.decode('utf-8') or '{}')

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ('/', '/index.html'):
                primary = hub.primary
                self._send(200, _html(primary.default_annotator, primary.db_label).encode('utf-8'),
                           'text/html; charset=utf-8')
                return
            if parsed.path == '/api/databases':
                self._json(200, {'databases': hub.catalog()})
                return
            if parsed.path == '/api/extend-defaults':
                qs = parse_qs(parsed.query)
                try:
                    picked = hub.get((qs.get('database') or [None])[0])
                except ValueError as e:
                    self._json(400, {'error': str(e)})
                    return
                self._json(200, picked.extend_defaults())
                return
            if parsed.path == '/api/labeled-packs':
                try:
                    for st in hub.states.values():
                        st.refresh_supervision()
                    self._json(200, {'packs': hub.primary.labeled_catalog()})
                except Exception as e:
                    self._json(400, {'error': str(e)})
                return
            if parsed.path == '/api/labeled':
                qs = parse_qs(parsed.query)
                pack_id = (qs.get('pack') or ['main'])[0]
                self._json(200, {
                    'items': hub.primary.labeled_items(pack_id),
                    'method_labels': hub.primary.method_labels(),
                })
                return
            parts = parsed.path.strip('/').split('/')
            if len(parts) == 3 and parts[0] == 'labeled-img':
                qs = parse_qs(parsed.query)
                pack_id = (qs.get('pack') or ['main'])[0]
                try:
                    index = int(parts[1])
                except ValueError:
                    self._json(404, {'error': 'image not found'})
                    return
                path = hub.primary.labeled_image(pack_id, index, parts[2])
                if path is None:
                    self._json(404, {'error': 'image not found'})
                    return
                data = path.read_bytes()
                ctype = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
                self._send(200, data, ctype)
                return
            if parsed.path == '/api/annotators':
                qs = parse_qs(parsed.query)
                try:
                    picked = hub.get((qs.get('database') or [None])[0])
                except ValueError as e:
                    self._json(400, {'error': str(e)})
                    return
                self._json(200, {'annotators': picked.list_annotators()})
                return
            if parsed.path == '/api/next':
                qs = parse_qs(parsed.query)
                aid = (qs.get('annotator_id') or [None])[0]
                if not aid:
                    self._json(400, {'error': 'Select an annotator'})
                    return
                try:
                    picked = hub.get((qs.get('database') or [None])[0])
                    with picked.lock:
                        cur = picked.current(aid)
                        if cur is None:
                            self._json(200, {'task': None, 'progress': picked.progress(aid)})
                        else:
                            self._json(200, {'task': picked.public_task(cur, aid)})
                except ValueError as e:
                    self._json(400, {'error': str(e)})
                return
            if parsed.path == '/api/status':
                qs = parse_qs(parsed.query)
                aid = (qs.get('annotator_id') or [None])[0]
                try:
                    picked = hub.get((qs.get('database') or [None])[0])
                except ValueError as e:
                    self._json(400, {'error': str(e)})
                    return
                if not aid:
                    self._json(200, {'annotators': picked.list_annotators()})
                    return
                try:
                    self._json(200, picked.progress(aid))
                except ValueError as e:
                    self._json(400, {'error': str(e)})
                return
            if parsed.path == '/img':
                qs = parse_qs(parsed.query)
                task_id = (qs.get('task_id') or [None])[0]
                side = (qs.get('side') or [None])[0]
                aid = (qs.get('annotator_id') or [None])[0]
                if not task_id or not side:
                    self._json(400, {'error': 'task_id and side required'})
                    return
                try:
                    picked = hub.get((qs.get('database') or [None])[0])
                except ValueError as e:
                    self._json(400, {'error': str(e)})
                    return
                path = picked.resolve_image(task_id, side, annotator_id=aid)
                if path is None:
                    self._json(404, {'error': 'image not found'})
                    return
                data = path.read_bytes()
                ctype = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
                self._send(200, data, ctype)
                return
            self._json(404, {'error': 'not found'})

        def do_POST(self):
            parsed = urlparse(self.path)
            try:
                payload = self._read_json()
                if parsed.path == '/api/session':
                    picked = hub.get(payload.get('database'))
                    info = picked.upsert_annotator(
                        payload.get('annotator_id', ''),
                        payload.get('expertise', ''),
                        payload.get('mode', 'label'),
                        only_task_ids=payload.get('only_task_ids') or None,
                        note=str(payload.get('note') or ''),
                    )
                    self._json(200, info)
                    return
                if parsed.path == '/api/extend':
                    picked = hub.get(payload.get('database'))
                    info = picked.extend_data(str(payload.get('kind') or ''), payload)
                    self._json(200, info)
                    return
                if parsed.path == '/api/label':
                    picked = hub.get(payload.get('database'))
                    result = picked.submit(
                        payload.get('annotator_id', ''),
                        payload['task_id'],
                        payload['choice'],
                    )
                    self._json(200, result)
                    return
                if parsed.path == '/api/correct':
                    picked = hub.get(payload.get('database'))
                    result = picked.correct_label(
                        payload.get('annotator_id', ''),
                        payload['task_id'],
                        payload['label'],
                    )
                    self._json(200, result)
                    return
                if parsed.path == '/api/heartbeat':
                    hub.primary.heartbeat(payload.get('client_id', ''))
                    self._json(200, {'ok': True})
                    return
                if parsed.path == '/api/bye':
                    hub.primary.browser_left(payload.get('client_id', ''))
                    self._json(200, {'ok': True})
                    return
                self._json(404, {'error': 'not found'})
            except Exception as e:
                self._json(400, {'error': str(e)})

    return Handler


def main() -> None:
    p = ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--annotator', type=str, default=None,
                   help='optional default id prefilled on the login screen')
    p.add_argument('--port', type=int, default=8766)
    p.add_argument('--split', type=str, default=None,
                   help='optional filter: train|val|test')
    p.add_argument('--allow-reannotate', action='store_true',
                   help='allow multi-round labels on the same task_id (append-only)')
    p.add_argument('--queue', type=Path, default=None,
                   help='optional JSON with random_reann / reannotation_pair_ids to restrict tasks')
    p.add_argument(
        '--extra', action='append', default=[],
        help='extra labeled pack for the view mode, as label=/path/to/supervision.jsonl',
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    queue_ids = None
    if args.queue is not None:
        q = json.loads(Path(args.queue).read_text(encoding='utf-8'))
        ids = set(q.get('random_reann') or []) | set(q.get('reannotation_pair_ids') or [])
        queue_ids = ids or None
    primary = AnnotateState(
        cfg,
        split=args.split,
        allow_reannotate=args.allow_reannotate,
        queue_task_ids=queue_ids,
        default_annotator=args.annotator,
        extras=args.extra,
        db_id='main',
        db_label='In-domain',
    )
    if not primary.pairs_path.is_file():
        raise SystemExit(f'missing {primary.pairs_path}; run sample_pairs first')
    states: dict[str, AnnotateState] = {'main': primary}
    used = {'main'}
    declared: list[tuple[str, str, str]] = []
    for item in cfg.get('browse_packs') or []:
        if isinstance(item, dict) and item.get('path'):
            key = str(item.get('id') or item.get('label') or 'extra')
            display = str(item.get('label') or item.get('id') or 'extra')
            declared.append((key, str(item['path']), display))
    for spec in args.extra:
        if '=' not in spec:
            raise SystemExit(f'--extra must be label=path, got {spec!r}')
        label, raw = spec.split('=', 1)
        declared.append((label.strip() or 'extra', raw.strip(), label.strip() or 'extra'))
    for label, raw, display in declared:
        root = Path(raw).resolve()
        if root.is_file():
            root = root.parent
        if not (root / 'pairs.jsonl').is_file() or root == primary.out:
            continue
        slug = re.sub(r'[^\w\-]+', '-', label.strip()).strip('-').lower() or 'pack'
        base = slug
        n = 2
        while slug in used:
            slug = f'{base}-{n}'
            n += 1
        used.add(slug)
        extra_cfg = dict(cfg)
        extra_cfg['out_root'] = str(root)
        extra_cfg['browse_packs'] = []
        states[slug] = AnnotateState(
            extra_cfg,
            split=None,
            allow_reannotate=args.allow_reannotate,
            default_annotator=args.annotator,
            db_id=slug,
            db_label=display,
        )
    hub = Hub(states, 'main')

    for key, st in states.items():
        print(f'[annotate] database {key} ({st.db_label}) pairs={len(st.pairs)} annotators={len(st.list_annotators())}')
    print(f'[annotate] reannotate={args.allow_reannotate}')
    print(f'[annotate] http://127.0.0.1:{args.port}/')
    print(f'[annotate] Cursor Ports → Forward {args.port}')
    print('[annotate] choose a database, then new annotation, a recommended review, or view')
    print('[annotate] labeling keys: 1=A, 2/Space=tie, 3=B; Escape=back; view keys: Left/Right ±1, Shift+arrows ±5, Page Up/Down ±10, R=random')
    print('[annotate] Closing the browser tab does not stop the server. Aggregation runs once on exit.')

    httpd = ThreadingHTTPServer(('0.0.0.0', args.port), make_handler(hub))
    primary.httpd = httpd

    def _request_stop(signum, _frame):
        raise KeyboardInterrupt

    def _sweep() -> None:
        while not primary._stopping:
            time.sleep(2.0)
            primary.sweep_clients()

    signal.signal(signal.SIGTERM, _request_stop)
    threading.Thread(target=_sweep, daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n[annotate] stopped')
    finally:
        httpd.server_close()
        wrote = False
        for st in states.values():
            if st.n_new <= 0:
                continue
            wrote = True
            try:
                info = st.refresh_supervision()
                print(f'[annotate] aggregated {st.n_new} new labels → {info["path"]}')
            except Exception as e:
                print(f'[annotate] aggregation failed for {st.db_label}: {e}')
        if not wrote:
            print('[annotate] no new labels, skip aggregation')
        time.sleep(0.05)


if __name__ == '__main__':
    main()
