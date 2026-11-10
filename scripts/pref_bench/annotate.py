#!/usr/bin/env python3
"""Preference pairwise annotation UI.

  python scripts/pref_bench/annotate.py --config /path/to/pref_bench/config.yaml --port 8766

Open the page, pick (or create) an annotator and expertise, then label.
Each browser stores its own identity, so several people can share one server.

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
    ):
        self.cfg = cfg
        self.out = Path(cfg['out_root']).resolve()
        self.split = split
        self.allow_reannotate = allow_reannotate
        self.queue_task_ids = queue_task_ids
        self.default_annotator = default_annotator
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

    def upsert_annotator(self, annotator_id: str, expertise: str, mode: str = 'label') -> dict:
        aid = valid_annotator_id(annotator_id)
        exp = valid_expertise(expertise)
        if mode not in ('label', 'recheck'):
            raise ValueError('mode must be label or recheck')
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
            else:
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

        # Adopt an existing queue written before this process (e.g. mid-pass refresh
        # before pass files existed): treat labels newer than the queue file as
        # already done for this pass.
        adopted = self._try_adopt_queue_locked(annotator_id, rounds)
        if adopted is not None:
            return adopted

        info = self._recommend_locked()
        self._resume_recheck_baseline(annotator_id, list(info['pair_ids']))
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

    def refresh_supervision(self) -> dict:
        from pref.io import export_supervision_jsonl

        eta = float(self.cfg.get('eta', 0.5))
        path = export_supervision_jsonl(self.out, eta=eta, cfg=self.cfg)
        return {'path': str(path), 'eta': eta}

    def _load_user(self, annotator_id: str) -> dict:
        rows = load_jsonl(self.labels_path(annotator_id))
        done = {r['task_id'] for r in rows if 'task_id' in r}
        round_counts = Counter(r['task_id'] for r in rows if 'task_id' in r)
        mode = self.user_mode.get(annotator_id, 'reannotate' if self.allow_reannotate else 'normal')
        if mode == 'recheck':
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
            pool = self.pairs
            queue = spread_by_scene([p for p in self.pairs if p['task_id'] not in done])
        user = {
            'done': done,
            'round_counts': round_counts,
            'queue': queue,
            'pool': pool,
            'target': len(pool) if mode == 'recheck' else (len(queue) if mode != 'normal' else len(self.pairs)),
            'mode': mode,
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
        if mode == 'normal':
            total = len(self.pairs)
            n_done = sum(1 for p in self.pairs if p['task_id'] in user['done'])
            remaining = total - n_done
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
            'recommend_note': self.recommend_meta.get('note') if mode == 'recheck' else '',
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
        return {
            'task_id': task['task_id'],
            'scene_id': task['scene_id'],
            'dataset': task['dataset'],
            'split': task['split'],
            'display_swapped': bool(assign['display_swapped']),
            'ir_url': f'/img?side=ir&task_id={tid_q}&annotator_id={aid_q}',
            'vi_url': f'/img?side=vi&task_id={tid_q}&annotator_id={aid_q}',
            'left_url': f'/img?side=left&task_id={tid_q}&annotator_id={aid_q}',
            'right_url': f'/img?side=right&task_id={tid_q}&annotator_id={aid_q}',
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
  header strong { font-size: 14px; white-space: nowrap; }
  .muted { color: var(--muted); font-size: 12px; }
  .bar {
    flex: 1; min-width: 80px; height: 6px; background: #222833; border-radius: 3px; overflow: hidden;
  }
  .bar > i { display: block; height: 100%; background: var(--accent); width: 0%; }
  #app {
    flex: 1; min-height: 0;
    display: flex; flex-direction: column;
    overflow: hidden;
  }
  .prompt {
    flex: 0 0 auto;
    margin: 4px 10px 0; padding: 4px 10px;
    background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
    font-size: 12px; line-height: 1.3;
  }
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
    margin: auto; max-width: 520px; text-align: center; padding: 24px;
    border: 1px solid var(--line); border-radius: 12px; background: var(--panel);
  }
  .done-box h1 { margin: 0 0 8px; font-size: 22px; color: var(--ok); }
  .gate {
    margin: auto; width: min(440px, calc(100% - 32px));
    padding: 22px; border: 1px solid var(--line); border-radius: 12px; background: var(--panel);
  }
  .gate h1 { margin: 0 0 6px; font-size: 18px; }
  .gate label { display: block; margin: 12px 0 4px; font-size: 12px; color: var(--muted); }
  .gate select, .gate input[type=text] {
    width: 100%; padding: 8px 10px; border-radius: 6px;
    border: 1px solid var(--line); background: #0e1116; color: var(--text); font-size: 14px;
  }
  .radios { display: flex; gap: 8px; }
  .radios label {
    flex: 1; margin: 0; padding: 10px 8px; text-align: center; color: var(--text);
    border: 1px solid var(--line); border-radius: 6px; cursor: pointer; font-size: 13px;
  }
  .radios input { margin-right: 4px; }
  .gate .go { width: 100%; margin-top: 16px; }
  #switch { margin-left: auto; min-width: 0; padding: 4px 10px; font-size: 12px; }
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
  <button type="button" id="switch" hidden>Switch user</button>
</header>
<div id="app"></div>
<script>
const app = document.getElementById('app');
const STORE = 'prefbench.session';
const DEFAULT_ID = __DEFAULT_ANNOTATOR__;
let task = null;
let busy = false;
let session = null;
let roster = [];

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function expLabel(e) { return e === 'non-professional' ? 'Non-professional' : 'Professional'; }

function setProgress(p) {
  const pct = p.total ? Math.round(100 * p.done / p.total) : 0;
  document.getElementById('pbar').style.width = pct + '%';
  document.getElementById('count').textContent = `${p.done} / ${p.total}`;
  document.getElementById('meta').textContent =
    `${p.annotator} · ${expLabel(p.expertise)} · ${p.remaining} left`
    + (p.mode === 'recheck' ? ' · review' : '');
  document.getElementById('switch').hidden = false;
}

function renderDone(p) {
  setProgress(p);
  const extra = p.recommend_note ? `<p class="muted">${esc(p.recommend_note)}</p>` : '';
  app.innerHTML = `<div class="done-box">
    <h1>This round is complete</h1>
    <p class="muted">${esc(p.annotator)} (${expLabel(p.expertise)}) has finished the current queue.</p>
    <p class="muted">Raw labels are in ${esc(p.labels_path)}. Closing this program recomputes and saves supervision.jsonl.</p>
    ${extra}
  </div>`;
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
  app.innerHTML = `
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
      <span class="hint">Clicking an image selects that side. For a tie, use the yellow button or press 2 / Space.</span>
    </div>
  `;
  document.getElementById('btn-left').onclick = () => submit('left');
  document.getElementById('btn-tie').onclick = () => submit('tie');
  document.getElementById('btn-right').onclick = () => submit('right');
}

function renderGate(err) {
  task = null;
  document.getElementById('switch').hidden = true;
  document.getElementById('meta').textContent = 'Select an annotator';
  document.getElementById('count').textContent = '—';
  document.getElementById('pbar').style.width = '0%';
  const opts = roster.map(a => {
    const sel = session && session.annotator_id === a.annotator_id ? ' selected' : '';
    return `<option value="${esc(a.annotator_id)}"${sel}>${esc(a.annotator_id)} · ${expLabel(a.expertise)} · ${a.n_labels} labels</option>`;
  }).join('');
  const cur = (session && session.annotator_id) || DEFAULT_ID || '';
  const exp = (session && session.expertise) || 'professional';
  app.innerHTML = `
    <form class="gate" id="gate">
      <h1>Select an annotator</h1>
      <p class="muted">One server can be shared. Identity stays in this browser. Expertise is stored on the annotator profile, not on each click.</p>
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
      ${err ? `<p class="muted" style="color:#f0b429">${esc(err)}</p>` : ''}
      <button class="primary go" type="submit" data-mode="label">Continue unlabeled items</button>
      <button class="go" type="submit" data-mode="recheck">Recommended review</button>
    </form>`;
  const who = document.getElementById('who');
  const box = document.getElementById('newid');
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
  document.getElementById('gate').onsubmit = async (e) => {
    e.preventDefault();
    const annotator_id = box.value.trim();
    const expertise = document.querySelector('input[name=exp]:checked').value;
    const mode = (e.submitter && e.submitter.dataset.mode) || 'label';
    try {
      const r = await fetch('/api/session', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ annotator_id, expertise, mode }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Could not save the user');
      session = { annotator_id: data.annotator_id, expertise: data.expertise, mode };
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

async function refreshRoster() {
  const r = await fetch('/api/annotators');
  const data = await r.json();
  roster = data.annotators || [];
}

async function loadNext() {
  if (!session) { renderGate(); return; }
  const q = encodeURIComponent(session.annotator_id);
  const r = await fetch('/api/next?annotator_id=' + q);
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

document.getElementById('switch').onclick = () => {
  task = null;
  renderGate();
};
document.addEventListener('keydown', (e) => {
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
  await refreshRoster();
  // Always show the gate so the user can pick identity / recommended review.
  // Remembered session only pre-fills the form.
  renderGate();
})();
</script>
</body>
</html>
"""


def _html(default_annotator: str | None) -> str:
    raw = json.dumps(default_annotator or '')
    return HTML.replace('__DEFAULT_ANNOTATOR__', raw)


def make_handler(state: AnnotateState):
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
                self._send(200, _html(state.default_annotator).encode('utf-8'),
                           'text/html; charset=utf-8')
                return
            if parsed.path == '/api/annotators':
                self._json(200, {'annotators': state.list_annotators()})
                return
            if parsed.path == '/api/next':
                qs = parse_qs(parsed.query)
                aid = (qs.get('annotator_id') or [None])[0]
                if not aid:
                    self._json(400, {'error': 'Select an annotator'})
                    return
                try:
                    with state.lock:
                        cur = state.current(aid)
                        if cur is None:
                            self._json(200, {'task': None, 'progress': state.progress(aid)})
                        else:
                            self._json(200, {'task': state.public_task(cur, aid)})
                except ValueError as e:
                    self._json(400, {'error': str(e)})
                return
            if parsed.path == '/api/status':
                qs = parse_qs(parsed.query)
                aid = (qs.get('annotator_id') or [None])[0]
                if not aid:
                    self._json(200, {'annotators': state.list_annotators()})
                    return
                try:
                    self._json(200, state.progress(aid))
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
                path = state.resolve_image(task_id, side, annotator_id=aid)
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
                    info = state.upsert_annotator(
                        payload.get('annotator_id', ''),
                        payload.get('expertise', ''),
                        payload.get('mode', 'label'),
                    )
                    self._json(200, info)
                    return
                if parsed.path == '/api/label':
                    result = state.submit(
                        payload.get('annotator_id', ''),
                        payload['task_id'],
                        payload['choice'],
                    )
                    self._json(200, result)
                    return
                if parsed.path == '/api/heartbeat':
                    state.heartbeat(payload.get('client_id', ''))
                    self._json(200, {'ok': True})
                    return
                if parsed.path == '/api/bye':
                    state.browser_left(payload.get('client_id', ''))
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
    args = p.parse_args()

    cfg = load_config(args.config)
    queue_ids = None
    if args.queue is not None:
        q = json.loads(Path(args.queue).read_text(encoding='utf-8'))
        ids = set(q.get('random_reann') or []) | set(q.get('reannotation_pair_ids') or [])
        queue_ids = ids or None
    state = AnnotateState(
        cfg,
        split=args.split,
        allow_reannotate=args.allow_reannotate,
        queue_task_ids=queue_ids,
        default_annotator=args.annotator,
    )
    if not state.pairs_path.is_file():
        raise SystemExit(f'missing {state.pairs_path}; run sample_pairs first')

    print(f'[annotate] pairs={len(state.pairs)} annotators={len(state.list_annotators())}')
    print(f'[annotate] reannotate={args.allow_reannotate}')
    print(f'[annotate] http://127.0.0.1:{args.port}/')
    print(f'[annotate] Cursor Ports → Forward {args.port}')
    print('[annotate] pick user + expertise in the page; keys: 1=A, 2/Space=tie, 3=B')
    print('[annotate] Closing the browser tab does not stop the server. Aggregation runs once on exit.')

    httpd = ThreadingHTTPServer(('0.0.0.0', args.port), make_handler(state))
    state.httpd = httpd

    def _request_stop(signum, _frame):
        raise KeyboardInterrupt

    def _sweep() -> None:
        while not state._stopping:
            time.sleep(2.0)
            state.sweep_clients()

    signal.signal(signal.SIGTERM, _request_stop)
    threading.Thread(target=_sweep, daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n[annotate] stopped')
    finally:
        httpd.server_close()
        if state.n_new > 0:
            try:
                info = state.refresh_supervision()
                print(f'[annotate] aggregated {state.n_new} new labels → {info["path"]}')
            except Exception as e:
                print(f'[annotate] aggregation failed: {e}')
        else:
            print('[annotate] no new labels, skip aggregation')
        time.sleep(0.05)


if __name__ == '__main__':
    main()
