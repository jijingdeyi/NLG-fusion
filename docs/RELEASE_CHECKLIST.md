# Before public release

- [ ] Replace `CITATION.cff` `repository-code` URL and author list
- [ ] Decide whether to release preference JSONL (Zenodo / Hugging Face); see `docs/DATA.md` / `docs/PACKS.md`
- [ ] Ensure `hpfm` weights synced: `python hpfm/scripts/sync_weights.py`
- [ ] Run `PYTHONPATH=. python -m pytest pref/tests` in CI
- [ ] Confirm only whitelisted scripts under `scripts/pref_bench/` are tracked
- [ ] Confirm `score.md` / eval scripts are **not** in the public commit
- [ ] Add paper / arXiv link to README once available
- [ ] Confirm third-party fusion method licenses for any shared fusion outputs
