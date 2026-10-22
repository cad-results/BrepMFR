# Hard Samples Expansion Plan

Goal: expand training data by (1) visualizing existing hard_samples with model inference + defeaturing, (2) hand-selecting the samples where inference produces good pseudo-labels, and (3) folding those "suitable_samples" into a larger training corpus alongside mftrcad for an upgraded run.

Label space: the **27-class MFTRCAD catalog** (see `brepformer/data/classes.py::CLASS_NAMES`). Selected samples are stored in the same directory layout as `brepformer/data/mftrcad/` (`steps/` + `labels/`).

## Working conventions

**STEP-only corpus.** `hard_samples/` and `hard_samples_xl_skipped/` hold *only* `.step`/`.stp` geometry. GrabCAD models ship renderings, STL, DXF, IGES and native CAD alongside the STEP; none of that is kept. `download_grabcad_api.py` enforces this at write time (it only ever writes `.step`, and validates the `ISO-10303-21` header before saving). The sole non-STEP files permitted under `hard_samples/` are this `PLAN.md` and the pseudo-label `.json`/`.seg` pairs inside `{yes,maybe,no}/labels/` — do not delete those, `merge_suitable_samples.py` reads them.

**Delegate repetitive stages to subagents.** Most of the remaining work is long, mechanical, and log-heavy: batch download, batch inference, batch defeature, per-face reclassification, preprocessing. Running these in the main session floods context with thousands of progress lines and risks losing the thread mid-phase. Dispatch each such stage to a **fresh subagent** (not a fork — a fork inherits the whole transcript and defeats the purpose) with a self-contained prompt that states the validated command, the resume/idempotency behaviour, and the exact counts to report back. Keep the orchestration and all decision-making in the main session.

Subagent prompts for these stages should always include:
- the exact command, pre-validated on a small `--limit` smoke test first;
- "re-run until it completes or stalls" — every batch script here is idempotent and resumable, so a 10-minute Bash timeout plus repeat is the right pattern for multi-hour runs;
- an explicit "do NOT modify source files; stop and report if blocked" constraint, so a subagent doesn't silently patch around a real bug;
- the specific file counts / failure tallies to report, so the main session can decide the next phase without re-reading logs.

**Launch these subagents with no `model` override.** Inheriting the session's model is the only reliable path in this environment; an explicit override fails.

Why (diagnosed 2026-08-28, worth not re-deriving): the Agent tool's `model` field accepts only the aliases `sonnet`/`opus`/`haiku`/`fable` — a full ID like `claude-sonnet-5` is rejected by input validation. The alias targets come from env vars, and this setup runs Bedrock (`CLAUDE_CODE_USE_BEDROCK=1`) with them mismatched:

```
ANTHROPIC_DEFAULT_OPUS_MODEL=us.anthropic.claude-opus-4-6-v1          # inference profile — works
ANTHROPIC_DEFAULT_SONNET_MODEL=anthropic.claude-sonnet-4-20250514-v1:0 # on-demand ID — 400s
```

On Bedrock the `us.anthropic.*` form is a cross-region *inference profile*; the bare `anthropic.*` form is the on-demand model ID, and these models are not enabled for on-demand throughput. So `model: "sonnet"` dies instantly with HTTP 400 `model_not_found` ("Retry your request with the ID or ARN of an inference profile"), while agents launched with no override inherit the session's already-resolved (prefixed) model and run fine.

Consequences:
- Sonnet 5 is **not** reachable here. `/model claude-sonnet-5` reports success but leaves `ANTHROPIC_MODEL` at `claude-sonnet-4-20250514`, so setting the session default does not help either — verified independently in a second session.
- Fixing the alias would mean pointing `ANTHROPIC_DEFAULT_SONNET_MODEL` at an inference-profile ID (`us.anthropic....`) enabled on the account. That is an environment/config change, not a repo change.
- A non-standard `CLAUDE_CODE_3P_PROBE_WROTE_SONNET_DEFAULT` var is present holding that same unprefixed ID, and is the likely source of the misconfiguration.

## Checkpoints used

- 27-class model (for pseudo-labels & visualization): `results/trial5/best-epoch=39-val/f1=0.8832.ckpt`
- 5-class defeature model (for defeature preview): `results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt`

## Progress checklist

- [x] **Phase 1 — Initial 5 hard_samples (already present)**
    - Files: `brake_master_center.step`, `print_in_place_box.step`, `push_clamp.step`, `screw_jack_asm.stp`, `speaker_52mm.step`
    - [x] Run 27-class inference → `brepformer/inference/hard_samples_27/`
    - [x] Run 5-class inference → `brepformer/inference/hard_samples_5/`
    - [x] Run defeature v2 using 5-class predictions → `brepformer/defeatured_output/hard_samples/`
        - 4/5 succeeded; `screw_jack_asm` timed out after 300s (see `batch_report.json`)
    - [x] Colored STEP exports (27-class) → `brepformer/pipeline_output/hard_samples_colored_27/`
    - [ ] User visually inspects each with `visualize_seg.py` and `visualize_defeature.py`
        - Both viewers were launched in the `brep_mfr` conda env (PyQt5 present there, not in `new_brepmfr`)
    - [ ] User names the subset that passed inspection (list in this file under "Suitable samples")
- [x] **Phase 2 — Bulk download from LINKS.md** (done 2026-08-19)
    - `download_grabcad.py` (curl/requests) was abandoned — GrabCAD sits behind CloudFront + AWS WAF, which blocks plain HTTP clients and even `curl_cffi` browser-impersonation (gets a WAF JS-challenge page, HTTP 202).
    - **Used instead**: `brepformer/data/download_grabcad_playwright.py` — a real headed Chromium (persistent profile at `~/.grabcad_playwright`) that the user manually signs into once per run; the real browser solves the WAF challenge natively.
    - Result of the full run over 476 unique slugs (550 links, deduped): **239 new files kept**, 8 already-present, ~396 slugs missed (their download-button DOM pattern didn't match any of the selectors tried — GrabCAD's page markup varies by model/uploader).
    - 251 STEP files total landed in `brepformer/data/hard_samples/`; all validated with `ISO-10303-21` headers (100% valid).
    - **33 files >10MB moved aside** to `brepformer/data/hard_samples_xl_skipped/` before inference (they dominate wall-clock and are usually noisy multi-body assemblies outside the model's training distribution). **223 files remain** in `hard_samples/` for inference.
    - ~~To pick up more of the missed 396 slugs later: rerun `download_grabcad_playwright.py` and iterate on the button-selector list in `top_button_selectors`.~~ **This diagnosis was wrong — see Phase 2b.** The selectors were fine; the download *flow* was the problem.
- [x] **Phase 2b — API-driven re-download of the missed 396** (done 2026-08-28)

    **Root cause of the 396 misses.** Not selectors. Three separate findings, in order:

    1. `probe_grabcad_dom.py` showed the "Download files" button is present and visible on essentially every page, and that headless Chromium gets a CloudFront `ERROR: The request could not be satisfied` interstitial while headed Chromium (WSLg, `DISPLAY=:0`) loads the page fine. So the WAF workaround was working and the button was findable — the old log's "no download button found" came from runs that hit the WAF page, and "no download fired" from the button itself.
    2. That button calls AngularJS `downloadArchive()`, which asks the server to build a ZIP of **every** file in the model — renderings, STL, native CAD, the lot. For most models that either never fires a client download event inside the timeout or yields a huge archive we'd throw away. Clicking it was always the wrong lever.
    3. `probe_grabcad_files.py` / `probe_grabcad_api.py` found the right lever: a public JSON manifest at `/community/api/v1/models/<slug>/files` listing every file with its `extension` and `download_url`. Reachable through Playwright's `context.request` (shares the profile's WAF + session cookies, skips page rendering — roughly 10x faster per slug).
    4. Final gotcha: `download_url` **does not serve the file**. It returns a small JSON envelope `{"url": "https://grabcad.com/cads/files/<cadid>/original.stp"}`, and the bytes live at that second URL. A downloader that treats step 1's response as file content rejects everything as "lacks ISO-10303-21 header". `resolve_download()` follows both hops.

    **Auth.** A red herring worth recording. The old `wait_for_signed_in()` polled for `.avatar`, which also matches GrabCAD's *anonymous* library page — so it reported success against a logged-out profile. The profile genuinely is logged out (`a[href*='logout']` = 0, three "Log in" links, anonymous `_grabcad_session`), and file downloads did 401 with `{"error":"You need to log in or sign up before continuing."}` at one point. They now return 200 + real STEP bytes anonymously, so login is **not** required for this corpus — but the check must verify by *actually fetching bytes*, never by sniffing the DOM. `grabcad_login.py::api_says_logged_in` does exactly that (manifest 200 proves nothing; it's public), and `download_grabcad_api.py` aborts up-front rather than burning hundreds of requests on 401s.

    **New files:**
    - `brepformer/data/download_grabcad_api.py` — the working downloader. Manifest-driven, recurses into `folders[]` (models bury CAD under "Renderings"-style subfolders), ranks candidates (real STEP > archive hinting at STEP/neutral CAD > any archive), unpacks ZIPs keeping only STEP entries, validates `ISO-10303-21`, idempotent via `--also_check`, `--stop_after` to cap a run.
    - `brepformer/data/grabcad_login.py` — headed sign-in helper + authoritative `--check_only` auth probe.
    - `brepformer/data/probe_grabcad_{dom,files,api,auth}.py` — the four diagnostics that produced the findings above. Kept because GrabCAD's markup and API drift; re-run these first if Phase 2 regresses rather than guessing at selectors again.

    Smoke test: 10 slugs → 8 STEP files kept, 3 missed (2 genuinely expose no STEP at all, only IGES/STL — an irreducible floor on yield).
- [x] **Phase 2c — Slug harvesting + corpus scale-out** (done 2026-08-28/29)
    - Built `harvest_grabcad_slugs.py` on the paginated community listing endpoint (see "Phase 2c" section below for the three API quirks that shape it). `LINKS.md`: **476 → 1881** distinct slugs across two harvests (902 + 503).
    - **`--max_per_query` was essential.** The first uncapped harvest returned 765 of 901 slugs from the single query `bracket` and never reached query 3 of 96 — recreating the concentration failure at harvest time. Reverted `LINKS.md` and re-harvested capped; the second run spread 902 slugs over 62 queries at 15 each.
    - Downloads, with the three hardening fixes in place: batch 1 **724/902 slugs (80.3%)**, batch 2 **380/503 (75.5%)**. Each converged in **two runs** versus nine before, because the 403 backoff held — batch 1 walked all 902 slugs end-to-end in ~23 min with 32 backoff events, all recovered, and zero slugs lost to WAF.
    - **Corpus: 269 → 1681 STEP files across 1443 distinct slugs (1.16 files/slug).** All valid `ISO-10303-21`. ~94% of remaining misses are genuine absences (pages offering only STL/IGES/images), so yield is effectively exhausted.
    - One file quarantined: `double-column-car-lift-1__car_jack.step` segfaults OCC's STEP reader (signal 11, uncatchable). See `hard_samples_quarantine/README.md`. **This hazard is unmitigated in `preprocess.py` / `infer.py` / `curate_corpus.py` — a future such file will kill those stages with no traceback.** Durable fix would be fork-per-file isolation in the reader path.
    - **Curation applied 2026-08-29.** 1643 top-level files fingerprinted → **KEEP 1185 files / 1142 distinct slugs** (72.3% of readable). Rejects moved (not deleted) to `hard_samples_rejected/<reason>/`: `too_many_faces` 239, `slug_overrepresented` 84, `family_variant` 53, `exact_duplicate` 41, `too_few_faces` 36. Largest remaining topology family: **2 files**. 1118 of 1142 slugs contribute exactly one file. Face counts: median 128, max 1470 — inside the 12/1500 band without piling at either edge.
    - 5 files were unreadable by OCC (cleanly, no crash) and are neither kept nor rejected by the script, so they were moved by hand to `hard_samples_unreadable/` to stop them consuming inference attempts: `40-iso-container__contain.stp`, `collars-1__Collar_Demo_01.step`, `mgu-h-model-motor-generator-unit-heat-f1-1__Part1.step`, `nema-17-coupler-1__NEMA17_Coupler.step`, `spacer-51__Disc_L5_spacer_1.step`.
    - **Assemblies decision (2026-08-29): KEEP them**, at 44.7% of the curated set. They are handled per-face by 0.7 confidence masking, which attenuates precisely the faces the teacher is least sure about, rather than discarding whole parts and with them the slug diversity the generalisation argument rests on. Do **not** pass `--drop_assemblies`.
- [x] **Phase 3 — Batch inference on all downloaded samples** (done via subagents, 2026-08-19 to 2026-08-21)
    - [x] 27-class inference (trial5) → `brepformer/inference/hard_samples_27/` — **219/223** succeeded. 2 files OOM'd (`normal-gauge-railway-track-1__UIC60_3m.stp`, `pipe-clamp-14__52-59 m10.step` — `numpy._ArrayMemoryError`, tried to allocate 26-48 GiB); 2 more silently missing.
    - [x] 5-class inference (trial1_ss1500) → `brepformer/inference/hard_samples_5/` — **188/223** succeeded, 31 CUDA OOM errors on large/complex parts.
    - [~] Defeature v2 (5-class preds) → `brepformer/defeatured_output/hard_samples/` — **in progress**, resumed 2026-08-25 after the prior run died without a `batch_report.json` (process was killed when the earlier session ended; no crash in the algorithm itself). Patched `defeature_v2.py`'s batch loop with a skip-existing check (mirrors the same fix already applied to `infer.py`) so the resume doesn't redo the 74 models already defeatured. 114 of 188 seg-backed models remained as of the resume.
    - Note: `infer.py` and `defeature_v2.py` both now have skip-existing logic in their batch loops — safe to interrupt and resume either at any time.
- [ ] **Phase 4 — Visual reclassification with Rotator + Claude** (new)
    - **Goal:** Use the rotator pipeline (tighter angles: 30° and 15° increments instead of 45°) to render dense multi-view images of each inferenced model, then feed each face's surrounding views to Claude's vision API for automated class verification/correction. This replaces (or supplements) manual visual analysis.
    - **Approach:**
        1. For each STEP file that has 27-class and/or 5-class inference results, run the rotator at 30° and 15° increments (12 and 24 views respectively) to produce a comprehensive set of orthographic renders.
        2. For each face, extract its geometric properties (surface type, area, adjacency, curvature) and generate face-highlighted views (the face colored distinctly against gray neighbors).
        3. Feed the face-highlighted views + geometric context to Claude's vision API with a structured prompt that asks: "Given this face's geometry and visual context, is the predicted class [X] correct? If not, what should it be?"
        4. Use geometric confirmation rules (from `defeature_v2.py`) as hard constraints: e.g., a planar face with >15% area fraction cannot be a hole; a cylinder co-axial with neighboring features is likely a hole; etc.
        5. Aggregate Claude's per-face verdicts and geometric rule outputs into a corrected label set. Only override predictions when both Claude and geometric rules agree (conservative).
    - **Angle configurations:**
        - Standard pass: 30° increments → 12 views (front/back/left/right + 8 diagonals)
        - Fine pass (for ambiguous faces): 15° increments → 24 views
    - **Files:**
        - `brepformer/reclassify_visual.py` — main orchestration: renders views, calls Claude API, applies geometric rules, outputs corrected labels
        - `brepformer/geometric_rules.py` — extracted geometric confirmation/denial logic (from defeature_v2.py patterns) applicable to both 27-class and 5-class label spaces
        - `rotator/` — pipeline.py, environment.yml, README.md (copied from ~/rotator)
    - **Output:** `brepformer/inference/hard_samples_27_corrected/` and `hard_samples_5_corrected/` — same JSON format as original inference, with a `corrections` field logging what changed and why.
    - **When to use:** Run on all files that have been inferenced (219 for 27-class, 188 for 5-class). This is Phase 4 because it refines pseudo-labels BEFORE the manual triage (Phase 5), making triage faster and more reliable.

- [ ] **Phase 5 — Manual selection** (tooling ready, not yet run by the user)
    - Once defeature finishes, launch `visualize_seg.py` (27-class overlay) and `visualize_defeature.py` (original vs defeatured, 5-class) in the `brep_mfr` conda env — same pattern as Phase 1, pointed at the full `hard_samples/` + `hard_samples_27`/`hard_samples_5`/`defeatured_output/hard_samples` dirs instead of just the 5 originals.
    - User sorts each model into `hard_samples/{yes,maybe,no}/steps/` (same convention as Phase 1) — no separate label-copying step needed up front; `merge_suitable_samples.py` (below) pulls labels automatically from the inference JSONs by matching STEP filename stem.
    - Verified inference JSON schema (`brepformer/inference/hard_samples_27/<id>.json`): top-level dict with `face_preds` (list[int], one 27-class id per face, in `TopExp_Explorer` face order) — `merge_suitable_samples.py` reads this key directly.
    - **Verified end-to-end** with a dry run against Phase 1's existing `yes`/`maybe` triage: 4/5 merged cleanly (`brake_master_center`, `push_clamp`, `speaker_52mm`, `print_in_place_box`). `screw_jack_asm` was correctly reported `missing_labels` — it's >10MB and got moved to `hard_samples_xl_skipped/` before the Phase 3 batch scan (which only globs the top level of `hard_samples/`, non-recursive), so it was never inferred. If the user wants `screw_jack_asm` in the final corpus, it needs a one-off inference run against its XL-skipped location before merging.
- [ ] **Phase 6 — Merged training run**
    - Combine `mftrcad/` + `suitable_samples/` into a merged preprocessed dataset
    - Run `train_preprocessed.py` (face segmentation, 27-class, tuned hyperparams from prior runs)
    - Compare against trial5 baseline

## Suitable samples — Phase 1 triage

User triage of the initial 5 hard_samples, sorted into `hard_samples/{yes,maybe,no}/{steps,labels}/`. Labels are the **27-class inference outputs from trial5** (both `.json` and `.seg`) — these are the pseudo-labels the model will train on.

- [x] `brake_master_center` — **yes**
- [x] `print_in_place_box` — **maybe**
- [x] `push_clamp` — **yes**
- [x] `screw_jack_asm` — **maybe** (also timed out during defeature)
- [x] `speaker_52mm` — **yes**

For downstream training, promote `yes/` (and optionally `maybe/` after a second review) into `brepformer/data/suitable_samples/` alongside mftrcad-format labels.

## Key commands

### Phase 1 — Inference + defeature on initial hard_samples

```bash
# 27-class inference (produces .seg + .json with 27-class labels — these become the pseudo-labels)
python -m brepformer.infer \
    --step_dir brepformer/data/hard_samples/ \
    --checkpoint "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --output_dir brepformer/inference/hard_samples_27/

# 5-class inference (used for defeaturing preview)
python -m brepformer.infer \
    --step_dir brepformer/data/hard_samples/ \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --output_dir brepformer/inference/hard_samples_5/

# Colored STEP export (27-class)
python -m brepformer.export_freecad \
    --step_dir brepformer/data/hard_samples/ \
    --seg_dir brepformer/inference/hard_samples_27/ \
    --output_dir brepformer/pipeline_output/hard_samples_colored_27/

# Defeature v2 with 5-class predictions
python -m brepformer.defeature_v2 \
    --step_dir brepformer/data/hard_samples/ \
    --seg_dir brepformer/inference/hard_samples_5/ \
    --output_dir brepformer/defeatured_output/hard_samples/ \
    --save_colored --verbose

# Visualize segmentation (27-class overlay)
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/hard_samples/ \
    --seg_dir brepformer/inference/hard_samples_27/

# Visualize defeature (original vs defeatured, 5-class overlay on original)
python -m brepformer.visualize_defeature \
    --step_dir brepformer/data/hard_samples/ \
    --defeatured_dir brepformer/defeatured_output/hard_samples/ \
    --seg_dir brepformer/inference/hard_samples_5/ \
    --report_dir brepformer/defeatured_output/hard_samples/
```

**Note on subdirectory scanning:** `infer.py --step_dir` and `defeature_v2.py --step_dir` both glob only the top level (`*.step`/`*.stp`, non-recursive). Once files are sorted into `hard_samples/{yes,maybe,no}/steps/` during Phase 5 triage, they drop out of any future top-level batch re-run automatically — no manual exclusion needed. This is also why `screw_jack_asm` (already in `maybe/steps/` from Phase 1, and additionally >10MB) was skipped by the Phase 3 full-corpus batch.

### Phase 4 — Visual reclassification with Rotator + Claude

```bash
# Geometric rules only (fast, no API calls — good first pass)
python -m brepformer.reclassify_visual \
    --step_dir brepformer/data/hard_samples/ \
    --seg_dir brepformer/inference/hard_samples_27/ \
    --output_dir brepformer/inference/hard_samples_27_corrected/ \
    --label_space 27 --no_claude --verbose

# Full pipeline with Claude vision (requires ANTHROPIC_API_KEY)
python -m brepformer.reclassify_visual \
    --step_dir brepformer/data/hard_samples/ \
    --seg_dir brepformer/inference/hard_samples_27/ \
    --output_dir brepformer/inference/hard_samples_27_corrected/ \
    --label_space 27 --angle_step 30 --verbose

# Fine-angle pass for maximum accuracy (15° increments, more views)
python -m brepformer.reclassify_visual \
    --step_dir brepformer/data/hard_samples/ \
    --seg_dir brepformer/inference/hard_samples_27/ \
    --output_dir brepformer/inference/hard_samples_27_corrected/ \
    --label_space 27 --angle_step 15 --verbose

# 5-class reclassification (for defeature predictions)
python -m brepformer.reclassify_visual \
    --step_dir brepformer/data/hard_samples/ \
    --seg_dir brepformer/inference/hard_samples_5/ \
    --output_dir brepformer/inference/hard_samples_5_corrected/ \
    --label_space 5 --verbose

# Standalone geometric rules check (no rendering, quick diagnostic)
python -m brepformer.geometric_rules \
    --step brepformer/data/hard_samples/some_model.step \
    --seg brepformer/inference/hard_samples_27/some_model.json \
    --label_space 27
```

**Note:** The reclassification pipeline requires:
- `rotator` conda env (pythonocc + vtk) for rendering face-highlighted multi-view images
- `ANTHROPIC_API_KEY` env var for Claude vision API calls (skip with `--no_claude`)
- Skip-existing logic built in — safe to interrupt and resume

**Launching the Phase 5 triage viewers on the full corpus** (must use `brep_mfr` conda env — PyQt5 lives there):

```bash
conda activate brep_mfr

# 27-class segmentation viewer, full corpus
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/hard_samples/ \
    --seg_dir brepformer/inference/hard_samples_27/

# Original vs defeatured comparison, full corpus (5-class overlay + defeature stats)
python -m brepformer.visualize_defeature \
    --step_dir brepformer/data/hard_samples/ \
    --defeatured_dir brepformer/defeatured_output/hard_samples/ \
    --seg_dir brepformer/inference/hard_samples_5/ \
    --report_dir brepformer/defeatured_output/hard_samples/
```

Both viewers use `--step_dir` (not a fixed `--step`), so D/RIGHT and A/LEFT step through every model in `hard_samples/` that has matching inference/defeature output. As the user sorts files into `yes/`, `maybe/`, `no/` (moving them out of the top level), subsequent launches of the same command automatically show a shrinking remaining set — no relaunch args need to change.

### Phase 2b — API-driven download (current, preferred)

```bash
# 0. Confirm the session can actually pull bytes (public manifest 200 proves nothing).
python brepformer/data/grabcad_login.py --check_only
# If it reports NOT authenticated, run without --check_only and sign in
# in the Chromium window it opens; it polls until a real download succeeds.

# 1. Build the list of slugs we don't already have (idempotent, but this
#    keeps the run short and makes the miss count meaningful).
python -c "
import re
from pathlib import Path
LINK_RE = re.compile(r'https://grabcad\.com/library/([A-Za-z0-9_\-]+)')
slugs=[]; seen=set()
for m in LINK_RE.finditer(Path('LINKS.md').read_text()):
    s=m.group(1)
    if s not in seen: seen.add(s); slugs.append(s)
have=set()
for r in [Path('brepformer/data/hard_samples'), Path('brepformer/data/hard_samples_xl_skipped')]:
    for f in r.rglob('*'):
        if f.is_file() and '__' in f.name: have.add(f.name.split('__')[0])
missed=[s for s in slugs if s not in have]
Path('/tmp/missed_slugs.txt').write_text('\n'.join(missed))
print('missed:', len(missed))
"

# 2. Smoke test before committing to the full run.
python brepformer/data/download_grabcad_api.py \
    --slugs_file /tmp/missed_slugs.txt --limit 10 --log_file /tmp/api_smoke.log

# 3. Full run. Headed browser required (do NOT pass --headless: AWS WAF
#    serves a CloudFront interstitial to headless clients). Idempotent and
#    resumable, so re-run freely until it completes or stalls.
python brepformer/data/download_grabcad_api.py \
    --slugs_file /tmp/missed_slugs.txt \
    --log_file hard_samples_download.log \
    --manifest_out /tmp/grabcad_api_manifest.json

# 4. Count + validate (corpus target: >= 1000 STEP files).
find brepformer/data/hard_samples brepformer/data/hard_samples_xl_skipped \
    -type f \( -iname '*.step' -o -iname '*.stp' \) | wc -l

# 5. Why did the rest miss?
grep -oE 'no STEP/archive candidates|manifest HTTP [0-9]+|manifest empty|step1 HTTP [0-9]+|step2 HTTP [0-9]+|not a valid ZIP|lacks ISO-10303-21' \
    hard_samples_download.log | sort | uniq -c | sort -rn
```

**Corpus target: 1000 _distinct_ parts. File count is explicitly NOT the objective.**

The `--all_step` / `--revisit` flags exist on the downloader but are **deliberately not used** for corpus growth. Measuring the corpus at 399 files / 199 slugs showed why:

| observation | value |
|---|---|
| files from the top 38 slugs | 238 of 399 (**60%**) |
| `pipe-clamp-14` alone | **44 files** (size variants of one clamp) |
| `butterfly-valves-wafer-type-1` | 21 files |
| bearing-unit family | 18 files |
| `wire-rope-clip-din-741-1` | 15 files |
| byte-identical duplicate groups | 25 |

Multiplying files-per-slug would have produced a corpus whose majority is parametric variants of a handful of product families. The model would learn pipe clamps, not machining features — and because size-variants of one part would straddle a random train/val split, validation F1 would *improve* while real generalisation got worse. That is the worst possible failure mode: it hides itself.

So growth comes from **more slugs**, not more files per slug, and every file passes geometric curation before training. Sequence:

```bash
# 4a. Curate: reject exact duplicates, cap parametric families and
#     over-represented slugs, drop unusable face counts. Dry-run first.
python brepformer/data/curate_corpus.py --step_dir brepformer/data/hard_samples
python brepformer/data/curate_corpus.py --step_dir brepformer/data/hard_samples \
    --apply --reject_dir brepformer/data/hard_samples_rejected

# 4b. If short of 1000 distinct parts, extend LINKS.md with NEW slugs
#     (see "Sourcing more slugs" below), then re-run the download from step 1.
```

`curate_corpus.py` groups on a rotation/translation-invariant signature (face/edge/solid counts + OCC surface-type histogram, plus log-quantised volume/area for the exact-duplicate test), because filename dedupe provably cannot separate these — the pipe-clamp variants have distinct names *and* distinct bytes. Two caps are needed, not one: `--max_per_family` catches same-topology variants, and `--max_per_slug` catches families whose sizes differ *topologically* and so evade it (an "M3–M12 socket head cap screw" library gives each diameter its own face count — verified: 7 survived family capping, the slug cap brought it to 3).

It also rejects `n_faces < 12` (too little signal) and `n_faces > 1500` (OOM'd in Phase 3, far outside MFTRCAD), and flags multi-body assemblies, which matter because the teacher model's confidence is measurably *worst* on assemblies — the lowest-confidence models in the corpus are all of them (`Allen bolt assembly`, `ER16_ColletChuck_Assembled`, bearing units, valve levers). Files already triaged into `{yes,maybe,no}/` are never touched; those are human decisions.

### Phase 2c — Programmatic slug harvesting (`harvest_grabcad_slugs.py`)

Hand-curating `LINKS.md` to 476 slugs topped out well short of 1000 distinct parts: **335 of 476 slugs yielded 578 files**, and of the remaining 141, roughly 130–140 are permanently dead (51 publish no STEP, 50 exceed the size cap, 40 serve a corrupt ZIP). More files per slug is not the answer (see above), so the list itself must grow.

`probe_grabcad_search.py` established that the community API has a paginated listing endpoint, and three properties of it drive the harvester's design:

| property | consequence |
|---|---|
| `per_page` is ignored — always 24 | pagination is fixed |
| pagination caps at 100 pages (page 100 == 101 == 150) | `total_entries: 2400` is that **cap**, identical for every query — not a match count. One query can never yield more than ~2400 records |
| `softwares=step` / `software=step` / `filter[softwares]=step` all return unfiltered results with unchanged totals | **server-side format filtering does not work**; filter client-side |

That client-side filter is the main win. Each record carries `softwares`, e.g. `[{"name": "STEP / IGES"}]`, and only ~29–37% of listed models publish a neutral format. Filtering *before* download turns the "no STEP/archive candidates" category — 51 wasted slugs last pass, each costing a full manifest round-trip — into zero wasted requests.

**`STEP / IGES` is necessary but not sufficient:** it is one GrabCAD tag covering both formats, so a matching model may publish IGES only. The downloader still verifies the `ISO-10303-21` header and curation still has the last word. The filter removes certain losses; it does not certify a win.

**`--max_per_query` is not politeness, it is the point.** An uncapped run for 900 slugs returned **765 from `bracket` alone** and never reached query 3 of 96 — reproducing the exact concentration failure `curate_corpus.py` exists to undo, at the harvest stage where it is far cheaper to prevent than to detect. Keep `target / max_per_query` well under `len(QUERIES)`.

```bash
# Discover new STEP-bearing slugs and append them to LINKS.md.
python brepformer/data/harvest_grabcad_slugs.py --target 900 --max_per_query 15 --append_links
# Then re-run the Phase 2b download over the extended list.
```

### Downloader hardening (from the 9-run bulk pass)

Three fixes, all from measured failures in that pass:

* **403 backoff-and-retry** (`get_with_backoff`). The WAF rate-limits in *bursts*: a run does 55–100 slugs of real work, then every remaining slug 403s within ~6 seconds. That produced **732 `manifest HTTP 403`** and inflated the miss count — in pass 1, 287 of 307 "misses" were a single burst, so only ~100 slugs were genuinely attempted. Those slugs were never unavailable; nine sequential runs recovered them. Retry with cooldown makes one run sufficient. (Note: each 403 also logs a redundant `manifest empty`, so those two counts are the same event.)
* **`Content-Length` pre-check before download.** The size cap was applied only *after* fetching. `dana-60-high-pinion-ford-1` (368 MB) and `ford-5-0-coyote-engine` (282 MB) were pulled in full and discarded every run at 18–28 min each, and sat at the head of the list, so they consumed whole runs' budgets — runs 3 and 4 yielded +4 and 0 almost entirely for this reason. A HEAD is also the only guard against responses Playwright cannot hold at all (one raised `Cannot create a string longer than 0x1fffffe8 characters`).
* **`grabcad_skip_slugs.txt`**, consulted by default via `--skip_slugs`. 29 confirmed-unattainable slugs. Nearly all are large multi-body assemblies (a motherboard, a truck, an engine, 3D printers, a container house) that curation would reject as assemblies even if they fit, so excluding them costs no training signal. Revisit if `--max_size_mb` is ever raised substantially.

**Validity checks must use `LC_ALL=C grep -a`.** These STEP files contain non-UTF-8 bytes in embedded strings, so plain `grep` treats them as binary and returns non-zero *without matching*. A first validation pass falsely condemned 35 of 724 good files (~5%) for "missing" an `ISO-10303-21` header that was at byte 0; `LC_ALL=C grep -a` cleared all 35. Any downstream validity gate that does not do this will silently discard ~5% of a perfectly good corpus.

One known wart: `keep_zip` still uses `_1`/`_2` suffix-increment naming, unlike the direct-STEP path, so ZIP-extracted files can accumulate byte-identical near-duplicates across revisit passes. Not urgent — `curate_corpus.py` catches them as `exact_duplicate` on geometry rather than filename — but it means raw file counts overstate the corpus.

### `infer.py` preprocessed-lookup fix (2026-08-29) — ~15x faster

`infer_single` called `load_sample_from_preprocessed` unconditionally, and that function scanned `mftrcad_processed/{test,val,train}` **unpickling up to all 28,555 files / 17 GB per STEP file** to find a matching `model_id`. Pickle filenames are sequential (`00000.pkl`), so the id really does live inside the file — but the scan was redone for every input.

Worse, it could never succeed here: mftrcad ids look like `20240116_231044_1009` while hard-samples ids are GrabCAD slugs, so the id spaces are **disjoint**. It always scanned everything, always returned `None`, then fell through to `step_to_graph`. That was ~90 s of the ~95 s per file with the GPU at 0% — a full-corpus pass would have taken ~25 h instead of ~4.

Fix: `_preprocessed_index()` builds a `model_id -> pickle path` map **once**, memoised in-process and cached to `mftrcad_processed/model_id_index.json` (28,555 entries, 2 MB). Plus `--no_preprocessed` to skip the lookup entirely when ids cannot possibly match.

Verified: predictions are **byte-identical** to the pre-fix outputs on the same file (`face_preds` equal, 218 faces), so no feature-extraction inconsistency was introduced across the corpus.

| path | per-file |
|---|---|
| before | ~95 s |
| `--no_preprocessed` | **8.3 s** |
| default, cold index | 32 s (one-off build) |
| default, warm cache | **6.3 s** |

The default path is now fast too, so this also speeds up genuine mftrcad inference — the flag is an optimisation, not a requirement.

### Pseudo-label confidence masking (decided 2026-08-28)

Training on trial5's raw argmax was rejected. Measured over 82,456 faces of `hard_samples_27`:

| top-1 confidence | share of faces |
|---|---|
| ≥ 0.5 | 66.9% |
| ≥ 0.7 | 38.9% |
| ≥ 0.8 | 25.1% |
| ≥ 0.9 | 11.8% |

Median top-1 is **0.62**, and the seven most-predicted classes cover 91% of faces while 3 classes are never predicted at all. Critically, **confidence is inversely correlated with class rarity** — the rarest classes average 0.23–0.34 (class 13: 0.232, class 2: 0.306, class 17: 0.316, class 8: 0.334) versus 0.75 for `stock`. Training on raw argmax would therefore preferentially relabel rare features as common ones, entrenching exactly the dead/rare-class problem this expansion is meant to fix.

`merge_suitable_samples.py --min_confidence 0.7` (the new default) writes `-1` for faces below threshold. **No model change is required**: the face head already uses `nn.CrossEntropyLoss(weight=..., ignore_index=-1)` (`models/brep_classifier.py:75`) and `face_labels` already reserves `-1` for padding, so masked faces contribute no gradient. `train_preprocessed.py`'s inverse-frequency weighting filters on `0 <= c < num_face_classes`, so masked faces are excluded from class statistics too.

`--min_kept_frac` (default 0.25) additionally drops models left mostly-unknown by masking; they cost a forward pass and teach almost nothing. Verified on Phase 1's triage: at 0.7, 46.7% of faces mask and all 4 models survive; at 0.9, 64.7% mask and 2 of 4 models fall below the sparsity floor — so **0.7, not 0.9**.

### Group-aware split — IMPLEMENTED 2026-08-29

`preprocess.py` split with `StratifiedShuffleSplit` on dominant class and **no grouping**, so parametric variants of one slug could land on both sides of the split. Verified on this corpus's real group distribution (853 models / 810 groups): the old split **leaked 12 groups** across train/val. Not theoretical.

Now: models are grouped by their `<slug>__` prefix and no group spans two splits. `StratifiedGroupKFold` is used when available (keeps both stratification and grouping); `--no_group_split` opts out. mftrcad ids contain no `__`, so each is its own group and grouping is a **no-op** for them — safe as a default.

Two things worth knowing:

* **sklearn was genuinely absent from `new_brepmfr`.** The pre-existing `except ImportError` therefore fell through to a plain random split over models — silently ungrouped. So the leak was the *actual* behaviour, not a dormant branch. `scikit-learn>=1.3` is now pinned in `modern_environment.yml`, and the fallback (`_numpy_group_split`) is itself group-aware, so a missing optional dependency can no longer reintroduce the leak. It drops stratification, not grouping — the right thing to sacrifice.
* The split **asserts** the no-overlap invariant and raises rather than writing a corpus whose validation score would be inflated. A silent leak is the failure mode this code exists to prevent, so it must not be able to fail quietly.

Verified on the real distribution — both paths, 0 leaks, ratios honoured:

| path | train/val/test | leaks |
|---|---|---|
| `StratifiedGroupKFold` | 682 / 85 / 86 (0.800/0.100/0.101) | **0** |
| `_numpy_group_split` fallback | 683 / 85 / 85 (0.801/0.100/0.100) | **0** |
| old ungrouped (for contrast) | — | **12 groups** |

### Group-aware train/val split (original requirement note)

The split must be grouped **by slug**, never random over files. Even after curation, up to `--max_per_slug` correlated variants share a slug; a random split leaks them across train and val and inflates val F1. This has to be enforced wherever the trial6 split is made.

### Phase 2 — Bulk download (superseded by Phase 2b; kept for history)

GrabCAD requires an authenticated session. The download helper reads a session cookie from `~/.grabcad_session` (single line: full `_grabcad_session=...` cookie value pasted from an authenticated browser session). See `brepformer/data/download_grabcad.py`.

```bash
python brepformer/data/download_grabcad.py \
    --links_md LINKS.md \
    --out_dir brepformer/data/hard_samples/ \
    --workers 4 \
    --max_size_mb 64
```

The downloader:
1. Parses all `https://grabcad.com/library/<slug>` URLs from LINKS.md
2. Hits GrabCAD's API to fetch the model download ZIP
3. Extracts only files with `.step` / `.stp` extension (case-insensitive)
4. Deposits them into `--out_dir` with the slug as filename
5. Skips ZIPs with total unzipped size > `--max_size_mb`
6. Logs per-slug success/failure to `hard_samples_download.log`

### Phase 5 — Merge triaged samples into suitable_samples/

Automated by `brepformer/data/merge_suitable_samples.py`:

```bash
# Merge yes-only (recommended first pass)
python brepformer/data/merge_suitable_samples.py \
    --triage_root brepformer/data/hard_samples \
    --inference_dir brepformer/inference/hard_samples_27 \
    --out_dir brepformer/data/suitable_samples \
    --include yes

# Or yes + maybe together for a larger corpus
python brepformer/data/merge_suitable_samples.py \
    --triage_root brepformer/data/hard_samples \
    --inference_dir brepformer/inference/hard_samples_27 \
    --out_dir brepformer/data/suitable_samples \
    --include yes maybe
```

The script writes mftrcad-format label JSONs (`{model_id}_result.json`) so
`preprocess.py` accepts the output directory without any adapter.

Then combine `suitable_samples/` with `mftrcad/`:

```bash
# Build a merged directory. Symlinks avoid duplication; the preprocess step
# only reads STEP + label JSON so links are fine.
mkdir -p brepformer/data/merged_train/{steps,labels}
for src in brepformer/data/mftrcad brepformer/data/suitable_samples; do
    for sub in steps labels; do
        for f in $src/$sub/*; do
            ln -sf "$(realpath "$f")" "brepformer/data/merged_train/$sub/$(basename "$f")"
        done
    done
done
```

### Phase 6 — Merged training

```bash
# Preprocess merged dataset (assumes suitable_samples/ and mftrcad/ are combined into merged_train/)
python brepformer/preprocess.py \
    --data_dir brepformer/data/merged_train \
    --output_dir brepformer/data/merged_train_processed \
    --compute_descriptors --d2_bins 64 --angle_bins 64

# Train on merged dataset (start from trial5 weights for a warm start)
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/merged_train_processed \
    --face_segmentation \
    --face_seg_weight 2.0 --model_cls_weight 0.5 \
    --max_faces 500 --max_epochs 300 \
    --batch_size 32 --learning_rate 0.0005 --num_workers 4 --precision "32" \
    --resume_from "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --output_dir results --exp_name trial6_merged
```

## Notes / decisions

- Pseudo-labels come from the **27-class** trial5 model, not the 5-class defeature model, because the goal is to grow the 27-class MFTRCAD training set. The 5-class predictions are only used to guide the defeaturing preview.
- We store inference JSONs as label files. `infer.py` writes labels in a compatible schema; if the on-disk format differs from mftrcad's `_result.json`, a small converter step is added in Phase 5 (see `brepformer/data/classes.py` and existing `_result.json` for reference).
- 64 MB is a soft ceiling on STEP size (see LINKS.md preamble). The downloader enforces this on the unzipped STEP size.
- If the user has no GrabCAD session cookie set up, Phase 2 pauses with a clear message rather than 550 failed HTTP requests.

## Resume points

- If interrupted after Phase 1 inference: rerun `visualize_*` commands; JSON/SEG outputs are already on disk.
- If interrupted mid-Phase-2 download: the downloader skips slugs whose output file exists.
- If interrupted before Phase 5 selection: the plan's "Suitable samples" section captures what was checked; resume from the next unchecked entry.
- If interrupted mid-Phase 4 reclassification: skip-existing logic means rerun picks up where it left off.

---

## Phase 4 outcome — visual reclassification is ABANDONED (measured 2026-08-30)

**Do not run `reclassify_visual.py` as a label source.** Both of its correction
mechanisms were measured against mftrcad ground truth and both make labels worse.
This section exists so the result is not rediscovered by re-running the pass.

### Why it could not have worked before, either

`query_claude_vision` returned `None` when `ANTHROPIC_API_KEY` was unset, and
`reclassify_single` treats `None` as "no opinion" and keeps the original label. This
machine has no `ANTHROPIC_API_KEY` — it authenticates via Bedrock. A full run would
have printed no error, written plausible output for all 853 models, and produced
**zero** visual relabelling. Now fixed: `_vision_client()` supports Bedrock and
*raises* on missing credentials, before any rendering starts.

### 1. Geometric rules corrupt correct labels

`brepformer/data/validate_geometric_rules.py`, 60 models / 1412 labelled faces.
Ground-truth labels fed in as predictions, so every DENY is provably false:

| true class | n | falsely denied | rewritten to |
|---|---|---|---|
| `chamfer_2` | 53 | **100%** | `through_hole` |
| `rectangular_passage_2` | 70 | **100%** | `through_hole` |
| `circular_end_pocket` | 86 | 59% | `stock` |
| `blind_hole` | 24 | 54% | `stock` |
| `stock` | 499 | **0%** | — |

**15.0% of correct labels overwritten overall.** Cylindrical → `through_hole`,
unplaceable → `stock`. Safe on stock, destructive on rare curved features — it would
take chamfer from dead to actively mislabelled. Hence `--geo_mode advise` (default):
denials only *select* faces for review, never rewrite. The rule's suggested class is
also kept out of the vision prompt on purpose, so its systematic errors cannot anchor
the vision model toward the same wrong answers.

### 2. Claude vision is at or below chance on feature faces

Three measurements, `validate_vision_labeling.py` (verify) and
`validate_vision_classify.py` (classify), 5-class defeature space, rare-biased
sampling, ~100-120 faces each:

| setup | harm | fix | net |
|---|---|---|---|
| verify, original renders | 76% | 39% | −0.37 / face |
| verify, fixed renders | 74% | 39% | −0.35 / face |

| classify (no verify anchor) | accuracy |
|---|---|
| all sampled faces | 30.8% |
| **excluding stock/random** | **17.3%** (chance = 20%) |

Per-class recall: `hole` 70%, `fillet` 67%, `chamfer` 10%, `cut` 3%. Failure mode is
consistent — feature faces collapse to `random`/`stock` (`cut→random` 22×,
`chamfer→random` 16×).

Note that DENY-rate alone is badly misleading: 91% of *wrong* labels were denied,
which looks like a working detector, but only 39% were replaced with the right class.
The rest is churn — one wrong label swapped for another. Always measure harm and
benefit together (see `validate_*.py` docstrings).

### Two real rendering bugs found and fixed (verdict unchanged)

Worth keeping, and worth knowing they were *not* the cause:

1. **No zoom to the face.** Views framed the whole model, so a real mftrcad chamfer
   was 280 of 360,000 px (0.078%, a 1px line) in its *best* view. A first attempt
   scaled *up* from the bbox's largest dimension — for a thin annular chamfer that is
   the ring diameter, not the chamfer width, so it re-framed the whole part and
   achieved nothing (0.078% → 0.16%). Framing slightly *inside* the bbox works:
   0.078% → 2.57% (33x), 5.9% → 18.4%.
2. **Views picked by camera order, not visibility.** Only the first 4 of 6 images are
   sent, in fixed camera order, so the face could be occluded in every image sent.
   Now ranked by highlight pixel count, with one whole-model context view kept first.

After both fixes, harm moved 76% → 74% and fix stayed at 39%. Visibility was real but
not the dominant cause.

### Performance fix (keep)

`render_face_highlighted` re-ran `load_step` + `align_shape` + `BRepMesh` and rebuilt
every triangle *per face* — 50 identical tessellations for a 50-face review, ~60 h
across the corpus. Now tessellated once per model and recoloured per face
(`_tessellate`, size-1 cache). Verified **byte-identical** PNGs, 2.3x faster on 4
faces, more at 50.

### What this means for the corpus

Train on the corpus **as it stands**: 27-class trial5 pseudo-labels with per-face
confidence masking at 0.7. That masking is the one label-quality mechanism here that
is defensible — it *removes* unreliable supervision rather than inventing replacement
labels, so it cannot introduce a wrong label.

The dead chamfer class remains unsolved, and visual relabelling is not the lever.
Remaining options, in rough order of expected value:
- class-balanced or focal loss on the 27-class head (cheap, no new labels);
- targeted hand-labelling of a small chamfer-rich set (expensive, reliable);
- oversampling mftrcad models that actually contain chamfer_2 ground truth.

### Methodological rule this established

Never let a component write labels on the pseudo-labelled corpus until its harm and
benefit rates are measured on mftrcad ground truth. On pseudo-labels a bad heuristic
is invisible — there is no correct label to disagree with, so it emits
confident-looking output and silently poisons the corpus. Both components here looked
productive by their own metrics (a third of faces "corrected"; 91% of wrong labels
"denied") while degrading label quality.

---

## Phase 5 — per-class masking replaces the global 0.7 threshold (2026-08-31)

Supersedes the "train on the corpus as it stands" recommendation above, and corrects
the claim in *What this means for the corpus* that confidence masking "cannot introduce
a wrong label". It cannot introduce a wrong *label*, but a **global** threshold silently
deletes whole classes, which is its own kind of corpus damage.

### What was wrong with one global cutoff

The teacher's confidence scale is not comparable across classes, so a single number
does not express one quality bar — it expresses "delete whichever classes this teacher
is shakiest on". Measured on the 853-model corpus at `--min_confidence 0.7`, keep rates
spanned two orders of magnitude:

| class | raw faces | kept @0.7 | keep rate |
|---|---|---|---|
| `rectangular_passage` | 4,758 | 3,893 | 81.8% |
| `round` | 14,583 | 11,307 | 77.5% |
| `chamfer` | 15,395 | 9,913 | 64.4% |
| `stock` | 8,550 | 1,871 | **21.9%** |
| `circular_blind_slot` | 6,113 | 167 | **2.7%** |
| `triangular_blind_step` | 2,390 | 22 | **0.9%** |
| `rectangular_blind_slot` | 277 | 1 | **0.4%** |

Three classes effectively annihilated, and `stock` — the majority class on real parts —
kept at 21.9%, which biases the student toward over-segmenting features. The
supervision was never missing; the merge was discarding it. `circular_blind_slot` had
6,113 raw faces on disk and contributed 167.

**The direction of the damage flips with the teacher**, which is the real argument
against relying on any global number. With the 27-class trial5 teacher, masking at 0.7
*raised* chamfer's share (median confidence 0.751). With the 5-class
`trial2_finetune_ss` teacher, chamfer was the weak class: mean confidence 0.606, only
35% of its predictions clearing 0.7 versus 73.5% for fillet, so its share went 5.68% →
3.80% while fillet's went 20.1% → 28.3%. Self-distillation then ratchets: masked class
→ no gradient → stays weak → masked harder. `trial2_finetune_ss` is 4.7x worse at
chamfer than the `trial1_ss` it descends from, while posting a *better* headline val F1.

### The fix (in the merge, not the loss)

`--per_class_top_frac` fits a cutoff per predicted class over the whole corpus in a
first pass and keeps the top N% *within each class*, equalising selectivity so weak
classes are thinned rather than erased. `--per_class_floor` refuses faces below an
absolute confidence even if a class is under quota. `--max_class_frac` optionally caps
any one class's share (`Oring` was 48.5% of kept faces). One mechanism fixes the rare
classes and `stock` together.

At `--per_class_top_frac 0.5 --per_class_floor 0.5` → `suitable_samples_pcq/`:
752 models, 103,448 supervised faces, **25 classes** (was 22). `circular_blind_slot`
167 → 1,268; `triangular_blind_step` 22 → 162; `stock` 1.5% → 2.9% share; `Oring`
48.5% → 40.0%. Floor 0.5 is the knee — it lifts the weakest admitted class from mean
confidence 0.496 → 0.598 for the cost of one class and 3.3k faces.

### OPEN QUESTION that trial6 decides — read this before interpreting the result

The recovered faces are **low-confidence by construction**: the classes that gained
most are admitted at ~0.60 mean top-1 confidence, versus ~0.93 for `Oring` or `round`.
In a 27-way problem that is well above chance but nowhere near reliable. So this trade
is deliberate and **unvalidated**: we exchanged fewer total faces (125,542 → 103,448)
and noisier labels on the rare classes for those classes existing at all.

Whichever way trial6 lands, the interpretation is already fixed — do not re-litigate it
after seeing the number:

- **If trial6 improves rare-class F1** (vs trial5 on the same held-out split): the
  trade paid off, noisy-but-present beats absent, and the next step is to push
  `--per_class_top_frac` higher / `--per_class_floor` lower and re-measure. There is
  more supervision still being discarded.
- **If trial6 degrades**, especially if the *strong* classes (`chamfer`, `round`,
  `Oring`, `rectangular_passage` — all trimmed to 0.6-0.8x here) get worse: the cause
  is most likely the noisy admissions, not the rebalancing. Do **not** conclude
  "per-class masking was wrong" and revert to a global threshold — that reintroduces
  the annihilation. Raise `--per_class_floor` first (0.6/0.65 were measured and cost
  little), and if that does not recover it, the principled fix is
  **soft-label distillation**: train on the teacher's full probability vector with
  KL divergence instead of hard argmax + `-1` mask. That expresses "the teacher thinks
  this is 60% a blind slot" honestly, whereas hard labels assert certainty the teacher
  never had, and masking throws the information away entirely. It is a loss change
  (`train_preprocessed.py`), which is why it was not done here.
- **If aggregate val F1 rises but rare-class F1 falls**, that is the failure mode this
  whole phase exists to catch. Aggregate F1 *increases* as weak classes die, because
  the classes that die are the hard ones. Score trial6 on per-class F1
  (`brepformer/eval_per_class.py --focus <class>`), never on the headline number.

Also enable `--weighted_crossentropy` — it defaults to **False** in
`train_preprocessed.py`, so nothing currently counteracts any of this.

### Corpus-size decision (2026-08-31): train, do not collect

Considered pausing to scrape another 500-1000 STEP files; rejected on the evidence.
The rare-class deficit was never scarcity — fixing the merge recovered 7.6x more
supervision from the *same* files than doubling the corpus would have under the old
threshold. More unlabelled parts add zero label information, since the ceiling is
trial5's accuracy on real parts and that does not move when parts are added. Only 2 of
27 classes are genuinely availability-limited (`6sides_pocket` n=1,
`triangular_pocket` n=68), and they are synthetic-taxonomy shapes that random GrabCAD
parts will not supply — that is a case for targeted labelling, not bulk collection.
Cheaper lever first: `--min_kept_frac` 0.25 → 0.15 yields 832 models / 109,945 faces
instead of 773 / 106,480, with no collection at all.

Checked and rejected: merging all 1,323 inferred models (1,071 pass the gate → 177,527
faces) looks like a 41% free gain, but ~470 of those were removed by
`curate_corpus.py` as duplicates and parametric-family members. Re-adding them buys
redundancy and train/val leakage, not diversity.
