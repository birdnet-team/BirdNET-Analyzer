# Review instructions for BirdNET-Analyzer

Python package (`birdnet_analyzer`) wrapping the `birdnet` library, with a CLI per
feature and a Gradio/pywebview GUI. General agent guidance lives in `AGENTS.md`; the
checks below are the diff-visible rules to apply when reviewing a PR. Everything here
can be judged from the diff text alone.

## Check on every PR

- **Localization is all-or-nothing.** A diff that adds a key to
  `birdnet_analyzer/lang/en.json`, or a `loc.localize("...")` call with a new key,
  must add that key to **all** files in `birdnet_analyzer/lang/` (currently 10) with a
  real translation — English text copied into `de.json` is a defect, not a
  placeholder. The files stay sorted with 4-space indent (a json load/dump round-trip
  with `ensure_ascii=False, indent=4, sort_keys=True`), and every translation keeps
  the `str.format` fields of its English source. `tests/gui/test_language.py` enforces
  all three, so flag it as a test that will fail rather than as a style opinion.
- **No module-level `import webview` in `birdnet_analyzer/gui/`.** CI's `gui-tests`
  environment installs gradio but not pywebview. A new top-level `import webview` in
  any GUI module reachable from a test breaks collection for the whole suite. Import
  it inside the function that needs it.
- **New subpackages and data files must be registered.** Packaging is an allow-list:
  a new directory under `birdnet_analyzer/` needs an entry in the `packages` list of
  `[tool.setuptools]`, and a new non-`.py` resource needs one in
  `[tool.setuptools.package-data]`. Miss it and the code is fine in a checkout but
  absent from an installed package — CI will not catch this.
- **Analysis/training parameters must round-trip.** `utils.save_params_file` writes a
  run's settings to `*-params.csv`, and `params.py` reads them back into keyword
  arguments through an explicit `parse("name", type, "Column header")` line per
  parameter. A diff that adds a parameter to `analyze()` or `train()` and writes it to
  the params file, but adds no matching `parse(...)` line, silently loses that setting
  when a previous run is loaded.
- **New CLI parsers need a docs section.** `docs/usage/cli.rst` renders each parser
  through sphinx-argparse. New *arguments* on an existing parser appear automatically
  and need no doc change; a new *parser/entry point* needs its own section there and
  an entry in `[project.scripts]`.
- **Don't accept a widened lint config as a fix.** Additions to `ignore`, removals
  from `select` in `[tool.ruff.lint]`, or new `# noqa` / `# ruff: noqa` comments need
  a stated reason in the PR. If ruff's version is bumped in the `[dev]` extra, the pin
  in `.github/workflows/lint.yml` has to move with it — they are two hand-maintained
  copies of one number.

## Context to avoid false positives

- **A cancelled "Tests" run is not a failure.** `ci.yml` sets
  `cancel-in-progress` for pull requests, so pushing a new commit kills the run for
  the previous one. Judge the newest run for the head SHA, not a cancelled older one.
- **Windows jobs take roughly three times as long as Linux** (about 6 minutes vs 2).
  That is the runner, not a performance regression introduced by the PR.
- **Model downloads dominate a cold run.** The `birdnet` dependency fetches roughly
  120 MB (acoustic), 35 MB (geo) and 380 MB (perch-v2) on first use, cached via
  `BIRDNET_APP_DATA`. Editing `pyproject.toml` changes the cache key, though the
  prefix restore-key usually still hits. A slow job or a test tripping the 120s
  per-test timeout right after a dependency change is expected, and is not evidence
  that the PR made the code slower.
- **Style and formatting are ruff's job** (line length 88). Do not comment on
  formatting, import order, or quote style that the pinned ruff version already
  governs.
- **Test audio comes from the `tests/data` submodule.** A diff that appears to
  reference fixture files that are not in the diff is normal; those live in the
  submodule, not in this repository.
