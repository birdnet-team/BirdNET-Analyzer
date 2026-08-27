# AGENTS.md

Guidance for coding agents working in this repository. It is the single source of
truth for agent instructions — `CLAUDE.md` just points here.

Personal or machine-specific preferences (which virtualenv to use, whether an agent
may commit) do **not** belong in this file. Put those in an untracked
`CLAUDE.local.md` / `AGENTS.local.md`, which `.gitignore` already covers.

## Setup

Python >= 3.11 (CI tests 3.11–3.13 on Ubuntu, macOS and Windows).

```bash
pip install -e .[dev]                              # ruff + pytest + docs
pip install -e .[dev,embeddings,train,gui-tests]   # what CI installs
git submodule update --init --recursive            # test audio fixtures
```

Extras: `gui`, `train`, `embeddings`, `docs`, `tests`, `gui-tests`, `dev`, `all`.

The test audio lives in the `tests/data` submodule; without it most tests cannot run.
`CONTRIBUTING.md` covers updating the submodule ref.

### Models are downloaded on first use

Inference comes from the `birdnet` dependency, which downloads model files on first
use — roughly 120 MB (acoustic), 35 MB (geo) and 380 MB (perch-v2). Two consequences
worth knowing before running the suite:

- `birdnet` reads `BIRDNET_APP_DATA` **at import time** and puts the downloads there,
  skipping any file that already exists. CI sets it to a workspace path and caches
  that directory; `Dockerfile` bakes the models into the image the same way.
- Every test has a 120s timeout, and a cold download can exceed it on a slow link.
  CI therefore pre-downloads outside pytest (`ci.yml`, "Pre-download birdnet models").
  Do the same locally if a first run times out:
  `python -c "import birdnet; birdnet.load('acoustic', '2.4', 'tf', lang='en_us')"`

## Commands

- **Tests**: `python -m pytest`. Single test:
  `python -m pytest tests/analyze/test_analyze.py::test_name`. Config lives in
  `[tool.pytest.ini_options]`.
- **Lint**: `ruff check` and `ruff format`. Line length 88. Ruff is pinned to the same
  version in the `[dev]` extra and in `.github/workflows/lint.yml` — bump both
  together.
- **Docs**: `pip install -e .[docs]` then `sphinx-build -E docs _build`. The CLI
  reference is generated from `birdnet_analyzer/cli.py` via sphinx-argparse, which is
  why the docs workflow also triggers on that file.
- **Run the GUI**: `python -m birdnet_analyzer.gui`
- **Run a CLI**: `python -m birdnet_analyzer.analyze` (likewise `.species`,
  `.segments`, `.embeddings`, `.search`, `.train`); installed as `birdnet-analyze`
  etc., plus `birdnet-evaluate` and the `birdnet-gui` GUI script.

## CI

Six workflows in `.github/workflows/`: `lint.yml` and `ci.yml` (the test matrix) run
on every PR to `main`, `documentation.yml` on docs changes and on release,
`docker-build.yml` on Dockerfile/`pyproject.toml` changes and on release, and
`publish.yml` / `test-publish.yml` on release. All of them use `paths:` filters, so a
PR touching only docs will not run the test matrix.

The published docs are **versioned**: `documentation.yml` deploys each release to
`/vX.Y.Z/` on `gh-pages` (mirrored at `/stable/`, the default landing spot), pushes
to `main` deploy to `/dev/`, and old releases can be backfilled via the workflow's
manual trigger (`tag`, plus `set_stable` when that tag is the newest release).
Prereleases get their own directory but never become `/stable/`. The site-root
redirect, 404 handler and version switcher live in `docs/_site/`, which is deployed
to the `gh-pages` root and excluded from the Sphinx build.

Two things worth knowing before touching this:

- **`/stable/` only exists once a release has been deployed to the versioned site.**
  Until then the root redirect and the 404 handler fall back to the newest version
  directory and then to `/dev/`, so the site still works, but it is serving
  unreleased docs. Backfill the current release (manual trigger, `set_stable`
  ticked) right after the versioning workflow first lands on `main`.
- **Backfilling only reaches `v2.1.0`–`v2.4.0`.** `v2.0.0` and `v2.0.0-rc` have a
  `docs/conf.py` but no `docs` extra, so no Sphinx gets installed; every `v1.x` tag
  predates `pyproject.toml` entirely. (The bare `1.4.0` tag is also rejected by the
  workflow's `vX.Y.Z` pattern.)

Run `ruff check` and `python -m pytest` before handing work back.

## Architecture

Python package (`birdnet_analyzer`) built on top of the `birdnet` library, which
provides the actual model inference.

- **Feature subpackages** (`analyze/`, `embeddings/`, `search/`, `species/`,
  `segments/`, `train/`): each follows the pattern `core.py` (public API, re-exported
  in `__init__.py`), `cli.py` (argparse entry point), `__main__.py`, plus an optional
  `utils.py`. Shared argparse helpers live in the top-level `cli.py`.
- **`evaluation/` is the exception**: it has no `core.py`/`cli.py` pair. Its `main()`
  lives in `__init__.py` and the logic sits under `assessment/` and `preprocessing/`.
  Don't assume the other subpackages' shape when working there.
- **Shared top-level modules**: `config.py` (global constants/defaults), `audio.py`,
  `model.py`/`model_utils.py`, `utils.py`, `settings.py` (paths, persisted settings),
  `logs.py` (logging setup — the package itself only attaches a `NullHandler` and
  leaves configuration to the CLI/GUI entry points), `params.py` (reads the
  `*-params.csv` files a run writes back into keyword arguments; shared by the CLI,
  where they become argument defaults, and the GUI).
- **GUI** (`gui/`): Gradio app wrapped in a pywebview window. One module per tab
  exposing a `build_*_tab` function, assembled in `gui/__init__.py:main` via
  `gui.utils.open_window`.
- **Tests** (`tests/`): mirror the package layout, e.g. `tests/analyze/`,
  `tests/gui/`. There is no `conftest.py`.

### State that lives outside the repo

`settings.APPDIR` resolves to a per-OS user data directory named
`BirdNET-Analyzer-GUI` (`%APPDATA%` on Windows, `~/.local/share` on Linux,
`~/Library/Application Support` on macOS). GUI settings (`gui-settings.json`),
tab state (`state.json`) and the log files live there, **not** in the checkout. So GUI
behaviour can differ between machines because of leftover local state, and deleting
that directory is the way to test a first-run experience.

## Conventions that are easy to get wrong

- **The GUI test environment has no pywebview.** CI's `gui-tests` extra installs
  gradio but not pywebview, so any GUI module reachable from a test must not
  `import webview` at module import time.
- **Localization is all-or-nothing.** `lang/*.json` holds one file per language
  (currently 10), located via `settings.LANG_DIR`. A new UI string has to be added to
  *every* language file with a real translation — not the English text copied over.
  Write the files with a json load/dump round-trip
  (`ensure_ascii=False, indent=4, sort_keys=True`). `tests/gui/test_language.py`
  enforces that formatting, key parity across languages, and that each translation
  keeps the `str.format` fields of its `en.json` source.
- **Don't widen the ruff config to make a fix pass.** The `select`/`ignore` lists in
  `pyproject.toml` are deliberate.
- **Packaging is an allow-list, not a deny-list.** What ships is decided by the
  explicit `packages` and `package-data` lists in `[tool.setuptools]`; a **new
  subpackage or data file has to be added there or it silently won't be installed**.
  Arbitrary top-level files are not distributed, so they need no `MANIFEST.in` entry
  (`prune tests` is the one line there that actually does work). `.dockerignore` is
  separate and does need updating for new top-level cruft.
