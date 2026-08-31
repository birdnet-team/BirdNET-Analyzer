# AGENTS.md

Guidance for coding agents working in this repository. It is the single source of
truth for agent instructions — `CLAUDE.md` just points here. It holds only what the
codebase does not already state: anything you can read off `pyproject.toml`, a
workflow file or a directory listing is deliberately not repeated here.

Personal or machine-specific preferences (which virtualenv to use, whether an agent
may commit) belong in an untracked `CLAUDE.local.md` / `AGENTS.local.md`, which
`.gitignore` already covers.

## Setup

`pip install -e .[dev]` gets ruff, pytest and the docs toolchain; CI additionally
installs `embeddings,train,gui-tests`.

**The test audio is a git submodule.** Without
`git submodule update --init --recursive` most tests cannot run. `CONTRIBUTING.md`
covers updating the ref.

**Models download on first use.** Inference comes from the `birdnet` dependency,
which fetches roughly 120 MB (acoustic), 35 MB (geo) and 380 MB (perch-v2) into the
directory named by `BIRDNET_APP_DATA` — **read at import time** — skipping files
that already exist. CI points it at a cached workspace path; `Dockerfile` bakes the
models into the image. Tests have a 120 s timeout that a cold download can exceed,
which is why CI pre-downloads outside pytest. Do the same locally if a first run
times out:
`python -c "import birdnet; birdnet.load('acoustic', '2.4', 'tf', lang='en_us')"`

## Before handing work back

Run `ruff check` and `python -m pytest`. **The ruff version is pinned in two
places** — the `[dev]` extra and `lint.yml` — and has to be bumped in both.

## Architecture

`birdnet_analyzer` wraps the `birdnet` library, which does the actual inference.

Feature subpackages (`analyze/`, `embeddings/`, `search/`, `species/`, `segments/`,
`train/`) follow `core.py` (public API, re-exported in `__init__.py`) + `cli.py` +
`__main__.py`. **`evaluation/` does not**: its `main()` lives in `__init__.py` and
the logic sits under `assessment/` and `preprocessing/`.

`params.py` reads the `*-params.csv` files a run writes back into keyword arguments,
shared by the CLI (where they become argument defaults) and the GUI. `logs.py`
leaves logging configuration to the CLI/GUI entry points — the package itself only
attaches a `NullHandler`. The GUI is a Gradio app in a pywebview window, one module
per tab exposing `build_*_tab`, assembled in `gui/__init__.py:main`.

### State that lives outside the repo

`settings.APPDIR` is a per-OS user data directory (`BirdNET-Analyzer-GUI`) holding
GUI settings, tab state (`state.json`) and the logs — **not** the checkout. GUI
behaviour can therefore differ between machines because of leftover local state, and
deleting that directory is how you test a first-run experience.

## Conventions that are easy to get wrong

- **The GUI test environment has no pywebview and no plotly** — `gui-tests` installs
  gradio, both others are `gui`-only. So no GUI module may import `webview` or
  `plotly` at module import time: `gui/utils.py` imports `webview` inside the dialog
  functions and `open_window` only, and its `_WINDOW` annotation is a string under
  `TYPE_CHECKING`. The tests import `gui.utils` unstubbed on purpose. A test that
  builds species-list controls must stub `gu.plot_map_scatter_mapbox` (it draws a
  plotly map). A local venv with the `gui` extra hides both problems, so run GUI
  tests once with `webview`/`plotly` made unimportable before pushing.
- **Localization is all-or-nothing.** `lang/*.json` holds one file per language. A
  new UI string has to be added to *every* file with a real translation — not the
  English text copied over. Write them with a json load/dump round-trip
  (`ensure_ascii=False, indent=4, sort_keys=True`); `tests/gui/test_language.py`
  enforces that formatting, key parity across languages, and that each translation
  keeps the `str.format` fields of its `en.json` source.
- **A GUI setting has three ways to get its value, not one.** Every persisted
  control (`TabState.persist`) is set (1) at build time from `state.json`, (2) by the
  user, and (3) programmatically by presets and loaded `*-params.csv` files
  (`TabState.updates_for`, which sets `value` only). So when one control depends on
  another (the sensitivity slider disabled for BirdNET 3.0/Perch, the locale dropdown
  offering only the model's languages, …), cover all three: pass the dependency into
  the builder for the initial state, handle it in a `.change` handler (gradio fires
  it for programmatic updates too — `.input` does not), and have the handler *reset*
  the dependent value when the loaded one is no longer valid, since the batch update
  landed before the handler ran. Doing only (2) is the classic gap.
- **Code comments: only when the code cannot say it.** Default to none. A comment
  earns its place for a constraint, a non-obvious *why*, or a measured value that
  justifies a bound. No history ("used to", "previously"), no narration of the next
  line — that belongs in commit messages and the changelog. Same bar for tests: the
  test name and its assertions are the explanation.
- **Don't widen the ruff config to make a fix pass.** The `select`/`ignore` lists in
  `pyproject.toml` are deliberate.
- **Packaging is an allow-list.** `[tool.setuptools]`'s `packages` / `package-data`
  decide what ships, so a new subpackage or data file that is not listed there is
  silently missing from the install. `.dockerignore` is separate and needs its own
  update for new top-level cruft.
