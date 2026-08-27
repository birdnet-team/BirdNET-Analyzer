# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(1, os.path.abspath(".."))

project = "BirdNET-Analyzer"
copyright = "%Y, BirdNET-Team"
author = "Stefan Kahl"
# Overridden with -D version=... when the docs workflow builds a release tag.
try:
    release = version = _dist_version("birdnet_analyzer")
except PackageNotFoundError:
    release = version = "dev"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinxarg.ext",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "_site", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# :github_url: is meta data used to force the "Edit on GitHub" link to point to the exact url.
# https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html#file-wide-metadata
rst_prolog = ":github_url: https://github.com/birdnet-team/BirdNET-Analyzer\n"
html_theme = "sphinx_rtd_theme"
html_favicon = "_static/birdnet-icon.ico"
html_logo = "_static/birdnet_logo.png"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
# Version switcher served from the gh-pages site root (docs/_site/switcher.js).
# One shared copy serves every published version; the script shows nothing on
# pages that are not under the published site (e.g. local builds).
html_js_files = ["https://birdnet-team.github.io/BirdNET-Analyzer/switcher.js"]
html_theme_options = {"style_external_links": True, "navigation_depth": 2}
html_show_sourcelink = False
html_show_sphinx = False
html_extra_path = ["projects.html", "projects_data.js"]
