# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(1, os.path.abspath(".."))

project = "BirdNET-Analyzer"
copyright = "%Y, BirdNET-Team"
author = "Stefan Kahl"
version = "1.5.1"

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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

rst_prolog = ":github_url: https://github.com/birdnet-team/BirdNET-Analyzer\n"
html_theme = "sphinx_rtd_theme"
html_favicon = "_static/birdnet-icon.ico"
html_logo = "_static/birdnet_logo.png"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_theme_options = {"style_external_links": True}
html_show_sourcelink = False
html_show_sphinx = False
html_extra_path = ["projects.html", "projects_data.js"]
