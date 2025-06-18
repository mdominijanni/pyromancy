# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Pyromancy"
copyright = "2025, Marissa Dominijanni"
author = "Marissa Dominijanni"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_design",
    "sphinx_remove_toctrees",
]

myst_enable_extensions = ["amsmath", "dollarmath", "smartquotes", "colon_fence"]
add_module_names = False
autosectionlabel_prefix_document = True
myst_heading_anchors = 2

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

remove_from_toctrees = ["api-reference/generated/*"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_title = "Pyromancy"
html_logo = "_static/images/pyromancy-logo.svg"

html_css_files = ["css/shape.css"]
