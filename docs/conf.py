# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from toml import load
sys.path.insert(0, os.path.abspath('../'))
project = 'PCA_for_time_series'
copyright = '2025, Samuel Gruffaz, Thibaut Germain'
author = 'Samuel Gruffaz, Thibaut Germain'
release = load(os.path.abspath("../pyproject.toml"))["project"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx.ext.mathjax",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
# autodoc_mock_imports=['jax',
# 'optax',
# 'matplotlib',
# 'numpy',
# 'scipy',
# 'pandas',
# 'scikit-learn']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
