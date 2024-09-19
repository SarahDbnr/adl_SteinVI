# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

# Add the stein_vi and its subdirectories to sys.path
sys.path.insert(0, os.path.abspath('../stein_vi'))
sys.path.insert(0, os.path.abspath('../stein_vi/parameter_search'))
sys.path.insert(0, os.path.abspath('../stein_vi/metrics'))
sys.path.insert(0, os.path.abspath('../stein_vi/algorithm'))
sys.path.insert(0, os.path.abspath('../stein_vi/data'))

project = 'Bayesian Neural Networks through Stein VI in JAX'
copyright = '2024, Sarah Deubner, Kilian Runnwerth and Luke-Liam Bergmeier'
author = 'Sarah Deubner, Kilian Runnwerth and Luke-Liam Bergmeier'
release = '23.09.2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon', # Support for Google-style docstrings
    'myst_parser',         # Support for markdown
]

templates_path = ['_templates']
exclude_patterns = []
autoclass_content = "class"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme_options = {
    'collapse_navigation': False,  # Keep the sidebar open and expand the current section
}