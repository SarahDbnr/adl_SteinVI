# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

# Add the src and its subdirectories to sys.path
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('../src/parameter_search'))
sys.path.insert(0, os.path.abspath('../src/metrics'))
sys.path.insert(0, os.path.abspath('../src/algorithm'))
sys.path.insert(0, os.path.abspath('../src/data'))

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



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}