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

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme_options = {
    'collapse_navigation': False,  # Keep the sidebar open and expand the current section
    'style_external_links': True,  # Improve external link styling
    'navigation_depth': 4,         # Set how many levels of TOC to display in the sidebar
    'prev_next_buttons_location': 'bottom',  # Display navigation buttons at the bottom
    'body_max_width': '80%',       # Use 80% of the available width, adjust as needed
}

html_static_path = ['_static']
html_css_files = [
    'custom.css',  # Custom CSS for additional width control
]