import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
# Project information
project = 'CleverHans'
copyright = '2025, Tamar Noselidze'
author = 'Tamar Noselidze'

# Extensions
extensions = [
    'sphinx.ext.autodoc',        # Auto-documentation from docstrings
    'sphinx.ext.viewcode',       # Add links to source code
    'sphinx.ext.napoleon',       # Support for NumPy and Google style docstrings
    'sphinx.ext.intersphinx',    # Link to other project's documentation
    'sphinx_rtd_theme',          # ReadTheDocs theme
    'sphinx.ext.todo',           # Support for TODO items
    'sphinx_autodoc_typehints',  # Use type hints for documentation
    'sphinx_copybutton',         # Add copy button to code blocks
]

# Theme configuration
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
}

templates_path = ['_templates']

exclude_patterns = []

htmlhelp_basename = 'CleverHansDoc'

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None


autodoc_member_order = 'bysource'