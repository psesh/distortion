import os
import sys

# Misc Sphinx settings
#########################
project = u'distorted'
copyright = u'2023 by Georgia Tech'
author = u''
outdir = sys.argv[-1]   #Bryn: Assuming last system argument is output directory
version = u'9.1'
release = u'v9.1.0'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'nbsphinx',
] #TODO - prune unused extensions
master_doc = 'index'
html_title = 'Distorted'
html_theme = 'pydata_sphinx_theme'
language = None
exclude_patterns = [u'_docsbuild', 'Thumbs.db', '.DS_Store']
source_suffix = '.txt'
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '12pt',
}


# Templates, static files, and css (bug with css being copied to _static/css, this must be done manually for now)
#########################
templates_path = ['_templates']
html_static_path = ['_static']
html_css_files = [
        'css/styles.css',
        'css/footer.css',
        'css/fonts.css'
]


# Sphinx panels
#########################
panels_add_bootstrap_css = False # sphinx-panels shouldn't add bootstrap css since the pydata-sphinx-theme already loads it

# nbsphinx setting
#########################
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=96",
]
nbsphinx_input_prompt = 'In [%s]:'
nbsphinx_output_prompt = 'Out[%s]:'



# sphinxcontrib.bibtex setting
##############################
bibtex_bibfiles = ['references.bib']

# pydata theme options
#########################
html_theme_options = {
    "show_prev_next": False,
    "search_bar_text": "Search documentation",
}
