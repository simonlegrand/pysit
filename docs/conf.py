# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Portions licenesed under a 3-clause BSD style license - see
# ASTROPY_SPHINXEXT_LICENSE.rst
#
# PySIT documentation build configuration file, created by
# sphinx-quickstart on Tue Sep 24 17:24:28 2013.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# import sys, os
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#sys.path.insert(0, os.path.abspath('.'))


from pysit._sphinx.conf import *

import pysit

html_theme = 'cloud-pysit'
# html_theme = 'redcloud'

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# The intersphinx_mapping in pysit._sphinx.conf refers to pysit for
# the benefit of extension packages who want to refer to objects in
# the pysit core.  However, we don't want to cyclically reference
# pysit in its own build so we remove it here.
del intersphinx_mapping['pysit']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns.append('_templates')
exclude_patterns.append('_exttemplate.rst')

# Add any paths that contain templates here, relative to this directory.
if 'templates_path' not in locals():  # in case parent conf.py defines it
    templates_path = []
templates_path.append('_templates')

# -- Project information ------------------------------------------------------

# General information about the project.
project = u'PySIT'
author = u'MIT, Russell J. Hewett, and The PySIT Developers'
copyright = u'2011 - 2013, ' + author

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '0.4'
# The full version, including alpha/beta/rc tags.
release = '0.4dev'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions += []


# -- Options for HTML output ---------------------------------------------------

# A NOTE ON HTML THEMES
#
# The global PySIT configuration uses a custom theme,
# 'cloud-pysit', which is installed along with PySIT.

# A different theme can be used, or other parts of this theme can be
# modified, by overriding some of the variables set in the global
# configuration. The variables set in the global configuration are
# listed below, commented out.

# Add any paths that contain custom themes here, relative to this directory.
# To use a different custom theme, add the directory containing the theme.
#html_theme_path = []

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes. To override the custom theme, set this to the
# name of a builtin theme or the name of a custom theme in html_theme_path.
#html_theme = None

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = ''

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = ''

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = '{0} v{1}'.format(project, release)

# Output file base name for HTML help builder.
htmlhelp_basename = project + 'doc'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [('index', project + '.tex', project + u' Documentation',
                    author, 'manual')]

latex_logo = '_static/pysit_logo.pdf'


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [('index', project.lower(), project + u' Documentation',
              [author], 1)]


# -- Options for the edit_on_github extension ----------------------------------------

# Doesn't work, as PySIT uses bitbucket and hg

# extensions += ['pysit._sphinx.from_astropy.ext.edit_on_github']

# # Don't import the module as "version" or it will override the
# # "version" configuration parameter
# from pysit import version as versionmod
# edit_on_github_project = "pysit/pysit"
# if versionmod.release:
#     edit_on_github_branch = "v{0}.{1}.x".format(
#         versionmod.major, versionmod.minor)
# else:
#     edit_on_github_branch = "master"
# edit_on_github_source_root = ""
# edit_on_github_doc_root = "docs"
