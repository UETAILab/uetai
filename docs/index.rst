=====
UETAI
=====

*Machine Learning analyzing and debugging tools integrated tracking experiment systems.*

.. image:: https://codecov.io/gh/UETAILab/uetai/branch/main/graph/badge.svg?token=9KY7UU1QNB
    :target: codecov
.. image:: https://github.com/UETAILab/uetai/actions/workflows/lint-test.yml/badge.svg
    :target: linting
.. image:: https://img.shields.io/pypi/v/uetai
    :target: release_pypi
.. image:: https://img.shields.io/pypi/pyversions/uetai
    :target: py_ver
.. image:: https://img.shields.io/github/license/UETAILab/uetai
    :target: license


UETAI is a customize PyTorch logger which will able to help users track machine learning experiment, \
and esily debug raw datasets and trained models.

UETAI provided tools for helping user tracking their experiment, visualizing the dataset, results, \
and debuging the model (and the raw dataset also) with little effort by integrated the tools into \
the dashboards which users are using for logging.

*In this beta version, we will only focus on integrated Comet ML, which is amazing dashboard \
with well-writen API and customable panel*

Guides
======
Starting to logging and debugging your model with :ref:`Quickstart <quickstart>`

- **Logging your Experiment**: Easy logging to your favourite dashboard with UETAI


Tables of Contents
==================

.. toctree::
   :maxdepth: 4

   Quickstart <start>

.. toctree::
   :maxdepth: 4
   :caption: Getting Started

   Best Practises <practises/index>
   Guides <guides/index>

.. toctree::
   :maxdepth: 4
   :caption: Advanced

   Reference <api/modules>
   Release <release>
   About us <about/index>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: http://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: http://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: http://sphinx-doc.org/domains.html#the-python-domain
.. _Sphinx: http://www.sphinx-doc.org/
.. _Python: http://docs.python.org/
.. _Numpy: http://docs.scipy.org/doc/numpy
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: http://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: http://scikit-learn.org/stable
.. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
.. _Google style: https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: http://www.sphinx-doc.org/en/stable/domains.html#info-field-lists
