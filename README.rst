UETAI
=====

.. BADGES_START

.. image:: https://github.com/UETAILab/uetai/actions/workflows/lint-test.yml/badge.svg
    :target: https://github.com/UETAILab/uetai/actions/workflows/lint-test.yml

.. image:: https://codecov.io/gh/UETAILab/uetai/branch/main/graph/badge.svg?token=9KY7UU1QNB
   :target: https://codecov.io/gh/UETAILab/uetai

.. BADGES_END


This is library for multiple utilities and templates for project in
UET-AILAB


Installation
------------
.. highlight:: bash

::

    pip install -e .

Getting started
---------------
.. highlight:: python

::

    import uetai
    from uetai.logger import SummaryWriter
    writer = SummaryWriter("some_experiment")

