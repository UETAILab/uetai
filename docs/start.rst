.. _quickstart:

===============
Quickstart
===============
To getting started with the library, we must notice that our library are using 3rd machine learning platform \
to upload, visualize our result, analysis or summary into their dashboard. With very first version of **uetai**, \
we only support `Comet ML <https://www.comet.ml/>`__ which is a great tracking experiment tools with very \
helpful customizable panel and well-writen documentation.

Log in or Sign up to `Comet ML <https://www.comet.ml/>`__ and start using tracking every though our library.

.. note::
    Did you know that you can do pretty much everything you can with Comet ML using our **uetai**?

Install uetai
===============
Installing **uetai** with ``pip`` by running:

.. code-block:: bash

    pip install uetai

Or by clone the repository and run:

.. code-block:: bash

    git clone git@github.com:UETAILab/uetai.git; cd uetai
    pip install -e .

Start logging
=============
To start logging your metrics, results or metadata, initialize a logger

.. code-block:: python

    from uetai.logger import CometLogger

    logger = CometLogger(project_name="uetai-example")
    logger.log({'metric': 0.3, 'acc': 0.9})  # logging
