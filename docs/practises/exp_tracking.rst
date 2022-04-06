.. _exp_tracking:

==================================
Logging your experiment with UETAI
==================================

It might feel strange at first, but everything in **uetai** is exactly the same
as there is in your favorite tracking experiment framework.

In this tutorial, we will show you how to log your experiment with **uetai** using
``CometLogger``, with other dashboard you can use exactly the same syntax.

Initialize experiment
=====================

Initialize your experiment with ``uetai.logger``

.. code-block:: python

    from uetai.logger import CometLogger

    logger = CometLogger(project_name='my-project', api_key='my-api-key')

.. note:: 

    If you don't pass API and your environment variable doesn't include any of them,
    it (the logger) will ask for your API key.

Logging parameters
==================

Every experiment usually come up with a set of parameters. 
Logging them using ``log_parameters``, passing value should be 
a ``Dict[str, Any]`` or a ``Namespace``:

.. code-block:: python

    # log Dict
    logger.log_parameters(
        {
            'bs': 64,
            'cuda': False,
            'dropout_rate': 0.4,
            'epoch': 10,
            'lr': 0.0001,
            'momentum': 0.9,
        }
    )

    # or Namespace
    logger.log_parameters(args_parser)

The hyperparameters will be logged after the run is finished. 
Find the it in the **hyperparameters** tabs.

.. image:: ../_static/images/param.png
    :alt: hyperparameters

Logging metrics
===============



Logging media
=============


Logging image
-------------

Logging text
------------

Logging html
------------

