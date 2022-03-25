.. _tracking:

===================
Experiment tracking
===================

.. note::

    We only integrated Comet ML as the dashboard for tracking experiment and logging media.

Intergrating UETAI in your script
=================================

We would like to introduce our very simple API to logging experiment and managing artifact \
which has been customized from original framework's API and carefully tested.

Initialize experiment
---------------------

Initializing every dashboard's experiment and using all its API through object build \
in ``uetai.logger``. While you call a new run at the top of your script, \
**uetai** will automatically initialize a Experiment object which will create a local directory \
to saved all log, files and streamed it to dashboard server (if the tools is hosted online).


.. code-block:: python

    from uetai.logger import CometLogger

    # Initialize experiment
    logger = CometLogger(
        project_name="your-project-name", 
        api_key="your-api-key",
        workspace="your-workspace-name", 
    )

.. note::

    To get your own ``API_KEY``, you need to sign up a `Comet ML <https://www.comet.ml/>`__ account, \
    go to Setting> Developer Information> Generate API key and copy the key.


Logging your experiment
-----------------------

In order to log your metrics, metadata like loss, accuracy in training loop, use ``log()``.\
You might find that different because we've re-designed your dashboard logging methods to \
be more synchronous and easy to use. 

You can log pretty much anything you want, including images, videos, audio, text, \
see more supported types in below table.

.. code-block:: python

    # Log your metrics
    logger.log({'loss'=loss, 'accuracy'=accuracy,}, step=step)


Supported data types
--------------------

We're still in developing progress, some data types might not be supported yet. \
However, you can still log it through dashboard's original API, example:

.. code-block:: python
    
    # Get experiment object
    exp = logger.get_experiment()

    # Using original API
    exp.log_confusion_matrix(labels=["one", "two", "three"],
        matrix=[[10, 0, 0],
        [ 0, 9, 1],
        [ 1, 1, 8]]
    )

**Currently supported data types**:

+------------------+---------------+-------+------+-------+-------+
| Dashboard        | Float metrics | Image | Text | Audio | Graph |
+==================+===============+=======+======+=======+=======+
| Comet ML         | ✓             | ✓     | ✓    | ✗     | ✗     |
+------------------+---------------+-------+------+-------+-------+
| Weights & Biases | ✗             | ✗     | ✗    | ✗     | ✗     |
+------------------+---------------+-------+------+-------+-------+
| MLFlow           | ✗             | ✗     | ✗    | ✗     | ✗     |
+------------------+---------------+-------+------+-------+-------+