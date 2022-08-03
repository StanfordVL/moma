Example usage
==============
Creating a MOMA Object
-----------------------
MOMA-LRG exports a simple to use API that allows users to access the dataset
via an easy-to-use interface.

To begin, create a MOMA-LRG object by passing in the path to the MOMA-LRG dataset
as follows:

.. code-block:: python

    import momaapi
    moma = momaapi.MOMA(dir_moma)

Getting access to the underlying data can be done via calling methods on
the ``moma`` object.

Few-shot experiments
--------------------
MOMA-LRG was designed to provide an abstraction to learn highly generalizable
video representations. As a result, the MOMA-LRG API supports a few-shot 
paradigm where different splits have non-overlapping activity classes and sub-activity classes.
This is in contrast to the standard evaluation paradigm, where different splits 
share the same sets of activity classes and sub-activity classes.

To evaluate on few-shot, create a MOMA object by running:

.. code-block:: python

    moma = momaapi.MOMA(dir_moma, paradigm='few-shot')

The interface is the same as in the standard paradigm.

