Example Usage
==============
Creating a MOMA Object
-----------------------
MOMA-LRG exports a simple to use API that allows users to access the dataset
via an easy-to-use interface.

To begin, create a MOMA-LRG object by passing in the path to the MOMA-LRG dataset
as follows:

.. code-block:: python

    import momaapi
    dir_moma = "my/moma/directory"
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

Working with the data
---------------------
After creating a MOMA object, you can interface with the dataset through
a very simple API. Let's say that you wanted to retrieve the annotations
for all videos containing the activity class ``"basketball game"`` in the
validation set. You could run

.. code-block:: python

    ids_act = moma.get_ids_act(split="val", cnames_act=["basketball game"])
    anns_act = moma.get_anns_act(ids_act)

``anns_act`` now contains a list of Activity annotations, each containing
metadata on a different instance of a basketball game. 