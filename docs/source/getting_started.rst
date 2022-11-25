Getting Started
===============
Installation
------------

To install the MOMA API, first clone the repository by running

.. code-block:: bash

    git clone https://github.com/StanfordVL/moma.git

Install the API and its dependencies by running

.. code-block:: bash

    cd moma
    pip install .
    pip install -r requirements.txt

Retrieving the dataset
-----------------------
The dataset will can be downloaded by following the instructions from the `MOMA website <https://moma.stanford.edu>`_.

Download the dataset into a directory titled ``dir_moma`` with the structure below. The anns directory requires roughly 1.8GB of space and the video directory requires 436 GB.

.. code-block:: bash

    $ tree dir_moma
    .
    ├── anns
    │    ├── anns.json
    │    ├── split_std.json
    │    ├── split_fs.json
    │    ├── clips.json
    │    └── taxonomy
    └── videos
        ├── all
        ├── raw
        ├── activity_fr
        ├── activity
        ├── sub_activity_fr
        ├── sub_activity
        ├── interaction
        ├── interaction_frames
        └── interaction_video
