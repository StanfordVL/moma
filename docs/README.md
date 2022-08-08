# Documentation for MOMA-LRG

To build locally, first install `sphinx` by running
```
python -m pip install sphinx
pip install furo
```
inside of a virtual or conda environment.

After installing, run 
```
mkdir source/_static
sphinx-build -b html source build/html
```
and open `docs/build/html/index.html` inside of your browser, where you 
should see a locally hosted version of the docs website.

After this has been built once, you can run
```
make clean
make html
```
every time you make changes. 

The [Sphinx](https://www.sphinx-doc.org/en/master/) package is what we use
to generate nicely-rendered, auto-generated documentation. The way it works
is to generate documentation based on docstrings (checkout `momaapi/moma.py`)
for an example.