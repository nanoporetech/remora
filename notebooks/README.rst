Remora API
""""""""""

The python jupyter notebooks provided in this directory introduce common use cases for the Remora API.
While the main target of the Remora code base is to train Nanopore modified base models, the interfaces are quite useful for orthogonal signal handling tasks.
The notebooks included here progress in the following order:

* basic_read_plotting.ipynb
* signal_mapping_refinement.ipynb
* metrics_api.ipynb

Basic Read Plotting
-------------------

In this notebook the basics of loading Nanopore signal data and associated sequences are introduced.

Signal Mapping Refinement
-------------------------

In this notebook the concept of signal mapping refinement (re-squiggling) is introduced.
While this method is not employed in modified base calling models it is very useful for exploring signal and aggregating results across reads mapped to the same reference location.

Metrics API
-----------

Finally Remora provides an API for the computation and extraction of signal based metrics.
These metrics are the core of many Nanopore signal analysis tools.
The notebook explores how these metrics might be plotted, how to input metrics into a testing framework and how to write out the statistics.
