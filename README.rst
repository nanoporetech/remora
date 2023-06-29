.. image:: /ONT_logo.png
  :width: 800
  :alt: [Oxford Nanopore Technologies]
  :target: https://nanoporetech.com/

Remora
""""""

Remora models predict methylation/modified base separated from basecalling.
The Remora repository is focused on the preparation of modified base training data and training modified base models.
Some functionality for running Remora models and investigation of raw signal is also provided.
For production modified base calling use `Dorado <https://github.com/nanoporetech/dorado>`_.
For recommended modified base downstream processing use `modkit <https://github.com/nanoporetech/modkit>`_.
For more advanced modified base data preparation from "randomers" see the `Betta release community note <https://community.nanoporetech.com/posts/betta-tool-release>`_ and reach out to customer support to inquire about access (customer.support@nanoporetech.com).

Installation
------------

Install from pypi:

::

   pip install ont-remora

Install from github source for development:

::

   git clone git@github.com:nanoporetech/remora.git
   pip install -e remora/[tests]

It is recommended that Remora be installed in a virtual environment.
For example ``python3 -m venv venv; source venv/bin/activate``.

See help for any Remora sub-command with the ``-h`` flag.

Getting Started
---------------

Remora models predict modified bases anchored to canonical basecalls or reference sequence from a nanopore read.

The Remora training/prediction input unit (referred to as a chunk) consists of:

1. Section of normalized signal
2. Canonical bases attributed to the section of signal
3. Mapping between these two

Chunks have a fixed signal length defined at data preparation/model training time.
These values are saved with the Remora model to extract chunks in the same manner at inference.
A fixed position within the chunk is defined as the "focus position" around which the fixed signal chunk is extracted.
By default, this position is the center of the "focus base" being interrogated by the model.

The canonical bases and mapping to signal (a.k.a. "move table") are combined for input into the neural network in several steps.
First each base is expanded to the k-mer surrounding that base (as defined by the ``--kmer-context-bases`` hyper parameter).
Then each k-mer is expanded according to the move table.
Finally each k-mer is one-hot encoded for input into the neural network.
This procedure is depicted in the figure below.

.. image:: images/neural_network_sequence_encoding.png
   :width: 600
   :alt: Neural network sequence encoding

Data Preparation
----------------

Remora data preparation begins from a POD5 file containing signal data and a BAM file containing basecalls from the POD5 file.
Note that the BAM file must contain the move table (``--emit-moves`` in Dorado) and the MD tag (default in Dorado with mapping and ``--MD`` argument for minimap2).

The following example generates training data from canonical (PCR) and modified (M.SssI treatment) samples in the same fashion as the releasd 5mC CG-context models.
Example reads can be found in the Remora respoitory (see ``test/data/`` directory).

K-mer tables for applicable conditions can be found in the `kmer_models repository <https://github.com/nanoporetech/kmer_models>`_.

.. code-block:: bash

  remora \
    dataset prepare \
    can_reads.pod5 \
    can_mappings.bam \
    --output-path can_chunks \
    --refine-kmer-level-table levels.txt \
    --refine-rough-rescale \
    --motif CG 0 \
    --mod-base-control
  remora \
    dataset prepare \
    mod_reads.pod5 \
    mod_mappings.bam \
    --output-path mod_chunks \
    --refine-kmer-level-table levels.txt \
    --refine-rough-rescale \
    --motif CG 0 \
    --mod-base m 5mC

The above commands each produce a core Remora dataset stored in the directory defined by ``--output-path``.
Each core dataset contains a memory mapped numpy file for each core array (chunk data) and a JSON format metadata config file.
These memory mapped files allow efficient access to very large datasets.

Before Remora, 3.0 datasets were stored as numpy array dictionaries.
Updating datasets can be accomplished with the ``scripts/update_dataset.py`` script included in the repository.

Model Training
--------------

Core datasets can be combined at training time via a JSON-format config file.
Datasets can even be merged hierarchically (a training dataset config may point to a core dataset or another config).
Metadata attributes are checked for compatibility and merged at training dataset loading time.
Chunk raw data are loaded from each core dataset at specified proportions to construct batches for training.

A training dataset config can be constructed from the above datasets with the following command.
The ``1`` weights will produce training batches with equal numbers of chunks from the two core datasets.

.. code-block:: bash

  echo '[["./can_chunks", 1], ["./mod_chunks", 1]]' > train_dataset.jsn

Models are trained with the ``remora model train`` command.
For example a model can be trained with the following command.

.. code-block:: bash

  remora \
    model train \
    train_dataset.jsn \
    --model remora/models/ConvLSTM_w_ref.py \
    --device 0 \
    --output-path train_results

This command will produce a "best" model in torchscript format for use in Bonito, ``remora infer``, or ``remora validate`` commands.
Models can be exported for use in Dorado with the ``remora model export`` command.

Model Inference
---------------

For testing purposes inference within Remora is provided.

.. code-block:: bash

  remora \
    infer from_pod5_and_bam \
    can_signal.pod5 \
    can_mappings.bam \
    --model train_results/model_best.pt \
    --out-file can_infer.bam \
    --device 0
  remora \
    infer from_pod5_and_bam \
    mod_signal.pod5 \
    mod_mappings.bam \
    --model train_results/model_best.pt \
    --out-file mod_infer.bam \
    --device 0

Finally, ``Remora`` provides tools to validate these results.
Ground truth BED files references positions where each read should be called as the modified or canonical base listed in the BED name field.

.. code-block:: bash

  remora \
    validate from_modbams \
    --bam-and-bed can_infer.bam can_ground_truth.bed \
    --bam-and-bed mod_infer.bam mod_ground_truth.bed \
    --full-output-filename validation_results.txt

Pre-trained Models
------------------

See the selection of current released models with ``remora model list_pretrained``.
Pre-trained models are stored remotely and can be downloaded using the ``remora model download`` command or will be downloaded on demand when needed.

Models may be run from `Bonito <https://github.com/nanoporetech/bonito>`_.
See Bonito documentation to apply Remora models.

More advanced research models may be supplied via `Rerio <https://github.com/nanoporetech/rerio>`_.
Note that older ONNX format models require Remora version < 2.0.

Python API and Raw Signal Analysis
----------------------------------

The Remora API can be applied to make modified base calls given a basecalled read via a ``RemoraRead`` object.

* ``dacs`` (Data acquisition values) should be an int16 numpy array.
* ``shift`` and ``scale`` are float values to convert dacs to mean=0 SD=1 scaling (or similar) for input to the Remora neural network.
* ``str_seq`` is a string derived from ``sig`` (can be either basecalls or other downstream derived sequence; e.g. mapped reference positions).
* ``seq_to_sig_map`` should be an int32 numpy array of length ``len(seq) + 1`` and elements should be indices within ``sig`` array assigned to each base in ``seq``.

.. code-block:: python

  from remora.model_util import load_model
  from remora.data_chunks import RemoraRead
  from remora.inference import call_read_mods

  model, model_metadata = load_model("remora_train_results/model_best.pt")
  read = RemoraRead(dacs, shift, scale, seq_to_sig_map, str_seq=seq)
  mod_probs, _, pos = call_read_mods(
    read,
    model,
    model_metadata,
    return_mod_probs=True,
  )

``mod_probs`` will contain the probability of each modeled modified base as found in model_metadata["mod_long_names"].
For example, run ``mod_probs.argmax(axis=1)`` to obtain the prediction for each input unit.
``pos`` contains the position (index in input sequence) for each prediction within ``mod_probs``.

The python API also enables access to per-read, per-site raw signal metrics for more advanced statistical analysis.
This API is primarily accessed via the ``remora.io.Read`` object.

The iPython notebooks (see ``notebooks`` directory) included in this repository exemplify some common analyses.

Raw signal plotting is also available via the ``remora analyze plot ref_region`` command.
Additional commands will be added to this group to enable more common raw signal analysis tasks.

The ``plot ref_region`` command is useful for gaining intuition into signal attributes and visualize signal shifts around modified bases.
As an example using the test data, the following command produces the plots below.

.. code-block:: bash

  remora \
    analyze plot ref_region \
    --pod5-and-bam can_reads.pod5 can_mappings.bam \
    --pod5-and-bam mod_reads.pod5 mod_mappings.bam \
    --ref-regions ref_regions.bed \
    --highlight-ranges mod_gt.bed \
    --refine-kmer-level-table levels.txt \
    --refine-rough-rescale \
    --log-filename log.txt

.. image:: images/plot_ref_region_fwd.png
   :width: 600
   :alt: Plot reference region image (forward strand)

.. image:: images/plot_ref_region_rev.png
   :width: 600
   :alt: Plot reference region image (reverse strand)

Terms and Licence
-----------------

This is a research release provided under the terms of the Oxford Nanopore Technologies' Public Licence.
Research releases are provided as technology demonstrators to provide early access to features or stimulate Community development of tools.
Support for this software will be minimal and is only provided directly by the developers. Feature requests, improvements, and discussions are welcome and can be implemented by forking and pull requests.
Much as we would like to rectify every issue, the developers may have limited resource for support of this software.
Research releases may be unstable and subject to rapid change by Oxford Nanopore Technologies.

© 2021-2023 Oxford Nanopore Technologies Ltd.
Remora is distributed under the terms of the Oxford Nanopore Technologies' Public Licence.

Research Release
----------------

Research releases are provided as technology demonstrators to provide early access to features or stimulate Community development of tools. Support for this software will be minimal and is only provided directly by the developers. Feature requests, improvements, and discussions are welcome and can be implemented by forking and pull requests. However much as we would like to rectify every issue and piece of feedback users may have, the developers may have limited resource for support of this software. Research releases may be unstable and subject to rapid iteration by Oxford Nanopore Technologies.
