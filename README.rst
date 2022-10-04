.. image:: /ONT_logo.png
  :width: 800
  :alt: [Oxford Nanopore Technologies]
  :target: https://nanoporetech.com/

Remora
""""""

Methylation/modified base calling separated from basecalling.
Remora primarily provides an API to call modified bases for basecaller programs such as Bonito.
Remora also provides the tools to prepare datasets, train modified base models and run simple inference.

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
For example ``python3.8 -m venv --prompt remora --copies venv; source venv/bin/activate``.

See help for any Remora sub-command with the ``-h`` flag.

Getting Started
---------------

Remora models are trained to perform binary or categorical prediction of modified base content of a nanopore read.
Models may also be trained to perform canonical base prediction, but this feature may be removed at a later time.
The rest of the documentation will focus on the modified base detection task.

The Remora training/prediction input unit (referred to as a chunk) consists of:

1. Section of normalized signal
2. Canonical bases attributed to the section of signal
3. Mapping between these two

Chunks have a fixed signal length defined at data preparation time and saved as a model attribute.
A fixed position within the chunk is defined as the "focus position".
By default, this position is the center of the "focus base" being interrogated by the model.

Pre-trained Models
------------------

See the selection of current released models with ``remora model list_pretrained``.
Pre-trained models are stored remotely and can be downloaded using the ``remora model download`` command or will be downloaded on demand when needed.

Models may be run from `Bonito <https://github.com/nanoporetech/bonito>`_.
See Bonito documentation to apply Remora models.

More advanced research models may be supplied via `Rerio <https://github.com/nanoporetech/rerio>`_.
Note that older ONNX format models require Remora version < 2.0.

Python API
----------

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

Data Preparation
----------------

Remora data preparation begins from a POD5 file (containing signal data) and a BAM file containing basecalls from the POD5 file.
Note that the BAM file must contain the move table (default in Bonito and ``--moves_out`` in Guppy).

The following example generates training data from canonical (PCR) and modified (M.SssI treatment) samples in the same fashion as the releasd 5mC CG-context models.

.. code-block:: bash

  remora \
    dataset prepare \
    can_signal.pod5 \
    can_basecalls.bam \
    --output-remora-training-file can_chunks.npz \
    --motif CG 0 \
   --mod-base-control
  remora \
    dataset prepare \
    mod_signal.pod5 \
    mod_basecalls.bam \
    --output-remora-training-file mod_chunks.npz \
    --motif CG 0 \
    --mod-base m 5mC
  remora \
    dataset merge \
    --input-dataset can_chunks.npz 10_000_000 \
    --input-dataset mod_chunks.npz 10_000_000 \
    --output-dataset chunks.npz

The resulting ``chunks.npz`` file can then be used to train a Remora model.

Model Training
--------------

Models are trained with the ``remora model train`` command.
For example a model can be trained with the following command.

.. code-block:: bash

  remora \
    model train \
    chunks.npz \
    --model remora/models/ConvLSTM_w_ref.py \
    --device 0 \
    --output-path train_results

This command will produce a "best" model in torchscript format for use in Bonito, or ``remora infer`` commands.

Model Inference
---------------

For testing purposes inference within Remora is provided.

.. code-block:: bash

  remora \
    infer from_pod5_and_bam \
    can_signal.pod5 \
    can_basecalls.bam \
    --model train_results/model_best.pt \
    --out-file can_infer.bam \
    --device 0
  remora \
    infer from_pod5_and_bam \
    mod_signal.pod5 \
    mod_basecalls.bam \
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

Terms and Licence
-----------------

This is a research release provided under the terms of the Oxford Nanopore Technologies' Public Licence.
Research releases are provided as technology demonstrators to provide early access to features or stimulate Community development of tools.
Support for this software will be minimal and is only provided directly by the developers. Feature requests, improvements, and discussions are welcome and can be implemented by forking and pull requests.
Much as we would like to rectify every issue, the developers may have limited resource for support of this software.
Research releases may be unstable and subject to rapid change by Oxford Nanopore Technologies.

© 2021 Oxford Nanopore Technologies Ltd.
Remora is distributed under the terms of the Oxford Nanopore Technologies' Public Licence.

Research Release
----------------

Research releases are provided as technology demonstrators to provide early access to features or stimulate Community development of tools. Support for this software will be minimal and is only provided directly by the developers. Feature requests, improvements, and discussions are welcome and can be implemented by forking and pull requests. However much as we would like to rectify every issue and piece of feedback users may have, the developers may have limited resource for support of this software. Research releases may be unstable and subject to rapid iteration by Oxford Nanopore Technologies.
