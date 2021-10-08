.. image:: /ONT_logo.png
  :width: 800
  :alt: [Oxford Nanopore Technologies]
  :target: https://nanoporetech.com/

Remora
""""""

Methylation/modified base calling separated from basecalling.

Detailed documentation for all ``remora`` commands and algorithms can be found on the `TODO Remora documentation page <https://nanoporetech.github.io/remora/>`_.

Installation
------------

To install from gitlab source for development, the following commands can be run.

::

   git clone git@git.oxfordnanolabs.local:algorithm/remora.git
   pip install -e remora/[tests]

See help for any Remora command with the ``-h`` flag.

Getting Started
---------------

Remora models are trained to perform binary or categorical prediction of modified base content of a nanopore read.
Models may also be trained to perform canonical base prediction, but this feature may be removed at a later time.
The rest of the documentation will focus on the modified base task.

The Remora training/prediction input unit (refered to as a chunk) consists of:
    1. Section of normalized signal
    2. Canonical bases attributed to the section of signal
    3. Mapping between these two
Chunks have a fixed signal length defined at data preparation time and saved as a model attribute.
A fixed position within the chunk is defined as the "focus position".
This position is the center of the base of interest for prediction.

Data Preparation
****************

Remora data preparation begins from Taiyaki mapped signal files generally produced from Megalodon containing modified base annotations.
An example dataset might be pre-processed with the following commands.

.. code-block:: bash

  megalodon \
    pcr_fast5s/ \
    --reference ref.mmi \
    --output-directory mega_res_pcr \
    --outputs mappings signal_mappings \
    --num-reads 10000 \
    --guppy-config dna_r9.4.1_450bps_fast.cfg \
    --devices 0 \
    --processes 40
  # Note the --ref-mods-all-motifs option defines the modified base characteristics
  megalodon \
    sssI_fast5s/ \
    --ref-mods-all-motifs m 5mC CG 0 \
    --reference ref.mmi \
    --output-directory mega_res_sssI \
    --outputs mappings signal_mappings \
    --num-reads 10000 \
    --guppy-config dna_r9.4.1_450bps_fast.cfg \
    --devices 0 \
    --processes 40

  python \
    taiyaki/misc/split_mappedsignalfiles.py \
    mega_res_pcr/signal_mappings.hdf5 \
    --output_basename mega_res_pcr/split_signal_mappings \
    --split_a_proportion 0.01 \
    --batch_format
  python \
    taiyaki/misc/split_mappedsignalfiles.py \
    mega_res_sssI/signal_mappings.hdf5 \
    --output_basename mega_res_sssI/split_signal_mappings \
    --split_a_proportion 0.01 \
    --batch_format

  python \
    taiyaki/misc/merge_mappedsignalfiles.py \
    mapped_signal_train_data.hdf5 \
    --input mega_res_pcr/split_signal_mappings.split_b.hdf5 None \
    --input mega_res_sssI/split_signal_mappings.split_b.hdf5 None \
    --allow_mod_merge \
    --batch_format

After the construction of a training dataset, chunks must be extracted and saved in a Remora-friendly format.
The following command performs this task in Remora.

.. code-block:: bash
  remora \
    prepare_train_data \
    mapped_signal_train_data.hdf5 \
    --output-remora-training-file remora_train_chunks.npz \
    --motif CG 0 \
    --mod-bases m \
    --chunk-context 50 50 \
    --kmer-context-bases 6 6 \
    --max-chunks-per-read 20 \
    --log-filename log.txt

The resulting ``remora_train_chunks.npz`` file can then be used to train a Remora model.

Model Training
**************

Models are trained with the ``remora train_model`` command.
For example a model can be trained with the following command.

.. code-block:: bash

  remora \
    train_model \
    remora_train_chunks.npz \
    --model remora/models/ConvLSTM_w_ref.py \
    --device 0 \
    --output-path remora_train_results

This command will produce a final model in ONNX format for exporting or using within Remora.

Model Inference
***************

For testing purposes inference within Remora is provided given Taiyaki mapped signal files as input.
The below command will call the held out validation dataset from the data preparation section above.

.. code-block:: bash

  remora \
    infer \
    mega_res_pcr/split_signal_mappings.split_a.hdf5 \
    remora_train_results/model_final.onnx \
    --output-path remora_infer_results_pcr.txt \
    --device 0
  remora \
    infer \
    mega_res_sssI/split_signal_mappings.split_a.hdf5 \
    remora_train_results/model_final.onnx \
    --output-path remora_infer_results_sssI.txt \
    --device 0

Note that in order to perfrom inference on a GPU device the ``onnxruntime-gpu`` package must be installed.

API
***

The Remora API can be applied to make modified base calls given a prepared read via a ``RemoraRead`` object.

.. code-block:: python
  from remora.data_chunks import RemoraRead
  from remora.model_util import load_onnx_model
  from remora.inference import call_read_mods

  model, model_metadata = load_onnx_model(
    "remora_train_results/model_final.onnx",
    device=0,
  )
  read = RemoraRead(sig, seq, sig_to_seq_map, read_id, labels)
  output, labels, read_data = call_read_mods(
    read,
    model,
    model_metadata,
    batch_size,
    focus_offset,
  )

``outputs`` will contain the categorical predictions from the neural network in a numpy array.
For example, run ``output.argmax(axis=1)`` to obtain the prediction for each input unit.
The ``read_data`` object contains the relative position for each prediction within outputs.

Terms and licence
-----------------

This is a research release provided under the terms of the Oxford Nanopore Technologies' Public Licence. 
Research releases are provided as technology demonstrators to provide early access to features or stimulate Community development of tools.
Support for this software will be minimal and is only provided directly by the developers. Feature requests, improvements, and discussions are welcome and can be implemented by forking and pull requests.
Much as we would like to rectify every issue, the developers may have limited resource for support of this software.
Research releases may be unstable and subject to rapid change by Oxford Nanopore Technologies.

© 2021 Oxford Nanopore Technologies Ltd.
Remora is distributed under the terms of the Oxford Nanopore Technologies' Public Licence.
