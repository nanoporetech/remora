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

The Remora training/prediction input unit (refered to as a chunk) consists of:

1. Section of normalized signal
2. Canonical bases attributed to the section of signal
3. Mapping between these two

Chunks have a fixed signal length defined at data preparation time and saved as a model attribute.
A fixed position within the chunk is defined as the "focus position".
This position is the center of the base of interest.

Pre-trained Models
------------------

Pre-trained models are included in the Remora repository.
To see the selection of models included in the current installation run ``remora model list_pretrained``.

Python API
----------

The Remora API can be applied to make modified base calls given a basecalled read via a ``RemoraRead`` object.
``sig`` should be a float32 numpy array.
``seq`` is a string derived from ``sig`` (can be either basecalls or other downstream derived sequence; e.g. mapped reference positions).
``seq_to_sig_map`` should be an int32 numpy array of length ``len(seq) + 1`` and elements should be indices within ``sig`` array assigned to each base in ``seq``.

.. code-block:: python

  from remora.model_util import load_model
  from remora.data_chunks import RemoraRead
  from remora.inference import call_read_mods

  model, model_metadata = load_model("remora_train_results/model_best.onnx")
  read = RemoraRead(sig, seq_to_sig_map, str_seq=seq)
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

Remora data preparation begins from Taiyaki mapped signal files generally produced from Megalodon containing modified base annotations.
This requires installation of Taiyaki via

.. code-block:: bash

  git clone https://github.com/nanoporetech/taiyaki
  pip install taiyaki/

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
    --processes 20
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
    --processes 20

  python \
    taiyaki/misc/merge_mappedsignalfiles.py \
    mapped_signal_train_data.hdf5 \
    --input mega_res_pcr/signal_mappings.hdf5 None \
    --input mega_res_sssI/signal_mappings.hdf5 None \
    --allow_mod_merge \
    --batch_format

After the construction of a training dataset, chunks must be extracted and saved in a Remora-friendly format.
The following command performs this task in Remora.

.. code-block:: bash

  remora \
    dataset prepare \
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
--------------

Models are trained with the ``remora model train`` command.
For example a model can be trained with the following command.

.. code-block:: bash

  remora \
    model train \
    remora_train_chunks.npz \
    --model remora/models/ConvLSTM_w_ref.py \
    --device 0 \
    --output-path remora_train_results

This command will produce a final model in ONNX format for use in Bonito, Megalodon or ``remora infer`` commands.

Model Inference
---------------

For testing purposes inference within Remora is provided given Taiyaki mapped signal files as input.
The below command will call the held out validation dataset from the data preparation section above.

.. code-block:: bash

  remora \
    infer from_taiyaki_mapped_signal \
    mega_res_pcr/split_signal_mappings.split_a.hdf5 \
    remora_train_results/model_best.onnx \
    --output-path remora_infer_results_pcr.txt \
    --device 0
  remora \
    infer from_taiyaki_mapped_signal \
    mega_res_sssI/split_signal_mappings.split_a.hdf5 \
    remora_train_results/model_best.onnx \
    --output-path remora_infer_results_sssI.txt \
    --device 0

Note that in order to perfrom inference on a GPU device the ``onnxruntime-gpu`` package must be installed.

GPU Troubleshooting
-------------------

Note that standard Remora models are small enough to run quite quickly on CPU resources and this is the primary recommandation.
Running Remora models on GPU compute resources is considered experimental with minimal support.

Deployment of Remora models is facilitated by the Open Neural Network Exchange (ONNX) format.
The ``onnxruntime`` python package is used to run the models.
In order to support running models on GPU resources the GPU compatible package must be installed (``pip install onnxruntime-gpu``).

Once installed the ``remora infer`` command takes a ``--device`` argument.
Similarly, the API ``remora.model_util.load_model`` function takes a ``device`` argument.
These arguments specify the GPU device ID to use for inference.

Once the ``device`` option is specified, Remora will attempt to load the model on the GPU resources.
If this fails a ``RemoraError`` will be raised.
The likely cause of this is the required CUDA and cuDNN dependency versions.
See the requirements on the `onnxruntime documentation page here <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements>`_.

To check the versions of the various dependencies see the following commands.

.. code-block:: bash

   # check cuda version
   nvcc --version
   # check cuDNN version
   grep -A 2 "define CUDNN_MAJOR" `whereis cudnn | cut -f2 -d" "`
   # check onnxruntime version
   python -c "import onnxruntime as ort; print(ort.__version__)"

These versions should match a row in the table linked above.
CUDA and cuDNN versions can be downloaded from the NVIDIA website (`cuDNN link <https://developer.nvidia.com/rdp/cudnn-archive>`_; `CUDA link <https://developer.nvidia.com/cuda-toolkit-archive>`_).
The cuDNN download can be specified at runtime as in the following example.

.. code-block:: bash

   CUDA_PATH=/path/to/cuda/include/cuda.h \
     CUDNN_H_PATH=/path/to/cuda/include/cudnn.h \
     remora \
     infer [arguments]

The ``onnxruntime`` dependency can be set via the python package install command.
For example `pip install "onnxruntime-gpu<1.7"`.

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
