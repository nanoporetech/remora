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

Getting Started
---------------

Model Training
**************

Remora models are trained to perform binary or categorical prediction of modified base content of a nanopore read.
In the initial implementation, the prediction/training unit is a modified base call at a single position within a read.
Later implementations may expand to calling many or all bases in a read at once.

The input to a Remora model are 1. a stretch of normalized signal, 2. the canonical bases attributed to the stretch of signal and 3. a mapping between these two.
Depending upon the model design/architecture the training units may be fix width of signal or a fixed width of sequence.
The Remora interface should provided training chunks for either fixed sequence length or fix signal lengths.

In the initial implementation the training data should be provided to Remora scripts in Taiyaki mapped signal format.
Reads in the mapped signal file should consist of fixed length sequence training units.
The "base of interest" should be at a fixed offset/position into each "read" in the mapped signal file.
The modified base content of the base of interest will be the training objective for Remora models.
Taiyaki mapped signal files represent modified bases with an alphabet defined within the mapped signal file format.
Modified base reads should contain a modified base at the central position of interest; canonical base training units should have a canonical base.
The fixed position of interest within each read must be provided to the Remora training scripts along with the mapped signal file.

The training units provided via the mapped signal file may derive from a number of sources, but should result in the above described training units.
The ``extract_toy_dataset.py`` script provides an example which extracts training data from a mapped signal file containing reads from E. coli native and PCR samples.
In this example native E. coli presents 6mA (single letter code ``a``) in the ``GATC`` sequence context.
Thus the first ``GATC`` site in each read is identified and sequence of the specified number of context bases is identified.
These training units are then output into a new Taiyaki mapped signal file which will be read for Remora model training.

Model Application
*****************

Remora modified base calling is not currently implemented.
Once implemented, Remora modified base calling should take a model produced from the above described training.
This model should specify the fixed sequence or signal length in order to apply to signal.

The core Remora API function will accept normalized nanopore signal, sequence (generally the reference sequence from the mapping for this read), and a mapping between the two and produce probabilities that each applicable base is modified.
Remora may provide a full pipeline to basecall, map, and call modified bases, but the optimized version of this pipeline will likely be implemented in other software (Megalodon, Tombo2 or Bonito).

Remora models and API will allow the simultaneous calling of multiple modified bases from a single model (e.g. 5mC, 5hmC, 5caC, 5fC as alternatives to C).

Terms and licence
-----------------

This is a research release provided under the terms of the Oxford Nanopore Technologies' Public Licence. 
Research releases are provided as technology demonstrators to provide early access to features or stimulate Community development of tools.
Support for this software will be minimal and is only provided directly by the developers. Feature requests, improvements, and discussions are welcome and can be implemented by forking and pull requests.
Much as we would like to rectify every issue, the developers may have limited resource for support of this software.
Research releases may be unstable and subject to rapid change by Oxford Nanopore Technologies.

© 2021 Oxford Nanopore Technologies Ltd.
Remora is distributed under the terms of the Oxford Nanopore Technologies' Public Licence.
