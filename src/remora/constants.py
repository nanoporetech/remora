DEFAULT_NN_SIZE = 64
DEFAULT_BATCH_SIZE = 1000
DEFAULT_CHUNK_CONTEXT = (50, 50)
DEFAULT_MIN_SAMPLES_PER_BASE = 5
DEFAULT_KMER_CONTEXT_BASES = (1, 1)
DEFAULT_KMER_LEN = sum(DEFAULT_KMER_CONTEXT_BASES) + 1
DEFAULT_FOCUS_OFFSET = 100
DEFAULT_VAL_PROP = 0.01
DEFAULT_EPOCHS = 50
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_CONF_THR = 0.8
DEFAULT_LR = 0.001

DEFAULT_SCHEDULER = "StepLR"
DEFAULT_SCH_VALUES = {"step_size": 10, "gamma": 0.6}
TYPE_CONVERTERS = {"str": str, "int": int, "float": float}

SGD_OPT = "sgd"
ADAM_OPT = "adam"
ADAMW_OPT = "adamw"
OPTIMIZERS = (ADAMW_OPT, SGD_OPT, ADAM_OPT)

FINAL_MODEL_FILENAME = "model_final.checkpoint"
FINAL_ONNX_MODEL_FILENAME = "model_final.onnx"
SAVE_DATASET_FILENAME = "remora_train_data.npz"

BEST_MODEL_FILENAME = "model_best.checkpoint"
BEST_ONNX_MODEL_FILENAME = "model_best.onnx"

# should be an int to store in onnx
MODEL_VERSION = 3

DEFAULT_BASECALL_MODEL_VERSION = "0.0.0"
DEFAULT_MOD_BASE = ["5mc"]
DEFAULT_MODEL_TYPE = "CG"
MODBASE_MODEL_NAME = "modbase_model.onnx"
MODEL_DICT = {
    "dna_r9.4.1_e8": {
        "fast": {
            DEFAULT_BASECALL_MODEL_VERSION: {
                "5mc": {"CG": [0]},
                "5hmc_5mc": {"CG": [0]},
            }
        },
        "hac": {
            DEFAULT_BASECALL_MODEL_VERSION: {
                "5mc": {"CG": [0]},
                "5hmc_5mc": {"CG": [0]},
            }
        },
        "sup": {
            DEFAULT_BASECALL_MODEL_VERSION: {
                "5mc": {"CG": [0]},
                "5hmc_5mc": {"CG": [0]},
            }
        },
    },
    "dna_r9.4.1_e8.1": {
        "fast": {
            DEFAULT_BASECALL_MODEL_VERSION: {
                "5mc": {"CG": [0]},
            }
        },
        "hac": {
            DEFAULT_BASECALL_MODEL_VERSION: {
                "5mc": {"CG": [0]},
            }
        },
        "sup": {
            DEFAULT_BASECALL_MODEL_VERSION: {
                "5mc": {"CG": [0]},
            }
        },
    },
    "dna_r10.4_e8.1": {
        "fast": {
            DEFAULT_BASECALL_MODEL_VERSION: {
                "5mc": {"CG": [0]},
                "5hmc_5mc": {"CG": [0]},
            }
        },
        "hac": {
            DEFAULT_BASECALL_MODEL_VERSION: {
                "5mc": {"CG": [0]},
                "5hmc_5mc": {"CG": [0]},
            }
        },
        "sup": {
            DEFAULT_BASECALL_MODEL_VERSION: {
                "5mc": {"CG": [0]},
                "5hmc_5mc": {"CG": [0]},
            },
        },
    },
}
