DEFAULT_NN_SIZE = 64
DEFAULT_BATCH_SIZE = 1000
DEFAULT_CHUNK_CONTEXT = (50, 50)
DEFAULT_MIN_SAMPLES_PER_BASE = 5
DEFAULT_KMER_CONTEXT_BASES = (4, 4)
DEFAULT_KMER_LEN = sum(DEFAULT_KMER_CONTEXT_BASES) + 1
DEFAULT_FOCUS_OFFSET = 100
DEFAULT_VAL_PROP = 0.01
DEFAULT_EPOCHS = 50
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_FILT_FRAC = 0.1
DEFAULT_LR = 0.001

DEFAULT_SCHEDULER = "StepLR"
DEFAULT_SCH_VALUES = {"step_size": 10, "gamma": 0.6}
TYPE_CONVERTERS = {"str": str, "int": int, "float": float}

SGD_OPT = "sgd"
ADAM_OPT = "adam"
ADAMW_OPT = "adamw"
OPTIMIZERS = (ADAMW_OPT, SGD_OPT, ADAM_OPT)

FINAL_MODEL_FILENAME = "model_final.checkpoint"
FINAL_TORCHSCRIPT_MODEL_FILENAME = "model_final.pt"
SAVE_DATASET_FILENAME = "remora_train_data.npz"

BEST_MODEL_FILENAME = "model_best.checkpoint"
BEST_TORCHSCRIPT_MODEL_FILENAME = "model_best.pt"

MODEL_VERSION = 3

DEFAULT_REFINE_SCALE_ITERS = -1
DEFAULT_REFINE_HBW = 5
DEFAULT_REFINE_BAND_MIN_STEP = 2

MODBASE_MODEL_NAME = "modbase_model.pt"
MODEL_DATA_DIR_NAME = "trained_models"

"""
The default model is the first key at every level after the pore and mod.
E.g. for "dna_r10.4.1_e8.2_400bps" and "5mc" the default model is
CG_sup_v3.5.1_2.
"""
MODEL_DICT = {
    "dna_r9.4.1_e8": {
        "5mc": {
            "CG": {
                "sup": {"v3.5.1": {0: "qedo6lilt29haqtdd97tic83lxoribfr"}},
                "hac": {"v3.5.1": {0: "icimz7z06ijdme9zkfletxl323nveunh"}},
                "fast": {"v3.5.1": {0: "ogtg6odxf9elj0mqjqxpx7xw82j5finz"}},
            }
        },
    },
    "dna_r10.4.1_e8.2_400bps": {
        "5mc": {
            "CG": {
                "sup": {
                    "v3.5.1": {
                        2: "6zo86p9z4me6hl4di12cimbqjd9n7p25",
                    }
                },
                "hac": {
                    "v3.5.1": {
                        2: "aub3do2tzhgzrhu2o100lg80d8yv9t91",
                    }
                },
                "fast": {
                    "v3.5.1": {
                        2: "e8zczcd15rhhs6eppuwmkehwubo848nu",
                    }
                },
            }
        },
        "5hmc_5mc": {
            "CG": {
                "sup": {"v4.0.0": {2: "whlux6wohu5fwyg4mreg5mz0a8iaugxk"}},
                "hac": {"v4.0.0": {2: "o1ah5qgd77l393gjnyrv44jt4wp6wxcl"}},
                "fast": {"v4.0.0": {2: "hms1t8ledf09p8ta9katj90l29uz2qll"}},
            }
        },
    },
}


DEFAULT_REFINE_SHORT_DWELL_PARAMS = (15, 5, 0.05)
REFINE_ALGO_VIT_NAME = "Viterbi"
REFINE_ALGO_DWELL_PEN_NAME = "dwell_penalty"
REFINE_ALGOS = (REFINE_ALGO_DWELL_PEN_NAME, REFINE_ALGO_VIT_NAME)
DEFAULT_REFINE_ALGO = REFINE_ALGO_DWELL_PEN_NAME

PA_TO_NORM_SCALING_FACTOR = 1.4826
