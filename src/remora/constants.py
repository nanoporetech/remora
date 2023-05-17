DEFAULT_NN_SIZE = 64
DEFAULT_BATCH_SIZE = 1024
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

DEFAULT_REFINE_HBW = 5

MODBASE_MODEL_NAME = "modbase_model.pt"
MODEL_DATA_DIR_NAME = "trained_models"

"""
The default model is the first key at every level after the pore and mod.
E.g. for "dna_r10.4.1_e8.2_400bps" and "5mc" the default model is
CG_sup_v3.5.1_2.
"""

# R9 5mC CG-context models
_R9_5mc_CG_models = {
    "sup": {"v3.5.1": {0: "dna_r9.4.1_e8_sup_v3.5.1_5mc_CG_v0"}},
    "hac": {"v3.5.1": {0: "dna_r9.4.1_e8_hac_v3.5.1_5mc_CG_v0"}},
    "fast": {"v3.5.1": {0: "dna_r9.4.1_e8_fast_v3.5.1_5mc_CG_v0"}},
}

# kit14 400bps 5mC CG-context models (contains 5kHz and 4kHz models)
_kit14_5mc_CG_models = {
    "sup": {
        "v4.2.0": {2: "dna_r10.4.1_e8.2_5khz_400bps_sup_v4.2.0_5mc_CG_v2"},
        "v4.1.0": {2: "dna_r10.4.1_e8.2_4khz_400bps_sup_v4.1.0_5mc_CG_v2"},
        "v3.5.1": {2: "dna_r10.4.1_e8.2_400bps_sup_v3.5.1_5mc_CG_v2"},
    },
    "hac": {
        "v4.2.0": {2: "dna_r10.4.1_e8.2_5khz_400bps_hac_v4.2.0_5mc_CG_v2"},
        "v4.1.0": {2: "dna_r10.4.1_e8.2_4khz_400bps_hac_v4.1.0_5mc_CG_v2"},
        "v3.5.1": {2: "dna_r10.4.1_e8.2_400bps_hac_v3.5.1_5mc_CG_v2"},
    },
    "fast": {
        "v4.2.0": {2: "dna_r10.4.1_e8.2_5khz_400bps_fast_v4.2.0_5mc_CG_v2"},
        "v4.1.0": {2: "dna_r10.4.1_e8.2_4khz_400bps_fast_v4.1.0_5mc_CG_v2"},
        "v3.5.1": {2: "dna_r10.4.1_e8.2_400bps_fast_v3.5.1_5mc_CG_v2"},
    },
}

# kit14 400bps 5hmc_5mC CG-context models (contains 5kHz and 4kHz models)
_kit14_5hmc_5mc_CG_models = {
    "sup": {
        "v4.2.0": {2: "dna_r10.4.1_e8.2_5khz_400bps_sup_v4.2.0_5hmc_5mc_CG_v2"},
        "v4.1.0": {2: "dna_r10.4.1_e8.2_4khz_400bps_sup_v4.1.0_5hmc_5mc_CG_v2"},
        "v4.0.0": {2: "dna_r10.4.1_e8.2_400bps_sup_v4.0.0_5hmc_5mc_CG_v2"},
    },
    "hac": {
        "v4.2.0": {2: "dna_r10.4.1_e8.2_5khz_400bps_hac_v4.2.0_5hmc_5mc_CG_v2"},
        "v4.1.0": {2: "dna_r10.4.1_e8.2_4khz_400bps_hac_v4.1.0_5hmc_5mc_CG_v2"},
        "v4.0.0": {2: "dna_r10.4.1_e8.2_400bps_hac_v4.0.0_5hmc_5mc_CG_v2"},
    },
    "fast": {
        "v4.2.0": {
            2: "dna_r10.4.1_e8.2_5khz_400bps_fast_v4.2.0_5hmc_5mc_CG_v2"
        },
        "v4.1.0": {
            2: "dna_r10.4.1_e8.2_4khz_400bps_fast_v4.1.0_5hmc_5mc_CG_v2"
        },
        "v4.0.0": {2: "dna_r10.4.1_e8.2_400bps_fast_v4.0.0_5hmc_5mc_CG_v2"},
    },
}

# kit14 260bps 5hmC_5mC CG-context models
_kit14_260bps_5hmc_5mc_CG_models = {
    "sup": {
        "v4.0.0": {2: "dna_r10.4.1_e8.2_260bps_sup_v4.0.0_5hmc_5mc_CG_v2"},
    },
    "hac": {
        "v4.0.0": {2: "dna_r10.4.1_e8.2_260bps_hac_v4.0.0_5hmc_5mc_CG_v2"},
    },
    "fast": {
        "v4.0.0": {2: "dna_r10.4.1_e8.2_260bps_fast_v4.0.0_5hmc_5mc_CG_v2"},
    },
}

# all-context models (contains 5kHz and 4kHz models)
_kit14_5mc_ac_models = {
    "sup": {
        "v4.2.0": {2: "dna_r10.4.1_e8.2_5khz_400bps_sup_v4.2.0_5mc_v2"},
        "v4.0.1": {2: "res_dna_r10.4.1_e8.2_4khz_400bps_sup_v4.0.1_5mc_v2"},
    },
}
_kit14_6ma_ac_models = {
    "sup": {
        "v4.2.0": {2: "dna_r10.4.1_e8.2_5khz_400bps_sup_v4.2.0_6ma_v2"},
        "v4.0.1": {2: "res_dna_r10.4.1_e8.2_4khz_400bps_sup_v4.0.1_6mA_v2"},
    },
}

MODEL_DICT = {
    "dna_r9.4.1_e8": {
        "5mc": {
            "CG": _R9_5mc_CG_models,
        },
    },
    "dna_r10.4.1_e8.2_400bps": {
        "5mc": {
            "C": _kit14_5mc_ac_models,
            "CG": _kit14_5mc_CG_models,
        },
        "6ma": {
            "A": _kit14_6ma_ac_models,
        },
        "5hmc_5mc": {
            "CG": _kit14_5hmc_5mc_CG_models,
        },
    },
    "dna_r10.4.1_e8.2_260bps": {
        "5hmc_5mc": {
            "CG": _kit14_260bps_5hmc_5mc_CG_models,
        },
    },
}

DEFAULT_REFINE_SHORT_DWELL_PARAMS = (4, 3, 0.5)
REFINE_ALGO_VIT_NAME = "Viterbi"
REFINE_ALGO_DWELL_PEN_NAME = "dwell_penalty"
REFINE_ALGOS = (REFINE_ALGO_DWELL_PEN_NAME, REFINE_ALGO_VIT_NAME)
DEFAULT_REFINE_ALGO = REFINE_ALGO_DWELL_PEN_NAME

PA_TO_NORM_SCALING_FACTOR = 1.4826

DEFAULT_PLOT_FIG_SIZE = (40, 10)
