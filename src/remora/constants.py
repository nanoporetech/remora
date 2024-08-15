DEFAULT_NN_SIZE = 64
DEFAULT_BATCH_SIZE = 2_048
DEFAULT_SUPER_BATCH_SIZE = 100_000
DEFAULT_SUPER_BATCH_SAMPLE_FRAC = 1.0
DEFAULT_CHUNKS_PER_EPOCH = 10_000_000
DEFAULT_NUM_TEST_CHUNKS = 10_000
DEFAULT_CHUNK_CONTEXT = (200, 200)
DEFAULT_MIN_SAMPLES_PER_BASE = 5
DEFAULT_KMER_CONTEXT_BASES = (4, 4)
DEFAULT_KMER_LEN = sum(DEFAULT_KMER_CONTEXT_BASES) + 1
DEFAULT_FILT_FRAC = 0.1
WARN_PROP_REMOVED_THRESH = 0.5

# basecall chunk defaults
DEFAULT_BASECALL_CHUNK_LEN = 3_600

# train args
DEFAULT_EPOCHS = 100
DEFAULT_EARLY_STOPPING = 10

TYPE_CONVERTERS = {"str": str, "int": int, "float": float}

# optimizer
DEFAULT_OPTIMIZER = "AdamW"
DEFAULT_OPT_VALUES = (("weight_decay", 1e-4, "float"),)

# learning rate scheduler
DEFAULT_LR = 0.001
DEFAULT_SCHEDULER = "CosineAnnealingLR"
DEFAULT_SCH_VALUES = (
    ("T_max", DEFAULT_EPOCHS, "int"),
    ("eta_min", 1e-6, "float"),
)
DEFAULT_SCH_COOL_DOWN_EPOCHS = 5
DEFAULT_SCH_COOL_DOWN_LR = 1e-7

FINAL_MODEL_FILENAME = "model_final.checkpoint"
FINAL_TORCHSCRIPT_MODEL_FILENAME = "model_final.pt"
SAVE_DATASET_FILENAME = "remora_train_data.npz"

BEST_MODEL_FILENAME = "model_best.checkpoint"
BEST_TORCHSCRIPT_MODEL_FILENAME = "model_best.pt"

MODEL_VERSION = 3

DEFAULT_REFINE_HBW = 5

MODBASE_MODEL_NAME = "modbase_model.pt"
MODEL_DATA_DIR_NAME = "trained_models"

# set string values for sequence output types from datasets
# encoded k-mers for standard Remora models
DATASET_ENC_KMER = "enc_kmer"
# sequences and lengths for basecaller
DATASET_SEQS_AND_LENS = "seq_and_len"
DATASET_SEQ_OUTPUTS = dict(
    [
        (DATASET_ENC_KMER, ["enc_kmer"]),
        (DATASET_SEQS_AND_LENS, ["seq", "seq_len"]),
    ]
)
DATASET_TYPE_MODBASE = "modbase"
DATASET_TYPE_SEQ = "sequence"
DATASET_TYPES = set((DATASET_TYPE_MODBASE, DATASET_TYPE_SEQ))


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
        "v4.3.0": {1: "dna_r10.4.1_e8.2_400bps_sup@v4.3.0_5mCG_5hmCG@v1"},
        "v4.2.0": {
            3: "dna_r10.4.1_e8.2_5khz_400bps_sup_v4.2.0_5hmc_5mc_CG_v3",
            2: "dna_r10.4.1_e8.2_5khz_400bps_sup_v4.2.0_5hmc_5mc_CG_v2",
        },
        "v4.1.0": {2: "dna_r10.4.1_e8.2_4khz_400bps_sup_v4.1.0_5hmc_5mc_CG_v2"},
        "v4.0.0": {2: "dna_r10.4.1_e8.2_400bps_sup_v4.0.0_5hmc_5mc_CG_v2"},
    },
    "hac": {
        "v4.3.0": {1: "dna_r10.4.1_e8.2_400bps_hac@v4.3.0_5mCG_5hmCG@v1"},
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
_kit14_5hmc_5mc_ac_models = {
    "sup": {
        "v5.0.0": {1: "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mC_5hmC@v1"},
        "v4.3.0": {1: "dna_r10.4.1_e8.2_400bps_sup@v4.3.0_5mC_5hmC@v1"},
        "v4.2.0": {1: "dna_r10.4.1_e8.2_5khz_400bps_sup_v4.2.0_5hmc_5mc_v1"},
    },
    "hac": {
        "v5.0.0": {1: "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_5mC_5hmC@v1"},
        "v4.3.0": {1: "dna_r10.4.1_e8.2_400bps_sup@v4.3.0_5mC_5hmC@v1"},
    },
}
_kit14_6ma_ac_models = {
    "sup": {
        "v5.0.0": {1: "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_6mA@v1"},
        "v4.3.0": {1: "dna_r10.4.1_e8.2_400bps_sup@v4.3.0_6mA@v1"},
        "v4.2.0": {
            3: "dna_r10.4.1_e8.2_5khz_400bps_sup_v4.2.0_6ma_v3",
            2: "dna_r10.4.1_e8.2_5khz_400bps_sup_v4.2.0_6ma_v2",
        },
        "v4.0.1": {2: "res_dna_r10.4.1_e8.2_4khz_400bps_sup_v4.0.1_6mA_v2"},
    },
    "hac": {
        "v5.0.0": {1: "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_6mA@v1"},
        "v4.3.0": {1: "dna_r10.4.1_e8.2_400bps_sup@v4.3.0_6mA@v1"},
    },
}
_kit14_4mc_5mc_ac_models = {
    "sup": {
        "v5.0.0": {1: "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mC_4mC@v1"},
        "v4.3.0": {1: "res_dna_r10.4.1_e8.2_400bps_sup@v4.3.0_4mC_5mC@v1"},
    },
    "hac": {
        "v5.0.0": {1: "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_5mC_4mC@v1"},
    },
}

_rna004_m6A_drach_models = {
    "sup": {
        "v3.0.1": {1: "rna004_130bps_sup@v3.0.1_m6A_DRACH@v1"},
    },
}

_rna004_m6A_ac_models = {
    "sup": {
        "v5.0.0": {1: "rna004_130bps_sup@v5.0.0_m6A@v1"},
    },
    "hac": {
        "v5.0.0": {1: "rna004_130bps_hac@v5.0.0_m6A@v1"},
    },
}

_rna004_pseU_ac_models = {
    "sup": {
        "v5.0.0": {1: "rna004_130bps_sup@v5.0.0_pseU@v1"},
    },
    "hac": {
        "v5.0.0": {1: "rna004_130bps_hac@v5.0.0_pseU@v1"},
    },
}

_rna004_inosine_ac_models = {
    "sup": {
        "v5.0.0": {1: "rna004_130bps_sup@v5.0.0_inosine@v1"},
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
            "C": _kit14_5hmc_5mc_ac_models,
        },
        "4mc_5mc": {
            "C": _kit14_4mc_5mc_ac_models,
        },
    },
    "dna_r10.4.1_e8.2_260bps": {
        "5hmc_5mc": {
            "CG": _kit14_260bps_5hmc_5mc_CG_models,
        },
    },
    "rna004_130bps": {
        "m6a": {
            "DRACH": _rna004_m6A_drach_models,
            "A": _rna004_m6A_ac_models,
        },
        "pseU": {
            "T": _rna004_pseU_ac_models,
        },
        "inosine": {
            "A": _rna004_inosine_ac_models,
        },
    },
}

DEFAULT_REFINE_SHORT_DWELL_PARAMS = (4, 3, 0.5)
REFINE_ALGO_VIT_NAME = "Viterbi"
REFINE_ALGO_DWELL_PEN_NAME = "dwell_penalty"
REFINE_ALGOS = (REFINE_ALGO_DWELL_PEN_NAME, REFINE_ALGO_VIT_NAME)
DEFAULT_REFINE_ALGO = REFINE_ALGO_DWELL_PEN_NAME
ROUGH_RESCALE_LEAST_SQUARES = "least_squares"
ROUGH_RESCALE_THEIL_SEN = "theil_sen"
ROUGH_RESCALE_METHODS = (ROUGH_RESCALE_LEAST_SQUARES, ROUGH_RESCALE_THEIL_SEN)
DEFAULT_ROUGH_RESCALE_METHOD = ROUGH_RESCALE_LEAST_SQUARES

PA_TO_NORM_SCALING_FACTOR = 1.4826

MAX_POINTS_FOR_THEIL_SEN = 1000
