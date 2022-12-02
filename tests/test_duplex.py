import pysam
import pytest
import numpy as np

from remora import (
    model_util,
    inference as infer,
    data_chunks as DC,
    util,
    duplex_utils as DU,
)


@pytest.mark.duplex
def test_fuzz_parasail():
    nucs = ["A", "C", "G", "T"]

    def random_sequence(seq_len):
        return "".join(np.random.choice(nucs, size=seq_len))

    def mutate_sequence(seq, p_err, p_indel):
        mutated_seq = []
        for idx in range(len(seq)):
            base = seq[idx]
            if uniform() <= p_err:
                base = random_state.choice(nucs)
            if uniform() <= p_indel:
                if np.random.uniform() > 0.5:
                    mutated_seq.append(base)
                    mutated_seq.append(random_state.choice(nucs))
                continue

            mutated_seq.append(base)

        return "".join(mutated_seq)

    random_state = np.random.RandomState(42)
    uniform = random_state.uniform

    for test_case in range(75):
        duplex = random_sequence(seq_len=5_000)
        simplex = mutate_sequence(duplex, p_err=0.05, p_indel=0.1)
        DU.map_simplex_to_duplex(simplex_seq=simplex, duplex_seq=duplex)

        # test ragged ends
        overhang = "T" * int(np.floor(uniform(low=5, high=100)))
        duplex_overhang = overhang + duplex + overhang
        DU.map_simplex_to_duplex(
            simplex_seq=simplex, duplex_seq=duplex_overhang
        )
        simplex_overhang = overhang + simplex + overhang
        DU.map_simplex_to_duplex(
            simplex_seq=simplex_overhang, duplex_seq=duplex
        )


@pytest.mark.unit
def test_duplex_alignment_to_signal_mapping_at_5prime_end():
    # Case 1: Simplex has extra sequence on the 5' end
    # TTTTTACGTACGTACG  [simplex]
    #      |||||||||||
    # -----ACGTACGTACG  [duplex]
    simplex = "TTTTTACGTACGTACG"
    duplex = "ACGTACGTACG"
    simplex_to_duplex_mapping = DU.map_simplex_to_duplex(
        simplex_seq=simplex, duplex_seq=duplex
    )
    assert (
        simplex_to_duplex_mapping.trimmed_duplex_seq == duplex
    ), "should not trim duplex read"
    assert simplex_to_duplex_mapping.duplex_offset == 0
    assert np.all(
        simplex_to_duplex_mapping.duplex_to_simplex_mapping
        == np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    )
    simplex_read_to_signal = np.arange(len(simplex))
    duplex_to_signal = DC.map_ref_to_signal(
        query_to_signal=simplex_read_to_signal,
        ref_to_query_knots=simplex_to_duplex_mapping.duplex_to_simplex_mapping,
    )
    assert np.all(
        duplex_to_signal
        == np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15])
    )

    # Case 2: Simplex is missing sequence on the 5' end
    simplex = "ACGTACGTACG"
    duplex = "TCGTTACGTACGTACG"
    # -----ACGTACGTACG
    #      |||||||||||
    # TCGTTACGTACGTACG
    #  ^^ this has no signal associated with it
    simplex_to_duplex_mapping = DU.map_simplex_to_duplex(
        simplex_seq=simplex, duplex_seq=duplex
    )
    assert (
        simplex_to_duplex_mapping.trimmed_duplex_seq == "ACGTACGTACG"
    ), "first 5 bases should be removed"
    assert simplex_to_duplex_mapping.duplex_offset == 5
    assert np.all(
        simplex_to_duplex_mapping.duplex_to_simplex_mapping
        == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    )
    simplex_read_to_signal = np.arange(len(simplex))
    duplex_to_signal = DC.map_ref_to_signal(
        query_to_signal=simplex_read_to_signal,
        ref_to_query_knots=simplex_to_duplex_mapping.duplex_to_simplex_mapping,
    )
    assert np.all(
        duplex_to_signal == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10])
    )

    # Case 3: Simplex is missing sequence at 5' end and starts with unpaired
    #   sequence
    # vv these bases are soft-clipped and their signal should not be used
    # GG-------GTACGTACG
    #          |||||||||
    # --TCGTTACGTACGTACG
    #    ^^ this has no signal associated with it

    simplex = "GGGTACGTACG"
    duplex = "TCGTTACGTACGTACG"

    simplex_to_duplex_mapping = DU.map_simplex_to_duplex(
        simplex_seq=simplex, duplex_seq=duplex
    )
    assert simplex_to_duplex_mapping.trimmed_duplex_seq == "GTACGTACG"
    assert simplex_to_duplex_mapping.duplex_offset == 7
    assert np.all(
        simplex_to_duplex_mapping.duplex_to_simplex_mapping
        == np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    )
    simplex_read_to_signal = np.arange(len(simplex))
    duplex_to_signal = DC.map_ref_to_signal(
        query_to_signal=simplex_read_to_signal,
        ref_to_query_knots=simplex_to_duplex_mapping.duplex_to_simplex_mapping,
    )
    assert np.all(
        duplex_to_signal == np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10])
    )


@pytest.mark.unit
@pytest.mark.duplex
def test_duplex_alignment_to_signal_mapping_at_3prime_end():
    # Case 4: Simplex is missing sequence at the 3' end
    # ACGTACGTACG------
    # |||||||||||
    # ACGTACGTACGTTTCGT
    #               ^^ Should not be classified
    simplex = "ACGTACGTACG"
    duplex = "ACGTACGTACGTTTCGT"
    simplex_to_duplex_mapping = DU.map_simplex_to_duplex(
        simplex_seq=simplex, duplex_seq=duplex
    )
    assert simplex_to_duplex_mapping.trimmed_duplex_seq == "ACGTACGTACG"
    assert simplex_to_duplex_mapping.duplex_offset == 0
    assert np.all(
        simplex_to_duplex_mapping.duplex_to_simplex_mapping
        == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    )
    simplex_read_to_signal = np.arange(len(simplex))
    duplex_to_signal = DC.map_ref_to_signal(
        query_to_signal=simplex_read_to_signal,
        ref_to_query_knots=simplex_to_duplex_mapping.duplex_to_simplex_mapping,
    )
    assert np.all(
        duplex_to_signal == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10])
    )

    # Case 5: Simplex is missing sequence at the end, but also has unaligned
    #   sequence
    #                  vv signal for these bases should not be used
    # ACGTACGTACG------AA
    # |||||||||||
    # ACGTACGTACGTTTCGT--
    #               ^^ these should not be classified
    simplex = "ACGTACGTACGAA"
    duplex = "ACGTACGTACGTTTCGT"
    simplex_to_duplex_mapping = DU.map_simplex_to_duplex(
        simplex_seq=simplex, duplex_seq=duplex
    )
    assert simplex_to_duplex_mapping.trimmed_duplex_seq == "ACGTACGTACG"
    assert simplex_to_duplex_mapping.duplex_offset == 0
    assert np.all(
        simplex_to_duplex_mapping.duplex_to_simplex_mapping
        == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    )
    simplex_read_to_signal = np.arange(len(simplex))
    duplex_to_signal = DC.map_ref_to_signal(
        query_to_signal=simplex_read_to_signal,
        ref_to_query_knots=simplex_to_duplex_mapping.duplex_to_simplex_mapping,
    )
    assert np.all(
        duplex_to_signal == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    )


@pytest.mark.unit
@pytest.mark.duplex
def test_duplex_alignment_to_signal_mapping_ragged_ends():
    # TTTTTACGTACGTACGTTTTTT [simplex]
    #      |||||||||||
    # -----ACGTACGTACG------ [duplex]
    # Test to make sure that duplex to simplex signal starts at position 5 and
    #   ends at 15

    simplex = "TTTTTACGTACGTACGTTTTTT"
    duplex = "ACGTACGTACG"
    simplex_to_duplex_mapping = DU.map_simplex_to_duplex(
        simplex_seq=simplex, duplex_seq=duplex
    )
    assert np.all(
        simplex_to_duplex_mapping.duplex_to_simplex_mapping
        == np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    )
    assert simplex_to_duplex_mapping.trimmed_duplex_seq == duplex
    assert simplex_to_duplex_mapping.duplex_offset == 0
    simplex_read_to_signal = np.arange(len(simplex))
    duplex_to_signal = DC.map_ref_to_signal(
        query_to_signal=simplex_read_to_signal,
        ref_to_query_knots=simplex_to_duplex_mapping.duplex_to_simplex_mapping,
    )
    assert np.all(
        duplex_to_signal
        == np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    )

    # -----ACGTACGTACG------
    #      |||||||||||
    # TCGTTACGTACGTACGTTTCGT
    #  ^^                ^^ should not be classified
    simplex = "ACGTACGTACG"
    duplex = "TCGTTACGTACGTACGTTTCGT"
    simplex_to_duplex_mapping = DU.map_simplex_to_duplex(
        simplex_seq=simplex, duplex_seq=duplex
    )
    assert np.all(
        simplex_to_duplex_mapping.duplex_to_simplex_mapping
        == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    )
    assert simplex_to_duplex_mapping.trimmed_duplex_seq == "ACGTACGTACG"
    assert simplex_to_duplex_mapping.duplex_offset == 5
    simplex_read_to_signal = np.arange(len(simplex))
    duplex_to_signal = DC.map_ref_to_signal(
        query_to_signal=simplex_read_to_signal,
        ref_to_query_knots=simplex_to_duplex_mapping.duplex_to_simplex_mapping,
    )
    assert np.all(
        duplex_to_signal == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10])
    )


@pytest.mark.unit
@pytest.mark.duplex
@pytest.mark.etl
def test_duplex_reads_data_etl(duplex_reads):
    pass


@pytest.mark.unit
@pytest.mark.duplex
def test_duplex_mod_infer_simple(duplex_reads, fw_mod_model_dir):
    FINAL_MODEL_FILENAME = "model_final.pt"
    model_path = str(fw_mod_model_dir / FINAL_MODEL_FILENAME)
    model, metadata = model_util.load_model(model_filename=model_path)
    motifs = [x[0] for x in metadata["motifs"]]
    duplex_caller = infer.DuplexReadModCaller(
        model=model, model_metadata=metadata
    )
    for duplex_read in duplex_reads:
        dat = duplex_caller.call_duplex_read_mod_probs(duplex_read)
        expected_sequence = (
            duplex_read.duplex_basecalled_sequence
            if not duplex_read.is_reverse_mapped
            else util.revcomp(duplex_read.duplex_basecalled_sequence)
        )
        assert dat["read_sequence"] == expected_sequence
        assert len(dat["template_positions"]) == len(dat["template_probs"])
        assert len(dat["complement_positions"]) == len(dat["complement_probs"])
        read_sequence = (
            duplex_read.duplex_basecalled_sequence
            if not duplex_read.is_reverse_mapped
            else util.revcomp(duplex_read.duplex_basecalled_sequence)
        )
        for template_position in dat["template_positions"]:
            assert (
                read_sequence[
                    template_position : template_position
                    + 2  # +2 because these should be CG motifs
                ]
                in motifs
            )
        for complement_position in dat["complement_positions"]:
            assert (
                read_sequence[complement_position - 1 : complement_position + 1]
                in motifs
            )


@pytest.mark.unit
@pytest.mark.duplex
@pytest.mark.skip(
    reason="causes deadlock with test_mod_infer_duplex in test_main.py which "
    "tests the same functionality"
)
def test_duplex_mod_infer_streaming(
    simplex_alignments,
    duplex_mapped_alignments,
    duplex_reads_and_pairs_pod5,
    fw_mod_model_dir,
    tmpdir_factory,
):
    reads_pod5, pairs = duplex_reads_and_pairs_pod5
    FINAL_MODEL_FILENAME = "model_final.pt"
    model_path = str(fw_mod_model_dir / FINAL_MODEL_FILENAME)
    model, metadata = model_util.load_model(model_filename=model_path)
    out_path = tmpdir_factory.mktemp("remora_tests") / "duplex_mod_infer.txt"

    infer.infer_duplex(
        simplex_pod5_path=reads_pod5,
        simplex_bam_path=simplex_alignments,
        duplex_bam_path=duplex_mapped_alignments,
        pairs_path=pairs,
        model=model,
        model_metadata=metadata,
        out_bam=str(out_path),
        num_extract_alignment_threads=1,
        num_infer_threads=1,
    )

    assert out_path.exists()
    n_expected_alignments = 0
    with pysam.AlignmentFile(
        duplex_mapped_alignments, "rb", check_sq=False
    ) as bam:
        for _aln in bam:
            n_expected_alignments += 1

    n_observed_alignments = 0
    with pysam.AlignmentFile(out_path, "rb", check_sq=False) as out_bam:
        for alignment in out_bam:
            # KeyError when not present
            alignment.get_tag("MM")
            alignment.get_tag("ML")
            n_observed_alignments += 1

    assert n_expected_alignments == n_observed_alignments
