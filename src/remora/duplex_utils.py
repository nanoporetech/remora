from collections import deque
from typing import List, Tuple
from dataclasses import dataclass

import parasail
from numpy import typing as npt

from remora import data_chunks as DC

CigarTuples = List[Tuple[int, int]]


@dataclass
class PairwiseAlignment:
    ref_start: int
    ref_end: int
    query_start: int
    query_end: int
    cigar: CigarTuples


def trim_parasail_alignment(alignment_result):
    ref_start = 0
    ref_end = alignment_result.len_ref
    query_start = 0
    query_end = alignment_result.len_query
    fixed_start = False
    fixed_end = False

    cigar_string = alignment_result.cigar.decode.decode()
    cigar_tuples = deque(DC.cigartuples_from_string(cigar_string))
    while not (fixed_start and fixed_end):
        fist_op, first_length = cigar_tuples[0]
        if fist_op in (1, 4):  # insert, soft-clip, increment query start
            query_start += first_length
            cigar_tuples.popleft()
        elif fist_op == 2:  # delete, increment reference start
            ref_start += first_length
            cigar_tuples.popleft()
        else:
            fixed_start = True

        last_op, last_length = cigar_tuples[-1]
        if last_op in (1, 4):  # decrement the query end
            query_end -= last_length
            cigar_tuples.pop()
        elif last_op == 2:  # decrement the ref_end
            ref_end -= last_length
            cigar_tuples.pop()
        else:
            fixed_end = True

    return PairwiseAlignment(
        ref_start=ref_start,
        ref_end=ref_end,
        query_start=query_start,
        query_end=query_end,
        cigar=list(cigar_tuples),
    )


def parasail_align(*, query, ref) -> PairwiseAlignment:
    """
    Semi-global alignment allowing for gaps at the start and end of the query
    sequence.

    :param query: str
    :param ref: str
    :return: PairwiseAlignment
    :raises RuntimeError when no matching/mismatching operations are found
    """
    alignment_result = parasail.sg_qx_trace_scan_32(
        query, ref, 10, 2, parasail.dnafull
    )
    assert alignment_result.len_ref == len(
        ref
    ), "alignment reference length is discordant with reference"
    assert alignment_result.len_query == len(
        query
    ), "alignment query length is discordant with query"
    assert alignment_result.end_ref == len(ref) - 1, "end_ref is not the end"

    try:
        return trim_parasail_alignment(alignment_result)
    except IndexError as e:
        raise RuntimeError(
            "failed to find match operations in pairwise alignment"
        ) from e


@dataclass
class SimplexDuplexMapping:
    duplex_to_simplex_mapping: npt.NDArray[int]
    trimmed_duplex_seq: str
    duplex_offset: int


def map_simplex_to_duplex(*, simplex_seq: str, duplex_seq: str):
    pairwise_alignment = parasail_align(query=simplex_seq, ref=duplex_seq)

    trimmed_duplex = duplex_seq[
        pairwise_alignment.ref_start : pairwise_alignment.ref_end
    ]
    duplex_to_simplex_mapping = (
        DC.make_sequence_coordinate_mapping(
            pairwise_alignment.cigar,
        ).astype(int)
        + pairwise_alignment.query_start
    )

    return SimplexDuplexMapping(
        duplex_to_simplex_mapping=duplex_to_simplex_mapping,
        trimmed_duplex_seq=trimmed_duplex,
        duplex_offset=pairwise_alignment.ref_start,
    )
