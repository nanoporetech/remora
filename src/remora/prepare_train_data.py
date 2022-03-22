import h5py
import pickle
import multiprocessing as mp

from tqdm import tqdm

from remora import log, RemoraError
from remora.data_chunks import RemoraRead, RemoraDataset
from remora.util import validate_mod_bases, get_can_converter, queue_iter

TAI_AVAIL = True
try:
    from taiyaki.mapped_signal_files import MappedSignalReader
except ImportError:
    TAI_AVAIL = False

LOGGER = log.get_logger()


def iter_reads_from_taiyaki(input_fn, base_pred, motifs, mod_base_control):
    with MappedSignalReader(input_fn) as input_msf:
        alphabet_info = input_msf.get_alphabet_information()
        can_conv = get_can_converter(
            alphabet_info.alphabet, alphabet_info.collapse_alphabet
        )
        if base_pred:
            label_conv = get_can_converter(
                alphabet_info.alphabet, alphabet_info.collapse_alphabet
            )
        else:
            label_conv = validate_mod_bases(
                alphabet_info.mod_bases,
                motifs,
                alphabet_info.alphabet,
                alphabet_info.collapse_alphabet,
                mod_base_control,
            )
        for read in input_msf:
            try:
                read = RemoraRead.from_taiyaki_read(read, can_conv, label_conv)
            except RemoraError as e:
                LOGGER.debug(
                    f"failed_taiyaki_remora_read_comv {read.read_id} {e}"
                )
                yield None
                continue
            read.add_motif_focus_bases(motifs)
            yield read


def iter_read_from_pickle(input_fn):
    try:
        with open(input_fn, "rb") as pkl_fp:
            while True:
                yield pickle.load(pkl_fp)
    except EOFError:
        pass
    except Exception as e:
        LOGGER.error(f"Failed to read RemoraReads pickle: {e}")


def fill_reads_q(
    reads_q,
    input_fn,
    base_pred=False,
    motifs=("N", 0),
    mod_base_control=True,
    num_proc=1,
    max_reads=None,
):
    put_reads = 0
    if h5py.is_hdf5(input_fn):
        try:
            for read in iter_reads_from_taiyaki(
                input_fn, base_pred, motifs, mod_base_control
            ):
                reads_q.put(read)
                put_reads += 1
                if max_reads is not None and put_reads >= max_reads:
                    break
        except RemoraError as e:
            LOGGER.error(f"Failed to read Taiyaki reads: {e}")
    else:
        for read in iter_read_from_pickle(input_fn):
            reads_q.put(read)
            put_reads += 1
            if max_reads is not None and put_reads >= max_reads:
                break
    for _ in range(num_proc):
        reads_q.put(StopIteration)


def reads_writer(remora_reads_q, out_reads_fn, num_proc):
    with open(out_reads_fn, "wb") as out_reads_fp:
        for read in queue_iter(remora_reads_q, num_proc):
            pickle.dump(read, out_reads_fp)


def extract_chunks_worker(
    reads_q,
    chunks_q,
    remora_reads_q,
    sig_map_refiner,
    max_chunks_per_read,
    chunk_context,
    kmer_context_bases,
    base_pred,
    base_start_justify,
    offset,
):
    for read in queue_iter(reads_q):
        if read is None:
            chunks_q.put([])
            continue
        read.refine_signal_mapping(sig_map_refiner)
        if remora_reads_q is not None:
            remora_reads_q.put(read)
        read.downsample_focus_bases(max_chunks_per_read)
        read_chunks = list(
            read.iter_chunks(
                chunk_context,
                kmer_context_bases,
                base_pred,
                base_start_justify,
                offset,
            )
        )
        LOGGER.debug(f"extracted {len(read_chunks)} chunks from {read.read_id}")
        chunks_q.put(read_chunks)
    if remora_reads_q is not None:
        remora_reads_q.put(StopIteration)
    chunks_q.put(StopIteration)


def check_alphabet(input_fn, base_pred, mod_base_control, mod_bases):
    if h5py.is_hdf5(input_fn):
        LOGGER.info("Extracting metadata from Taiyaki mapped signal file")
        with MappedSignalReader(input_fn) as input_msf:
            num_reads = len(input_msf.get_read_ids())
            alphabet_info = input_msf.get_alphabet_information()
        if base_pred and alphabet_info.alphabet != "ACGT":
            raise RemoraError(
                "Base prediction is not compatible with modified base "
                "training data. It requires a canonical alphabet (found "
                f"'{alphabet_info.alphabet}')."
            )
        mod_bases = alphabet_info.mod_bases
        mod_long_names = alphabet_info.mod_long_names
    else:
        LOGGER.info("Counting reads from RemoraRead pickle file")
        num_reads = sum(1 for _ in iter_read_from_pickle(input_fn))
        if mod_bases is None:
            if mod_base_control:
                mod_bases = ""
                mod_long_names = []
            else:
                raise RemoraError(
                    "Must provide modbases with RemoraReads pickle"
                )
        else:
            mod_bases, mod_long_names = zip(*mod_bases)
            mod_bases = "".join(mod_bases)
    num_types_specified = sum(
        (
            int(base_pred),
            int(mod_bases != ""),
            int(mod_base_control),
        )
    )
    if num_types_specified == 0:
        raise RemoraError(
            "Must specify one of modified base(s), modified base control, or "
            "base prediction model type option."
        )
    elif num_types_specified > 1:
        raise RemoraError(
            "Must specify only one of modified base(s), modified base "
            "control, and base prediction model type option."
        )

    return mod_bases, mod_long_names, num_reads


def extract_chunk_dataset(
    input_fn,
    out_fn,
    out_reads_fn,
    mod_bases,
    motifs,
    mod_base_control,
    chunk_context,
    min_samps_per_base,
    max_chunks_per_read,
    sig_map_refiner,
    base_pred,
    kmer_context_bases,
    base_start_justify,
    offset,
    num_proc,
):
    mod_bases, mod_long_names, num_reads = check_alphabet(
        input_fn, base_pred, mod_base_control, mod_bases
    )

    LOGGER.info("Allocating memory for output tensors")
    # initialize empty dataset with pre-allocated memory
    dataset = RemoraDataset.allocate_empty_chunks(
        num_chunks=max_chunks_per_read * num_reads,
        chunk_context=chunk_context,
        kmer_context_bases=kmer_context_bases,
        min_samps_per_base=min_samps_per_base,
        base_pred=base_pred,
        mod_bases=mod_bases,
        mod_long_names=mod_long_names,
        motifs=[mot.to_tuple() for mot in motifs],
        sig_map_refiner=sig_map_refiner,
        base_start_justify=base_start_justify,
        offset=offset,
    )

    LOGGER.info("Processing reads")
    reads_q = mp.Queue()
    filler_p = mp.Process(
        target=fill_reads_q,
        args=(reads_q, input_fn, base_pred, motifs, mod_base_control, num_proc),
        daemon=True,
        name="ReadQueueFiller",
    )
    filler_p.start()

    chunk_workers = []
    chunks_q = mp.Queue()
    remora_reads_q = mp.Queue(maxsize=1000) if out_reads_fn else None
    for chunk_pi in range(num_proc):
        chunk_workers.append(
            mp.Process(
                target=extract_chunks_worker,
                args=(
                    reads_q,
                    chunks_q,
                    remora_reads_q,
                    sig_map_refiner,
                    max_chunks_per_read,
                    chunk_context,
                    kmer_context_bases,
                    base_pred,
                    base_start_justify,
                    offset,
                ),
                daemon=True,
                name=f"ChunkExtractor{chunk_pi:03d}",
            )
        )
        chunk_workers[-1].start()

    if out_reads_fn is not None:
        reads_writer_p = mp.Process(
            target=reads_writer,
            args=(
                remora_reads_q,
                out_reads_fn,
                num_proc,
            ),
            daemon=True,
            name="RemoraReadWriter",
        )
        reads_writer_p.start()
    num_fail_chunks = 0
    for read_chunks in tqdm(
        queue_iter(chunks_q, num_proc),
        total=num_reads,
        smoothing=0,
        unit="Reads",
        desc="Extracting chunks",
    ):
        for chunk in read_chunks:
            try:
                dataset.add_chunk(chunk)
            except RemoraError:
                num_fail_chunks += 1
    if num_fail_chunks > 0:
        LOGGER.info(f"{num_fail_chunks} chunks removed by max seq len")

    filler_p.join()
    for c_worker in chunk_workers:
        c_worker.join()
    if out_reads_fn is not None:
        reads_writer_p.join()

    dataset.clip_chunks()
    dataset.shuffle()
    dataset.save(out_fn)

    LOGGER.info(f"Extracted {dataset.nchunks} chunks from {num_reads} reads.")
    LOGGER.info(f"Label distribution: {dataset.get_label_counts()}")
