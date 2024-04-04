from typing import List, Union
import numpy.typing as npt
import numpy as np
from Bio.Seq import Seq
from Bio import SeqIO
import re

def base_count(fastafile: str) -> List[int]:
    # 課題 1-1
    for seq_record in SeqIO.parse(fastafile, "fasta"):
        seq = seq_record.seq
    ATGC_content = [seq.count("A"),
                    seq.count('T'),
                    seq.count("G"),
                    seq.count("C")]
    return ATGC_content # A, T, G, C

def gen_rev_comp_seq(fastafile: str) -> str:
    # 課題 1-2
    for seq_record in SeqIO.parse(fastafile, "fasta"):
        seq = seq_record.seq
    rev_comp = seq.reverse_complement()
    return rev_comp

def gc_percent_from_sequence(seq: Seq):
    gc_content = (seq.count("G") + seq.count("C"))

    return gc_content / len(seq)

def calc_gc_content(fastafile: str, window: int=1000, step: int=300) -> Union[npt.NDArray[np.float_], List[float]]:
    # 課題 1-3
    for seq_record in SeqIO.parse(fastafile, "fasta"):
        seq = seq_record.seq
    gc_content = [round(100*gc_percent_from_sequence(seq[i:i+window]), 1) for i in range(0, len(seq) - window + 1, step)]
    # 値を出力するところまで。matplotlibを使う部分は別途実装してください。
    return gc_content

def search_motif(fastafile: str, motif: str) -> List[str]:
    # 課題 1-4
    for seq_record in SeqIO.parse(fastafile, "fasta"):
        seq = seq_record.seq
        inv_complement = seq.reverse_complement()
    iters = [f'F{i.span()[0]+1}' for i in re.finditer(motif, str(seq))]
    inv_iters = [f'R{len(seq) - i.span()[0]}' for i in re.finditer(motif, str(inv_complement))]
    return iters + inv_iters


def translate_with_cut_unused(seq):
    rest = len(seq) % 3
    if rest:
        seq = seq[0:-rest]
    return seq.translate(stop_symbol='_')

def translate(fastafile: str) -> List[str]:
    # 課題 1-5
    for seq_record in SeqIO.parse(fastafile, "fasta"):
        seq = seq_record.seq
        inv_comp = seq.reverse_complement()

    candidates = []
    for i in range(3):
        candidates.append(translate_with_cut_unused(seq[i:]))
        candidates.append(translate_with_cut_unused(inv_comp[i:]))

    amino_seqs = []
    for candidate in candidates:
        amino_seqs += re.findall(r'M[^_]*$|M[^_]*_', str(candidate))

    return amino_seqs

if __name__ == "__main__":
    # filepath = "data/NT_113952.1.fasta"
    filepath = "data/NC_000012.fasta"
    # filepath = "data/ATGCCGT.fasta"
    # 課題 1-1
    print(base_count(filepath))
    # 課題 1-2
    print(gen_rev_comp_seq(filepath))
    # 課題 1-3/
    print(calc_gc_content(filepath))
    # 課題 1-4
    print(search_motif(filepath, "ATG"))
    # 課題 1-5
    print(translate(filepath))
