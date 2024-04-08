from typing import List, Tuple, Union
import numpy.typing as npt
import numpy as np
from Bio.Seq import Seq
from Bio import SeqIO
import re
import itertools

def get_joinable_pairs(seq: Seq):
    A_idx = [i.start() + 1 for i in re.finditer("A", str(seq))]
    T_idx = [i.start() + 1 for i in re.finditer("T", str(seq))]
    G_idx = [i.start() + 1 for i in re.finditer("G", str(seq))]
    C_idx = [i.start() + 1 for i in re.finditer("C", str(seq))]

    AT_idx = itertools.product(A_idx, T_idx)
    GC_idx = itertools.product(G_idx, C_idx)
    pairs = []
    for at in AT_idx:
        pairs.append((at[1], at[0]) if at[0] > at[1] else (at[0], at[1]))
    for gc in GC_idx:
        pairs.append((gc[1], gc[0]) if gc[0] > gc[1] else (gc[0], gc[1]))
    return pairs

def enumerate_pairs(fastafile: str) -> List[Tuple[int, int]]:
    # 課題 2-1
    seq = list(SeqIO.parse(fastafile, "fasta"))[0].seq
    pairs = get_joinable_pairs(seq)
    return pairs

def enumerate_possible_pairs(fastafile: str, min_distance: int=4) -> List[Tuple[int, int]]:
    # 課題 2-2
    seq = list(SeqIO.parse(fastafile, "fasta"))[0].seq
    pairs = get_joinable_pairs(seq)
    # ここまでは 2-1
    pairs = list(filter(lambda p: p[0]+min_distance<p[1], pairs))
    return pairs

def enumerate_continuous_pairs(fastafile: str, min_distance: int=4, min_length: int=2) -> List[Tuple[int, int, int]]:
    # 課題 2-3
    seq = list(SeqIO.parse(fastafile, "fasta"))[0].seq
    pairs = get_joinable_pairs(seq)
    pairs = list(filter(lambda p: p[0]+min_distance<p[1], pairs))
    # ここまでは 2-2
    stems = []
    is_exist = [[False] * (len(seq)+1) for i in range(len(seq)+1)]
    # 存在判定をO(1)で行うための準備
    for pair in pairs:
        is_exist[pair[0]][pair[1]] = True
    
    for pair in sorted(pairs):
        now = pair
        stem_length = 1
        while is_exist[now[0]+1][now[1]-1]:
            now = (now[0]+1, now[1]-1)
            stem_length += 1
        if stem_length >= min_length:
            stems.append((pair[0], pair[1], stem_length))
    
    return stems

def create_dotbracket_notation(fastafile: str, min_distance: int=4, min_length: int=2) -> str:
    # 課題 2-4
    seq = list(SeqIO.parse(fastafile, "fasta"))[0].seq
    pairs = get_joinable_pairs(seq)
    pairs = list(filter(lambda p: p[0]+min_distance<p[1], pairs))
    stems = []
    is_exist = [[False] * (len(seq)+1) for i in range(len(seq)+1)]
    for pair in pairs:
        is_exist[pair[0]][pair[1]] = True
    for pair in sorted(pairs):
        now = pair
        stem_length = 1
        while is_exist[now[0]+1][now[1]-1]:
            now = (now[0]+1, now[1]-1)
            stem_length += 1
        if stem_length >= min_length:
            stems.append((pair[0], pair[1], stem_length))
    # ここまでは 2-3
    dot_list = ["."] * len(seq)
    for stem in stems:
        for i in range(stem[2]):
            dot_list[stem[0]-1+i] = '('
            dot_list[stem[1]-1-i] = ')'
    dot_str = "".join(dot_list)

    return dot_str

if __name__ == "__main__":
    filepath = "data/AUCGCCAU.fasta"
    # filepath = "data/NM_014495.4.fasta"
    # 課題 2-1
    print(enumerate_pairs(filepath))
    # 課題 2-2
    print(enumerate_possible_pairs(filepath))
    # 課題 2-3
    print(enumerate_continuous_pairs(filepath, 2))
    # 課題 2-4
    print(create_dotbracket_notation(filepath, 2))


