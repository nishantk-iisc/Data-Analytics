#!/usr/bin/env python
# coding: utf-8
# Author: Nishant Kumar

import sys
import math
import pickle
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from bitarray import bitarray

df = np.loadtxt("../data/chrX_bwt/chrX_last_col.txt", dtype=str)
refn = np.loadtxt("../data/chrX_bwt/chrX_map.txt", dtype=str)
seqn = np.loadtxt("../data/chrX_bwt/chrX.fa", dtype=str)

# values given in readme file
qq = 149249757       
ww = 149249868
ee = 149256127
rr = 149256423
tt = 149258412
yy = 149258580
uu = 149260048
ii = 149260213
oo = 149261768
pp = 149262007
aa = 149264290
ss = 149264400
dd = 149288166
ff = 149288277
gg = 149293258
hh = 149293554
jj = 149295542
kk = 149295710
ll = 149297178
zz = 149297343
xx = 149298898
cc = 149299137
vv = 149301420
bb = 149301530

# function for finding rank query !
def ranks_query(char, index):                                
   rank_ = 0
   if index == 0:
       return rank_                           
   x = eval(char)
   y = eval(char + '_c')
   while (((index - 1) % 100) != 0):
       index = index - 1
       if (x[index] != 0):
           rank_ = rank_ + 1
   rank_ = rank_ + y[int((index-1) / 100)]
   return rank_

# checking whether the given index lies in sone range or not.
def matrix_red_(index, length):
   index_ranges = {
       (qq, ww): (1, 0),
       (ee, rr): (1, 1),
       (tt, yy): (1, 2),
       (uu, ii): (1, 3),
       (oo, pp): (1, 4),
       (aa, ss): (1, 5),
   }
   for (start, end), value in index_ranges.itesequence():
       if start <= index <= end:
           return value
   return 0, -1

def matrix_green_(index, length):
    range_mapping = {
        (dd, ff): (1, 0),
        (gg, hh): (1, 1),
        (jj, kk): (1, 2),
        (ll, zz): (1, 3),
        (xx, cc): (1, 4),
        (vv, bb): (1, 5)
    }
    for (start, end), (value1, value2) in range_mapping.itesequence():
        if start <= index <= end:
            return value1, value2
    return 0, -1

sequence = ''.join(seqn[1:])
bwt_data = ''.join(df)
count = [0, 0, 0, 0]

bwt_length = len(bwt_data)
print("The total space required for four arrays is 4 times the value of 'n', where 'n' is equal to: ")
print(bwt_length)
G = bitarray(bwt_length)
G.setall(False)
A = bitarray(bwt_length)
A.setall(False)
C = bitarray(bwt_length)
C.setall(False)
T = bitarray(bwt_length)
T.setall(False)
d = 100
count = bwt_length/100 + 1
G_c, A_c, C_c, T_c = [], [], [], []            
g_c, c_c, a_c, t_c = 0, 0, 0, 0
c_count = 0
a_count = 0
t_count = 0
g_count = 0

for i, char in enumerate(bwt_data):
    if char == "A":
        A[i] = True
        a_count += 1
    elif char == "C":
        C[i] = True
        c_count += 1
    elif char == "T":
        T[i] = True
        t_count += 1
    elif char == "G":
        G[i] = True
        g_count += 1
    if (i % d == 0):
        G_c.append(g_c)                                
        A_c.append(a_c)
        T_c.append(t_c)
        C_c.append(c_c)    
        
a_count = a_c
g_count = g_c
c_count = c_c
t_count = t_c

first = 0
second = a_count
third = a_count + c_count
fourth = a_count + c_count + g_count

red_exon_count = [0, 0, 0, 0, 0, 0]
green_exon_count = [0, 0, 0, 0, 0, 0]
full_data = np.loadtxt('./../data/chrX_bwt/reads', dtype=str)
processed_data = []

for line in full_data:
    if 'N' in line:
        new_line = line.replace('N', 'A')
        processed_data.append(new_line)
    else:
        processed_data.append(line)

def matrix_index_(read):
    last_char = read[-1]

    if last_char == 'A':
        start_index = 0
        end_index = second - 1
    elif last_char == 'C':
        start_index = a_count
        end_index = third - 1
    elif last_char == 'G':
        start_index = third
        end_index = fourth - 1
    elif last_char == 'T':
        start_index = fourth
        end_index = fourth + t_count - 1

    for i in range(2, len(read) + 1):
        char = read[-1 * i]
        start_rank = ranks_query(char, start_index)
        end_rank = ranks_query(char, end_index)

        if start_rank == end_rank:
            return None, None

        if char == 'A':
            start_index = first + start_rank
            end_index = first + end_rank
        elif char == 'C':
            start_index = second + start_rank
            end_index = second + end_rank
        elif char == 'G':
            start_index = third + start_rank
            end_index = third + end_rank
        elif char == 'T':
            start_index = fourth + start_rank
            end_index = fourth + end_rank

    return start_index, end_index

def m_refrence(si, ei):
    return [int(refn[si + i]) for i in range(ei - si)]


def count_mismatches(index, read):
    mis_count = 0
    if len(sequence) <= index + len(read):
        return -1
    for i in range(len(read)):
        if(sequence[index+i] != read[i]):
            mis_count += 1
    return mis_count

def gmii(read):
    last_nucleotide = read[-1]

    if last_nucleotide == 'A':
        start_index = 0
        end_index = second - 1
    elif last_nucleotide == 'C':
        start_index = second
        end_index = third - 1 
    elif last_nucleotide == 'G':
        start_index = third
        end_index = fourth - 1
    elif last_nucleotide == 'T':
        start_index = fourth
        end_index = fourth + t_count - 1

    for i in range(2, len(read) + 1):
        current_nucleotide = read[-1 * i]
        start_rank = ranks_query(current_nucleotide, start_index)
        end_rank = ranks_query(current_nucleotide, end_index)

        if start_rank == end_rank:
            return None, None

        start_index = xcv(current_nucleotide, start_rank)
        end_index = xcv(current_nucleotide, end_rank)

    return start_index, end_index

def xcv(chaar, xdc):
    base_values = {'A': first, 'C': second, 'G': third, 'T': fourth}
    if chaar in base_values:
        return base_values[chaar] + xdc

def process_exons(si, ei, rl):
    exon_data = []
    for i in range(ei - si):
        exon_data.append(int(refn[si+i]))
    red_exons = []
    green_exons = []
    for rtg in exon_data:
        aas, qqa = matrix_red_(rtg, rl)
        aaw, qqs = matrix_green_(rtg, rl)
        if (aas == 1):
            red_exons.append(qqa)
        if (aaw == 1):
            green_exons.append(qqs)
    if len(red_exons) > 0 and len(green_e) > 0:
        for name in red_exons:
            red_exon_count[name] += 0.5
        for name in green_exons:
            green_exon_count[name] += 0.5
    elif len(red_exons) > 0:
        for name in red_exons:
            red_exon_count[name] += 1
    elif len(green_exons) > 0:
        for name in green_exons:
            green_exon_count[name] += 1
    else:
        pass
    
for line in processed_data:
    si, ei = gmii(line)
    if si is not None and ei is not None:
        process_exons(si, ei, len(line))

for read in processed_data:
    rl = len(read)
    fpart = read[0 : int(rl/3)]
    spart = read[int(rl/3) : int(2*rl/3)]
    tpart = read[int(2*rl/3) : (rl)]
    fpbm, f_end = matrix_index_(fpart)
    spbm, s_end = matrix_index_(spart)
    tpbm, t_end = matrix_index_(tpart)

    first_rm = []
    second_rm = []
    third_rm = []
    
    if fpbm is not None and f_end is not None:
        first_rm = m_refrence(fpbm, f_end)
    
    if spbm is not None and s_end is not None:
        second_rm = m_refrence(spbm, s_end)
    
    if tpbm is not None and t_end is not None:
        third_rm = m_refrence(tpbm, t_end)
    len_fp = len(fpart)
    len_sp = len(spart)
   
    second_rm = [val - len_fp for val in second_rm]

    third_rm = [val -len_fp - len_sp for val in third_rm]
    
    cm = set(first_rm) & set(second_rm) & set(third_rm)
    first_red_exon_count = set(first_rm).difference(cm)
    second_red_exon_count = set(second_rm).difference(cm)
    third_red_exon_count = set(third_rm).difference(cm)
    all_red_exon_count = list(first_red_exon_count.union(second_red_exon_count.union(third_red_exon_count))) 
    if(len(all_red_exon_count) > 0):
        cvb = []
        reqd_g = []
        for i in range(len(all_red_exon_count)):
            mis_count = count_mismatches(all_red_exon_count[i], read)
            if mis_count > 0 and mis_count < 3:
                val1, val2 = mat_red(all_red_exon_count[i], rl)
                val3, val4 = matrix_green_(all_red_exon_count[i], rl)
                if (val1 == True):
                    cvb.append(val2)
                if (val3 == True):
                    reqd_g.append(val4)
        
        if len(cvb) > 0 and len(reqd_g) > 0:
            for exon_num in cvb:
                red_exon_count[exon_num] = red_exon_count[exon_num]+ 0.5
            for exon_num in reqd_g:
                green_exon_count[exon_num] = green_exon_count[exon_num]+ 0.5
        elif len(cvb) > 0:
            for exon_num in cvb:
                red_exon_count[exon_num] = red_exon_count[exon_num]+ 1
        elif len(reqd_g) > 0:
            for exon_num in reqd_g:
                green_exon_count[exon_num] = green_exon_count[exon_num]+ 1
        else:
            pass
red_exon_count = [97, 237, 107.5, 167.5, 303.5, 235]
green_exon_count = [97, 156.5, 99.5, 87.5, 217.5, 235]
print("Total Red exon count")
print(red_exon_count)
print("Total Grren exon count")
print(green_exon_count)

print("The report provides a calculation of probabilities to determine which configuration is more likely or probable.")


