#!/usr/bin/env python
# coding: utf-8

## Author :: Nishant Kumar (21495) ##

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def import_data(filepath):
    data = list()
    with open(filepath) as file:
        # iterate over data file line by line
        for line in file:
            # strip and split line over tab symbols
            line = line.strip().split('\t')
            # convert map object into list of string values
            line = list(map(str, line))
            data.append(line)
    data = np.array(data, dtype=object)

    for i in range(1, len(data)):
        for j in range(len(data[1])):
            if( j != 0 ):
                data[i][j] = float(data[i][j])

    return data

# range of smokers and Non smokers
# range of Male Non smokers => (106 - 117)
# range of Male smokers => (117 - 129)
# range of Female Non smokers => (130 - 141)
# range of Female smokers => (114 -  153)

def smoker_nonsmoker(data):
    # non_smokers(Male)
    non_sm_male = [[0.0]*12 for _ in range(1, len(data))]
    for j in range(1, len(data)):
        for k in range(1, 13):
            non_sm_male[j-1][k-1] = data[j][k]

    # smokers(Male)
    sm_male = [[0.0]*12 for _ in range(1, len(data))]
    x_ = -1
    for j in range(1, len(data)):
        x_ = x_ + 1
        y_ = 0
        for k in range(13, 25):
            sm_male[x_][y_] = data[j][k]
            y_ = y_ + 1

    #non_smokers(Female)   
    non_sm_female=[[0.0]*12 for _ in range(1,len(data))]
    x_ = -1
    for j in range(1,len(data)):
        x_ = x_ + 1
        y_ = 0
        for k in range(25,37):
            non_sm_female[x_][y_] = data[j][k]
            y_ = y_ + 1

    #smokers(Female)
    sm_female=[[0.0]*12 for p in range(1,len(data))]
    x_ = -1
    for j in range(1,len(data)):
        x_ = x_ + 1
        y_ = 0
        for k in range(37,49):
            sm_female[x_][y_] = data[j][k]
            y_ = y_ + 1
    return non_sm_male, sm_male, non_sm_female, sm_female

def p_value_histogram(all_male_female_data):
    # Null Hypothesis
    # null_matrix and A_matrix 
    null_matrix = [[0]*4 for _ in range(48)]
    for row in range(12):
        null_matrix[row][0] = 1
        null_matrix[row][2] = 1
    
    for row in range(12,24):
        null_matrix[row][0] = 1
        null_matrix[row][3] = 1

    for row in range(24,36):
        null_matrix[row][1] = 1
        null_matrix[row][2] = 1
    
    for row in range(36,48):
        null_matrix[row][1] = 1
        null_matrix[row][3] = 1

    # Alternative Hypothesis
    # null_matrix, A_matrix
    A_matrix = [[0]*4 for _ in range(48)]
    for row in range(12):
        A_matrix[row][0] = 1
    
    for row in range(12,24):
        A_matrix[row][1] = 1

    for row in range(24,36):
        A_matrix[row][2] = 1
    
    for row in range(36,48):
        A_matrix[row][3] = 1

    A_matrix = np.matrix(A_matrix)
    null_matrix = np.matrix(null_matrix)

    # Rank Calculation
    rank1 = np.linalg.matrix_rank(null_matrix)
    rank2 = np.linalg.matrix_rank(A_matrix)

    # F statistics
    N = 48
    I = np.identity(48) # identity matrix 48 X 48
    F = list()
    for i in range(len(all_male_female_data)):
        H = all_male_female_data[i]
        mat1 = np.matmul(H.T, I - (np.matmul(np.matmul(null_matrix, np.linalg.pinv(np.matmul(null_matrix.T, null_matrix))), null_matrix.T)))
        mat2 = np.matmul(H.T, I - (np.matmul(np.matmul(A_matrix, np.linalg.pinv(np.matmul(A_matrix.T, A_matrix))), A_matrix.T)))
    
        if np.matmul(mat2, H) == 0:
            f_stat = ((np.matmul(mat1, H)/0.0000000000000000001) - 1) * (N - rank2)/(rank2 - 1)
            F.append(f_stat)
        else:
            f_stat = ((np.matmul(mat1, H)/np.matmul(mat2, H)) - 1) * (N - rank2)/(rank2 - rank1)
            F.append(f_stat)

    F = np.array(F)
    F = F.tolist()

    for i in range(len(F)):
        F[i] = F[i][0]

    ### p-value calculation ###
    d_f_d = 48 - rank2
    d_f_n = rank2 - rank1
    p_score = 1 - stats.f.cdf(F, d_f_n, d_f_d)
    print("p-Values Generated")

    ### p-value histogram ###
    plt.hist(p_score, bins=15, density=True, facecolor='orange')
    plt.savefig("histogram.png")
    print("P-Values Histogram Saved\n")
    return p_score

def genes_symbols(data):
    # Gene symbols
    g_symbol = []
    for i in range(len(data)):
        if (len(data[i]) > 49):
            g_symbol.append(data[i][49])
        else:
            g_symbol.append('NA')

    index = 0
    g_dict = dict()
    g_list = list()
    g_symbol_lst = list()
    for i in range(len(p_score)):
        if p_score[i] < 0.05:
            index = index + 1
            g_dict[i] = [data[i][0], p_score[i][0]]
            g_list.append(data[i][0])
            if g_symbol[i] != 'NA':
                g_symbol_lst.append(g_symbol[i])

    # save genes symbol list 
    with open('genes_symbols_list.txt', 'w') as g_file:
        for line in g_symbol_lst:
            g_file.write("".join(line) + "\n")
    print("Genes symbols file saved with name genes_symbols_list.txt")

    # save genes probe list
    with open('genes_porbe_list.txt', 'w') as g_file:
        for line in g_list:
            g_file.write("".join(line) + "\n")
    print("Genes probe list saved with name genes_probe_list.txt")
    return g_dict, g_list, g_symbol_lst, g_symbol
# false discovery rate
def FDR(p_score):
    q = 0.05
    n0 = 811 * 0.2
    fdr = list()
    for i in range(len(p_score)):
        fdr.append(p_score[i][0])
    fdr = np.sort(fdr)
    c = 0
    for i in range(0, len(fdr)):
        if n0 * fdr[i]/(i+1) <= q:
            c = c + 1
    return fdr

# Intersect with the following gene lists: Xenobiotic metabolism, Free Radical Response, DNA Repair, Natural Killer Cell Cytotoxicity.
##### Find Intersection (File 1) #####
def genes_and_Xenobiotic(g_symbol_lst):
    Xenobiotic_data_file = list()
    with open('../data/XenobioticMetabolism1.txt') as Xenobiotic_file:
        next(Xenobiotic_file)
        next(Xenobiotic_file)
        # iterate over file line by line
        for line in Xenobiotic_file:
            # strip and split line over tab symbols
            line = line.strip().split('\n')
            # convert map object into list of string values
            line = list(map(str, line))
            Xenobiotic_data_file.append(line)

    check = list()
    for i in range(len(Xenobiotic_data_file)):
        check.append(Xenobiotic_data_file[i][0])

    gsIntxs = list()
    for gs in g_symbol_lst:
        for xs in check:
            if gs == xs:
                gsIntxs.append(gs)
    print("Genes Symbol Intersection with Xenobiotic ",gsIntxs)
    print("Numbers of Genes that are intersecting with Xenobiotic Metabolism: ", len(gsIntxs))
    return gsIntxs

##### Find Intersection (File 2) #####
def genes_and_freeRadical(g_symbol_lst):
    free_readical_data_file = list()
    with open("../data/FreeRadicalResponse.txt") as free_readical_file:
        next(free_readical_file)
        next(free_readical_file)
        # iterate over file line by line
        for line in free_readical_file:
            # strip and split line over tab symbols
            line = line.strip().split('\t')
            # convert map object into list of string values
            line = list(map(str, line))
            free_readical_data_file.append(line)

    check = list()
    for i in range(len(free_readical_data_file)):
        check.append(free_readical_data_file[i][0])

    gsIntFrs = list()
    for gs in g_symbol_lst:
        for frs in check:
            if gs == frs:
                gsIntFrs.append(gs)

    print("Genes Symbol Intersection with Free Radical Response ", gsIntFrs)
    print("Numbers of Genes that are intersecting with free Radical Response: ", len(gsIntFrs))
    return gsIntFrs

##### Find Intersection (File3) #####
def genes_and_DNA(g_symbol_lst):
    DNA_data_file = list()
    with open('../data/DNARepair1.txt') as DNA_file:
        next(DNA_file)
        next(DNA_file)
        # iterate over file line by line
        for line in DNA_file:
            # strip and split line over tab symbols
            line = line.strip().split('\t')
            # convert map object into list of string values
            line = list(map(str, line))
            DNA_data_file.append(line)

    check = list()
    for i in range(len(DNA_data_file)):
        check.append(DNA_data_file[i][0])

    gsIntDnas = list()
    for gs in g_symbol_lst:
        for dnas in check:
            if gs == dnas:
                gsIntDnas.append(gs)

    print("Genes Symbol Intersetion with DNA Repair:", gsIntDnas)
    print("Numbers of Genes that are intersecting with DNA Repair: ", len(gsIntDnas))
    return gsIntDnas

##### Find Intersection (File 4) #####
def genes_and_NKCell(g_symbol_lst):
    NK_data_file = list()
    with open('../data/NKCellCytotoxicity.txt') as NK_file:
        next(NK_file)
        next(NK_file)
        # iterate over file line by line
        for line in NK_file:
            # strip and split line over tab symbols
            line = line.strip().split('\t')
            # convert map object into list of string values
            line = list(map(str, line))
            NK_data_file.append(line)

    check = list()
    for i in range(len(NK_data_file)):
        check.append(NK_data_file[i][0])

    gsIntNks = list()
    for gs in g_symbol_lst:
        for nks in check:
            if gs == nks:
                gsIntNks.append(gs)

    print("Genes Symbol Intersetion with NKCellCytotoxicity : ", gsIntNks)
    print("Numbers of Genes that are intersecting with NKCellCytotoxicity: ", len(gsIntNks))
    return gsIntNks

def group_with_another_data(gxs,gfrs,gDnas,gNks,data):
    # Report intersection counts for each list, split into four groups; 
    # Going down in women smokers vs non-smokers/going up in women smokers vs non-smokers.

    ##### Grouping Lists #####
    female_sm_up = []
    female_sm_down = []
    male_sm_up = []
    male_sm_down = []

    #### Grouping (1) ####
    rows_values = list()
    t3mp_genes = list()
    for i in range(len(g_symbol)):
        for j in range(len(gxs)):
            if(g_symbol[i] == gxs[j]):
                t3mp_genes.append(gxs[j])
                rows_values.append(i)
                break 

    print("\nGrouping(Xenobiotic Metabolism): ")
    print("Probes-Names","Genes-Symbols","Male-NonSm","Male-Sm","Female-NonSm","Female-Sm")
    for i in rows_values:
        male_nonsm=all_male_female_data[i,0:12]
        male_sm=all_male_female_data[i,12:24]
        female_nonsm=all_male_female_data[i,24:36]
        female_sm=all_male_female_data[i,36:48]
        male_nonsm_mean=np.mean(male_nonsm)
        male_sm_mean=np.mean(male_sm)
        female_nonsm_mean=np.mean(female_nonsm)
        female_sm_mean=np.mean(female_sm)
        if(male_nonsm_mean<male_sm_mean):
            male_sm_up.append(g_symbol[i])
        if(male_nonsm_mean>male_sm_mean):
            male_sm_down.append(g_symbol[i])
        if(female_nonsm_mean<female_sm_mean):
            female_sm_up.append(g_symbol[i])
        if(female_nonsm_mean>female_sm_mean):
            female_sm_down.append(g_symbol[i])

        print(data[i+1][0],g_symbol[i],male_nonsm_mean,male_sm_mean,female_nonsm_mean,female_sm_mean)

    #### Grouping (2) ####

    rows_values = []
    t3mp_genes = []
    for i in range(len(g_symbol)):
        for j in range(len(gfrs)):
            if(g_symbol[i] == gfrs[j]):
                t3mp_genes.append(gfrs[j])
                rows_values.append(i)
                break 

    #### Grouping (3) ####

    rows_values = []
    t3mp_genes = []
    for i in range(len(g_symbol)):
        for j in range(len(gDnas)):
            if(g_symbol[i] == gDnas[j]):
                t3mp_genes.append(gDnas[j])
                rows_values.append(i)
                break 

    print("\nGrouping:(DNA Repair)")
    print("Probes-Names","Genes-Symbols","Male-NonSm","Male-Sm","Female-NonSm","Female-Sm")
    for i in rows_values:
        male_nonsm=all_male_female_data[i,0:12]
        male_s=all_male_female_data[i,12:24]
        female_ns=all_male_female_data[i,24:36]
        female_s=all_male_female_data[i,36:48]
        male_nonsm_mean=np.mean(male_nonsm)
        male_sm_mean=np.mean(male_sm)
        female_nonsm_mean=np.mean(female_nonsm)
        female_sm_mean=np.mean(female_sm)
        if(male_nonsm_mean<male_sm_mean):
            male_sm_up.append(g_symbol[i])
        if(male_nonsm_mean>male_sm_mean):
            male_sm_down.append(g_symbol[i])
        if(female_nonsm_mean<female_sm_mean):
            female_sm_up.append(g_symbol[i])
        if(female_nonsm_mean>female_sm_mean):
            female_sm_down.append(g_symbol[i])
        print(data[i+1][0],g_symbol[i],male_nonsm_mean,male_sm_mean,female_nonsm_mean,female_sm_mean)

    #### Grouping (4) ####
    rows_values = []
    t3mp_genes = []
    for i in range(len(g_symbol)):
        for j in range(len(gNks)):
            if(g_symbol[i]==gNks[j]):
                t3mp_genes.append(gNks[j])
                rows_values.append(i)
                break 

    print("\nGrouping:(Natural Killer Cell)")
    print("Probes-Names","Genes-Symbols","Male-NonSm","Male-Sm","Female-NonSm","Female-Sm")
    for i in rows_values:
        male_nonsm = all_male_female_data[i,0:12]
        male_sm = all_male_female_data[i,12:24]
        female_nonsm = all_male_female_data[i,24:36]
        female_sm = all_male_female_data[i,36:48]
        male_nonsm_mean = np.mean(male_nonsm)
        male_sm_mean = np.mean(male_sm)
        female_nonsm_mean = np.mean(female_nonsm)
        female_sm_mean = np.mean(female_sm)
        if(male_nonsm_mean < male_sm_mean):
            male_sm_up.append(g_symbol[i])
        if(male_nonsm_mean > male_sm_mean):
            male_sm_down.append(g_symbol[i])
        if(female_nonsm_mean < female_sm_mean):
            female_sm_up.append(g_symbol[i])
        if(female_nonsm_mean > female_sm_mean):
            female_sm_down.append(g_symbol[ i])
        print(data[i+1][0],g_symbol[i],male_nonsm_mean,male_sm_mean,female_nonsm_mean,female_sm_mean)


    print("\n############### Final Grouping ###############")
    print("Female Smokers Up Genes" ,set(female_sm_up))
    print("Female Smokers Down Genes", set(female_sm_down))
    print("Male Smokers Up Genes" ,set(male_sm_up))
    print("Male Smokers Down Genes" ,set(male_sm_down))


if __name__ == "__main__":
    # Read Data 
    genes_data = import_data("../data/Raw Data_GeneSpring.txt")    
    
    non_sm_male, sm_male, non_sm_female, sm_female = smoker_nonsmoker(genes_data)

    male_data = np.hstack((non_sm_male, sm_male))
    female_data = np.hstack((non_sm_female, sm_female))
    all_male_female_data = np.hstack((male_data, female_data))

    p_score = p_value_histogram(all_male_female_data)
    g_dict, g_list, g_symbol_lst, g_symbol = genes_symbols(genes_data)

    print("\nFalse Discovery Rate:",FDR(p_score),"\n")

    gxs = genes_and_Xenobiotic(g_symbol_lst)

    gfrs = genes_and_freeRadical(g_symbol_lst)

    gDnas = genes_and_DNA(g_symbol_lst)

    gNks = genes_and_NKCell(g_symbol_lst)

    group_with_another_data(gxs,gfrs,gDnas,gNks,genes_data)