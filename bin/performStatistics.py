#performs kruskal-wallis, wilcoxon/Mann-whitney/chi square tests.
#Usage : python3 performStatistics.py /Users/aniket/Desktop/Work/PROGRESS_20231101/PROGRESS/CleanAndProcess_Progress/outFiles/collateral_trial_20241113.xlsx /Users/aniket/Desktop/Work/PROGRESS_20231101/PROGRESS/CleanAndProcess_Progress/inFiles/performStatisticalTests.txt outFiles/trial_statsPackage_20241119.txt
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, wilcoxon
import sys
import os
from colorama import Fore, Back, Style



inputDataFile = sys.argv[1]
input_statsFile = sys.argv[2]
output_statsResult = sys.argv[3]
#df_subset_collaterals = pd.read_excel('/Users/aniket/Desktop/Work/PROGRESS_20231101/PROGRESS/CleanAndProcess_Progress/outFiles/collateral_trial_20241113.xlsx')

df = pd.read_excel(inputDataFile)

def main():
    statsPackage()
    #a = mann_whitney()
    #print (a)
    #kruskal_wallis()
    #chisq()
    #print (df_subset_collaterals)





def statsPackage(inputStatsFile = input_statsFile, matrix = df, outputFile = output_statsResult):
    inputFileRead = open(inputStatsFile, 'r')
    resultFile = open (outputFile, 'w')
    for line in inputFileRead.readlines():
        line = line.rstrip()
        if not line.startswith('#'):
            testName = line.split(', ')[2]
            colName1 = line.split(', ')[0]
            colName2 = line.split(', ')[1]
            #print (colName1)
            #print (colName2)
            #print (testName)
            try:
                if testName == 'MW-U':
                    result = mann_whitney(matrix= matrix, column1= colName1, column2= colName2)
                    print (Fore.GREEN + 'Mann-Whittney test performed for ' + colName1 + ', ' + colName2 + ' successfully' + Fore.RESET)
                    newLine = line + ', statistic : ' + str(result[0]) + ', p-value: ' + str(result[1])
                    resultFile.write(newLine + '\n')
                elif testName == 'kruskal':
                    result = kruskal_wallis(matrix= matrix, column1= colName1, column2= colName2)
                    print (Fore.GREEN + 'Kruskal-Wallis test performed for ' + colName1 + ', ' + colName2 + ' successfully' + Fore.RESET)
                    newLine = line + ', statistic : ' + str(result[0]) + ', p-value: ' + str(result[1])
                    resultFile.write(newLine + '\n')
                elif testName == 'chisq':
                    result = chisq(matrix= matrix, column1= colName1, column2= colName2)
                    print (Fore.GREEN + 'Chi-square test performed for ' + colName1 + ', ' + colName2 + ' successfully' + Fore.RESET)
                    #print (result)
                    newLine = line + ', statistic : ' + str(result[0]) + ', p-value: ' + str(result[1] )
                    resultFile.write(newLine + '\n')
                else:
                    print(Fore.RED + 'Error in test identification. Please check the input File' + Fore.RESET) 
            except:
                print (Fore.RED + 'Error in performing statistical tests. Please check all the column names.' + Fore.RESET)
        
    return()

def mann_whitney(matrix = df, column1 = 'chol_a_total', column2 = 'nikotin'):
    matrix_small = matrix[[column1, column2]]
    matrix_small = matrix_small.dropna()
    groups = matrix_small.groupby(column2)[column1].apply(list)
    #print(groups)

    stat, pval = (mannwhitneyu(*groups))
    return(stat, pval)

def kruskal_wallis(matrix = df, column1 = 'unters_alter', column2 = 'rentrop_grading'):
    matrix_small = matrix[[column1, column2]]
    matrix_small = matrix_small.dropna()
    groups = matrix_small.groupby(column2)[column1].apply(list)
    #print(groups)

    stat, pval = (kruskal(*groups))
    
    return(stat, pval)

def chisq(matrix = df, column1 = 'nikotin', column2 = 'diabetes'):
    # Create contingency table
    matrix_small = matrix[[column1, column2]]
    matrix_small = matrix_small.dropna()
    contingency_table = pd.crosstab(matrix_small[column1], matrix_small[column2])
    #print(contingency_table)
    # Perform Chi-Square test
    #print(chi2_contingency(contingency_table))
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return(chi2, p, dof, expected)


if __name__ == "__main__":
    main()