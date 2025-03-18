## Writing a script to automate annotation calculations
## February 07, 2023
## We are using templates created by Dr. Vilne in November 2022
##USAGE: python3 CleanOneLineDataTable_clean.py <inputFile.xlsx> <outputFilePrefix>
import os
import sys
import pandas as pd
import numpy as np
import math
from colorama import Fore, Back, Style
from scipy.stats import kurtosis 
from scipy.stats import skew 
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from datetime import date
from typing import Dict, List
import seaborn as sns
import subprocess
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, wilcoxon
inputFile = sys.argv[1]
#inputFile = "/Users/aniket/Desktop/Work/PROGRESS_20231101/PROGRESS/Scripts/one_line_table_with_annotations_21.03.2024.xlsx"
outputFolder = sys.argv[2]
outputFile_prefix = sys.argv[3]
outputFile = outputFolder + '/' + outputFile_prefix + '_main.tsv'

try:
    #os.makedirs(outputFolder, exist_ok=True)
    #os.system(str('mkdir + ' outputFolder))
    os.makedirs(outputFolder + '/plots', exist_ok=True)
    os.makedirs(outputFolder + '/temp', exist_ok=True)
    

except:
    print ("The output folders are not empty. Kindly make sure that te existing files are removed to avoid loss of analyses.\n")
#outputFile = "./outFiles/cleaned_oneLineTable_20240610.txt"
#outputFile = '/Users/aniket/Desktop/Work/PROGRESS_20231101/PROGRESS/Scripts/template/clinical_data/2022-11-14/cleaned_one_line_table_with_annotations_14.11.2022.tsv'
#Step 1 : opening excel file
print("Loading Data...")
df = pd.read_excel(inputFile)


str_out = 'Number of Rows :' + str((df.shape)[0]) + " \nNumber of Columns: " + str((df.shape)[1])


def main():
    
    #Step 2 : converting column names to lower case. Replacing ' ' with '_'
    colnames = df.columns
    dict_colnames = {}
    for i in range(0,len(colnames)):
        col1 = colnames[i].lower().replace(' ','_')
        dict_colnames[colnames[i]] = col1

    #print(dict_colnames)
    df.rename(columns=dict_colnames, inplace=True)
    #print(df.columns)

    #Step 3 : #clean data
    newdf = cleanDataset(df)
    
    #Step 4 : Get the cleaned dataset to output file for further use. However, remove new line characters first as they are observed to be present
    #newdf = newdf.applymap(lambda x: x.replace("\n", "") if isinstance(x, str) else x)
    newdf = newdf.apply(lambda col: col.map(lambda x: x.replace("\n", "") if isinstance(x, str) else x))

    # In the above step, we did not use the replace function directly as it gives a warning.
    newdf.to_csv( outputFile, sep='\t', index=False, header=True, na_rep='NA')
    #print(df)
    

    #Step 5: Process to calculate number of collaterals
    calculate_numberOfCollaterals(inputFileName= outputFile)

    #Step 6: get Stenosis Grades in tempFiles
    procClinDataStenGrad()

    #Step 7: get Stent Grade Individual Matrix in tempFiles
    getStentGradeIndfile()

    #Step 8 : Impute stent info
    #We can use a for loop here, but we are intentionally avoiding this as our data has an anamoly where segment 15 may be represented as segment 17 
    #Here we noticed that the old data fills the excluded_stent_impl_sten_seg_1_15.tsv but the new data does not.
    getStentInfo(stenSegFileName= outputFolder + '/temp/sten_seg_1_sten_grad.tsv', stenSegNo= '1', outFileName = outputFolder + '/temp/stent_impl_sten_seg_1_sten_grad.tsv')
    getStentInfo(stenSegFileName= outputFolder + '/temp/sten_seg_2_sten_grad.tsv', stenSegNo= '2', outFileName = outputFolder + '/temp/stent_impl_sten_seg_2_sten_grad.tsv')
    getStentInfo(stenSegFileName= outputFolder + '/temp/sten_seg_3_sten_grad.tsv', stenSegNo= '3', outFileName = outputFolder + '/temp/stent_impl_sten_seg_3_sten_grad.tsv')
    getStentInfo(stenSegFileName= outputFolder + '/temp/sten_seg_4_sten_grad.tsv', stenSegNo= '4', outFileName = outputFolder + '/temp/stent_impl_sten_seg_4_sten_grad.tsv')
    getStentInfo(stenSegFileName= outputFolder + '/temp/sten_seg_5_sten_grad.tsv', stenSegNo= '5', outFileName = outputFolder + '/temp/stent_impl_sten_seg_5_sten_grad.tsv')
    getStentInfo(stenSegFileName= outputFolder + '/temp/sten_seg_6_sten_grad.tsv', stenSegNo= '6', outFileName = outputFolder + '/temp/stent_impl_sten_seg_6_sten_grad.tsv')
    getStentInfo(stenSegFileName= outputFolder + '/temp/sten_seg_7_sten_grad.tsv', stenSegNo= '7', outFileName = outputFolder + '/temp/stent_impl_sten_seg_7_sten_grad.tsv')
    getStentInfo(stenSegFileName= outputFolder + '/temp/sten_seg_8_sten_grad.tsv', stenSegNo= '8', outFileName = outputFolder + '/temp/stent_impl_sten_seg_8_sten_grad.tsv')
    getStentInfo(stenSegFileName= outputFolder + '/temp/sten_seg_9_sten_grad.tsv', stenSegNo= '9', outFileName = outputFolder + '/temp/stent_impl_sten_seg_9_sten_grad.tsv')
    getStentInfo(stenSegFileName= outputFolder + '/temp/sten_seg_10_sten_grad.tsv', stenSegNo= '10', outFileName = outputFolder + '/temp/stent_impl_sten_seg_10_sten_grad.tsv')
    getStentInfo(stenSegFileName= outputFolder + '/temp/sten_seg_11_sten_grad.tsv', stenSegNo= '11', outFileName = outputFolder + '/temp/stent_impl_sten_seg_11_sten_grad.tsv')
    getStentInfo(stenSegFileName= outputFolder + '/temp/sten_seg_12_sten_grad.tsv', stenSegNo= '12', outFileName = outputFolder + '/temp/stent_impl_sten_seg_12_sten_grad.tsv')
    getStentInfo(stenSegFileName= outputFolder + '/temp/sten_seg_13_sten_grad.tsv', stenSegNo= '13', outFileName = outputFolder + '/temp/stent_impl_sten_seg_13_sten_grad.tsv')
    getStentInfo(stenSegFileName= outputFolder + '/temp/sten_seg_14_sten_grad.tsv', stenSegNo= '14', outFileName = outputFolder + '/temp/stent_impl_sten_seg_14_sten_grad.tsv')
    getStentInfo(stenSegFileName= outputFolder + '/temp/sten_seg_15_sten_grad.tsv', stenSegNo= '17', outFileName = outputFolder + '/temp/stent_impl_sten_seg_15_sten_grad.tsv')
    
    #Step 9: Could be run in loop
    for i in range(1,16):
        imputeStenosisGrade(inputFileName= outputFolder + '/temp/stent_impl_sten_seg_' + str(i) + '_sten_grad.tsv')

    #Step 10: Apply Annotations function and performing basic stats.
    applyingAnnotationFunction(inputFileName = outputFile, outputFileName = outputFolder + '/temp/' + outputFile_prefix + '_main.xlsx') #this is temporary file as we are now going to drop all unwanted columns here.
    df_temp = pd.read_excel(outputFolder + '/temp/' + outputFile_prefix + '_main.xlsx')
    print(df_temp)
    
    #dropping unwanted columns.
    df_temp = dropColumns(fileName= './inFiles/dropColumns.txt', matrix=df_temp)
    df_temp.to_excel(outputFolder + '/' + outputFile_prefix + '_main.xlsx', index=False) #main output File


    df_subset_collaterals = getCollateralDataSubset(inputFile = outputFolder + '/' + outputFile_prefix + '_main.xlsx')
    
    basicStats_continuous(fileName = './inFiles/statsFile_continuous.txt', matrix = df_subset_collaterals, outFileName = outputFolder + '/collaterals_statisticalAnalysis_continuousData.csv', prefix = 'collateral_data')
    basicStats_categorical(fileName = './inFiles/statsFile_categorical.txt', matrix = df_subset_collaterals, outFileName = outputFolder + '/collaterals_statisticalAnalysis_categorical.csv', prefix = 'collateral_data')


    #perform Statistics:
    os.system('python3 bin/performStatistics.py ' + outputFolder + '/' + outputFile_prefix + '_main.xlsx inFiles/performStatisticalTests.txt ' + outputFolder + '/statistics_results_' + outputFile_prefix + '.txt' )
    os.system('python3 bin/performStatistics.py '+ outputFolder + '/collateral_' + outputFile_prefix + '_main.xlsx inFiles/performStatisticalTests.txt ' + outputFolder + '/collateralData_statistics_results_' + outputFile_prefix+'.txt' )
    #Step 11: Final message.
    print("\n...Finished.")

def cleanDataset(dataset):
    #check for duplicates and remove them
    #print(dataset.duplicated().unique()) # we see that there are no entirely duplicated entries
    #Check for duplicate progress ID

    progress_id_repeated = dataset.duplicated(subset=['progress_id'])
    repeat_count = 0
    repeat_idx = []
    for i in range(0, len(progress_id_repeated)):
        if(progress_id_repeated[i] == True):
            repeat_count = repeat_count + 1
            repeat_idx.append(i)
            print ("Repeated entry found: " + str(dataset['progress_id'][i]))
            print("Either handle these or the script will drop the later entry")
    print ('Total repeated Progress IDs found are: ' + str(repeat_count))
    dataset = dataset.drop_duplicates(subset=['progress_id'], keep='first') #keeping the last repeated values
    #print(dataset['report_vorh'])
    #selecting patients with report available: report_vorh==1
    df_reports = dataset[dataset['report_vorh']==1]
    #df_reports = dataset.query('report_vorh==1')
    #print (df_reports)    
    #print(df_reports['report_vorh'])
    #dataset = dataset.drop_duplicates(subset=['progress_id'], keep='last') #keeping the last repeated values
    count = dataset['report_vorh'].value_counts()[1]
    print('Number of entries with reports (report_vorh): ' + str(count))
    #print(dataset['report_vorh'].unique()) #report_vorh == has only 2 unique values 0 and 1
    #replacing frame rate = NA; dicom_cleaning, dicom_export as NA. Not there in the dataset
    #colnames = df_reports.columns
    #print(sorted(colnames))

    #Checking age distribution below 33, count_age
    
    count = len(np.where(df_reports['unters_alter'] <= 33)[0])
    print ('Individual entries with age <= 33 : ' + str(count))
    temp = df_reports['stemi'].value_counts()
    #print(temp)
    print('Entries found in column stemi with value = 9: ' + str(temp[9]) +  '\n\t' + 'Replacing these with 0' )
    #Replace stemi = 9 and nstemi = 9 with 0
    df_reports.loc[df_reports["stemi"] == 9, "stemi"] = 0
    temp = df_reports['nstemi'].value_counts()
    print('Entries found in column stemi with value = 9: ' + str(temp[9]) +  '\n\t' + 'Replacing these with 0' )
    df_reports.loc[df_reports["nstemi"] == 9, "nstemi"] = 0
    
    # columns DTBT and STBT replace 0 with NA
    temp = df_reports['dtbt'].value_counts()
    #print(temp)
    print('Entries found in column DTBT with value = 0: ' + str(temp[0]) +  '\n\t' + 'Replacing these with NA' )
    
    df_reports.loc[df_reports["dtbt"] == 0, "dtbt"] = np.nan
    temp = df_reports['stbt'].value_counts()
    print('Entries found in column stbt with value = 0: ' + str(temp[0]) +  '\n\t' + 'Replacing these with NA' )
    df_reports.loc[df_reports["stbt"] == 0, "stbt"] = np.nan
    #temp = df_reports['stbt'].value_counts()
    #print(temp)

    #### replaceThresholds(thresholdFile = './inFiles/thresholdFile.txt', matrix = df)
    #### Used to replace all the below mentioned thresholds
    df_reports = replaceThresholds(thresholdFile = './inFiles/thresholdFile.txt', matrix = df_reports)


    # EF replace 0  or > 80 with NA
    #print('Replacing EF values greater than 80 and equal to 0 as NA' )

    #print('Entries found in column EF with value > 80: ' + str(temp[80]) +  '\n\t' + 'Replacing these with NA' )

    #df_reports.loc[df_reports["ef"] == 0, "ef"] = np.nan
    
    #df_reports.loc[df_reports["ef"] > 80, "ef"] = np.nan
    #temp = df_reports['ef'].value_counts()
    #print(temp)

    # GROESSE replace 0 or < 145 with NA # Should be resolved now, however double-check
    #temp = df_reports['groesse'].value_counts()
    #print(temp)
    #print('Replacing entries in Groesse (Height) with entries < 145 and =0 with NA' )
    
    #df_reports.loc[df_reports["groesse"] == 0, "groesse"] = np.nan
    #df_reports.loc[df_reports["groesse"] < 145, "groesse"] = np.nan
    
    # gewicht replace 0 or < 45 with NA # Should be resolved now, however double-check
    #temp = df_reports['gewicht'].value_counts()
    #print(temp)
    #print('Replacing entries in Gewicht (Weight) with entries < 45 with NA' )
    #df_reports.loc[df_reports["gewicht"] < 45, "gewicht"] = np.nan

    #Calculate BMI: wt (kg) / height(m)^2
    BMI = df_reports['gewicht'] * 10000/ (df_reports['groesse'] * df_reports['groesse'])
    #print(BMI)
    #df_reports['BMI'] = BMI
    #print(df_reports)
    df_reports.insert(loc = 2, column = 'BMI', value =  BMI, allow_duplicates = False)
    # syst_pressure and Diast_pressure, replace 0 with NA
    print('Replacing entries in Systolic and Diastolic BP with entries == 0 and less than 20 with NA' )
    df_reports.loc[df_reports['syst_pressure'] < 20, "syst_pressure"] = np.nan
    df_reports.loc[df_reports["diast_pressure"]  < 20, "diast_pressure"] = np.nan

    max_sys_bp = df_reports['syst_pressure'].max()
    min_sys_bp = df_reports['syst_pressure'].min()
    max_dias_bp = df_reports['diast_pressure'].max()
    min_dias_bp = df_reports['diast_pressure'].min()

    print('Range of values in syst_pressure is between : ' + str(max_sys_bp) + ' and ' + str(min_sys_bp))
    print('Range of values in diast_pressure is between : ' + str(max_dias_bp) + ' and ' + str(min_dias_bp) + '\n')

    # heartrate, set 0 to NA
    print('Replacing entries in heartrate with entries == 0 with NA' )
    df_reports.loc[df_reports['heartrate']  == 0, "heartrate"] = np.nan

    # nodcv, set 0 to NA
    print('Replacing entries in nodcv with entries == 0 with NA' )
    df_reports.loc[df_reports['nodcv'] == 0, "nodcv"] = np.nan


    ###CHECK and Process Pace-maker data
    # oth_devices, create 3 new columns: pacemaker, icd, laa_occluder # check, whether this is still necessary, i.e. is there free text in the column dat$pacemaker?
    print (df_reports['oth_devices'].value_counts())
    pacemaker_values = ['ICD', 'Schrittmacher', 'DDD ICD', 'SM', 'DDD-SM', 'DDD-ICD', 'HSM', 'AICD', 'Schrittmacher DDD', 'icd', 'DDDR-SM', 'DDR SM', 'Einkammer SM Typ Talos']
    icd_values = ['ICD', 'DDD ICD', 'DDD-ICD', 'AICD', 'icd']
    laa_occluder_values = ['LAA okkluder', 'LAA-Occluder']

    pacemaker = [0] * len(df_reports['oth_devices'])
    laa_occluder = [0] * len(df_reports['oth_devices'])
    icd_device = [0] * len(df_reports['oth_devices'])

    pacemaker_idx = df_reports['oth_devices'].isin(pacemaker_values).tolist()
    laa_occluder_idx = df_reports['oth_devices'].isin(laa_occluder_values).tolist()
    icd_device_idx = df_reports['oth_devices'].isin(icd_values).tolist()
    null_idx = df_reports['oth_devices'].isna().tolist()
    #count = 0
    for i in range(0, len(pacemaker_idx)):
        if (null_idx[i] == True):
            pacemaker[i] = np.nan
            laa_occluder[i] = np.nan
            icd_device[i] = np.nan
        
        else:
            if (pacemaker_idx[i] == True):
                pacemaker[i] = 1
            if (laa_occluder_idx[i] == True):
                laa_occluder[i] = 1
            if (icd_device_idx[i] == True):
                icd_device[i] = 1
    
    df_reports.insert(loc = 2, column = 'pacemaker', value =  pacemaker, allow_duplicates = False)
    df_reports.insert(loc = 2, column = 'laa_occluder', value =  laa_occluder, allow_duplicates = False)
    df_reports.insert(loc = 2, column = 'icd_device', value =  icd_device, allow_duplicates = False)
    #print(df_reports['pacemaker'].value_counts())
    #print(df_reports['laa_occluder'].value_counts())
    #print(df_reports['icd_device'].value_counts())

    # Following Lines can now be done by replaceThresholds()
    # gluc_a < 20 ; gluc_a > 1000 --> NA
    #print('Replacing entries in Gluc_a less than 20 and greater than 999 with NA' )
    #df_reports.loc[df_reports['gluc_a'] < 20, "gluc_a"] = np.nan
    #df_reports.loc[df_reports["gluc_a"]  > 999 , "gluc_a"] = np.nan

    # ldh_max, replace values > 6000 with NA
    #df_reports.loc[df_reports['ldh_max'] > 6000, "ldh_max"] = np.nan
    
    # platelets, replace values >1000 with NA
    #df_reports.loc[df_reports['platelets'] > 1000, "platelets"] = np.nan

    # leuk_a replace values > 50 000 or < 500 with NA
    #df_reports.loc[df_reports['leuk_a'] > 50000, "leuk_a"] = np.nan
    #df_reports.loc[df_reports['leuk_a'] < 500 , "leuk_a"] = np.nan
    
    #Replace males = 1; females = 0
    df_reports.loc[df_reports['sex'] == 'm', "sex"] = 1
    df_reports.loc[df_reports['sex'] == 'w', "sex"] = 0
    print( 'Encoding Males == 1 and Females == 0. \n')

    #df_reports = dropColumns(fileName= './inFiles/dropColumns.txt', matrix=df_reports)
    basicStats_continuous(fileName = './inFiles/statsFile_continuous.txt', matrix = df_reports, outFileName = outputFolder + '/basicStatisticalValues_continuousdata.csv')
    basicStats_categorical(fileName='./inFiles/statsFile_categorical.txt', matrix=df_reports, outFileName=outputFolder + '/CategoricalData_Distribution.csv')


    #print (df_reports)


    #Implement this later
    #plotSurvival(time_col='days_between_last_examen_and_death', event_col='death', matrix=df_reports)
    #calculate_numberOfCollaterals(df_reports)
    return(df_reports)

#  CTO.ID, CAD, CAD.ID,  NBMI, pr_acvb, valve_op, asd_cl 
def dropColumns(fileName = './inFiles/dropColumns.txt', matrix = df):
    file = open(fileName, 'r')
    cols_drop = []
    for line in file.readlines():
        line = line.rstrip()
        if not(line.startswith('#')):
            cols_drop.append(line)
    
    
    for i in cols_drop:
        try:
            matrix.drop(str(i), axis = 1)
            print(Fore.GREEN + 'Dropping Column :' + str(i) + Fore.RESET)
        except:
            print(Fore.RED + 'Dropping of a column :' + str(i) + ' resulted in error: ' + str(i) + '. Please check the column names in file dropColumns.txt' + Fore.RESET)
    file.close()   
    return(matrix)

# colname,mean,median,skewness,kurtosis,missingness 
def basicStats_continuous(fileName = './inFiles/statsFile_continuous.txt', matrix = df, outFileName = outputFolder + '/statisticalAnalysis.csv', prefix = 'whole_data'):
    file = open (fileName, 'r')
    outFile = open(outFileName , 'w')
    for line in file.readlines():
        line = line.rstrip()
        line_arr = line.split(',')
        col_name = line_arr[0].lower().replace(' ','_')
        line_arr.pop(0)
        statsPack = line_arr
        if col_name in matrix.columns:
            try:
                new_arr = [col_name, str(calculateMissingness(matrix, colname=str(col_name))), str(calculateSkewness(matrix=matrix, colname=str(col_name))), str(calculateKurtosis(matrix=matrix, colname=str(col_name)))]
                if ( 'mean' in statsPack):
                    #print(matrix[col_name].dropna().tolist())
                    new_arr.append(str(statistics.mean(matrix[col_name].dropna().tolist())))
                if ('median' in statsPack):
                    new_arr.append(str(statistics.median(matrix[col_name].dropna().tolist())))
                plotContinuous(values = matrix[col_name].dropna().tolist(), colName= col_name, outputPrefix= prefix)
            except:
                print('Cannot implement basicStats_continuous on the column: ' + str(col_name))
            
            new_line = ','.join(new_arr)
            outFile.write(new_line + '\n')
        else:
            print (col_name + ' is not present in the dataset. Please check the files again.')
    file.close()
    outFile.close()
    return()

#missingness, category 1 count, category 2 count, ... and so on
def basicStats_categorical(fileName = './inFiles/statsFile_categorical.txt', matrix = df, outFileName = 'Categorical_Distribution.csv', prefix = 'whole_data'):
    file = open(fileName, 'r')
    outFile = open(outFileName, 'w')
    for line in file.readlines():
        line = line.rstrip().lower().replace(' ','_')
        if (line in matrix.columns):
            try:
                dict_print = matrix[line].value_counts()
                keys = dict_print.keys()
                newLine = [line]
                values = []
                for key in keys:
                    #print(str(key) + ': ' + str(dict_print[key]))
                    values.append(dict_print[key]) 
                    newLine.append(str(key) + ': ' + str(dict_print[key]))
                newLine = ','.join(newLine)
                plotCategorical(keys, values, line, outputPrefix= prefix)
                outFile.write(newLine + '\n')
                print(newLine)
            except:
                print(Exception + '\n')
                print (Fore.RED + 'basicStats_categorical() failed to run for ' + line + '\n' + Fore.RESET)
    file.close()
    outFile.close()
    return()

def plotCategorical(keys, values, colName, outputPrefix = 'wholeData'):
    key_char = []
    for key in keys:
        key_char.append(str(key))
    #print(key_char, values)
    plt.bar(key_char, values, color = "red")
    
    addlabels(key_char, values)
    plt.title(colName)
    plt.savefig(outputFolder + '/plots/' + outputPrefix + '_' +str(colName) + '.png')
    plt.clf()
    return()

def plotContinuous(values, colName, outputPrefix = 'wholeData'):
    plt.hist(values)
    plt.title(colName)
    plt.savefig(outputFolder + '/plots/' + outputPrefix + '_' +  str(colName) + '.png')
    plt.clf()
    return()
#only for barplots
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i]) 
    return()

def calculateMissingness(matrix = df, colname = 'gewicht'):
    col = matrix[str(colname)].tolist()
    col_new = matrix[str(colname)].dropna().tolist()
    #print(len(col_new))
    #print(len(col))
    missingness = round(1 - (len(col_new) / len(col)), ndigits=4)
    #print('missingness : ' + str(missingness))

    
    return(missingness)

def calculateSkewness(matrix = df, colname = 'gewicht'):
    col_new = matrix[str(colname)].dropna().tolist()
    skew_value = round((skew(col_new, axis=0, bias=True)), ndigits= 3)
    return(skew_value)

def calculateKurtosis(matrix = df, colname = 'gewicht'):
    col_new = matrix[str(colname)].dropna().tolist()
    kurt = round((skew(col_new, axis=0, bias=True)), ndigits= 3)
    return(kurt)

"""
DESCRIPTION:
-------------
Here, we realized that the column "nr_of_collaterals" is not reliable and we need to calculate
the number of collaterals ourselves, based on the number of start and end vessel
numbered 1 to 15, i.e. the unique pairs of coordinates for each collateral

"""


def getCoordinates(p_id, vals, col_n, ix_l, c_map):
    
    
    for ix in ix_l:
        #print(ix)
        c1, c2, colat = "donor_segm"+ix, "receiving_segm"+ix, "nr_of_collaterals_"+ix.split("_")[1]
        if vals[col_n[c1]] != "NA" and vals[col_n[c2]] != "NA":
            fillMap(c_map, p_id, (vals[col_n[c1]], vals[col_n[c2]]))

def genIndex():
    ixs = []
    ix1Lst = [1,2,3,4,5,6]
    ix2Lst = [1,2,3,4]
    for ix1 in ix1Lst:
        for ix2 in ix2Lst:
            ixs.append("_"+str(ix1)+"_"+str(ix2))
    return ixs  

def getcolNum(inLst):

    outMap = {}
    for ix, col_name in enumerate(inLst):
        outMap[col_name] = ix
    return outMap

def fillMap(aMap, key, val):

        """
        Add another profile to the profile count.
        """

        if key not in aMap:
                aMap[key] = [val]
        else:
                aMap[key].append(val)

def fillEmpty(inputFile):

    emptyMap = {}
    inputFile = open(inputFile)
    header = inputFile.readline().replace(" ", "_").lower().rstrip("\n").split("\t")
    colNum = getcolNum(header)
    for line in inputFile:
        line = line.rstrip("\n")
        fields = line.split("\t")
        progress_id = fields[colNum["progress_id"]]
        progress_patient_id, progress_coro_id = str(int(progress_id.split("-")[0])), progress_id.split("-")[1]
        createEmptyMap(emptyMap, progress_patient_id)
    inputFile.close()
    return emptyMap

def fillSegStenGrad(pp_id, co_id, sg_l, s_map, e_map):

    if (pp_id, co_id) in e_map:
        s_map[(pp_id,co_id)] = convertListToString(sg_l, "\t")

def getStenGrad(p_id, vals, col_n):

    sten_grad = []
    seg_ix = list(range(1, 16))
    for ix in seg_ix:
        seg = "sten_seg_"+str(ix)+"_sten_grad"
        sten = int(vals[col_n[seg]])
        sten_grad.append(sten)
    return sten_grad

def createEmptyMap(e_map, pp_id):

    exams = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    for exam in exams:
        e_map[(pp_id, str(exam))] = convertListToString(['NA'] * 15, "\t")

def fillMap(aMap, key, val):

        """
        Add another profile to the profile count.
        """

        if key not in aMap:
                aMap[key] = [val]
        else:
                aMap[key].append(val)

def fillMap2(aMap, key, val1, val2):

        """
        Add another profile to the profile count.
        """

        if key not in aMap:
                aMap[key] = [[val1],[val2]]
        else:
                aMap[key][0].append(val1)
                aMap[key][1].append(val2)

def convertListToString(lst, sep):

        string = ""
        for i,el in enumerate(lst):
            if i != len(lst)-1:
                string+=str(el)+sep
            if i == len(lst)-1:
                string+=str(el)
        return string

def getStent(s_no, inputFileName = outputFile):

    sMap = {}
    #inputFile = open('/Users/aniket/Desktop/Work/PROGRESS_20231101/PROGRESS/Scripts/cleaned_oneLineTable_20240610.txt')
    inputFile = open(inputFileName, 'r')
    header = inputFile.readline().replace(" ", "_").lower().rstrip("\n").split("\t")
    colNum = getcolNum(header)
    ixPairs = genIndex()
    #colMap = {}
    for line in inputFile:
        line = line.rstrip("\n")
        fields = line.split("\t")
        progress_id, stent_segment = fields[colNum["progress_id"]], "NA"
        patient_id, exam = int(progress_id.split("-")[0]), int(progress_id.split("-")[1])
        stent_segment_of_interest = "stent_segment_no_"+ str(s_no)
        if stent_segment_of_interest in colNum:
            stent_segment = fields[colNum["stent_segment_no_"+str(s_no)]]
        fillMap2(sMap, patient_id, exam, stent_segment)
    inputFile.close()
    return sMap

def calculate_numberOfCollaterals(inputFileName = outputFile, outputFileName = outputFolder + '/numberOfCollaterals.txt'):
    outFile = open(outputFileName, 'w')
    outFile.write("progress_id" + '\t' + "nr_of_collaterals_1\n")
    inputFile = open(inputFileName, 'r')
    header = inputFile.readline().replace(" ", "_").lower().rstrip("\n").split("\t")
    #print(header)
    colNum = getcolNum(header)
    ixPairs = genIndex()
    #print(ixPairs)
    colMap = {}
    #print(colNum)
    progress_id = ''
    for line in inputFile:
        line = line.rstrip("\n")
        fields = line.split("\t")
        progress_id = fields[colNum['progress_id']]
        #nr_of_collaterals_1 = fields[colNum['nr_of_collaterals_1']]
        getCoordinates(progress_id, fields, colNum, ixPairs, colMap)
    inputFile.close()
    for  progress_id2 in colMap:
        outFile.write("\t".join([str(x) for x in [progress_id2, len(set(colMap[progress_id2]))]])+'\n')
    inputFile.close()
    outFile.close()
    return()

def plotSurvival(time_col = 'days_between_last_examen_and_death', event_col = 'death', matrix = df):
    df = matrix.dropna(subset=time_col)
    print(df)
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], event_observed = df[event_col])
    # Plot survival curve
    plt.figure(figsize=(10, 6))
    kmf.plot(label='Survival Curve')
    plt.title('Survival Plot')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.grid()
    plt.legend()
    plt.show()
    return()

#use defineThresholds matrix to set thresholds for real matrix
def replaceThresholds(thresholdFile = './inFiles/thresholdFile.txt', matrix = df):
    in_thresholdFile = open(thresholdFile,'r')
    header = in_thresholdFile.readline().rstrip()
    #print(header)
    for line in in_thresholdFile.readlines():
        line = line.rstrip().split(';')
        if (line[0] in matrix.columns):
            matrix.loc[matrix[line[0]] < int(line[1]), line[0]] = np.nan
            matrix.loc[matrix[line[0]] > int(line[2]), line[0]] = np.nan
            print('Matrix edited for ' + str(line[0]))
        
    #print(matrix)
    in_thresholdFile.close() 
    return(matrix)
#replaceThresholds()

def impStenGrad(toImp, impVal):

    outImp = []
    outImp.append(toImp[0])
    for el in toImp[1:]:
        outImp.append(impVal)
    return outImp

def reconstruct(orig, nna, nna_imp):

    out = orig.copy()
    c = 0
    for i,el in enumerate(orig):
        if el != 'NA':
            out[i] = str(nna_imp[c])
            c = c + 1
    return out

##############################################
############################################## writing 04_procClinDataStenGrad.py as function
def procClinDataStenGrad(inputFileName = outputFile):
    inputFile = open(inputFileName, 'r')
    output = open(outputFolder + '/temp/stenosisGradesCombined_seg1-15.tsv', 'w')
    #output = open(outputFolder + '/sten_seg_1-15_sten_grad_14.11.2022.csv')
    header = inputFile.readline().replace(" ", "_").lower().rstrip("\n").split("\t")
    colNum = getcolNum(header)
    #ixPairs = genIndex()
    #colMap = {}
    emptyMap = fillEmpty(inputFileName)
    segStenGradMap = emptyMap.copy()
    for line in inputFile:
        line = line.rstrip("\n")
        fields = line.split("\t")
        progress_id = (fields[colNum["progress_id"]])
        #print(progress_id)
        progress_patient_id, progress_coro_id = str(int(progress_id.split("-")[0])), (progress_id.split("-")[1])
        stenGradLst = getStenGrad(progress_id, fields, colNum)
        fillSegStenGrad(progress_patient_id, progress_coro_id, stenGradLst, segStenGradMap, emptyMap)      
    inputFile.close()
    
    output.write("\t".join(["progress_patient_id", "progress_coro_id", "sten_seg_1_sten_grad", \
                    "sten_seg_2_sten_grad", "sten_seg_3_sten_grad", "sten_seg_4_sten_grad", \
                    "sten_seg_5_sten_grad", "sten_seg_6_sten_grad", "sten_seg_7_sten_grad", \
                    "sten_seg_8_sten_grad", "sten_seg_9_sten_grad", "sten_seg_10_sten_grad", \
                    "sten_seg_11_sten_grad", "sten_seg_12_sten_grad", "sten_seg_13_sten_grad", \
                    "sten_seg_14_sten_grad", "sten_seg_15_sten_grad"]) + '\n')
    
    for (pat_id, coro_id) in segStenGradMap:
        output.write("\t".join([str(x) for x in [pat_id.zfill(4), coro_id, segStenGradMap[(pat_id, coro_id)]]]) + '\n')
    output.close()
    return()

def getStentGradeIndfile(inputRscriptFile = './bin/StenGradToMatrix_impScript.R', outputFolder1 = outputFolder):
    filePath = inputRscriptFile 
    os.system(f'Rscript {filePath} "{outputFolder1}"')

    return()
###############################################
#this is to write script 05_getStentInfo.py
def getStentInfo(stenSegFileName, stenSegNo, outFileName):
    outFile = open (outFileName, 'w')
    outFile.write("\t".join(["stent_y_n", "patient_id", "stent_visits", "stent_impl", "sten_grad_visits"]) + '\n')
    #stenSegFileName, stenSegNo = sys.argv[1:]
    stentMap = getStent(stenSegNo)
    #print(stentMap)
    stenSegFile = open(stenSegFileName, 'r')
    header = stenSegFile.readline().replace(" ", "_").lower().rstrip("\n").split("\t")
    for line in stenSegFile:
        line = line.rstrip("\n")
        fields = line.split("\t")
        patient_id, exams = int(fields[0]), fields[1:]
        exams_num = [int(exam) for exam in exams if str(exam) != 'NA']
        exam_diffs = [i - j for i, j in zip(exams_num[1:], exams_num)]
        if any(exam_diff < 0 for exam_diff in exam_diffs):
            if patient_id in stentMap:
                if "1" in stentMap[patient_id][1]:
                    #print (patient_id)
                    outFile.write("\t".join([str(x) for x in ["Y", patient_id, "|".join(str(x) for x in stentMap[patient_id][0]), \
                         "|".join(str(x) for x in stentMap[patient_id][1]), \
                         "|".join(exams)]]) + '\n')
                else:
                    outFile.write("\t".join([str(x) for x in ["N", patient_id, "|".join(str(x) for x in stentMap[patient_id][0]), \
                         "|".join(str(x) for x in stentMap[patient_id][1]), \
                         "|".join(exams)]]) + '\n')
    stenSegFile.close()
    outFile.close()
    return()

###############################################
#writing 06_impStenGrad.py as function

def imputeStenosisGrade(inputFileName):
    #stenSegFileName,impStenSegFileName,stentFileName = sys.argv[1:]
    segNo = os.path.basename(inputFileName).split("_")[4]
    impStenSegFile = open(outputFolder + '/imputed_stent_impl_sten_seg_1_15_sten_grad.tsv', "a")
    stentFile = open(outputFolder + '/excluded_stent_impl_sten_seg_1_15.tsv', "a")
    stenSegFile = open(inputFileName, 'r')
    header = stenSegFile.readline().replace(" ", "_").lower().rstrip("\n").split("\t")
    #print("\t".join(["segNo", convertListToString(header[1:], "\t")]))
    #print("\t".join(["segNo","patient_id","sten_grad_visits","imputing","sten_grad_visits_imp"]))
    for line in stenSegFile:
        line = line.rstrip("\n")
        fields = line.split("\t")
        stent_y_n, patient_id, stent_visits, stent_impl, sten_grad_visits = fields
        sten_grad_visits = sten_grad_visits.split("|")
        sten_grad_visits_nna = [int(sten_grad_visit) for sten_grad_visit in sten_grad_visits if str(sten_grad_visit) != 'NA']
        sten_grad_visits_nna_diffs = [i - j for i, j in zip(sten_grad_visits_nna[1:], sten_grad_visits_nna)]
        if any(sten_grad_visits_nna_diff < 0 for sten_grad_visits_nna_diff in sten_grad_visits_nna_diffs):
            sten_grad_visits_nna_imp, sten_grad_visits_imp = [], []
            if stent_y_n == "N":
                if sten_grad_visits_nna[0] < max(sten_grad_visits_nna[1:]):
                    sten_grad_visits_nna_imp = impStenGrad(sten_grad_visits_nna, max(sten_grad_visits_nna[1:]))
                    sten_grad_visits_imp = reconstruct(sten_grad_visits, sten_grad_visits_nna, sten_grad_visits_nna_imp)
                else:
                    if sten_grad_visits_nna[0] >= max(sten_grad_visits_nna[1:]):
                        sten_grad_visits_nna_imp = impStenGrad(sten_grad_visits_nna, sten_grad_visits_nna[0])
                        sten_grad_visits_imp = reconstruct(sten_grad_visits, sten_grad_visits_nna, sten_grad_visits_nna_imp)
                print("\t".join([segNo, patient_id, convertListToString(sten_grad_visits, "|"), "imputing", convertListToString(sten_grad_visits_imp, "|")]), file=impStenSegFile)
            else:
                if stent_y_n == "Y":
                    print("\t".join([segNo, patient_id, stent_visits, stent_impl, convertListToString(sten_grad_visits, "|")]), file=stentFile)
                    #sten_grad_visits_nna_imp = sten_grad_visits_nna
                    #sten_grad_visits_imp = reconstruct(sten_grad_visits, sten_grad_visits_nna, sten_grad_visits_nna_imp)
        else:
            print(stent_y_n, patient_id, stent_visits, stent_impl, sten_grad_visits)
    stenSegFile.close()
    impStenSegFile.close()
    stentFile.close()
    return()
################################################################################################################

def applyingAnnotationFunction(inputFileName = outputFile, outputFileName = outputFolder +'/' + outputFile_prefix + '.xlsx'):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(outputFileName, engine='openpyxl')
    # Function body goes here
    # Get the date at which Troponin T measurement was being done:
    
    coro_date = getCoroDate('./inFiles/coro_date_10.02.2023.csv')
    
    # Impute (i.e. replace erronious data) stenosis decrease (clinically impossible) with max value across time points:
    imp = getImpStenGrad(outputFolder + '/imputed_stent_impl_sten_seg_1_15_sten_grad.tsv')
    # exclDueToStent = ..
    # Open the cleaned clinical data  file in read mode and assign the file object to the variable 'inputFile'
    inputFile = open(inputFileName, 'r')
    # Read the first line from inputFile, replace spaces with underscores,
    # convert to lowercase, remove trailing newline character, and split into a list
    # using the tab character as a delimiter.
    header = inputFile.readline().replace(" ", "_").lower().rstrip("\n").split("\t")
    out_header = "\t".join([convertListToString(header[1:10], "\t"),str(header[570]), convertListToString(header[11:48], "\t"),"tnt_pos", convertListToString(header[50:276], "\t"), str(header[571]), str(header[572]),str(header[573]), "sis_score","sss_score","nr_of_collaterals","rentrop_grading","pathways","collateral_flow_grade", "blush_grade","number_of_bifurcations_distal","collateral_artery_size_(mm)","collateral_artery_size_(px)"])
    ###print(out_header)
    # Create a DataFrame with the header
    df = pd.DataFrame(columns=out_header.split('\t'))
    #print(sorted(df.columns))
    # Convert the DataFrame to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    # Get a dictionary mapping column names to their indices in the input list:
    colNum = getcolNum(header)
    # Generate a list of strings representing the Cartesian product of two lists of integers:
    ixPairs = genIndex()
    colMap, casm_l, casx_l = {}, [], []
    for line in inputFile:
        line = line.rstrip("\n")
        fields = line.split("\t")
        progress_id, tnt_max = fields[colNum["progress_id"]], fields[colNum["tnt_max"]]
        # Classify each patient as Troponin T positive ('1') or negative ('='), based on the measurement date and value:
        try:
            tnt_pos = getTNTpos(progress_id, tnt_max, coro_date)
        except:
            print(Fore.RED + 'The column tnt_max includes bad format. Kindly remove special characters from this column and try again. \n Currently ignoring these entries. ' + Fore.RESET)
        # Collects stenosis grades for 15 segments:
        stenGrad = collectStenGrad(progress_id, fields, colNum)
        # For patients with a stenosis decrease (clinically impossible), replace the original stenosis grades with the imputed ones:
        impStenGrad = replaceWithImpStenGrad(progress_id, stenGrad, imp)
        # Update the fields with the values from impStenGrad
        fields[list(impStenGrad)[0]] = impStenGrad[list(impStenGrad)[0]]
        fields[list(impStenGrad)[1]] = impStenGrad[list(impStenGrad)[1]]
        fields[list(impStenGrad)[2]] = impStenGrad[list(impStenGrad)[2]]
        fields[list(impStenGrad)[3]] = impStenGrad[list(impStenGrad)[3]]
        fields[list(impStenGrad)[4]] = impStenGrad[list(impStenGrad)[4]]
        fields[list(impStenGrad)[5]] = impStenGrad[list(impStenGrad)[5]]
        fields[list(impStenGrad)[6]] = impStenGrad[list(impStenGrad)[6]]
        fields[list(impStenGrad)[7]] = impStenGrad[list(impStenGrad)[7]]
        fields[list(impStenGrad)[8]] = impStenGrad[list(impStenGrad)[8]]
        fields[list(impStenGrad)[9]] = impStenGrad[list(impStenGrad)[9]]
        fields[list(impStenGrad)[10]] = impStenGrad[list(impStenGrad)[10]]
        fields[list(impStenGrad)[11]] = impStenGrad[list(impStenGrad)[11]]
        fields[list(impStenGrad)[12]] = impStenGrad[list(impStenGrad)[12]]
        fields[list(impStenGrad)[13]] = impStenGrad[list(impStenGrad)[13]]
        fields[list(impStenGrad)[14]] = impStenGrad[list(impStenGrad)[14]]
        # Calculate the Segment Involvement Score (SIS), by designating a score of 1 for each one of the coronary artery 
        # segments with a detectable atherosclerotic coronary plaque, irrespective of the plaque size or individual plaque 
        # burden in that particular segment:
        sis = calcSIS(progress_id, impStenGrad)
        # Calculate the Segment Stenosis Score (SSS), by grading each coronary segment based on plaque stenosis severity 
        # (i.e., grades 0-5) and summing the grades from all segment
        sss = calcSSS(progress_id, impStenGrad)
        # Extract collaterals using coordinates, TO DO: replace original annotation fields   
        getCoordinates(progress_id, fields, colNum, ixPairs, colMap)
        # Take a unique set of coordinates, to determine the number of collaterals
        colat = countColat(progress_id, colMap)
        # Extract Rentrop grading measurements across all DICOMs and
        # take the median Rentrop grading measurement across all DICOMs: #Replacing by Maximum; Aniket 20230918
        rg = getAnnotations(progress_id, fields, colNum, ixPairs, "rentrop_grading")
        # Extract Pathways measurements across all DICOMs and
        # take the median Pathways measurement across all DICOMs:#Replacing by Maximum; Aniket 20230918
        pw = getAnnotations(progress_id, fields, colNum, ixPairs, "pathways")
        # Extract Collateral flow grade measurements across all DICOMs and 
        # take the median Collateral flow grade  measurement across all DICOMs: #Replacing by Maximum; Aniket 20230918
        cfg = getAnnotations(progress_id, fields, colNum, ixPairs, "collateral_flow_grade")
        # Extract Blush grade measurements across all DICOMs and 
        # take the median Blush grade measurement across all DICOMs: #Replacing by Maximum; Aniket 20230918
        bg = getAnnotations(progress_id, fields, colNum, ixPairs, "blush_grade")
        # Extract the number of bifurcations (distal) measurements across all DICOMs and 
        # take the number of bifurcations (distal) measurement across all DICOMs:
        nbd = getAnnotations(progress_id, fields, colNum, ixPairs, "number_of_bifurcations_distal")
        # Extract the collateral artery size (mm) measurements across all DICOMs and 
        # take the collateral artery size (mm) measurement across all DICOMs:
        casm = getAnnotations(progress_id, fields, colNum, ixPairs, "collateral_artery_size_(mm)")
        if casm != "NA":
            casm_l.append(float(casm))
        # Extract the collateral artery size (px) measurements across all DICOMs and 
        # take the collateral artery size (px) measurement across all DICOMs:
        casx = getAnnotations(progress_id, fields, colNum, ixPairs, "collateral_artery_size_(px)")
        if casx != "NA":
            casx_l.append(float(casx))
        # Print modified columns to a new file:
        out_line = "\t".join([convertListToString(fields[1:10], "\t"), str(fields[570]), convertListToString(fields[11:48], "\t"), str(tnt_pos), convertListToString(fields[50:276], "\t"), str(fields[571]), str(fields[572]), str(fields[573]), str(sis), str(sss), str(colat), str(rg), str(pw), str(cfg), str(bg), str(nbd), str(casm), str(casx)])
        ##print(out_line)
        # Write data to the Excel file line by line
        df = pd.DataFrame([out_line.split('\t')])
        df.to_excel(writer, sheet_name='Sheet1', index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    inputFile.close()
    # Close the Pandas Excel writer and output the Excel file.
    writer.close()
    #corrPlot(casm_l, casx_l,'Collateral size', '(mm)', '(px)', './plots/corr_col_size_mm_vs_px.png')

    #Get subset of data for which annotations are calculated. Taking 'nr_of_collaterals' as the factor to distinguish
    #print(sorted(df.columns))
    #df.columns = out_header.split('\t')
    #print(df)
    #df_subset = df[int(df["nr_of_collaterals"]) > 0]
    return()


def getCollateralDataSubset(inputFile = outputFolder + '/' + outputFile_prefix + '.xlsx'):
    df_annotated = pd.read_excel(inputFile)
    #print(df_annotated)
    df_annotated = df_annotated.fillna(np.nan)
    #print(df_annotated)
    #print(df_annotated['nr_of_collaterals'])
    df_annotated_subset = df_annotated[df_annotated["nr_of_collaterals"].astype(int) > 0]
    outputExcel = 'collateral_' + inputFile.split('/')[-1]
    df_annotated_subset.to_excel(outputFolder + '/' + outputExcel)
    return(df_annotated_subset)


# Higher-level functions to perform more complex operations  

################################################################################################################

################################################################################################################
# getCoroDate
################################################################################################################
def getCoroDate(inFileName):

    # Define a function named getCoroDate that takes one argument: inFileName
    # Open the file specified by inFileName
    inFile = open(inFileName)
    # Create an empty dictionary named oMap
    oMap = {}
    # Read the first line of the file and store it in the variable header
    header = inFile.readline()
    # Iterate over the remaining lines in the file
    for line in inFile:
        # Remove the newline character from the end of the line
        line = line.rstrip("\n")
        # Split the line into fields using ";" as the delimiter
        fields = line.split(";")
        # Check if the number of fields is equal to 2
        if len(fields) == 2:
            # Unpack the fields into two variables: progress_id and coro_date
            progress_id, coro_date = fields
            # Split coro_date into day, month, and year using "." as the delimiter and convert them to integers
            dd, mm, yyyy = [int(x) for x in [coro_date.split('.')[0], coro_date.split('.')[1], '20' + coro_date.split('.')[2]]]
            # Add an entry to oMap with progress_id as the key and a date object constructed from yyyy, mm, and dd as the value
            oMap[progress_id] = date(yyyy, mm, dd)
    # Close the file
    inFile.close()
    # Return oMap
    return oMap

def collectStenGrad(p_id, vals, col_n):
    """
    Collects stenosis grades for 15 segments.

    :param p_id: The patient ID.
    :type p_id: str
    :param vals: The list of values.
    :type vals: list
    :param col_n: The column number mapping.
    :type col_n: dict
    :return: A dictionary mapping column numbers to stenosis grades.
    :rtype: dict
    """
    sten_grad = {}
    for ix in range(1, 16):
        seg0 = f"sten_seg_{ix}_sten_grad"
        sten0 = int(vals[col_n[seg0]])
        sten_grad[col_n[seg0]] = sten0
    return dict(sorted(sten_grad.items()))

def replaceWithImpStenGrad(p_id, sg, isg):
    """
    This function takes in three arguments: p_id, sg and isg. It returns a modified copy of the sg dictionary.
    
    :param p_id: An identifier used to check if it exists as a key in the isg dictionary.
    :type p_id: Any hashable type
    :param sg: A dictionary to be copied and modified.
    :type sg: dict
    :param isg: A dictionary containing values that are lists of tuples. Each tuple contains two elements where 
    the first element is used as an index to modify the value in the copied sg dictionary.
    :type isg: dict
    :return: A modified copy of the sg dictionary.
    :rtype: dict
    """
    o = sg.copy()
    if p_id in isg:
        for s, st in isg[p_id]:
            o[int(s) - 1] = st
    return o




################################################################################################################
################################################################################################################
# calcSIS 
################################################################################################################
def calcSIS(p_id, vals):
    """
    Calculate the Segment Involvement Score (SIS) for a given patient ID and dictionary of values.

    :param p_id: An identifier for the patient.
    :type p_id: Any hashable type
    :param vals: A dictionary containing values to be used in the calculation of SIS.
    :type vals: dict
    :return: The calculated Segment Involvement Score (SIS).
    :rtype: int
    """
    sis = [int(v) for v in vals.values() if int(v) > 0]
    return len(sis)





################################################################################################################
# calcSSS 
################################################################################################################
def calcSSS(p_id, vals):
    """
    Calculate the Segment Stenosis Score (SSS) for a given patient ID and dictionary of values.

    :param p_id: An identifier for the patient.
    :type p_id: Any hashable type
    :param vals: A dictionary containing values to be used in the calculation of SSS.
    :type vals: dict
    :return: The calculated Segment Stenosis Score (SSS).
    :rtype: int
    """
    sss = []
    for v in vals.values():
        sten = int(v)
        if sten == 0:
            sss.append(0)
        elif 1 <= sten <= 24:
            sss.append(1)
        elif 25 <= sten <= 49:
            sss.append(2)
        elif 50 <= sten <= 69:
            sss.append(3)
        elif 70 <= sten <= 99:
            sss.append(4)
        elif sten == 100:
            sss.append(5)
    return sum(sss)



def getImpStenGrad(inFileName):
    
    """

    This function reads data from a file and processes it to create a 
    dictionary that maps patient IDs to tuples containing segment 
    number and stenosis (the narrowing / restriction of a blood 
    vessel / valve that reduces blood flow) grade (i.e., the severity 
    of narrowing / restriction in %) values

    """

    # Use a with statement to open the file and ensure it is closed when done
    with open(inFileName) as inFile:
        # Read the first line of the file, replace spaces with underscores, 
        # convert to lowercase, remove trailing newline, and split into fields
        header = next(inFile).replace(" ", "_").lower().rstrip().split("\t")
        oMap = {}
        for line in inFile:
            # Remove trailing newline and split into fields
            fields = line.rstrip().split("\t")
            segno, patient_id, sten_grad_visits, imputing, sten_grad_visits_imp = fields
            # Split sten_grad_visits_imp into a list using "|" as the delimiter
            sten_grad_visits_imp = sten_grad_visits_imp.split("|")
            # Create a list of visit indices
            visits_ix = list(range(1, len(sten_grad_visits_imp)+1))
            for n, ix in enumerate(visits_ix):
                # Call the fillMap function with the appropriate arguments
                fillMap(oMap, f"{patient_id}-{ix}", (segno, sten_grad_visits_imp[n]))
        #does not require closing
    return oMap

################################################################################################################
def countColat(p_id, inMap):
    """
    Counts the number of unique collaterals for a given patient ID in the input dictionary.

    Args:
        p_id (str): The ID of the patient to count collaterals for.
        inMap (dict): A dictionary mapping patient IDs to lists of collaterals.

    Returns:
        int: The number of unique collaterals for the given patient ID. Returns 0 if the patient ID is not in the input dictionary.
    """
    colat = 0
    if p_id in inMap:
        colat = len(set(inMap[p_id]))
    return colat


################################################################################################################
# getAnnotations
################################################################################################################
def getAnnotations(p_id: str, vals: List[str], col_n: Dict[str, int], ix_l: List[int], anno: str) -> float:
    """
    This function calculates the median value of the annotations for a given patient ID. #Replacing by Maximum; Aniket 20230918

    :param p_id: Patient ID
    :param vals: List of values
    :param col_n: Dictionary mapping column names to column indices
    :param ix_l: List of indices
    :param anno: Annotation string
    :return: Median value of the annotations for the given patient ID #Replacing by Maximum; Aniket 20230918
    """
    a_map = {}
    for ix in ix_l:
        at = f"{anno}{ix}"
        if vals[col_n[at]] != "NA":
            fillMap(a_map, p_id, vals[col_n[at]])
    if p_id in a_map:
        #return statistics.median(float(i) for i in a_map[p_id])
        return max(float(i) for i in a_map[p_id]) #Replacing by Maximum; Aniket 20230918
    else:
        return "NA"


################################################################################################################
################################################################################################################
# getTNTpos
################################################################################################################
def getTNTpos(p_id, t_max, c_date):

    """
    On June 15, 2010, the measurement method for Troponin T was changed from ug/L to ng/L. For patients with a 
    coro date before June 15, 2010, we consider them to be Troponin T positive ('1') if their value is greater 
    than or equal to 0.1. On and after June 15, 2010, we consider a patient to be Troponin T positive ('1') if 
    their value is greater than or equal to 14.0.
    """
    if t_max == 'NA':
        return 'NA'
    
    c_d = c_date.get(p_id)
    if c_d is None:
        return 'NA'
    
    if ( '<0,0' in t_max):
        return (0) #here we consider <0,01;0,003 as <0.01 and since its less than threshold, returning 0
    
    #There were certain entries where '.' was replaced by ','. Correct them in excel. Also, there is 1 entry 3238-1 with tnt_max > 10. Considering this value as 10 and less than 14. This is only one value. Returning as 0. We can also choose to return it as 'NA'
    if ('>10' in t_max):
        return(0) #or return 'NA'
    
    threshold = 14.0 if c_d >= date(2010,6,15) else 0.1
    return '1' if float(t_max) >= threshold else '0'


def corrPlot(x, y, p_title, n_x, n_y, filename):

    """
    Create a scatter plot with a regression line and save it to a file with a user-defined filename.

    Example usage of the function:
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    corrPlot(x, y, 'filename.png')
    """
    # Create a scatter plot with a regression line
    sns.regplot(x=x, y=y) # color='blue'

    # Add labels and title
    plt.xlabel(n_x)
    plt.ylabel(n_y)
    plt.title(p_title+": "+n_x+' vs. '+n_y)

    # Save the plot to a file with the user-defined filename
    plt.savefig(filename, dpi=300)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()

##Module 1: Clean data --> R3eference Script : 01_cleanClinData.R
##Steps: Algorithm / Pseudocode:
# ==> Step 1: open excel sheet, convert all col names to lower case for better code writing, replace 'space' by '_'. 
# ==> Step 2: get patient id counts, check for duplicates
# ==> Step 3: # Remove duplicates 
# ==> Step 4: # Select only patients where the REPORT is available report_vorh = 1 
# ==> Step 5: Process problematic columns # Some of them may not be problematic any more, however better double-check!
# ==> Step 6: Dropping DICOM cleaning, Dicom export and framerate
# If Age < 33, set to "NA" # later we discovered that this is unfortunately not an error but really young patients\# Still please double check how many, but do not set to 'NA'
# STEMI and NSTEMI replace 9 with 0, as this is a typo
# Again, please double-check first, whethere there are still such cases with '9'
# DTBT and STBT replace 0 with NA
# EF replace 0  or > 80 with NA
# CTO.ID, CAD, CAD.ID not meaningful, remove
# GROESSE replace 0 or < 145 with NA # Should be resolved now, however double-check
# GEWICHT replace 0 or < 45 with NA # Should be resolved now, however double-check  
# BMI recalculate using corrected GROESSE and GEWICHT
# NBMI, pr_acvb all 0s, remove
# syst_pressure, replace 0 with NA
# divide values greater as 1000 with 10 # Should be resolved now, however double-check
# what about values < 20 ? # Should be resolved now, however double-check
# diast_pressure, replace 0 with NA
# divide values greater as 600 with 10 # Should be resolved now, however double-check
# what about values < 20 ?
# heartrate, set 0 to NA
# nodcv, set 0 to NA
# valve_op, asd_cl all NAs, exclude
# oth_devices, create 3 new columns: pacemaker, icd, laa_occluder # check, whether this is still necessary, i.e. is there free text in the column dat$pacemaker?
# Double-check why we set those to 'NULL': free text, missing values, non-informative ('1' or 'O' only)?
# crea_absolute, what to do with values > 140 or smaller than 18 # Are there such values still?
# crp_a 0.03 to 305, define range?
#tnt_max, convert to numeric, remove <, replace all , with .
# ldh_max, replace values > 6000 with NA
# platelets, replace values >1000 with NA
# leuk_a replace values > 50 000 or < 500 with NA
    

