![image](https://github.com/user-attachments/assets/36fb842a-3425-4e2a-875f-d6777a8127e3)


# What is the tool for?
The clean_annotate_data_v1_03 tool allows you to perform basic cleaning and annotation of the clinical data as obtained from the PROGRESS project.
The tool provides for an efficient and portable way to process, clean and annotate the clinical data from PROGRESS project and can be used by the clinical collaborators to gain insights from their data. Moreover, the tool also performs basic statistical tests over the required variables!

# What are the pre-requisites to run this tool?
The user is expected to have basic computational skills and some experience with terminal commands.
The system on which the tool is install must have the following:
1. A stable internet connection to allow installation of python3 and R packages.
2. Docker tool installed over the machine. You can find the steps to install docker on your machine [here](https://www.docker.com/). 
3. The input clinical data file in xlxs format and other input files as mentioned below.

# What are the dependencies that need to be installed to run the tool?
The tool takes care of installing the required dependencies. All you need is a stable connection that can access the python3 and R packages from respective repositories. However, if you plan to run the python3 script independently (without installing the docker system), you would require the following:
```
rpy2
rbase
pandas
numpy
colorama
scipy
statistics
pandas
matplotlib
lifelines
datetime
typing
seaborn
openpyxl

```
> [!NOTE]
> The above python3 packages can be installed using the **pip install** command.

Apart from these, the tool also requires three R packages to be installed viz. 'dplyr', 'ggplot2', 'tidyr'.
> [!NOTE]
> To install these, in the R prompt, run the command: 

```
install.packages(c('dplyr', 'ggplot2', 'tidyr'), dependencies=TRUE)
```
# Run the Script:
## Step 1: Download the docker image and place in a separate folder. 
If the docker image is not available, you may create the same using the steps mentioned in the last part of the document.

## Step 2: Load the docker image in the system.

```
docker load -i clean_annotate_data_v1_03.tar
```
## Step 3: Run the image.

```
docker run --rm -v /path/to/inFiles:/app/inFiles -v /path/to/outFiles:/app/outFiles clean_annotate_data_v1_02 python3 /app/bin/CleanOneLineDataTable_clean.py /app/inFiles/<inputFile> /app/outFiles <outputPrefix>
```
The command maps the/path/to/inFiles to /app/inFiles /path/to/outFiles to /app/outFiles and the present in the docker image. \

i. path/to/inFiles: All the input files are placed in this folder. Make sure that the name and format of these files are the same as those described below. \
ii. path/to/outFiles: All the output files are stored in this folder. \
iii. /app/inFiles/&lt;inputFile&gt;: This file name can change to the latest input file. Please note that the file needs to be present at the /path/to/inFiles location and needs to be an excel spreadsheet. \
iv.&lt;outputPrefix&gt; : Prefix to the output files. \
> [!NOTE]
> DO NOT replace /app/inFiles and /app/outFiles with anything else. The paths /app/outFiles and /app/inFiles are created in the Docker image.

## Input Files

The input files are to be placed in a single folder. The path of this folder is hereby denoted as /path/to/inFiles/. The code requires 5 modifiable input files: \

**i. &lt;inputFile&gt;**:  The target clinical data file that needs to be cleaned

**ii. statsFile_categorical.txt**: List of categorical variables in the &lt;inputFile&gt;. The categorical variables present in this file are further used to perform basic statistical analysis such as identifying value counts per category and missing data proportions.

**iii. statsFile_continuous.txt**: This file includes information that needs to be extracted for the continuous variables. Each line is considered as a separate entry for individual column. The format of the entry should be as follows: 
>
 >Unters_alter,mean,median 
 >  OR 
 >Unters_alter,mean 
 >  OR 
 >Unters_alter,median 

> [!WARNING]
> It is imperative that there should be no space character in the entry. 

**iv. thresholdFile.txt**: This file includes information to govern the minimum and maximum acceptable data for individual columns. This was made particularly to deal with entries that may contain erroneous data. The format of this file is as follows: 

  &lt;columnName&gt;;&lt;min&gt;;&lt;max&gt;

For example, in order to set limits to the gluc_a values from 20 to 999, the entry should be: 
 
  gluc_a;20;999

**v. dropColumns.txt**: This file includes the list of columns that need to be dropped from the output files. 

**vi. performStatisticalTests.txt**: This file includes the list of statistical tests to perform. Current version of the tool performs 3 statistical tests:

    a.	Mann-Whitney U test
    b.	Kruskal-Wallis test
    c.	Chi-square test for goodness of fit.

The comments in this file are marked by ‘#’  as the start of the line.

## Each line that is not marked by ‘#’ at the start is considered as a separate entry. For example:
    bmi, diabetes, MW-U
    chol_a_total, rentrop_grading, kruskal
    bmi, collateral_flow_grade, kruskal
    unters_alter, collateral_flow_grade, Kruskal
    collateral_flow_grade, diabetes, chisq 

Furthermore, while using this function, special care needs to be taken while addressing the columns and tests.
## The following codes are required to perform respective tests.
  a. to perform Mann-Whitney U --> MW-U \
  b. to perform Kruskal-Wallis test --> kruskal \
  c. to perform chi-square test --> chisq.

As seen in the example, the first column, second column and the test names are separated by ‘, ’ [&lt; comma &gt; &lt; space_character &gt;].  

> [!NOTE]
> While performing tests like the Kruskal-Wallis and the Mann-Whitney U test, the first column needs to be continuous while the second column needs to be categorical.

## Output Files 

The &lt;output prefix&gt; is added to all the output files. These output files are created and stored in the ‘/path/to/outFiles’ folder as defined by the user.

a.	**&lt;output prefix&gt;_main.xlsx** : The entire cleaned, processed and annotated file in excel spreadsheet format.
b. **&lt;output prefix&gt;_main.tsv** The entire cleaned, processed and annotated file in .tsv format.\
c. **basicStatisticalValues_continuousdata.csv**: Basic Statistical values as calculated by script for entire dataset. The variable names need to be given to the script via statsFile_continuous.txt input file. \
d.	**CategoricalData_Distribution.csv**: The number of individuals in each category of the categorical variable. This variable is given by the user in the statsFile_categorical.txt input file. \
e.	**statistics_results_&lt;output prefix&gt;.txt** : This output file contains all the results of the statistical tests performed. These tests are given by the performStatisticalTests.txt input file. \
f.	**collateral_&lt;output prefix&gt;.xlsx**: The cleaned, processed and annotated file in excel spreadsheet format for only individuals identified with collateral arteries. \
g.	**collateral_&lt;output prefix&gt;.tsv**: The cleaned, processed and annotated file in .tsv format for only individuals identified with collateral arteries. \
h.	**collateralData_statistics_results_&lt;output prefix&gt;.txt**: This output file contains all the results of the statistical tests performed. These tests are given by the performStatisticalTests.txt input file and run only for individuals identified with collateral arteries. \
i.	**collaterals_statisticalAnalysis_categorical.csv**: CategoricalData_Distribution.csv: The number of individuals in each category of the categorical variable. This variable is given by the user in the statsFile_categorical.txt input file. \
j.	**collaterals_statisticalAnalysis_continuousData.csv**: Basic Statistical values as calculated by script for dataset including individuals identified with collateral arteries. The variable names need to be given to the script via statsFile_continuous.txt input file. \
k.	**imputed_stent_impl_sten_seg_1_15_sten_grad.tsv**: Temporary file that includes information about imputed stent implantations. \
l.	**numberOfCollaterals.txt**: File includes the information for number of collaterals identified in each individual with collaterals.

> [!NOTE]
>If the docker image is not available, download entire folder /docker and perform the following steps:
```
 	cd ./docker/
 	docker build -t clean_annotate_data_v1_03:latest ./
 	docker save -o clean_annotate_data_v1_03.tar clean_annotate_data_v1_03:latest
```
The docker image would be saved in the folder. This can then be used by following steps 2 and later.

# 🤝 Contributing
Your contributions are valued! Please open issues or submit pull requests to enhance this tool.

In case of any queries or concerns, please feel free to email me at ani.saw@rsu.edu.lv or our group head, Dr. Vilne at baiba.vilne@rsu.lv.

Happy QC-ing! 🎉
