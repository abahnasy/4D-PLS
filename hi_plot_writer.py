""" Write the experiments results in csv format to be readible by Hi-Plot
"""
import glob
import csv
from os import isatty
import re

# importing pandas package
import pandas as pandasForSortingCSV

def parse_exp(path):
    row = []
    with open('{}/run_testing.log'.format(path), mode='r') as f:
        lines = f.readlines()
        
        for line in lines:
            # print(">>>>", line)
            if 'rot_x' in line:
                row.append(int(re.search(r'\d+', line).group()))
                continue
            if 'rot_y' in line:
                row.append(int(re.search(r'\d+', line).group()))
                continue
            if 'rot_z' in line:
                row.append(int(re.search(r'\d+', line).group()))
                continue
            
            line = line.split('[INFO]')[-1]
            match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')

            if 'LSTQ' in line:
                # row.append(float(re.search(r'\d+\.\d+', line).group()))
                row.append(float(re.search(match_number, line).group()))
                continue
            if 'S_assoc' in line:
                # row.append(float(re.search(r'\d+\.\d+', line).group()))
                row.append(float(re.search(match_number, line).group()))
                continue
            if 'things_iou' in line:
                # row.append(float(re.search(r'\d+\.\d+', line).group()))
                row.append(float(re.search(match_number, line).group()))
                continue
            if 'stuff_iou' in line:
                # row.append(float(re.search(r'\d+\.\d+', line).group()))
                row.append(float(re.search(match_number, line).group()))
                continue
            if 'S_cls' in line:
                # row.append(float(re.search(r'\d+\.\d+', line).group()))
                row.append(float(re.search(match_number, line).group()))
                continue
    assert len(row) == 8, 'error parsing !'
    # convert into string
    row = [ "{:.4f}".format(i) if isinstance(i, float)==True else str(i) for i in row]
    return row
    

header = ['rot_x', 'rot_y', 'rot_z', 'LSTQ', 'S_assoc', 'things_iou', 'stuff_iou', 'S_cls']


exp_path_list = glob.glob('./multirun/2022-01-07/19-22-55/*/')


with open('experiments_z_100.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for exp_path in exp_path_list:
        row = parse_exp(exp_path)
        # write the data
        writer.writerow(row)




# assign dataset
csvData = pandasForSortingCSV.read_csv("experiments_z_100.csv")
# sort data frame
csvData.sort_values(["LSTQ"], 
                    axis=0,
                    ascending=[False], 
                    inplace=True)
print(csvData)
csvData.to_csv('experiments_z.csv', index=False)
print("written successfully")

# print(parse_exp('multirun/2022-01-01/14-35-50/11/'))

# s = '[2022-01-01 16:03:46,572][__main__][INFO] - things_iou: 3.962523334859639e-05'
# match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
# print()
# print(float(re.search(r'\d+\.\d+', s).group()))


