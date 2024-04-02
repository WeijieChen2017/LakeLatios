# here we need to control the data integrity for multiple models on the same dataset.
# 1, load the dataset folder, check if there are data control files
# 2, if not, create the data control files
# 3, if yes, the function will receive the required file numbers and return available file paths

import os
import glob
import numpy as np

def create_data_control(data_folder_name):
    # create the data control file
    data_folder = "data/"+data_folder_name
    data_control_file = data_folder+"_data_control.txt"
    # load all case folders
    # the case folder is the folder that contains .nii.gz
    case_list = sorted(glob.glob(data_folder+"/*/*.nii.gz"))
    # extract the case folder name and write to the data control file
    with open(data_control_file, "w") as f:
        for case_folder in case_list:
            case_folder_name = case_folder.split("/")[-2]
            # write the none as the this case is not used yet
            f.write(case_folder_name+":none\n")

def acquire_data_from_control(data_folder_name, required_case_numbers, experiment_name):

    # go to "data" folder to see whether the data control file exists
    data_control_file = "data/"+data_folder_name+"_data_control.txt"
    if not os.path.exists(data_control_file):
        create_data_control(data_folder_name)
        print(f"Data control file created for {data_folder_name}")
    
    # load the data control file
    with open(data_control_file, "r") as f:
        lines = f.readlines()
    # create the data control dictionary
    data_control_dict = {}
    for line in lines:
        case_folder_name, occupiant = line.split(":")
        data_control_dict[case_folder_name] = occupiant.strip()
    
    # find the availe data
    available_cases = [x for x in data_control_dict.keys() if data_control_dict[x] == "none"]
    # if the required number is larger than the available number, return None
    if required_case_numbers > len(available_cases):
        print(f"Required case number {required_case_numbers} is larger than available case number {len(available_cases)}")
        return None
    # randomly select the required number of cases
    selected_cases = np.random.choice(available_cases, required_case_numbers, replace=False)
    
    # replace the data_control_file with info
    # first write the case with current experiment name
    # second write the case with other occupied experiment name
    # last write the case with none
    first_to_write = []
    second_to_write = []
    last_to_write = []
    for case in data_control_dict.keys():
        if case in selected_cases:
            first_to_write.append(case+":"+experiment_name)
        elif data_control_dict[case] == "none":
            last_to_write.append(case+":none")
        else:
            second_to_write.append(case+":"+data_control_dict[case])
    # write to the data control file and replace the old one
    # delete the old one and create a new one
    os.system(f"rm {data_control_file}")
    with open(data_control_file, "w") as f:
        for line in first_to_write:
            f.write(line+"\n")
        for line in second_to_write:
            f.write(line+"\n")
        for line in last_to_write:
            f.write(line+"\n")
    print(f"Data control file updated for {data_folder_name}")
    
    # print the selected cases
    print(f"Selected cases for {experiment_name}:")
    # for each case in selected_cases, add "data/data_folder_name/case_folder_name" to the list
    selected_cases = [f"data/{data_folder_name}/{case}" for case in selected_cases]
    for case in selected_cases:
        print(case)
    return selected_cases
    

def remove_data_occupation(data_folder_name, experiment_name):
    # search the data control file
    data_control_file = "data/"+data_folder_name+"_data_control.txt"
    if not os.path.exists(data_control_file):
        print(f"Data control file {data_control_file} does not exist")
        return
    # load the data control file
    with open(data_control_file, "r") as f:
        lines = f.readlines()
    # create the data control dictionary
    data_control_dict = {}
    for line in lines:
        case_folder_name, occupiant = line.split(":")
        data_control_dict[case_folder_name] = occupiant.strip()
    # remove the experiment_name from the data control file
    for case in data_control_dict.keys():
        if data_control_dict[case] == experiment_name:
            data_control_dict[case] = "none"
    # write the data control file
    os.system(f"rm {data_control_file}")
    with open(data_control_file, "w") as f:
        for case in data_control_dict.keys():
            f.write(case+":"+data_control_dict[case]+"\n")
    print(f"Data control file updated for {data_folder_name}")

    