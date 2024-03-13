cfg = {
    "data_folder": "data/MIMRTL_Brain",
    "batch_size": 4,
    "training_case" : 27,
    "validation_case" : 5,
    "random_seed": 42,
}



# load all cases and select the first part for tarining and the second part for validation
training_case = cfg["training_case"]
validation_case = cfg["validation_case"]
data_folder = cfg["data_folder"]
import glob
case_list = sorted(glob.glob(data_folder+"/*"))
training_list = case_list[:training_case]
validation_list = case_list[training_case:training_case+validation_case]
# output the training and validation list
for idx, item in enumerate(training_list):
    print(f"Training {idx+1}: {item}")
for idx, item in enumerate(validation_list):
    print(f"Validation {idx+1}: {item}")

n_train = len(training_list)
n_val = len(validation_list)
batch_size = cfg["batch_size"]

# search_list = training_list + validation_list
search_list = validation_list

import h5py
import glob

for idx_batch, case in enumerate(search_list):
    
    # load every file like pack_000.hdf5
    pack_list = sorted(glob.glob(case+"/pack_*.hdf5"))

    # load the data
    for idx_pack, pack_path in enumerate(pack_list):
        data_hdf5 = h5py.File(pack_path, "r")
        
        only_key = list(data_hdf5.keys())[0]
        modalities = list(data_hdf5[only_key].keys())
        n_modalities = len(modalities)

        # construct the new hdf5 file with the same only key and float32 to float16
        new_hdf5_path = pack_path.replace(".hdf5", "_float16.hdf5")
        new_hdf5 = h5py.File(new_hdf5_path, "w")
        for key in data_hdf5.keys():
            new_hdf5.create_group(key)
            for modality in modalities:
                new_hdf5[key].create_dataset(modality, data=data_hdf5[key][modality][()], dtype="float16")
        new_hdf5.close()
        data_hdf5.close()
        print(f"[{idx_batch+1}/{n_train}] {pack_path} -> {new_hdf5_path}")
