from torch.utils.data import Dataset
import nibabel as nib

class simple_nifti_dataset(Dataset):
    def __init__(self, case_list = []):
        """
        Args:
            file_paths (list of str): List of HDF5 file paths.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.case_list = case_list

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        folder_path = self.case_list[idx]
        mr_path = folder_path + "/mr.nii.gz"
        ct_path = folder_path + "/ct.nii.gz"
        # try load both
        try:
            mr_file = nib.load(mr_path)
            ct_file = nib.load(ct_path)
        except:
            # if failed, raise an error
            raise ValueError(f"Error loading {mr_path} or {ct_path}")
        
        mr_data = mr_file.get_fdata()
        ct_data = ct_file.get_fdata()
        return {"mr": mr_data, "ct": ct_data}
