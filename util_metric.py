# here we compute the main metrics for the evaluation of the model
# ground truth folder: /data/MR2CT/nifty/CT
# ground truth is from 0 to 4000

prediction_folder = [
    "results/nifti_test_UNETR_v1",
    "results/nifti_test_Deconv_v1",
    "results/nifti_test_PP_v1",
    "results/nifti_test_PP_pct20_v1",
    "results/nifti_test_PP_pct100_v1",
]

metric = [
    "RMSE",
    "MAE",
    "PSNR",
    "SSIM",
    "DSC_AIR",
    "DSC_SOFT",
    "DSC_BONE",
]

savename = "metrics.xlsx"

# load the ground truth
import os
import glob
import numpy as np
import nibabel as nib
import xlsxwriter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

ground_truth_folder = "data/MR2CT/nifty/CT"
ground_truth_list = sorted(glob.glob(os.path.join(ground_truth_folder, "*.nii.gz")))
print("Find {} ground truth cases.".format(len(ground_truth_list)))

model_name = [
    "UNETR",
    "Deconv",
    "PP_pct5",
    "PP_pct20",
    "PP_pct100",
]
n_model = len(model_name)
n_metric = len(metric)

# create a workbook and add a worksheet
workbook = xlsxwriter.Workbook(savename)
worksheet = workbook.add_worksheet()

# the first row is space, RMSE, MAE, PSNR, SSIM, DSC_AIR, DSC_SOFT, DSC_BONE
worksheet.write(0, 0, "Model")
worksheet.write(0, 1, "RMSE")
worksheet.write(0, 2, "MAE")
worksheet.write(0, 3, "PSNR")
worksheet.write(0, 4, "SSIM")
worksheet.write(0, 5, "DSC_AIR")
worksheet.write(0, 6, "DSC_SOFT")
worksheet.write(0, 7, "DSC_BONE")

# the first column is model name
worksheet.write(1, 0, "UNETR")
worksheet.write(2, 0, "Deconv")
worksheet.write(3, 0, "PP_pct5")
worksheet.write(4, 0, "PP_pct20")
worksheet.write(5, 0, "PP_pct100")

for idx_model in range(n_model):
    model_folder = prediction_folder[idx_model]
    prediction_list = sorted(glob.glob(os.path.join(model_folder, "*.nii.gz")))

    print("Find {} prediction cases for model: {}".format(len(prediction_list), model_name[idx_model]))

    n_sample = len(prediction_list)
    metrics = np.zeros((n_sample, n_metric))

    print("Start evaluating model: ", model_name[idx_model])

    for idx_sample in range(n_sample):
        
        # load the ground truth
        ground_truth = nib.load(ground_truth_list[idx_sample]).get_fdata()
        prediction = nib.load(prediction_list[idx_sample]).get_fdata()

        # remove last 4 slices
        ground_truth = ground_truth[:, :, :-4]
        prediction = prediction[:, :, :-4]
        
        # shift ground truth from 0 to 4000 to -1000 to 3000
        ground_truth = ground_truth - 1000

        # compute the metrics
        metrics[idx_sample, 0] = np.sqrt(np.mean((ground_truth - prediction) ** 2))
        metrics[idx_sample, 1] = np.mean(np.abs(ground_truth - prediction))
        metrics[idx_sample, 2] = psnr(ground_truth, prediction, data_range=4000)
        metrics[idx_sample, 3] = ssim(ground_truth, prediction, data_range=4000)
        
        mask_air_ground_truth = ground_truth < -500
        mask_air_prediction = prediction < -500
        mask_soft_ground_truth = (ground_truth >= -500) & (ground_truth < 500)
        mask_soft_prediction = (prediction >= -500) & (prediction < 500)
        mask_bone_ground_truth = ground_truth >= 500
        mask_bone_prediction = prediction >= 500

        metrics[idx_sample, 4] = 2 * np.sum(mask_air_ground_truth & mask_air_prediction) / (np.sum(mask_air_ground_truth) + np.sum(mask_air_prediction))
        metrics[idx_sample, 5] = 2 * np.sum(mask_soft_ground_truth & mask_soft_prediction) / (np.sum(mask_soft_ground_truth) + np.sum(mask_soft_prediction))
        metrics[idx_sample, 6] = 2 * np.sum(mask_bone_ground_truth & mask_bone_prediction) / (np.sum(mask_bone_ground_truth) + np.sum(mask_bone_prediction))

        print(f"[{idx_sample+1}/{n_sample}] RMSE: {metrics[idx_sample, 0]:.4f}, MAE: {metrics[idx_sample, 1]:.4f}, PSNR: {metrics[idx_sample, 2]:.4f}, SSIM: {metrics[idx_sample, 3]:.4f}, DSC_AIR: {metrics[idx_sample, 4]:.4f}, DSC_SOFT: {metrics[idx_sample, 5]:.4f}, DSC_BONE: {metrics[idx_sample, 6]:.4f}")

    # save the metrics
    metrics_mean = np.mean(metrics, axis=0)
    metrics_std = np.std(metrics, axis=0)
    # the output should be mean +/- std with four decimal places
    for idx_metric in range(n_metric):
        worksheet.write(idx_model+1, idx_metric+1, "{:.4f} +/- {:.4f}".format(metrics_mean[idx_metric], metrics_std[idx_metric]))

workbook.close()