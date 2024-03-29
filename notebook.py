# train_1 -> GPU:6 killed
# train_2 -> GPU:3 killed
# train_3 -> GPU:7 killed

# Mar 5 2:42 PM
# eval_1 -> GPU:2 killed
# eval_2 -> GPU:6 killed
# eval_3 -> GPU:7 killed

# load from checkpoint ./proj/decoder_UNETR/best_model.pth
# Start testing on 2233 samples.
# Test Loss: 38.6859182846043

# load from checkpoint ./proj/decoder_Deconv/best_model.pth
# Start testing on 2233 samples.
# Test Loss: 44.63018013609501

# load from checkpoint ./proj/decoder_PyramidPooling/model_270.pth
# Start testing on 2233 samples.
# Test Loss: 39.183466535143346

# load from checkpoint ./proj/decoder_PP_pct20/best_model.pth
# Start testing on 2233 samples.
# Test Loss: 37.103882594512875

# load from checkpoint ./proj/decoder_PP_pct100/best_model.pth
# Start testing on 2233 samples.
# Test Loss: 36.82605276931957

# eval_30 -> GPU:6 killed
# eval_31 -> GPU:7 killed
# eval_32 -> GPU:7 killed
# UNETR_pct20 -> GPU:0 killed
# UNETR_pct100 -> GPU:2 killed

# Mar 7
# train_GPU_5 killed
# train_GPU_1 -> GPU:1 killed
# data_embed -> GPU:5 killed
# SynthRad_Brain: 181 cases
# train:val:test = 0.7:0.15:0.15
# train: 181 * 0.7 = 126
# val: 181 * 0.15 = 27
# test: 181 * 0.15 = 27
# 5% train: 126 * 5% = 6
# 20% train: 126 * 20% = 25
# 100% train: 126 * 100% = 126


# MIMRTL_Brain: 777 cases
# train:val:test = 0.7:0.15:0.15
# train: 777 * 0.7 = 543
# val: 777 * 0.15 = 116
# test: 777 * 0.15 = 116
# 5% train: 543 * 5% = 27
# 20% train: 543 * 20% = 108
# 100% train: 543 * 100% = 543

# Mar 8:
# GPU_playground -> GPU:0
# GPU_00 -> GPU:0
# GPU_10 -> GPU:1
# GPU_20 -> GPU:2
# GPU_30 -> GPU:3
# GPU_40 -> GPU:4
# GPU_50 -> GPU:5
# GPU_60 -> GPU:6
# GPU_70 -> GPU:7
# GPU_71 -> GPU:7
# GPU_72 -> GPU:7
# GPU_73 -> GPU:7
# GPU_74 -> GPU:7
# GPU_75 -> GPU:7


# Mar 14:
# 3495093.GPU_11	(03/14/2024 10:37:46 PM)	(Detached) -> UNet_pct1_cv
# 3470023.GPU_34	(03/14/2024 10:30:32 PM)	(Detached) -> UNet_pct1_cv
# 3468772.GPU_33	(03/14/2024 10:30:26 PM)	(Detached) -> UNet_pct1_cv
# 3467883.GPU_32	(03/14/2024 10:30:23 PM)	(Detached) -> UNet_pct1_cv
# 3467264.GPU_31	(03/14/2024 10:30:11 PM)	(Detached) -> UNet_pct1_cv
# 3185008.GPU_20	(03/09/2024 02:12:41 AM)	(Detached) -> PP_slice_pct5
# 3182296.GPU_10	(03/09/2024 02:11:57 AM)	(Detached) -> Deconv_slice_pct5
# 3177737.GPU_00	(03/09/2024 02:10:34 AM)	(Detached) -> UNETR_slice_pct5
# 34853.playground	(03/08/2024 12:50:38 AM)	(Detached)
