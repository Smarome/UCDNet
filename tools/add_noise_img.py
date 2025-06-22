import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle

"""add noise"""
# def add_gaussian_noise(image, mean=0, std=25):
#     """
#     Add Gaussian noise to an image.
#
#     Parameters:
#     - image: Input image in which noise is to be added.
#     - mean: Mean of the Gaussian noise.
#     - std: Standard deviation of the Gaussian noise.
#
#     Returns:
#     - noisy_image: Image with added Gaussian noise.
#     """
#     # Generate Gaussian noise
#     gaussian_noise = np.random.normal(mean, std, image.shape).astype('uint8')
#
#     # Add the Gaussian noise to the image
#     noisy_image = cv2.add(image, gaussian_noise)
#
#     return noisy_image
#
#
# # Define the source and destination directories
# source_root = '/workspace/data/TEST23/samples'
# destination_root = '/workspace/data/TEST23_noise1/samples'
#
# # List of subdirectories
# subdirectories = ['UAV_1', 'UAV_2', 'UAV_3', 'UAV_4', 'UAV_5', 'UAV_6']
#
# # Iterate through each subdirectory
# for subdir in subdirectories:
#     source_dir = os.path.join(source_root, subdir)
#     destination_dir = os.path.join(destination_root, subdir)
#
#     # Create destination directory if it doesn't exist
#     if not os.path.exists(destination_dir):
#         os.makedirs(destination_dir)
#
#     # List all jpg files in the current subdirectory
#     image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
#
#     # Process each image in the subdirectory with a progress bar
#     for filename in tqdm(image_files, desc=f"Processing {subdir}"):
#         source_path = os.path.join(source_dir, filename)
#         destination_path = os.path.join(destination_dir, filename)
#
#         # Load the image
#         image = cv2.imread(source_path)
#
#         # Add Gaussian noise to the image
#         noisy_image = add_gaussian_noise(image, mean=0, std=1)
#
#         # Save the noisy image
#         cv2.imwrite(destination_path, noisy_image)
#
# print("All images have been processed.")

"""change pkl"""
with open("/workspace/data/TEST23/bevdetv2-nuscenes-test23_infos_val.pkl", 'rb') as file:
    pkl_data = pickle.load(file)
infos = pkl_data['infos']
for ann_info in infos:
    for UAV_name in ['UAV_1', 'UAV_2', 'UAV_3', 'UAV_4', 'UAV_5', 'UAV_6']:
        trans = ann_info['cams'][UAV_name]['sensor2ego_translation']
        trans = np.array(trans)
        noise = np.random.normal(0, 0.5, trans.shape)
        trans_noisy = trans + noise
        trans_noisy_list = trans_noisy.tolist()
        ann_info['cams'][UAV_name]['sensor2ego_translation'] = trans_noisy_list
with open("/workspace/data/TEST23/bevdetv2-nuscenes-test23_infos_val_noise.pkl", 'wb') as file:
    pickle.dump(pkl_data, file)



