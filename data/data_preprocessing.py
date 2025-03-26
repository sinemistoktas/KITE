import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, morphology, restoration, transform, registration, exposure, feature
from scipy.ndimage import map_coordinates, binary_fill_holes, median_filter, gaussian_laplace, uniform_filter, generic_filter, gaussian_filter, laplace
from skimage.io import imread
from scipy.stats import trim_mean
import cv2
import os

input_directory = "./duke_original/image"
ground_truth_directory = "./duke_original/lesion"

# the preprocessing examples were done with the help of Par Kragsterman:
# https://about.cmrad.com/articles/the-ultimate-guide-to-preprocessing-medical-images-techniques-tools-and-best-practices-for-enhanced-diagnosis

def original_vs_preprocessed_plot(original, processed, title1= "Original Image", title2= "Preprocessed Image"):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title(title1)
    plt.imshow(original ,cmap= "gray")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title(title2)
    plt.imshow(processed, cmap= "gray")
    plt.axis("off")
    plt.show()

# threshold -> which pixels are considered the "black region"
# np.max(image) -> basically the maximum intensity of the image.
def thresholding(image, threshold):
    binary_image = image > (threshold * np.max(image))  # pixels above the threshold will be white, else they will be black, possibly corresponding to the fluid regions.
    binary_image = morphology.closing(binary_image, morphology.square(5))  # fill small gaps (may be removed later if the small gaps are retinal fluids too)
    binary_image = morphology.remove_small_objects(binary_image, min_size=500)  # remove noise

    return binary_image.astype(np.uint8)

def remove_background_otsu(image): # uses otsu thresholding to remove the background.
    _, otsu_threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_threshold

def plot_multiple_thresholds(image, ground_truth, thresholds):
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, len(thresholds)//2 + 1, 1)
    plt.imshow(image, cmap="gray")
    plt.contour(ground_truth, colors="red", linewidths= 2)
    plt.title("Original Image")
    plt.axis("off")

    for idx, threshold in enumerate(thresholds, start=2):
        bg_removed = thresholding(image, threshold)
        plt.subplot(2, len(thresholds)//2 + 1, idx)
        plt.imshow(bg_removed, cmap="gray")
        plt.contour(ground_truth, colors="red", linewidths= 1)
        plt.title(f"Threshold = {threshold}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

threshold_values = [0.15, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.25]


def denoise_image(image):
    return restoration.denoise_wavelet(image, method= "BayesShrink", mode="soft", rescale_sigma= True) #wavelet based denoising

def resample_image(image, target_shape): # resampling = changing the pixel size of an image without altering its resolution.
    return transform.resize(image, target_shape, order=3, mode= "reflect", anti_aliasing= True)

# Image Registration: The process of aligning two images from different modalities or time points.
# I don't think this is needed in our application but here it is:

def register_images(fixed_image, moving_image):
    v, u = registration.optical_flow_tvl1(fixed_image, moving_image) # this created a displacement field, meaning a vector field that shows how each pixel should be shifted to match the fixed image.
    coords = np.meshgrid(np.arange(moving_image.shape[0]),
                         np.arange(moving_image.shape[1]),
                         indexing="ij")
    registered_image = map_coordinates(moving_image, 
                                       [coords[0] + v, coords[1] + u], 
                                       order=1)
    return registered_image

# test function to move the image.
def translate_image(image, tx=30, ty=20):
    moved_image = np.zeros_like(image)
    max_y, max_x = image.shape
    for y in range(max_y):
        for x in range(max_x):
            new_y, new_x = y + ty, x + tx
            if 0 <= new_y < max_y and 0 <= new_x < max_x:
                moved_image[new_y, new_x] = image[y, x]
    
    return moved_image


def normalize_intensity(image, min_percentile= 0.5, max_percentile=99.5): # normalize the image intensities to ensure consistency across the dataset.
    min_val = np.percentile(image, min_percentile)
    max_val = np.percentile(image, max_percentile)
    normalized_img = (image - min_val) / (max_val - min_val)
    return np.clip(normalized_img, 0, 1) # further ensures that no possible out-of-range values are present.

def adapted_thresholding(image): # uses adapted thresholding to remove the background.
    smoothed_image = cv2.blur(image, (49, 49))
    bw_image = np.where(image > smoothed_image, 255, 0).astype(np.uint8)
    return bw_image

def apply_median_filter(image):
    return median_filter(image, size=5)

def apply_log_filter(image):
    return gaussian_laplace(image, sigma=2)

def apply_box_filter(image):
    return uniform_filter(image, size=5)

def apply_a_trimmed_mean_filter(image, alpha, kernel_size):
    return generic_filter(image, lambda x: trim_mean(x, proportiontocut=alpha), size= kernel_size)

def apply_gaussian_filter(image, sigma= 2):
    return gaussian_filter(image, sigma)

def apply_laplacian_filter(image):
    return laplace(image)

def preprocess_image(image): # note to mislina: call this to preprocess the image ! 
    image = apply_median_filter(image)
    return normalize_intensity(image)

def handle_npz_images(npz_path, filename): # this function is only for .npz files.
    np_data = np.load(npz_path)
    if filename.startswith("TEST"):
        for key in np_data:
            image = np_data[key]
            ground_truth = np.zeros_like(image)

    elif filename.startswith("TRAIN"):
        image = np_data['image'] # I've noticed that the "train" files include two fields, image and label.
        ground_truth = np_data['label'] # label is for the labeled retina fluid.

    else:
        print(f"Skipping unknown file type: {filename}")
    
    return ground_truth

def find_ground_truth(filename, image): # this function can be used to find the ground truth label of an image for the DUKE
    # dataset.
    ground_truth_path = os.path.join(ground_truth_directory, filename)

    if os.path.exists(ground_truth_path):
        return imread(ground_truth_path, as_gray= True)
    else:
        print(f"Ground Truth not found for {filename}")
        return np.zeros_like(image)

for filename in os.listdir(input_directory):
    if filename.lower().endswith(".npz"):  # RETOUCH dataset files
        npz_path = os.path.join(input_directory, filename)
        try:
            ground_truth = handle_npz_images(npz_path, filename)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue

    else: # any other file type
        image_path = os.path.join(input_directory, filename)
        image = imread(image_path, as_gray= True)
        ground_truth = find_ground_truth(filename, image)

    image = preprocess_image(image)
    plot_multiple_thresholds(image, ground_truth, threshold_values)
