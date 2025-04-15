import os
import csv
import cv2
from skimage import io
import numpy as np
from cellpose import models
from skimage.measure import block_reduce, label, regionprops
from skimage.color import label2rgb
import tifffile as tiff
import matplotlib.pyplot as plt

##############################################################################
# 1) GATHER ALL IMAGE PATHS RECURSIVELY
##############################################################################
def gather_image_paths_in_subfolders(root_path):
    """
    Recursively finds all .tif/.tiff files in `root_path` and returns a list of full file paths.
    """
    image_files = []
    for current_dir, subdirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
                image_files.append(os.path.join(current_dir, file))
    return image_files


##############################################################################
# 2) COMPUTE BATCH-WIDE CHANNEL STATS (RAW SPACE)
##############################################################################
def compute_batch_channel_stats(all_image_paths, channels=4, lower_percentile=0.05, upper_percentile=99.95):
    """
    Collect pixel values across all images (in raw space),
    compute global min/max (percentiles) for each channel.
    Returns min_vals[ch], max_vals[ch].
    """
    all_channel_data = [[] for _ in range(channels)]

    for path in all_image_paths:
        im = tiff.imread(path).astype('float32')
        # Move channel axis -> shape: (H, W, channels)
        im = np.moveaxis(im, 0, -1)

        for ch_idx in range(channels):
            all_channel_data[ch_idx].extend(im[:, :, ch_idx].ravel())

    min_vals = []
    max_vals = []
    for ch_idx in range(channels):
        ch_data = np.array(all_channel_data[ch_idx])
        lower_val = np.percentile(ch_data, lower_percentile)
        upper_val = np.percentile(ch_data, upper_percentile)
        min_vals.append(lower_val)
        max_vals.append(upper_val)

    return min_vals, max_vals


##############################################################################
# 3) HELPER TO LOAD & NORMALIZE ONE IMAGE (using global min/max)
##############################################################################
def load_and_normalize_image(image_path, min_vals, max_vals):
    """
    Loads image from path, downsamples by 2, and uses global (min_vals, max_vals)
    to normalize each channel to [0, 1].
    """
    # Load raw
    image = tiff.imread(image_path).astype('float32')
    image = np.moveaxis(image, 0, -1)  # shape: (H, W, 4)

    # Downsample
    image = block_reduce(image, block_size=(2, 2, 1))

    # Normalize each channel to [0,1] using global min/max
    for ch_idx in range(4):
        image[:, :, ch_idx] = np.clip(image[:, :, ch_idx], min_vals[ch_idx], max_vals[ch_idx])
        denom = max_vals[ch_idx] - min_vals[ch_idx]
        if denom > 0:
            image[:, :, ch_idx] = (image[:, :, ch_idx] - min_vals[ch_idx]) / denom
        else:
            raise ValueError(f"Error: min and max are the same for channel {ch_idx}")
    
    return image


##############################################################################
# 4) GATHER NORMALIZED MITO & 2SC PIXELS ACROSS ALL IMAGES
##############################################################################
def gather_normalized_mito_and_sc_pixels(all_image_paths, min_vals, max_vals,
                                         mito_channel_idx=2, sc_channel_idx=3):
    """
    1) Loads each image, normalizes it using the global min_vals/max_vals.
    2) Extracts the (normalized) pixels from the mitochondria channel and 2SC channel.
    3) Returns two big lists: all_mito_pixels_norm, all_sc_pixels_norm.
    """
    all_mito_pixels_norm = []
    all_sc_pixels_norm = []

    for path in all_image_paths:
        # Load + normalize
        norm_image = load_and_normalize_image(path, min_vals, max_vals)
        # Extract channels
        mito_pixels = norm_image[:, :, mito_channel_idx].ravel()
        sc_pixels = norm_image[:, :, sc_channel_idx].ravel()
        all_mito_pixels_norm.extend(mito_pixels)
        all_sc_pixels_norm.extend(sc_pixels)
    
    return all_mito_pixels_norm, all_sc_pixels_norm


##############################################################################
# 5) COMPUTE GLOBAL THRESHOLDS IN NORMALIZED SPACE
##############################################################################
def compute_global_thresholds_in_normalized_space(mito_pixels_norm, sc_pixels_norm,
                                                  mito_percentile=90, sc_percentile=90):
    """
    Given two lists of normalized intensities, one for mitochondria channel and
    one for 2SC channel, compute global threshold (e.g. 90th percentile) in normalized space.
    """
    mito_pixels_norm = np.array(mito_pixels_norm)
    sc_pixels_norm = np.array(sc_pixels_norm)

    global_mito_threshold = np.percentile(mito_pixels_norm, mito_percentile)
    global_sc_threshold = np.percentile(sc_pixels_norm, sc_percentile)

    return global_mito_threshold, global_sc_threshold


##############################################################################
# UTILITY
##############################################################################
def save_image(output_path, image):
    tiff.imwrite(output_path, image)
    print(f"Image saved to {output_path}")


##############################################################################
# CELLPOSE SEGMENTATION, ETC.
##############################################################################
def segment_instance(image, model_type='cyto', diameter=None, channels=[0, 0]):
    cyto_model = models.Cellpose(gpu=True, model_type=model_type)
    image_float = image.astype(np.float32)
    masks, flows, styles, diams = cyto_model.eval(
        image_float, diameter=diameter, channels=channels, do_3D=False
    )
    colored_masks = label2rgb(masks, bg_label=0)
    return masks, colored_masks


def overlay_segmentation(image, cytoplasm_mask, nucleus_mask):
    grayscale_background = np.mean(image[:, :, :3], axis=2)
    overlay = np.stack([grayscale_background]*3, axis=-1).astype(np.uint8)
    cytoplasm_color = [0, 255, 0]
    nucleus_color = [255, 0, 255]
    overlay[cytoplasm_mask > 0] = np.array(cytoplasm_color, dtype=np.uint8)
    overlay[nucleus_mask > 0] = np.array(nucleus_color, dtype=np.uint8)
    return overlay


def overlay_segmentation_with_mito_in_cytoplasm(image, cytoplasm_mask, nucleus_mask, mito_mask, sc_mask):
    grayscale_background = np.mean(image[:, :, :3], axis=2)
    overlay = np.stack([grayscale_background]*3, axis=-1).astype(np.uint8)
    
    cytoplasm_color = [0, 255, 0]  # Green
    nucleus_color = [255, 0, 255]  # Magenta
    mito_color = [0, 0, 255]       # Blue
    sc_color = [255, 0, 0]         # Red

    mito_mask_cyto = np.logical_and(mito_mask, cytoplasm_mask)
    sc_mask_cyto = np.logical_and(sc_mask, cytoplasm_mask)
    
    overlay[cytoplasm_mask > 0] = cytoplasm_color
    overlay[nucleus_mask > 0] = nucleus_color
    overlay[mito_mask_cyto > 0] = mito_color
    overlay[sc_mask_cyto > 0] = sc_color
    
    return overlay


def extract_2sc_intensity_excluding_mito(image, mask, mito_mask, sc_channel_idx):
    sc_image = image[:, :, sc_channel_idx]
    mask_exclusive = np.logical_and(mask, np.logical_not(mito_mask))
    masked_image = np.multiply(sc_image, mask_exclusive)
    total_intensity = np.mean(masked_image[mask_exclusive > 0])
    return total_intensity


def extract_cytoplasmic_2sc_intensity(image, cytoplasm_mask, nucleus_mask, mito_mask, sc_channel_idx):
    sc_image = image[:, :, sc_channel_idx]
    cytoplasm_mask_exclusive = np.logical_and(
        cytoplasm_mask, 
        np.logical_not(np.logical_or(nucleus_mask, mito_mask))
    )
    masked_image = np.multiply(sc_image, cytoplasm_mask_exclusive)
    total_intensity = np.mean(masked_image[cytoplasm_mask_exclusive > 0])
    return total_intensity


def calculate_2sc_ratio(cytoplasm_intensity, nucleus_intensity):
    if cytoplasm_intensity == 0:
        return 0
    return nucleus_intensity / cytoplasm_intensity


def calculate_colocalisation_in_cytoplasm(mito_mask, sc_mask, cyto_mask):
    mito_mask_cyto = np.logical_and(mito_mask, cyto_mask)
    sc_mask_cyto = np.logical_and(sc_mask, cyto_mask)
    overlap = np.sum(np.logical_and(mito_mask_cyto, sc_mask_cyto))
    mito_area = np.sum(mito_mask_cyto)
    sc_area = np.sum(sc_mask_cyto)
    if mito_area == 0 or sc_area == 0:
        return 0, 0
    mito_colocalisation_fraction = overlap / mito_area
    sc_colocalisation_fraction = overlap / sc_area
    return mito_colocalisation_fraction, sc_colocalisation_fraction


def calculate_cellwise_areas(cytoplasm_mask, nucleus_mask, mito_mask):
    cytoplasm_labels = label(cytoplasm_mask)
    cellwise_areas = []
    for region in regionprops(cytoplasm_labels):
        cell_cytoplasm_mask = (cytoplasm_labels == region.label)
        cell_cytoplasm_exclusive = np.logical_and(
            cell_cytoplasm_mask,
            np.logical_not(np.logical_or(nucleus_mask, mito_mask))
        )
        cell_nucleus_area = np.sum(np.logical_and(nucleus_mask, cell_cytoplasm_mask))
        cell_mito_area = np.sum(np.logical_and(mito_mask, cell_cytoplasm_mask))
        cell_cytoplasm_area = np.sum(cell_cytoplasm_exclusive)
        cellwise_areas.append({
            'cell_label': region.label,
            'cytoplasm_area': cell_cytoplasm_area,
            'nucleus_area': cell_nucleus_area,
            'mitochondria_area': cell_mito_area
        })
    return cellwise_areas


##############################################################################
# 6) MAIN PROCESS FUNCTION
##############################################################################
def process_image(
    image_path, 
    output_base_path, 
    cyto_channel, 
    nuclei_channel, 
    sc_channel, 
    mito_channel,
    min_vals, 
    max_vals,
    global_mito_threshold_norm,
    global_sc_threshold_norm
):
    """
    Loads & normalizes a single image, segments cytoplasm/nuclei with Cellpose,
    and uses the precomputed global thresholds (in normalized space) for mitochondria & 2SC.
    """
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    image_output_folder = os.path.join(output_base_path, base_filename)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    # --------- (A) LOAD & NORMALIZE
    image = load_and_normalize_image(image_path, min_vals, max_vals)
    save_image(os.path.join(image_output_folder, f'{base_filename}_Processed_Image.tiff'), image)
    
    # --------- (B) CELLPOSE for cytoplasm
    cytoplasm_mask, colored_cytoplasm_mask = segment_instance(
        image, model_type='cyto2', diameter=None, channels=[nuclei_channel, cyto_channel]
    )
    save_image(os.path.join(image_output_folder, f'{base_filename}_Cytoplasm_Mask.tiff'), 
               (colored_cytoplasm_mask * 255).astype(np.uint8))

    # --------- (C) CELLPOSE for nucleus
    nucleus_mask, colored_nucleus_mask = segment_instance(
        image, model_type='cyto2', diameter=None, channels=[cyto_channel, 0]
    )
    save_image(os.path.join(image_output_folder, f'{base_filename}_Nucleus_Mask.tiff'), 
               (colored_nucleus_mask * 255).astype(np.uint8))
    
    # --------- (D) Mitochondria Mask (Normalized threshold)
    mito_channel_data = image[:, :, mito_channel]
    mito_mask = mito_channel_data > global_mito_threshold_norm
    mito_mask_labeled = label(mito_mask)
    colored_mito_mask = label2rgb(mito_mask_labeled, bg_label=0)
    save_image(os.path.join(image_output_folder, f'{base_filename}_Mitochondria_Mask.tiff'), 
               (colored_mito_mask * 255).astype(np.uint8))
    
    print(f"[{base_filename}] Global Mitochondria threshold in normalized space: {global_mito_threshold_norm:.3f}")

    # --------- (E) 2SC Mask (Normalized threshold)
    sc_channel_data = image[:, :, sc_channel]
    sc_mask = sc_channel_data > global_sc_threshold_norm
    print(f"[{base_filename}] Global 2SC threshold in normalized space: {global_sc_threshold_norm:.3f}")

    # --------- (F) Extract intensities
    cytoplasm_2sc_intensity = extract_cytoplasmic_2sc_intensity(
        image, cytoplasm_mask, nucleus_mask, mito_mask, sc_channel
    )
    nucleus_2sc_intensity = extract_2sc_intensity_excluding_mito(
        image, nucleus_mask, mito_mask, sc_channel
    )
    ratio = calculate_2sc_ratio(cytoplasm_2sc_intensity, nucleus_2sc_intensity)
    print(f"Mean 2SC Intensity (cyto, excl. nucleus & mito): {cytoplasm_2sc_intensity:.3f}")

    # --------- (G) Cellwise areas
    cellwise_areas = calculate_cellwise_areas(cytoplasm_mask, nucleus_mask, mito_mask)
    for area_info in cellwise_areas:
        print(f"Cell {area_info['cell_label']}: "
              f"Cytoplasm Area = {area_info['cytoplasm_area']}, "
              f"Nucleus Area = {area_info['nucleus_area']}, "
              f"Mitochondria Area = {area_info['mitochondria_area']}")
    print("\n")
    # Save cellwise areas to CSV
    with open(os.path.join(image_output_folder, f'{base_filename}_cellwise_areas.csv'), 'w', newline='') as csvfile:
        fieldnames = ['cell_label', 'cytoplasm_area', 'nucleus_area', 'mitochondria_area']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in cellwise_areas:
            writer.writerow(row)

    # --------- (H) Overlay (cyto + nucleus)
    overlay_image = overlay_segmentation(image, cytoplasm_mask, nucleus_mask)
    save_image(os.path.join(image_output_folder, f'{base_filename}_Segmentation_Overlay.tiff'), overlay_image)
    
    # --------- (I) Co-localization with global 2SC mask
    mito_colocalisation_fraction, sc_colocalisation_fraction = calculate_colocalisation_in_cytoplasm(
        mito_mask, sc_mask, cytoplasm_mask
    )
    print(f"Mito->2SC co-local: {mito_colocalisation_fraction:.3f}, 2SC->Mito co-local: {sc_colocalisation_fraction:.3f}")

    # Overlay (cyto + nucleus + mito + sc)
    overlay_image = overlay_segmentation_with_mito_in_cytoplasm(
        image, cytoplasm_mask, nucleus_mask, mito_mask, sc_mask
    )
    save_image(os.path.join(image_output_folder, f'{base_filename}_MitoCyto2SC_Overlay.tiff'), overlay_image)
    
    return cytoplasm_2sc_intensity, nucleus_2sc_intensity, mito_colocalisation_fraction, sc_colocalisation_fraction


##############################################################################
# 7) MAIN SCRIPT
##############################################################################
if __name__ == '__main__':
    # Top-level raw_data folder containing subfolders
    raw_data_path = '/mnt/sda/Seema/raw_data/'
    # Where you want your results to be saved
    top_level_output_path = '/mnt/sda/Seema/'
    
    nuclei_channel = 0
    cyto_channel = 1
    mito_channel = 2
    sc_channel = 3

    # ---------------------------------------------------------------
    # (A) Gather ALL images for computing global stats
    # ---------------------------------------------------------------
    all_image_paths = gather_image_paths_in_subfolders(raw_data_path)

    # 1) Global channel stats (raw space) -> for normalization
    min_vals, max_vals = compute_batch_channel_stats(all_image_paths, channels=4)

    # 2) Gather normalized mito & sc pixels across all images
    #    so that thresholds are computed in normalized space
    all_mito_pixels_norm, all_sc_pixels_norm = gather_normalized_mito_and_sc_pixels(
        all_image_paths,
        min_vals, 
        max_vals,
        mito_channel_idx=mito_channel,
        sc_channel_idx=sc_channel
    )

    # 3) Compute global thresholds in normalized space
    global_mito_threshold_norm, global_sc_threshold_norm = compute_global_thresholds_in_normalized_space(
        all_mito_pixels_norm, 
        all_sc_pixels_norm,
        mito_percentile=90, 
        sc_percentile=95
        
    )

    print("\n====== Global Normalized Thresholds ======")
    print(f"Mitochondria channel threshold (normalized): {global_mito_threshold_norm:.3f}")
    print(f"2SC channel threshold (normalized): {global_sc_threshold_norm:.3f}\n")

    # ---------------------------------------------------------------
    # (B) Iterate over subfolders (cell types), process each image
    # ---------------------------------------------------------------
    subfolders = sorted(
    d for d in os.listdir(raw_data_path) 
    if os.path.isdir(os.path.join(raw_data_path, d))
    )
    
    for sub in subfolders:
        celltype_folder = os.path.join(raw_data_path, sub)
        celltype_output_folder = os.path.join(top_level_output_path, sub)
        if not os.path.exists(celltype_output_folder):
            os.makedirs(celltype_output_folder)
        
        image_files = gather_image_paths_in_subfolders(celltype_folder)
        
        # Make sure we sort these in alphabetical order, if desired
        # (You already sorted inside gather_image_paths_in_subfolders, but you can double-check)
        image_files.sort()
    
        cytoplasm_2sc_intensities = []
        nucleus_2sc_intensities = []
        mito_colocalisation_fractions = []
        sc_colocalisation_fractions = []
    
        # -----------------------
        # Process each image
        # -----------------------
        for img_path in image_files:
            print(f"\n\n------ Processing image: {img_path} ------\n")
            cyto_int, nuc_int, mito_coloc, sc_coloc = process_image(
                image_path=img_path,
                output_base_path=celltype_output_folder,
                cyto_channel=cyto_channel,
                nuclei_channel=nuclei_channel,
                sc_channel=sc_channel,
                mito_channel=mito_channel,
                min_vals=min_vals,
                max_vals=max_vals,
                global_mito_threshold_norm=global_mito_threshold_norm,
                global_sc_threshold_norm=global_sc_threshold_norm
            )
            cytoplasm_2sc_intensities.append(cyto_int)
            nucleus_2sc_intensities.append(nuc_int)
            mito_colocalisation_fractions.append(mito_coloc)
            sc_colocalisation_fractions.append(sc_coloc)
    
        # ----------------------------------------
        # Add image filenames to the summary CSV
        # ----------------------------------------
        # 1) Extract the base filenames from the paths
        image_names = [os.path.basename(path) for path in image_files]
    
        # 2) Combine image names + results in a single structure
        #    Each row: (image name, cytoplasm, nucleus, mito-coloc, sc-coloc)
        summary_data = list(zip(
            image_names,
            cytoplasm_2sc_intensities,
            nucleus_2sc_intensities,
            mito_colocalisation_fractions,
            sc_colocalisation_fractions
        ))
    
        # 3) Write this to CSV with headers
        summary_csv_path = os.path.join(celltype_output_folder, 'Intensity_Colocalisation_data.csv')
        with open(summary_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write a header row:
            writer.writerow(['ImageName', 'Cytoplasm', 'Nucleus', 'Mito-2SC_Colocalisation', '2SC-Mito_Colocalisation'])
            # Write each row
            for row in summary_data:
                writer.writerow(row)
    
        print(f"\nSaved summary CSV: {summary_csv_path}")
        print(f"Done processing subfolder: {sub}\n")
