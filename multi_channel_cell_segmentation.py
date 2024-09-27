import os
import cv2
from skimage import io
import numpy as np
from cellpose import models
from skimage.measure import block_reduce, label, regionprops
from skimage.color import label2rgb
import tifffile as tiff
import matplotlib.pyplot as plt

# Save image function
def save_image(output_path, image):
    tiff.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

# Function to load RGB microscopy images
def load_image(image_path):
    image = tiff.imread(image_path).astype('float32')
    image = np.moveaxis(image, 0, -1)  # Move channels from first to last
    
    if image is None:
        raise ValueError(f"Error: Unable to load image {image_path}")
    
    for idx in range(4):
        lp = np.percentile(image[:,:,idx], 0.05)
        up = np.percentile(image[:,:,idx], 99.95)
        image[:,:,idx] = np.clip(image[:,:,idx], lp, up)
    
    image = block_reduce(image, block_size=(2,2,1))
    
    if image.shape[2] != 4:
        raise ValueError(f"Error: The image does not have 4 channels. It has {image.shape[2]} channels.")

    for i in range(4):
        min_val = np.min(image[:, :, i])
        max_val = np.max(image[:, :, i])

        if max_val > min_val:
            image[:, :, i] = (image[:, :, i] - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"NO BUENO: min/max values are the same for channel {i}!")
            
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for i in range(4):
        ax = axs[i // 2, i % 2]
        ax.imshow(image[:, :, i], cmap='gray')
        ax.set_title(f'Channel {i + 1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    return image

# Function to segment predefined feature using Cellpose for instance segmentation
def segment_instance(image, model_type='cyto', diameter=None, channels=[0, 0]):
    cyto_model = models.Cellpose(gpu=True, model_type=model_type)
    image_float = image.astype(np.float32)
    masks, flows, styles, diams = cyto_model.eval(image_float, diameter=diameter, channels=channels, do_3D=False)
    
    # Apply distinct colors to each instance using label2rgb from skimage
    colored_masks = label2rgb(masks, bg_label=0)
    
    return masks, colored_masks

# Generate an overlay of the segmentation mask on the original image
def overlay_segmentation(image, cytoplasm_mask, nucleus_mask):
    grayscale_background = np.mean(image[:, :, :3], axis=2)
    overlay = np.stack([grayscale_background]*3, axis=-1).astype(np.uint8)
    cytoplasm_color = [0, 255, 0]
    nucleus_color = [255, 0, 255]
    overlay[cytoplasm_mask > 0] = (np.array(cytoplasm_color)).astype(np.uint8)
    overlay[nucleus_mask > 0] = (np.array(nucleus_color)).astype(np.uint8)
    return overlay

# Extract 2SC intensity values from segmented regions
def extract_2sc_intensity(image, mask, sc_channel_idx):
    sc_image = image[:, :, sc_channel_idx]
    masked_image = np.multiply(sc_image, mask)
    total_intensity = np.mean(masked_image)
    return total_intensity

# Extract cytoplasmic 2SC intensity excluding nucleus and mitochondria
def extract_cytoplasmic_2sc_intensity(image, cytoplasm_mask, nucleus_mask, mito_mask, sc_channel_idx):
    sc_image = image[:, :, sc_channel_idx]
    
    # Exclude the nucleus and mitochondria from the cytoplasm mask
    cytoplasm_mask_exclusive = np.logical_and(cytoplasm_mask, np.logical_not(np.logical_or(nucleus_mask, mito_mask)))
    
    masked_image = np.multiply(sc_image, cytoplasm_mask_exclusive)
    total_intensity = np.mean(masked_image[cytoplasm_mask_exclusive > 0])
    return total_intensity

# Calculate the ratio of intensities between the nucleus and the cytoplasm
def calculate_2sc_ratio(cytoplasm_intensity, nucleus_intensity):
    if cytoplasm_intensity == 0:
        return 0
    ratio = nucleus_intensity / cytoplasm_intensity
    return ratio

# Function to perform image-wise percentile thresholding for mitochondria
def segment_mitochondria(image, mito_channel_idx, percentile=90):
    mito_image = image[:, :, mito_channel_idx]
    threshold = np.percentile(mito_image, percentile)
    mito_mask = mito_image > threshold  # Binary mask for mitochondria
    return mito_mask, threshold

# Function to calculate co-localisation between mitochondria and 2SC within the cytoplasm
def calculate_colocalisation_in_cytoplasm(mito_mask, sc_mask, cyto_mask):
    # Restrict mitochondria and 2SC to cytoplasm area
    mito_mask_cyto = np.logical_and(mito_mask, cyto_mask)
    sc_mask_cyto = np.logical_and(sc_mask, cyto_mask)

    overlap = np.sum(np.logical_and(mito_mask_cyto, sc_mask_cyto))  # Co-localisation count
    mito_area = np.sum(mito_mask_cyto)
    sc_area = np.sum(sc_mask_cyto)
    
    if mito_area == 0 or sc_area == 0:
        return 0, 0
    
    # Fraction of mitochondria overlapping with 2SC and vice versa
    mito_colocalisation_fraction = overlap / mito_area
    sc_colocalisation_fraction = overlap / sc_area
    
    return mito_colocalisation_fraction, sc_colocalisation_fraction


# Calculate areas on a per-cell basis using instance segmentation
# Areas need to be multiplied by pixel dimensions
def calculate_cellwise_areas(cytoplasm_mask, nucleus_mask, mito_mask):
    # Label the cytoplasm mask to get individual cells
    cytoplasm_labels = label(cytoplasm_mask)
    
    cellwise_areas = []
    
    for region in regionprops(cytoplasm_labels):
        # Get the individual cell's cytoplasm mask
        cell_cytoplasm_mask = cytoplasm_labels == region.label
        
        # Exclude nucleus and mitochondria within the cell
        cell_cytoplasm_exclusive = np.logical_and(cell_cytoplasm_mask, np.logical_not(np.logical_or(nucleus_mask, mito_mask)))
        
        # Get nucleus and mitochondria areas specific to this cell
        cell_nucleus_area = np.sum(np.logical_and(nucleus_mask, cell_cytoplasm_mask))
        cell_mito_area = np.sum(np.logical_and(mito_mask, cell_cytoplasm_mask))
        cell_cytoplasm_area = np.sum(cell_cytoplasm_exclusive)
        
        cellwise_areas.append({
            'cytoplasm_area': cell_cytoplasm_area,
            'nucleus_area': cell_nucleus_area,
            'mitochondria_area': cell_mito_area,
            'cell_label': region.label
        })
    
    return cellwise_areas

# Generate an overlay of mitochondria, cytoplasm, and 2SC segmentation masks within cytoplasm
def overlay_segmentation_with_mito_in_cytoplasm(image, cytoplasm_mask, nucleus_mask, mito_mask, sc_mask):
    grayscale_background = np.mean(image[:, :, :3], axis=2)
    overlay = np.stack([grayscale_background]*3, axis=-1).astype(np.uint8)
    
    cytoplasm_color = [0, 255, 0]  # Green
    nucleus_color = [255, 0, 255]  # Magenta
    mito_color = [0, 0, 255]  # Blue
    sc_color = [255, 0, 0]  # Red
    
    # Restrict mitochondria and 2SC masks to the cytoplasm
    mito_mask_cyto = np.logical_and(mito_mask, cytoplasm_mask)
    sc_mask_cyto = np.logical_and(sc_mask, cytoplasm_mask)
    
    # Apply colors to the respective masks
    overlay[cytoplasm_mask > 0] = (np.array(cytoplasm_color)).astype(np.uint8)
    overlay[nucleus_mask > 0] = (np.array(nucleus_color)).astype(np.uint8)
    overlay[mito_mask_cyto > 0] = (np.array(mito_color)).astype(np.uint8)
    overlay[sc_mask_cyto > 0] = (np.array(sc_color)).astype(np.uint8)
    
    return overlay

# Main function to process an image
def process_image(image_path, output_base_path, cyto_channel, nuclei_channel, sc_channel, mito_channel):
    # Extract the filename without extension:
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Create a folder for this image inside the output_base_path
    image_output_folder = os.path.join(output_base_path, base_filename)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    # Load the image
    image = load_image(image_path)
    save_image(os.path.join(image_output_folder, f'{base_filename}_Processed_Image.tiff'), image)
    
    # Perform instance segmentation of cytoplasm
    cytoplasm_mask, colored_cytoplasm_mask = segment_instance(image, model_type='cyto2', diameter=None, channels=[nuclei_channel, cyto_channel])
    save_image(os.path.join(image_output_folder, f'{base_filename}_Cytoplasm_Mask.tiff'), (colored_cytoplasm_mask * 255).astype(np.uint8))

    # Perform instance segmentation of nucleus
    nucleus_mask, colored_nucleus_mask = segment_instance(image, model_type='cyto2', diameter=None, channels=[cyto_channel, 0])
    save_image(os.path.join(image_output_folder, f'{base_filename}_Nucleus_Mask.tiff'), (colored_nucleus_mask * 255).astype(np.uint8))
    
    # Perform segmentation of mitochondria using percentile thresholding
    mito_mask, mito_threshold = segment_mitochondria(image, mito_channel)
    
    # Convert mito_mask to colored version for visualization
    mito_mask_labeled = label(mito_mask)  # Label connected components of mitochondria
    colored_mito_mask = label2rgb(mito_mask_labeled, bg_label=0)
    
    save_image(os.path.join(image_output_folder, f'{base_filename}_Mitochondria_Mask.tiff'), (colored_mito_mask * 255).astype(np.uint8))
    
    print(f"\nMitochondria threshold value: {mito_threshold}")

    # Extract 2SC intensities, excluding nucleus and mitochondria for cytoplasmic 2SC
    cytoplasm_2sc_intensity = extract_cytoplasmic_2sc_intensity(image, cytoplasm_mask, nucleus_mask, mito_mask, sc_channel)
    nucleus_2sc_intensity = extract_2sc_intensity(image, nucleus_mask, sc_channel)

    # Calculate the 2SC ratio
    ratio = calculate_2sc_ratio(cytoplasm_2sc_intensity, nucleus_2sc_intensity)

    print(f"Mean 2SC Intensity in Cytoplasm (excluding Nucleus and Mitochondria): {cytoplasm_2sc_intensity:.3f}\n")
    
    # **Updated**: Calculate cell-wise areas for cytoplasm, nucleus, and mitochondria
    cellwise_areas = calculate_cellwise_areas(cytoplasm_mask, nucleus_mask, mito_mask)
    for area_info in cellwise_areas:
        print(f"Cell {area_info['cell_label']}: Cytoplasm Area = {area_info['cytoplasm_area']}, Nucleus Area = {area_info['nucleus_area']}, Mitochondria Area = {area_info['mitochondria_area']}")
    print("\n")

    # Create an overlay image
    overlay_image = overlay_segmentation(image, cytoplasm_mask, nucleus_mask)
    save_image(os.path.join(image_output_folder, f'{base_filename}_Segmentation_Overlay.tiff'), overlay_image)
    
    # Perform co-localisation of mitochondria with 2SC
    sc_mask = image[:, :, sc_channel] > np.percentile(image[:, :, sc_channel], 90)  # Thresholding 2SC mask
    mito_colocalisation_fraction, sc_colocalisation_fraction = calculate_colocalisation_in_cytoplasm(mito_mask, sc_mask, cytoplasm_mask)
    
    print(f"\nMitochondria and 2SC Co-localisation (Mito -> 2SC): {mito_colocalisation_fraction:.3f}")
    print(f"Mitochondria and 2SC Co-localisation (2SC -> Mito): {sc_colocalisation_fraction:.3f}\n")
    
    # Create an overlay image with mitochondria, cytoplasm, and 2SC
    overlay_image = overlay_segmentation_with_mito_in_cytoplasm(image, cytoplasm_mask, nucleus_mask, mito_mask, sc_mask)
    save_image(os.path.join(image_output_folder, f'{base_filename}_MitoCyto2SC_Overlay.tiff'), overlay_image)
    
    return cytoplasm_2sc_intensity, nucleus_2sc_intensity, mito_colocalisation_fraction, sc_colocalisation_fraction



if __name__ == '__main__':
    
    img_path = '/path/to/images/'  # Directory with input images
    output_path = '/path/to/output/'  # Directory to save output
    
    cyto_channel = 1  # Channel for cytoplasm 
    nuclei_channel = 0  # Channel for nucleus 
    mito_channel = 2  # Channel for mitochondria
    sc_channel = 3  # Channel for 2SC staining
    
    cytoplasm_2sc_intensities = []
    nucleus_2sc_intensities = []
    mito_colocalisation_fractions = []
    sc_colocalisation_fractions = []
    
    imgs = os.listdir(img_path)
    imgs = [img for img in imgs if img.endswith('.tiff') or img.endswith('.tif')]
    for img in imgs:
        print(f"\n\n ------ Processing image: {img} ------\n")

        cytoplasm_2sc_intensity, nucleus_2sc_intensity, mito_colocalisation, sc_colocalisation = process_image(
            os.path.join(img_path, img), 
            output_path,
            cyto_channel,
            nuclei_channel,
            sc_channel,
            mito_channel
        )
        
        cytoplasm_2sc_intensities.append(cytoplasm_2sc_intensity)
        nucleus_2sc_intensities.append(nucleus_2sc_intensity)
        mito_colocalisation_fractions.append(mito_colocalisation)
        sc_colocalisation_fractions.append(sc_colocalisation)
    
    intensity_array = np.array([cytoplasm_2sc_intensities, nucleus_2sc_intensities, mito_colocalisation_fractions, sc_colocalisation_fractions]).T   
    
    # Save data as CSV
    np.savetxt('Intensity_Colocalisation_data.csv', intensity_array, delimiter=',', 
               header='Cytoplasm,Nucleus,Mito-2SC_Colocalisation,2SC-Mito_Colocalisation', comments='', fmt='%f')
    
    print(f"\n\n ------ Summary Statistics ------\n")
    print("Cytoplasm 2SC Intensities, Nucleus 2SC Intensities, and Mitochondria-2SC Co-localisation:")
    for row in intensity_array:
        print(f'{row[0]:.3f} {row[1]:.3f} {row[2]:.3f} {row[3]:.3f}')


