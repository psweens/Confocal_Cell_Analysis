# 2D Confocal Microscopy Cell Image Segmentation and Quantitative Analysis

This repository contains a Python script for processing multi-channel microscopy images. It uses libraries such as `cellpose`, `skimage`, `tifffile`, and `numpy` to perform instance segmentation and quantitative analysis of cytoplasm, nuclei, mitochondria, and 2SC staining. The script generates processed images, segmentation overlays, and a CSV file with intensity and colocalisation data.

## Features

- **Image Preprocessing**: Loads `.tiff` microscopy images with 4 channels, scales pixel intensities, and normalises each channel.
- **Instance Segmentation**:
  - Uses **Cellpose** for segmenting cytoplasm and nuclei.
  - Segments mitochondria via percentile-based intensity thresholding.
- **Quantitative Analysis**:
  - Extracts 2SC intensity from cytoplasm and nucleus.
  - Calculates colocalisation between mitochondria and 2SC staining within the cytoplasm.
  - Computes cell-wise areas for cytoplasm, nucleus, and mitochondria.
- **Overlay Generation**: Creates and saves overlay images of segmented regions.
- **Batch Processing**: Handles multiple `.tiff` images, processing each in sequence and saving the results in structured folders.

## Dependencies

To run this script, install the following Python libraries:
```bash
pip install numpy tifffile scikit-image matplotlib opencv-python cellpose
```

## Usage
### Step 1: Prepare Image Data
Place your .tiff microscopy images in a directory (e.g., /path/to/images/). Ensure the images contain four channels (e.g., cytoplasm, nucleus, mitochondria, and 2SC staining).

### Step 2: Modify the Script
Before running the script, modify the following parameters to correspond to the correct image channels:

cyto_channel: Cytoplasm channel (e.g., 1)
nuclei_channel: Nucleus channel (e.g., 0)
mito_channel: Mitochondria channel (e.g., 2)
sc_channel: 2SC stain channel (e.g., 3)

### Step 3: Run the Script
```bash
python process_images.py
```
Example usage within the script:
```bash
if __name__ == '__main__':
    
    img_path = '/path/to/images/'  # Directory with input images
    output_path = '/path/to/output/'  # Directory to save output
    
    cyto_channel = 1  # Channel for cytoplasm 
    nuclei_channel = 0  # Channel for nucleus 
    mito_channel = 2  # Channel for mitochondria
    sc_channel = 3  # Channel for 2SC staining
    
    # Process the images and output results
    process_images(img_path, output_path, cyto_channel, nuclei_channel, mito_channel, sc_channel)
```

### Step 4: Output
The script generates the following outputs:

Processed .tiff images with segmentation masks for cytoplasm, nucleus, and mitochondria.  
Overlay images combining segmented regions for visualisation.  
A CSV file (Intensity_Colocalisation_data.csv) containing quantitative data for intensity and colocalisation analysis.  

### Example Output
Sample output files include:

Processed Images: *_Processed_Image.tiff  
Segmentation Masks: *_Cytoplasm_Mask.tiff, *_Nucleus_Mask.tiff, *_Mitochondria_Mask.tiff  
Overlay Images: *_Segmentation_Overlay.tiff, *_MitoCyto2SC_Overlay.tiff  
CSV File: Intensity_Colocalisation_data.csv  

### Example CSV Output
The CSV file contains the following columns:

Cytoplasm: Mean 2SC intensity in the cytoplasm.  
Nucleus: Mean 2SC intensity in the nucleus.  
Mito-2SC Colocalisation: Fraction of mitochondria colocalising with 2SC.  
2SC-Mito Colocalisation: Fraction of 2SC colocalising with mitochondria.  

## Key Functions
*load_image(image_path)*: Loads and normalises 4-channel .tiff images.  
*segment_instance(image, model_type, diameter, channels)*: Segments cytoplasm or nucleus using Cellpose.  
*segment_mitochondria(image, mito_channel_idx, percentile)*: Segments mitochondria based on intensity thresholding.  
*extract_cytoplasmic_2sc_intensity(image, cytoplasm_mask, nucleus_mask, mito_mask, sc_channel_idx)*: Extracts cytoplasmic 2SC intensity, excluding the nucleus and mitochondria.  
*calculate_colocalisation_in_cytoplasm(mito_mask, sc_mask, cyto_mask)*: Calculates colocalisation between mitochondria and 2SC staining.  
*overlay_segmentation(image, cytoplasm_mask, nucleus_mask)*: Generates overlay images of segmented regions.  
