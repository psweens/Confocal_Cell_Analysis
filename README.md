# 🧬 2D Confocal Microscopy Image Segmentation & Quantitative Analysis

This repository provides a fully automated pipeline for analyzing **multi-channel confocal microscopy images** of cells. It uses:

- [`Cellpose`](https://github.com/MouseLand/cellpose) for **instance segmentation** of cytoplasm and nuclei  
- **Percentile-based thresholding** for **mitochondria and 2SC** detection  
- Quantitative extraction of **intensity metrics** and **colocalization statistics**  

The pipeline operates on `.tiff` images with 4 channels and outputs per-image segmentation masks, overlays, and summary CSV reports.

---

## 🧠 Workflow Summary

1. 🖼️ **Preprocessing**
   - Recursively loads `.tiff` images and normalizes channels using global percentile thresholds.

2. 🧫 **Segmentation**
   - Cytoplasm & nuclei via Cellpose  
   - Mitochondria & 2SC via global thresholding (e.g. 90th / 95th percentile in normalized space)

3. 📊 **Quantification**
   - Calculates 2SC intensity in cytoplasm (excluding nucleus & mitochondria) and in nucleus (excluding mitochondria)  
   - Computes mitochondrial colocalization with 2SC  
   - Estimates per-cell areas for cytoplasm, nuclei, and mitochondria

4. 🎨 **Overlay Generation**
   - Produces visual overlays of segmented structures for interpretation

5. 📁 **Batch Processing**
   - Automatically processes all `.tiff` images in subdirectories  
   - Stores outputs in organized folders  
   - Saves CSV summary reports

---

## 🧬 Input Format

Each input image must be a `.tiff` file with **4 channels**, representing:

| Channel Index | Target Structure  |
|---------------|-------------------|
| `0`           | Nucleus           |
| `1`           | Cytoplasm         |
| `2`           | Mitochondria      |
| `3`           | 2SC Staining      |

Images can be organized into subfolders (e.g., by experimental condition or cell type). The script handles recursive traversal and processing.

---

## 🧱 Requirements

Install dependencies with:

```bash
pip install numpy tifffile scikit-image matplotlib opencv-python cellpose
```

Cellpose will optionally use a GPU if available.

---

## ▶️ How to Run

Modify the paths and channel indices in the `__main__` section of [`multi_channel_cell_segmentation.py`](multi_channel_cell_segmentation.py):

```python
raw_data_path = '/path/to/raw_images/'  # input directory (recursively scanned)
top_level_output_path = '/path/to/output/'

nuclei_channel = 0
cyto_channel = 1
mito_channel = 2
sc_channel = 3
```

Then, run the script:

```bash
python multi_channel_cell_segmentation.py
```

---

## 📂 Output Structure

For each image:

| Output File                          | Description                                    |
|-------------------------------------|------------------------------------------------|
| `*_Processed_Image.tiff`            | Normalized and downsampled image               |
| `*_Cytoplasm_Mask.tiff`             | Segmented cytoplasm overlay                    |
| `*_Nucleus_Mask.tiff`               | Segmented nucleus overlay                      |
| `*_Mitochondria_Mask.tiff`         | Mitochondria mask from thresholding            |
| `*_Segmentation_Overlay.tiff`       | Overlay of cytoplasm and nucleus               |
| `*_MitoCyto2SC_Overlay.tiff`        | Combined overlay of all structures             |
| `*_cellwise_areas.csv`              | Per-cell area stats (cytoplasm, nucleus, mito) |
| `Intensity_Colocalisation_data.csv` | Summary stats per image (2SC, colocalisation)  |

---

## 🧪 Use Cases

This pipeline is ideal for:

- Quantifying 2SC localization patterns across cell types
- Evaluating mitochondrial-2SC colocalization metrics
- Extracting cell-wise morphometrics for statistical analysis
- Generating overlay visualizations for publication or QC

---

## 📬 Contact

For questions, bugs, or contributions, feel free to open an issue or reach out!
