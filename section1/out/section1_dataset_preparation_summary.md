#  Section 1 Summary: Hippocampus Dataset Curation and Preparation

This document summarizes the steps taken in Section 1 of the project to inspect, clean, and prepare the Medical Decathlon hippocampus dataset for model training.

---

##  Tasks Completed

### 1. **Loading and Inspection**
- Loaded image and label volumes from NIfTI files using NiBabel
- Verified volume dimensions and array shapes
- Visualized multiple 2D slices across sagittal, coronal, and axial planes
- Overlayed segmentation masks on image slices to confirm anatomical alignment

### 2. **Metadata Exploration**
- Confirmed NIfTI file format via `.header_class`
- Extracted voxel size from header to determine physical units (1mm³)
- Verified grid regularity and decoded spatial/temporal units

### 3. **3D Visualization**
- Rendered labeled voxels in 3D using `matplotlib`'s voxel plotting
- Helped validate hippocampal shape and segmentation coverage

### 4. **Volume Analysis**
- Computed hippocampal volume (mm³) for each label using voxel count × voxel size
- Plotted a histogram of hippocampal volumes across the dataset
- Identified and investigated outlier volumes (e.g., ~20,000 mm³)

### 5. **Data Cleaning**
- Removed known invalid or mislabeled volumes based on visual inspection and histogram outliers
- Saved clean image and label files to the `section1/out/` directory
- Created a ZIP archive of the cleaned dataset and uploaded it to Google Drive for safe storage

---

##  Output Artifacts

- Cleaned images and labels are saved in: section1/out/images/ section1/out/

