*Note: put the MedSAM model checkpoint in this folder.*

## Model Weights (MedSAM)

This project integrates the [MedSAM](https://github.com/bowang-lab/MedSAM) segmentation model as an optional backend for medical image analysis.

Due to GitHub’s file size limit, the model weight file `medsam_vit_b.pth` (357 MB) is **not included** in the repository and is listed in `.gitignore`.

### How to Set It Up

1. Download the MedSAM model weights manually from this link:  
   [Download from Google Drive](https://drive.google.com/drive/folders/1mZpTaCjz7Vq77XAVeQzgISPXVfnLAOAK?usp=sharing)

2. Place the downloaded file in the following directory:

```
models/
```

3. Now you're ready to run the MedSAM-based segmentation!

> ℹ️ This file is not tracked in Git and should not be committed to the repository.
