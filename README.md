# KITE
Segmentation and Characterization of Retinal Fluids/Deposits in OCT Images

This repository contains the implementation of a web-based semi-automated tool for segmenting and characterizing retinal fluids/deposits in optical coherence tomography (OCT) images. The project is developed as part of COMP 491 - Computer Engineering Design Project at Koç University.

## Setup Instructions

To set up and run the KITE project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/KITE.git
cd KITE
````

### 2. Create and Activate a Virtual Environment

Using `venv`:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Or using `conda`:

```bash
conda create --name kite-env python=3.10
conda activate kite-env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Model Weights (MedSAM)

To use the MedSAM segmentation model, you need to manually download the model weights.

**Instructions are provided in:**
[`data/medsam/model/README.md`](data/medsam/model/README.md)

---

## Contributors

* Duru Tandoğan
* Mislina Akça
* Sinemis Toktaş
* Yamaç Ömür

**Project Advisor:** Çiğdem Gündüz Demir – Koç University

## License

This project is licensed under the MIT License.
