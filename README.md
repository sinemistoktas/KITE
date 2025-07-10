# KITE: Web-Based Semi-Automated Retinal Fluid Segmentation Tool

<div align="center">

<img src="media/kite-logo.png" alt="Light Theme" width="200" />

**An interactive web application for segmenting retinal fluid regions in OCT images**

<!--add in the future-->
<!--[Demo Video](#demo) • [Documentation](#documentation) • [Installation](#installation) • [Usage](#usage)-->

</div>

## 🩺 Overview

KITE is a  web-based tool designed to assist medical professionals in segmenting retinal fluid regions in Optical Coherence Tomography (OCT) images. The tool combines traditional image processing techniques with state-of-the-art deep learning models to provide accurate, efficient, and user-friendly medical image annotation.

### Key Features

- 🖼️ **Interactive Annotation Tools**: Point, line, box, and fill tools for precise manual annotation
- 🤖 **Multiple Segmentation Algorithms**:
    - **KITE Algorithm**: Custom region-growing approach using OpenCV
    - **U-Net**: Deep learning model trained on DUKE OCT dataset
    - **MedSAM**: Foundation model for medical image segmentation
- 🌐 **Web-Based Interface**: Django based
- 📊 **Real-Time Processing**: Fast segmentation with immediate visual feedback
- 💾 **Export Capabilities**: Download results in PNG and NPY formats
- 🎨 **Layer Management**: Organize and edit multiple segmentation layers

## 🎯 Applications

- **Clinical Research**: Generate annotated datasets for training ML models
- **Medical Education**: Teaching tool for understanding retinal pathology
- **Ophthalmology Practice**: Assist in diagnosis and treatment planning

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │     Backend      │    │   AI Models     │
│                 │    │                  │    │                 │
│ • HTML/CSS/JS   │◄──►│ • Django REST    │◄──►│ • U-Net         │
│ • Konva.js      │    │ • Image Pipeline │    │ • MedSAM        │
│ • Annotation    │    │ • File Handling  │    │ • Traditional   │
│   Tools         │    │                  │    │   CV Methods    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Node.js (for frontend dependencies)
- CUDA-compatible GPU (optional, for faster deep learning inference)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/KITE.git
cd KITE
```

### 2. Set Up Python Environment

Using **conda** (recommended):
```bash
conda create --name kite-env python=3.10
conda activate kite-env
```

Using **venv**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Weights

To use the MedSAM segmentation model, you need to manually download the model weights.

**Instructions are provided in:**
[`models/README.md`](models/README.md)

### 5. Database Setup

```bash
cd website
python manage.py migrate
python manage.py collectstatic
```

### 6. Run the Application

```bash
python manage.py runserver
```

Visit `http://localhost:8000` to access the application.

## 📖 Usage

### Quick Start

1. **Select Algorithm**: Choose between KITE, U-Net, or MedSAM
2. **Upload Image**: Upload your OCT image (JPEG format) or choose provided the demo image
3. **Annotate** (if using KITE / MedSAM): Use annotation tools to mark regions of interest
4. **Segment**: Click "Ready to Segment!" to process the image
5. **Edit**: Refine results using manual editing tools
6. **Export**: Download segmentation masks in your preferred format

**Note:** It is possible to upload previous annotations or mask results and edit them in KITE mode.

### Detailed Workflows

#### KITE Algorithm Workflow
```
Upload Image → Annotate ROI → Traditional Segmentation → Manual Refinement → Export
```

#### U-Net Workflow
```
Upload Image → Automatic Segmentation → Manual Refinement → Export
```

#### MedSAM Workflow
```
Upload Image → Draw Bounding Boxes → AI Segmentation → Manual Refinement → Export
```

## 🛠️ Technology Stack

### Frontend
- **HTML5/CSS3**: Modern web standards
- **JavaScript**: Interactive functionality
- **Konva.js**: Canvas-based annotation tools

### Backend
- **Django**: Web framework and REST API
- **Python**: Core programming language
- **OpenCV**: Traditional image processing
- **NumPy/SciPy**: Scientific computing

### Deep Learning
- **PyTorch**: Deep learning framework
- **U-Net**: Custom trained model on DUKE dataset
- **MedSAM**: Foundation model integration


## 📄 License

This project is licensed under the MIT License.

## 👥 Team

**Koç University COMP 491 Capstone Project Team**

- **Duru Tandoğan**
- **Mislina Akça**
- **Sinemis Toktaş**
- **Yamaç Ömür**

**Project Advisor:** Çiğdem Gündüz Demir – Koç University

## 🙏 Acknowledgments

- DUKE University OCT Dataset
- U-Net and MedSAM research communities
- OpenCV and PyTorch communities

## 📈 Future Work

- [ ] Integration with additional clinical data (text, metadata)
- [ ] Support for other medical imaging modalities (MRI, CT)
- [ ] Real-time collaborative annotation
- [ ] Advanced AI model fine-tuning capabilities
- [ ] Clinical validation studies

## 📚 References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation.
2. Ma, J., He, Y., Li, F., et al. (2024). Segment anything in medical images. Nature Communications.

---

<div align="center">


[⬆ Back to Top](#kite-web-based-semi-automated-retinal-fluid-segmentation-tool)

</div>