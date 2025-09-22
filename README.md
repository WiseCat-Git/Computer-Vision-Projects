# Computer Vision Projects - Master's in Applied Artificial Intelligence

This repository contains computer vision projects completed as part of my Master's program in Applied Artificial Intelligence. The projects demonstrate a comprehensive progression from fundamental image processing techniques to advanced object detection using state-of-the-art deep learning models.

## Projects Overview

### Unit 1: Fundamental Image Processing with OpenCV
**Files:** `solucion_caso_practico_iep_iaa_cv_u1.py`

#### Basic Image Operations and Preprocessing
- **Objective:** Master fundamental computer vision operations and preprocessing techniques
- **Framework:** OpenCV, Python
- **Key Operations Implemented:**
  - Image loading, validation, and format conversion
  - Color space transformation (BGR to RGB, RGB to Grayscale)
  - Geometric transformations (resize, rotation, cropping)
  - Noise reduction using Gaussian blur filtering
  - Edge detection with Canny algorithm
  - Image persistence and file management

**Technical Pipeline:**
1. **Image Acquisition:** Download and load images with error handling
2. **Preprocessing:** Convert to grayscale for simplified analysis
3. **Filtering:** Apply Gaussian blur to reduce noise
4. **Feature Extraction:** Use Canny edge detection for contour identification
5. **Output Management:** Save processed images at each stage

**Key Results:**
- Successfully implemented complete image processing pipeline
- Demonstrated proper handling of different image formats
- Applied fundamental CV operations essential for advanced applications

---

### Unit 2: Object Detection with YOLOv8
**Files:** `solucion_caso_práctico_iep_iaa_cv_u2.py`

#### Real-time Object Detection System
- **Dataset:** COCO128 (80 object classes)
- **Architecture:** YOLOv8 (You Only Look Once v8)
- **Objective:** Implement and train a state-of-the-art object detection model
- **Key Components:**
  - Model training from scratch using Ultralytics framework
  - Advanced data augmentation with Albumentations
  - Performance evaluation and metrics analysis
  - Real-time inference pipeline

**Data Augmentation Pipeline:**
- Random brightness and contrast adjustment
- Horizontal flipping and rotation
- Motion blur simulation
- Random resized cropping
- Normalization for neural network optimization

**Model Performance:**

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.6116 |
| mAP@0.5:0.95 | 0.4541 |
| Inference Time | 170.6 ms |

**Technical Achievements:**
- Successful training of YOLOv8 on limited hardware resources
- Implementation of robust data augmentation strategy
- Integration of modern CV frameworks (Ultralytics, Albumentations)
- Real-time object detection capabilities

**Applications:**
- Security and surveillance systems
- Industrial quality control
- Traffic monitoring and analysis
- Automated inventory management

---

### Unit 3: Advanced Computer Vision Techniques
**Files:** `solución_caso_práctico_iep_iaa_cv_u3.py`

#### Comprehensive CV Framework Integration
- **Frameworks:** OpenCV, Scikit-Image
- **Objective:** Master advanced image processing and analysis techniques
- **Advanced Techniques Implemented:**
  - Face detection using Haar Cascade classifiers
  - Image segmentation with multiple approaches
  - Advanced filtering and enhancement methods
  - Clustering-based image analysis

**Core Techniques:**

**1. Object Detection:**
- Haar Cascade implementation for face detection
- Real-time detection capabilities
- Bounding box visualization

**2. Image Segmentation:**
- **Global Thresholding:** Otsu's method for automatic threshold selection
- **Adaptive Thresholding:** Local threshold calculation for varying illumination
- **K-means Clustering:** Pixel-based segmentation into k regions

**3. Advanced Filtering:**
- Gaussian filtering for noise reduction
- Canny edge detection for contour analysis
- Morphological operations

**Medical and Security Applications:**
- Automated medical image analysis
- Anomaly detection in radiographs
- Tissue and structure identification
- Surveillance and monitoring systems

**Key Insights:**
- Demonstrated effectiveness of classical CV methods
- Integration of multiple frameworks for comprehensive solutions
- Practical applications in healthcare and security domains

---

## Technical Skills Demonstrated

### Programming & Libraries
- **Python:** OpenCV, Scikit-Image, NumPy, Matplotlib
- **Deep Learning:** Ultralytics YOLOv8, PyTorch
- **Data Augmentation:** Albumentations
- **Visualization:** Matplotlib, real-time display systems

### Computer Vision Techniques
- **Image Preprocessing:** Filtering, enhancement, format conversion
- **Feature Extraction:** Edge detection, contour analysis
- **Object Detection:** YOLO architecture, Haar Cascades
- **Image Segmentation:** Thresholding, clustering methods
- **Geometric Transformations:** Rotation, scaling, cropping

### Advanced Methodologies
- **Deep Learning:** Neural network training and optimization
- **Data Augmentation:** Synthetic data generation for model robustness
- **Performance Evaluation:** mAP metrics, inference time analysis
- **Real-time Processing:** Optimized pipelines for live applications

---

## Project Structure

Each project follows a structured computer vision pipeline:

1. **Image Acquisition:** Loading and validation of input data
2. **Preprocessing:** Format conversion, noise reduction, normalization
3. **Feature Extraction:** Edge detection, pattern recognition
4. **Analysis/Detection:** Object detection, segmentation, classification
5. **Visualization:** Results display and interpretation
6. **Evaluation:** Performance metrics and validation

---

## Repository Contents

```
├── solucion_caso_practico_iep_iaa_cv_u1.py    # Unit 1: Fundamental Image Processing
├── solucion_caso_práctico_iep_iaa_cv_u2.py    # Unit 2: YOLOv8 Object Detection
├── solución_caso_práctico_iep_iaa_cv_u3.py    # Unit 3: Advanced CV Techniques
└── README.md                                   # This file
```

---

## Getting Started

To run these projects:

1. Clone the repository
2. Install required dependencies:
   ```bash
   # Core computer vision libraries
   pip install opencv-python-headless scikit-image
   
   # Deep learning and object detection
   pip install torch torchvision ultralytics
   
   # Data augmentation and visualization
   pip install albumentations matplotlib
   
   # Additional utilities
   pip install numpy scipy
   ```
3. Run individual Python files in your preferred environment (Jupyter, Google Colab, etc.)

**Note:** For YOLOv8 training, GPU acceleration is recommended but not required.

---

## Applications and Use Cases

### Healthcare
- Medical image analysis and diagnosis
- Anomaly detection in radiological images
- Automated tissue classification

### Security & Surveillance
- Real-time face detection and recognition
- Object tracking and monitoring
- Intrusion detection systems

### Industrial
- Quality control and defect detection
- Automated visual inspection
- Production line monitoring

### Research & Development
- Computer vision algorithm benchmarking
- Custom model development and training
- Academic research applications

---

## Academic Context

These projects were completed as part of the Master's program in Applied Artificial Intelligence, specifically within the Computer Vision track. The progression demonstrates:

**Foundation Building:** Starting with fundamental image operations and OpenCV mastery
**Modern Techniques:** Advancing to state-of-the-art deep learning models like YOLOv8  
**Framework Integration:** Combining multiple specialized libraries for comprehensive solutions
**Real-world Applications:** Focusing on practical implementations with measurable performance metrics

The combination of classical computer vision techniques with modern deep learning approaches provides a robust foundation for tackling diverse visual recognition challenges in professional and research contexts.

---

## Performance Highlights

- **YOLOv8 Model:** Achieved 61.16% mAP@0.5 on COCO128 dataset
- **Real-time Capability:** 170ms average inference time per image
- **Multi-framework Integration:** Successful combination of OpenCV, Scikit-Image, and Ultralytics
- **Comprehensive Pipeline:** End-to-end solutions from raw images to actionable insights

---

## License

This project is part of academic coursework for educational purposes.

---

## Contact

For questions about these projects or collaboration opportunities, please feel free to reach out through GitHub.
