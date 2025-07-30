
# Barnacle Counting - Computer Vision Approach

A comprehensive solution to automate barnacle counting in tide pool images for National Park Service scientists using computer vision techniques.

## Project Overview

Scientists with the National Park Service manually count barnacles in tide pool images - a time-consuming process that can involve counting over 1000 barnacles per image. This project develops an automated system to speed up their research pipeline.

### Problem Statement
- **Manual Process**: Scientists place a fixed-size frame over barnacle colonies and manually count individuals
- **Scale**: Images often contain 1000+ barnacles requiring manual counting
- **Goal**: Develop an automated system to accelerate the counting process
- **Dataset**: 2 training images with ground truth masks, 2 unseen test images

## Approach Evolution

### 1. Initial Deep Learning Approach
- **Implementation**: Patch-based CNN classification using PyTorch
- **Architecture**: Custom CNN with convolutional layers, pooling, and fully connected layers
- **Results**: 100% training accuracy
- **Problem Identified**: Perfect accuracy indicated overfitting/oversimplified task
- **Decision**: Pivoted to more suitable approach

### 2. Computer Vision Solution
Implemented a multi-method detection system combining:
- **Hough Circle Detection**: Identifies circular barnacle openings
- **Contour Analysis**: Detects barnacle shapes through adaptive thresholding
- **Parameter Optimization**: Iterative tuning to reduce false positives

## Key Results

| Image | Predicted Count | Ground Truth | Error |
|-------|----------------|--------------|-------|
| img1 | 1,200 | 1,718 | 30.2% |
| img2 | 1,224 | 397 | 208.3% |
| unseen_img1 | 877 | N/A | - |
| unseen_img2 | 28 | N/A | - |

## Technical Implementation

### Dependencies
```python
cv2              # Computer vision operations
numpy            # Numerical computations
matplotlib       # Visualization
torch            # Deep learning framework (initial approach)
PIL              # Image processing
scikit-learn     # Evaluation metrics
```

### Core Components

#### 1. Green Frame Detection
```python
def detect_green_frame(image):
    """Detect wire frame boundary using HSV color filtering"""
```

#### 2. Multi-Method Barnacle Detection
```python
class BarnacleDetector:
    def detect_circles(self, image):      # Hough circle detection
    def detect_contours(self, image):     # Adaptive thresholding + contour analysis
    def detect_barnacles(self, image):    # Combined detection with weighted results
```

#### 3. Evaluation System
- Ground truth comparison using provided mask files
- Error analysis and performance metrics
- Visual validation of detection results

## Project Structure

```
barnacle-counting/
├── README.md
├── barnacle_counting_notebook.ipynb    # Main implementation
├── data/
│   ├── img1.png                       # Training image 1
│   ├── img2.png                       # Training image 2
│   ├── mask1.png                      # Ground truth mask 1
│   ├── mask2.png                      # Ground truth mask 2
│   ├── unseen_img1.png               # Test image 1
│   └── unseen_img2.png               # Test image 2
└── results/
    └── detection_visualizations/      # Output visualizations
```

## How to Run

### Setup
1. Clone the repository:
```bash
git clone https://github.com/Blessing27-oss/Barnacle-Counting.git
cd Barnacle-Counting
```

2. Install dependencies:
```bash
pip install opencv-python numpy matplotlib torch torchvision pillow scikit-learn
```

3. Download and extract the dataset to the `data/` folder

### Execution
1. Open `Barnacle_Counting.ipynb` in Jupyter Notebook or JupyterLab
2. Update the `DATA_PATH` variable to point to your data folder
3. Run all cells sequentially

The notebook is structured to show the complete development process:
- Data loading and exploration
- Deep learning approach and analysis
- Computer vision implementation
- Results comparison and evaluation

## Key Learnings

### 1. Problem Recognition
- **100% accuracy is often a red flag**, not a success
- Learned to question "too good to be true" results
- Importance of validating model behavior beyond accuracy metrics

### 2. Approach Selection
- **Domain knowledge matters**: Barnacles have circular openings → circle detection makes sense
- **Interpretability**: Scientists need to understand and trust the system
- **Limited data**: Computer vision can work with fewer labeled examples than deep learning

### 3. Iterative Improvement
- **Parameter tuning is crucial**: Reduced error from 500-2000% to 30-200%
- **Multi-method validation**: Combining approaches increases robustness
- **Visual debugging**: Essential for understanding algorithm behavior

### 4. Real-World Considerations
- **Deployment practicality**: Interpretable methods are easier to debug and maintain
- **Scientific applications**: Domain experts need explainable results
- **Resource constraints**: Works with minimal training data

## Technical Highlights

### Computer Vision Techniques
- **HSV color space filtering** for green frame detection
- **Hough Circle Transform** for circular pattern detection
- **Adaptive thresholding** for robust edge detection
- **Morphological operations** for noise reduction
- **Contour analysis** with shape filtering

### Detection Optimization
- **Multi-parameter testing** to find optimal settings
- **Intensity-based validation** to reduce false positives
- **Size and shape constraints** based on barnacle morphology
- **Conservative combination** using minimum of detection methods

## Future Improvements

1. **Image-specific parameter tuning** based on lighting conditions
2. **Multi-scale detection** for varying barnacle sizes
3. **Machine learning enhancement** using detected features
4. **User interface** for scientist feedback and validation
5. **Batch processing** capabilities for large datasets

## Why This Approach Works

**Interpretable**: Scientists can see exactly what's being detected  
**Debuggable**: Visual output allows for easy validation  
**Adaptable**: Parameters can be tuned for different conditions  
**Efficient**: Works with limited training data  
**Practical**: Suitable for real scientific deployment  

## Challenges Overcome

1. **Overfitting Recognition**: Identified perfect accuracy as problematic
2. **Parameter Sensitivity**: Systematic tuning to reduce false positives  
3. **Method Integration**: Combining multiple detection approaches effectively
4. **Performance Optimization**: Balancing precision vs. recall for scientific use

.
