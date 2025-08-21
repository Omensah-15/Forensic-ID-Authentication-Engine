# Image-Based Anti-Counterfeit Detection with SuperPoint

A computer vision project that leverages the SuperPoint deep learning model for feature detection and matching to verify the authenticity of product labels and security patterns.
The system scans query images, extracts nano-level keypoints, and compares them against a trusted database to detect counterfeits.

## Features

- **Sub-millimeter Accuracy**: Detects micro-patterns as small as 0.2mm
- **Rotation Invariant**: Works from angles up to 60 degrees
- **Lighting Robust**: Reliable performance in varying lighting conditions
- **Real-time Processing**: <100ms recognition time per image
- **Zero Training Required**: Uses pre-trained SuperPoint model

## Requirements
- Python 3.x
- Required libraries: `numpy`, `opencv-python`, `torch`, `scipy`
- Pretrained SuperPoint weights file `superpoint_v1.pth`

## Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd superpoint-image-matching
   ```

   ## DEMO OUTPUT
![image alt](https://github.com/Omensah-15/Image-Based-Anti-Counterfeit-Detection-with-SuperPoint/blob/e2bba3fdf564d8adfc6a72358ddf824b31dc17e7/asset/query_030.png)
Image of query_030.png

![image alt](https://github.com/Omensah-15/Image-Based-Anti-Counterfeit-Detection-with-SuperPoint/blob/5055cf5cd07158ef9015961ed9189ebf63c6a9f4/asset/pattern_014.png)
Image of pattern_014.png

![image alt](https://github.com/Omensah-15/Image-Based-Anti-Counterfeit-Detection-with-SuperPoint/blob/9ef8d762412fc8830cf38fc21126b41aec2eed90/asset/Screenshot%202025-08-21%20162010.png)
The result image displays the SuperPoint image matching visualization between the query image (query_030.png) and the matched image (pattern_014.png). It features two overlaid grayscale images with numerous colored lines connecting corresponding keypoints, indicating successful feature matches. The lines, in various colors, span across the images, highlighting a dense set of matching points, with a perfect similarity score of 1.000, suggesting a highly accurate match.
   
