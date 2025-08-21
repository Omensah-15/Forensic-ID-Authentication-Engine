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
   
   A score of 1.000 indicates a near-perfect match (query_030.png and pattern_014.png).


   
