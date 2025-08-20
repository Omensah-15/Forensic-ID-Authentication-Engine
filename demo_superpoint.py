# Import libraries

import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch
from scipy.ndimage import maximum_filter

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self):
    super(SuperPointNet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Shared Encoder.
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head.
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Shared Encoder.
    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)
    # Descriptor Head.
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    return semi, desc



class ImageScanner:
    """Optimized SuperPoint image scanner with vectorized NMS, keypoint limit, and half-precision support."""

    def __init__(self, weights_path, conf_thresh=0.015, nms_dist=4, border_remove=4, cuda=False):
        self.conf_thresh = conf_thresh
        self.nms_dist = nms_dist
        self.cell = 8
        self.border_remove = border_remove
        self.cuda = cuda

        # Load network
        self.net = SuperPointNet()
        device = torch.device('cuda' if cuda else 'cpu')
        if cuda:
            self.net.load_state_dict(torch.load(weights_path))
        else:
            self.net.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(device)
        self.net.eval()

    def load_image(self, image_path, target_size=None):
        """Load image and optionally resize."""
        grayim = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if grayim is None:
            raise Exception(f"Error reading image {image_path}")
        if target_size is not None:
            grayim = cv2.resize(grayim, target_size, interpolation=cv2.INTER_AREA)
        return grayim.astype('float32') / 255.0

    def scan_image(self, image_path, target_size=None, max_keypoints=1000):
        """Scan single image and return keypoints and descriptors efficiently."""
        # Load and preprocess
        img = self.load_image(image_path, target_size)
        H, W = img.shape

        inp = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        device = next(self.net.parameters()).device
        inp = inp.to(device)

        # Half precision for GPU
        if self.cuda:
            inp = inp.half()
            self.net.half()

        # Forward pass
        with torch.no_grad():
            semi, coarse_desc = self.net(inp)

        semi = semi.cpu().numpy().squeeze()
        coarse_desc = coarse_desc.cpu().numpy().squeeze()

        # Heatmap and points
        dense = np.exp(semi - np.max(semi))
        dense /= (np.sum(dense, axis=0) + 1e-8)
        nodust = dense[:-1, :, :]

        Hc, Wc = semi.shape[1], semi.shape[2]
        heatmap = nodust.transpose(1, 2, 0).reshape(Hc, Wc, self.cell, self.cell)
        heatmap = heatmap.transpose(0, 2, 1, 3).reshape(Hc*self.cell, Wc*self.cell)

        ys, xs = np.where(heatmap >= self.conf_thresh)
        if len(xs) == 0:
            return np.zeros((3, 0)), np.zeros((256, 0))

        pts = np.zeros((3, len(xs)))
        pts[0, :] = xs
        pts[1, :] = ys
        pts[2, :] = heatmap[ys, xs]

        # Vectorized NMS
        grid = np.zeros((H, W))
        grid[pts[1, :].astype(int), pts[0, :].astype(int)] = pts[2, :]
        max_filt = maximum_filter(grid, size=self.nms_dist*2+1)
        keep_mask = (grid == max_filt) & (grid >= self.conf_thresh)
        keep_indices = np.where(keep_mask)
        vals = grid[keep_indices]

        pts = np.zeros((3, len(vals)))
        pts[0, :], pts[1, :], pts[2, :] = keep_indices[1], keep_indices[0], vals

        # Limit max keypoints
        if pts.shape[1] > max_keypoints:
            inds = np.argsort(-pts[2, :])[:max_keypoints]
            pts = pts[:, inds]

        # Remove border points
        bord = self.border_remove
        valid_mask = ~((pts[0, :] < bord) | (pts[0, :] >= (W - bord)) |
                       (pts[1, :] < bord) | (pts[1, :] >= (H - bord)))
        pts = pts[:, valid_mask]

        # Descriptor sampling
        if pts.shape[1] == 0:
            desc = np.zeros((256, 0))
        else:
            samp_pts = pts[:2, :].copy().T
            samp_pts[:, 0] = (samp_pts[:, 0]/(W-1))*2 - 1
            samp_pts[:, 1] = (samp_pts[:, 1]/(H-1))*2 - 1

            samp_pts_tensor = torch.from_numpy(samp_pts).float().unsqueeze(0).unsqueeze(0).to(device)
            coarse_desc_tensor = torch.from_numpy(coarse_desc).float().unsqueeze(0).to(device)

            if self.cuda:
                coarse_desc_tensor = coarse_desc_tensor.half()
                samp_pts_tensor = samp_pts_tensor.half()

            desc = torch.nn.functional.grid_sample(
                coarse_desc_tensor, samp_pts_tensor,
                mode='bilinear', padding_mode='zeros', align_corners=True
            )
            desc = desc.squeeze().cpu().numpy()
            if desc.ndim == 1:
                desc = desc.reshape(-1, 1)
            desc /= (np.linalg.norm(desc, axis=0, keepdims=True) + 1e-8)

        return pts, desc



scanner = ImageScanner('superpoint_v1.pth', cuda=True)
keypoints, descriptors = scanner.scan_image('the_sticker_pattern.jpg', target_size=(640, 480))

print(f"Found {keypoints.shape[1]} keypoints")
print(f"Descriptor shape: {descriptors.shape}")
