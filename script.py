# Import libraries

import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch
from scipy.ndimage import maximum_filter


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



def find_image_with_superpoint(scanner, query_img_path, folder_path, match_threshold=0.99, visualize=True):
    # --- Load query ---
    q_pts, q_desc = scanner.scan_image(query_img_path)

    if q_desc is None or q_desc.shape[1] == 0:
        print("No features found in query image")
        return None

    best_score, best_path, best_match_data = -1, None, None
    qd = q_desc.T / (np.linalg.norm(q_desc.T, axis=1, keepdims=True) + 1e-8)

    # --- Loop through folder ---
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        cand_path = os.path.join(folder_path, fname)
        db_pts, db_desc = scanner.scan_image(cand_path)

        if db_desc is None or db_desc.shape[1] == 0:
            continue

        dd = db_desc.T / (np.linalg.norm(db_desc.T, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity
        similarity = np.dot(qd, dd.T)
        matches_q_to_db = np.argmax(similarity, axis=1)
        matches_db_to_q = np.argmax(similarity, axis=0)

        reciprocal_matches = [(i, j, similarity[i, j]) 
                              for i, j in enumerate(matches_q_to_db) 
                              if matches_db_to_q[j] == i]

        if reciprocal_matches:
            avg_score = np.mean([s for _, _, s in reciprocal_matches])
            if avg_score > best_score:
                best_score = avg_score
                best_path = cand_path
                best_match_data = (q_pts, db_pts, reciprocal_matches)

    # --- Results ---
    if best_score > match_threshold:
        print(f"Found match: {best_path} (score: {best_score:.3f})")
        if visualize and best_match_data:
            q_pts, c_pts, matches = best_match_data
            q_img = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)
            c_img = cv2.imread(best_path, cv2.IMREAD_GRAYSCALE)

            kp1 = [cv2.KeyPoint(float(q_pts[0, i]), float(q_pts[1, i]), 1) for i in range(q_pts.shape[1])]
            kp2 = [cv2.KeyPoint(float(c_pts[0, i]), float(c_pts[1, i]), 1) for i in range(c_pts.shape[1])]
            dmatches = [cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=1 - s) for i, j, s in matches]

            match_vis = cv2.drawMatches(q_img, kp1, c_img, kp2, dmatches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("SuperPoint Matches", match_vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return best_path
    else:
        print(f"No good match found (best score: {best_score:.3f})")
        return None


# --- Example Usage ---
if __name__ == "__main__":
    # INITIALIZE THE SCANNER
    scanner = ImageScanner(
        weights_path='superpoint_v1.pth',  
        conf_thresh=0.003,                 
        nms_dist=3,                        
        cuda=False                         
    )
    
    # SET YOUR PATHS
    query_path = r"C:\your_path\query_030.png"
    folder_path = r"C:\your_path\dataset\database"
    
    # RUN THE SEARCH
    best_match = find_image_with_superpoint(
        scanner=scanner,  
        query_img_path=query_path,
        folder_path=folder_path,
        visualize=True
    )
    
    print("Best match:", best_match)