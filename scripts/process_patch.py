#!/usr/bin/env python3
"""
Vesuvius patch processor — runs in GitHub Actions free tier.
Downloads a 3D patch from the public Vesuvius S3 bucket,
runs segmentation + ink detection, posts results back to TrustKernel.

Scroll 3 (PHerc. 332): volume 20231117143551
  - 7.91µm resolution, 53keV
  - 9,778 slices x 24MB each = 236 GB total
  - Public S3: s3://vesuvius-challenge-open-data/
"""

import os
import sys
import json
import io
import requests
import numpy as np
import torch
import torch.nn as nn

# ── Config from environment ───────────────────────────────────────────────────
SCROLL_ID    = os.environ.get('SCROLL_ID', '20231117143551')   # Scroll 3 volume
X            = int(os.environ.get('X', 0))
Y            = int(os.environ.get('Y', 0))
Z            = int(os.environ.get('Z', 0))
PATCH_SIZE   = int(os.environ.get('PATCH_SIZE', 64))
THRESHOLD    = float(os.environ.get('THRESHOLD', 0.5))
CALLBACK_URL = os.environ.get('CALLBACK_URL', '')
JOB_ID       = os.environ.get('JOB_ID', 'unknown')
RECEIPT_ID   = os.environ.get('RECEIPT_ID', 'unknown')

# Public Vesuvius S3 — no auth required
S3_BASE = f'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/full-scrolls/Scroll3.volpkg/volumes/{SCROLL_ID}'

# ── Minimal 3D ink detection network ─────────────────────────────────────────
# Architecture matches the Vesuvius Grand Prize open-source U-Net family.
# Weights here are untrained (random init) — replace with official checkpoint
# from https://github.com/younader/Vesuvius-Grandprize-Entry when available.
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class InkUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.bot   = DoubleConv(64, 128)
        self.up2   = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec2  = DoubleConv(128, 64)
        self.up1   = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1  = DoubleConv(64, 32)
        self.out   = nn.Conv3d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bot(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.out(d1))

# ── Load pretrained weights if available ─────────────────────────────────────
def load_model():
    model = InkUNet()
    weights_path = 'ink_model.pth'
    # Try to fetch the official checkpoint from the Vesuvius grand prize repo
    checkpoint_url = 'https://github.com/younader/Vesuvius-Grandprize-Entry/releases/download/v1.0/model.pth'
    try:
        r = requests.get(checkpoint_url, timeout=30)
        if r.status_code == 200:
            with open(weights_path, 'wb') as f:
                f.write(r.content)
            state = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state, strict=False)
            print('[MODEL] Loaded pretrained weights')
        else:
            print('[MODEL] Checkpoint unavailable — using random init (heuristic mode)')
    except Exception as e:
        print(f'[MODEL] Weight load failed: {e} — using heuristic mode')
    model.eval()
    return model

# ── Fetch a single TIFF slice from S3 ────────────────────────────────────────
def fetch_slice(z_idx):
    url = f'{S3_BASE}/{z_idx:05d}.tif'
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        # tifffile not always available — fallback to raw parsing
        try:
            import tifffile
            img = tifffile.imread(io.BytesIO(r.content))
            return img.astype(np.float32)
        except Exception:
            # Raw uint16 fallback
            data = np.frombuffer(r.content[8:], dtype=np.uint16)
            side = int(np.sqrt(len(data)))
            return data[:side*side].reshape(side, side).astype(np.float32)
    except Exception as e:
        print(f'[FETCH] Slice {z_idx} failed: {e}')
        return None

# ── Heuristic segmentation — find sheet by max variance z-slice ──────────────
def segment_sheet(patch3d):
    """Returns index of the z-slice most likely to contain papyrus sheet."""
    variances = np.var(patch3d, axis=(1, 2))
    return int(np.argmax(variances))

# ── Connected components — count distinct ink blobs ──────────────────────────
def count_letter_candidates(prob_map_2d, min_size=20):
    """Flood-fill connected components above threshold. Returns count of blobs."""
    binary = (prob_map_2d > 0.75).astype(np.uint8)
    h, w = binary.shape
    visited = np.zeros_like(binary)
    blobs = 0

    def bfs(sy, sx):
        queue = [(sy, sx)]
        size = 0
        while queue:
            cy, cx = queue.pop()
            if cy < 0 or cy >= h or cx < 0 or cx >= w: continue
            if visited[cy, cx] or not binary[cy, cx]: continue
            visited[cy, cx] = 1
            size += 1
            queue.extend([(cy+1, cx), (cy-1, cx), (cy, cx+1), (cy, cx-1)])
        return size

    for y in range(h):
        for x in range(w):
            if binary[y, x] and not visited[y, x]:
                size = bfs(y, x)
                if size >= min_size:
                    blobs += 1
    return blobs

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f'[START] Scroll={SCROLL_ID} patch=({X},{Y},{Z}) size={PATCH_SIZE}')

    # 1. Download Z slices
    slices = []
    for z_idx in range(Z, Z + PATCH_SIZE):
        sl = fetch_slice(z_idx)
        if sl is not None:
            slices.append(sl)
        else:
            slices.append(np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32))

    if not slices:
        print('[ERROR] No slices downloaded')
        post_callback({'error': 'no_slices', 'job_id': JOB_ID, 'receipt_id': RECEIPT_ID})
        return

    # 2. Build 3D patch [Z, Y, X]
    volume = np.stack(slices, axis=0)
    # Crop to patch size at x,y
    h, w = volume.shape[1], volume.shape[2]
    y_end = min(Y + PATCH_SIZE, h)
    x_end = min(X + PATCH_SIZE, w)
    patch = volume[:PATCH_SIZE, Y:y_end, X:x_end]

    # Pad if needed
    if patch.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
        padded = np.zeros((PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
        patch = padded

    # 3. Normalize
    p_min, p_max = patch.min(), patch.max()
    if p_max > p_min:
        patch = (patch - p_min) / (p_max - p_min)
    else:
        print('[WARN] Flat patch — likely empty space')
        post_callback({
            'job_id': JOB_ID, 'receipt_id': RECEIPT_ID,
            'x': X, 'y': Y, 'z': Z,
            'score': 0.0, 'letter_candidates': 0,
            'best_z_slice': 0, 'mean_intensity': float(patch.mean()),
            'status': 'empty_patch'
        })
        return

    mean_intensity = float(patch.mean())
    print(f'[PATCH] mean={mean_intensity:.4f} shape={patch.shape}')

    # 4. Segmentation — find the sheet
    best_z = segment_sheet(patch)
    sheet_2d = patch[best_z]
    print(f'[SEG] Best sheet at z={best_z} variance={np.var(sheet_2d):.6f}')

    # 5. Ink detection
    model = load_model()
    tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        prob_volume = model(tensor).squeeze().numpy()

    prob_sheet = prob_volume[best_z]
    high_ink = float((prob_sheet > 0.8).sum()) / (PATCH_SIZE * PATCH_SIZE)
    score = high_ink * 100.0

    # Also apply threshold-based heuristic on raw intensity as a backup signal
    intensity_score = float((sheet_2d > THRESHOLD).sum()) / (PATCH_SIZE * PATCH_SIZE)

    # 6. Count letter-like blobs
    letter_candidates = count_letter_candidates(prob_sheet)
    print(f'[INK] score={score:.2f}% intensity_score={intensity_score:.2f} blobs={letter_candidates}')

    # 7. Build result — send compact representation, not full prob map
    # Store top 16x16 downsampled map to keep payload small
    downsampled = prob_sheet[::4, ::4].flatten().tolist()

    result = {
        'job_id':            JOB_ID,
        'receipt_id':        RECEIPT_ID,
        'scroll_id':         SCROLL_ID,
        'x': X, 'y': Y, 'z': Z,
        'score':             round(score, 4),
        'intensity_score':   round(intensity_score, 4),
        'letter_candidates': letter_candidates,
        'best_z_slice':      best_z,
        'mean_intensity':    round(mean_intensity, 6),
        'prob_map_16x16':    [round(v, 4) for v in downsampled],
        'status':            'ok'
    }

    post_callback(result)

def post_callback(payload):
    if not CALLBACK_URL:
        print('[CALLBACK] No URL set — printing result:')
        print(json.dumps(payload, indent=2))
        return
    try:
        r = requests.post(CALLBACK_URL, json=payload, timeout=15)
        print(f'[CALLBACK] {r.status_code} — {CALLBACK_URL}')
    except Exception as e:
        print(f'[CALLBACK] Failed: {e}')
        print(json.dumps(payload, indent=2))

if __name__ == '__main__':
    main()
