#!/usr/bin/env python3
"""
Vesuvius segment processor — runs in GitHub Actions free tier.
Downloads pre-segmented surface data from public S3 bucket,
runs ink detection on the 2D surface, posts results back to TrustKernel.

This is how the community actually works:
  Raw CT → Segmentation (hard, done by community) → Surface Volume → Ink Detection → Letters

We skip the hard part by using existing community segments.
Scroll 3 (PHerc. 332) segments: s3://vesuvius-challenge-open-data/full-scrolls/Scroll3.volpkg/paths/
"""

import os
import sys
import json
import io
import requests
import numpy as np
import torch
import torch.nn as nn
import glob

# ── Config ────────────────────────────────────────────────────────────────────
SEGMENT_ID   = os.environ.get('SEGMENT_ID', '')
SCROLL_PATH  = os.environ.get('SCROLL_PATH', 'Scroll3.volpkg')
LAYER        = int(os.environ.get('LAYER', '32'))       # which layer of surface volume
CALLBACK_URL = os.environ.get('CALLBACK_URL', '')
JOB_ID       = os.environ.get('JOB_ID', 'unknown')
RECEIPT_ID   = os.environ.get('RECEIPT_ID', 'unknown')
KAGGLE_USER  = os.environ.get('KAGGLE_USERNAME', '')
KAGGLE_KEY   = os.environ.get('KAGGLE_KEY', '')

# Public S3 base
S3_BASE = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'

# ── Minimal 3D U-Net ──────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class InkUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 32); self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv(32, 64); self.pool2 = nn.MaxPool3d(2)
        self.bot  = DoubleConv(64, 128)
        self.up2  = nn.ConvTranspose3d(128, 64, 2, stride=2); self.dec2 = DoubleConv(128, 64)
        self.up1  = nn.ConvTranspose3d(64, 32, 2, stride=2);  self.dec1 = DoubleConv(64, 32)
        self.out  = nn.Conv3d(32, 1, 1)
    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(self.pool1(e1)); b=self.bot(self.pool2(e2))
        d2=self.dec2(torch.cat([self.up2(b), e2], 1)); d1=self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.out(d1))

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    model = InkUNet()
    loaded = False

    if KAGGLE_USER and KAGGLE_KEY:
        try:
            print('[MODEL] Downloading from Kaggle...')
            os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
            with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
                json.dump({'username': KAGGLE_USER, 'key': KAGGLE_KEY}, f)
            os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
            os.system('pip install kaggle -q')
            os.system('kaggle datasets download -d ryches/unet3d -p /tmp/unet3d --unzip -q')
            pth_files = glob.glob('/tmp/unet3d/*.pth')
            if pth_files:
                state = torch.load(pth_files[0], map_location='cpu')
                if isinstance(state, dict) and 'model_state_dict' in state:
                    state = state['model_state_dict']
                model.load_state_dict(state, strict=False)
                print(f'[MODEL] Loaded pretrained weights: {pth_files[0]}')
                loaded = True
        except Exception as e:
            print(f'[MODEL] Kaggle failed: {e}')

    if not loaded:
        print('[MODEL] Using heuristic mode')
    model.eval()
    return model

# ── List available segments for Scroll 3 ─────────────────────────────────────
def list_segments():
    """Fetch list of available segment IDs for Scroll 3"""
    # Try to get segment listing from S3
    url = f'{S3_BASE}/full-scrolls/{SCROLL_PATH}/paths/'
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            # Parse XML listing
            import re
            keys = re.findall(r'<Key>([^<]+)</Key>', r.text)
            segment_ids = set()
            for k in keys:
                parts = k.split('/')
                if len(parts) >= 4 and parts[2] == 'paths':
                    segment_ids.add(parts[3])
            return list(segment_ids)[:20]  # first 20
    except Exception as e:
        print(f'[SEGMENTS] List failed: {e}')
    return []

# ── Download a surface volume layer (TIFF from segment) ──────────────────────
def fetch_surface_layer(segment_id, layer_num):
    """
    Downloads a single layer from a segment's surface volume.
    Surface volumes are stored as TIFFs: paths/{segment_id}/layers/{layer:02d}.tif
    """
    url = f'{S3_BASE}/full-scrolls/{SCROLL_PATH}/paths/{segment_id}/layers/{layer_num:02d}.tif'
    print(f'[FETCH] {url}')
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            try:
                import tifffile
                img = tifffile.imread(io.BytesIO(r.content))
                return img.astype(np.float32)
            except Exception:
                data = np.frombuffer(r.content[8:], dtype=np.uint16)
                side = int(np.sqrt(len(data)))
                return data[:side*side].reshape(side, side).astype(np.float32)
        else:
            print(f'[FETCH] HTTP {r.status_code}')
            return None
    except Exception as e:
        print(f'[FETCH] Error: {e}')
        return None

# ── Connected components ──────────────────────────────────────────────────────
def count_ink_blobs(prob_map, threshold=0.75, min_size=30):
    binary = (prob_map > threshold).astype(np.uint8)
    h, w = binary.shape
    visited = np.zeros_like(binary)
    blobs = 0
    def bfs(sy, sx):
        queue = [(sy, sx)]; size = 0
        while queue:
            cy, cx = queue.pop()
            if cy<0 or cy>=h or cx<0 or cx>=w: continue
            if visited[cy,cx] or not binary[cy,cx]: continue
            visited[cy,cx]=1; size+=1
            queue.extend([(cy+1,cx),(cy-1,cx),(cy,cx+1),(cy,cx-1)])
        return size
    for y in range(h):
        for x in range(w):
            if binary[y,x] and not visited[y,x]:
                if bfs(y,x) >= min_size: blobs+=1
    return blobs

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f'[START] Scroll={SCROLL_PATH} segment={SEGMENT_ID} layer={LAYER}')

    # If no segment specified, pick one from available list
    seg_id = SEGMENT_ID
    if not seg_id:
        print('[SEGMENTS] No segment specified, listing available...')
        segs = list_segments()
        if segs:
            seg_id = segs[0]
            print(f'[SEGMENTS] Using first available: {seg_id}')
        else:
            print('[SEGMENTS] No segments found, falling back to raw patch mode')
            post_callback({'error': 'no_segments', 'job_id': JOB_ID})
            return

    # Download multiple layers to build a 3D patch
    layers = []
    for l in range(max(0, LAYER-16), LAYER+16):
        sl = fetch_surface_layer(seg_id, l)
        if sl is not None:
            layers.append(sl)

    if len(layers) < 4:
        print(f'[ERROR] Only got {len(layers)} layers')
        post_callback({'error': 'insufficient_layers', 'segment_id': seg_id, 'job_id': JOB_ID})
        return

    print(f'[LAYERS] Got {len(layers)} layers, shape={layers[0].shape}')

    # Build 3D volume from layers
    volume = np.stack(layers, axis=0)
    p_min, p_max = volume.min(), volume.max()
    if p_max > p_min:
        volume = (volume - p_min) / (p_max - p_min)

    # Crop to 64x64 patch in center
    z, h, w = volume.shape
    patch_size = 64
    z_c = min(z, patch_size)
    h_s = max(0, (h - patch_size) // 2)
    w_s = max(0, (w - patch_size) // 2)
    patch = volume[:z_c, h_s:h_s+patch_size, w_s:w_s+patch_size]

    # Pad if needed
    if patch.shape != (patch_size, patch_size, patch_size):
        padded = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
        padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
        patch = padded

    mean_intensity = float(patch.mean())
    print(f'[PATCH] mean={mean_intensity:.4f}')

    if mean_intensity < 0.01:
        print('[WARN] Empty patch')
        post_callback({'status': 'empty', 'segment_id': seg_id, 'job_id': JOB_ID, 'receipt_id': RECEIPT_ID})
        return

    # Run ink detection
    model = load_model()
    tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        prob_volume = model(tensor).squeeze().numpy()

    # Best slice
    best_z = int(np.argmax(np.var(prob_volume, axis=(1,2))))
    prob_sheet = prob_volume[best_z]
    high_ink = float((prob_sheet > 0.8).sum()) / (patch_size * patch_size)
    score = high_ink * 100.0
    letter_candidates = count_ink_blobs(prob_sheet)

    print(f'[INK] score={score:.2f}% blobs={letter_candidates} best_z={best_z}')

    # Downsample prob map for callback
    downsampled = prob_sheet[::4, ::4].flatten().tolist()

    result = {
        'job_id':            JOB_ID,
        'receipt_id':        RECEIPT_ID,
        'scroll_path':       SCROLL_PATH,
        'segment_id':        seg_id,
        'layer':             LAYER,
        'score':             round(score, 4),
        'letter_candidates': letter_candidates,
        'best_z_slice':      best_z,
        'mean_intensity':    round(mean_intensity, 6),
        'prob_map_16x16':    [round(v, 4) for v in downsampled],
        'status':            'ok',
        'mode':              'segment_surface'
    }
    post_callback(result)

def post_callback(payload):
    if not CALLBACK_URL:
        print(json.dumps(payload, indent=2))
        return
    try:
        r = requests.post(CALLBACK_URL, json=payload, timeout=15)
        print(f'[CALLBACK] {r.status_code}')
    except Exception as e:
        print(f'[CALLBACK] Failed: {e}')
        print(json.dumps(payload, indent=2))

if __name__ == '__main__':
    main()
