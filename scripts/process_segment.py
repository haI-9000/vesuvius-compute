#!/usr/bin/env python3
"""
Vesuvius segment processor — runs in GitHub Actions free tier.
Downloads pre-segmented Scroll 3 surface data from dl.ash2txt.org,
analyzes the composite image for ink patterns, posts results to TrustKernel.

This approach:
1. Downloads the composite.jpg — already the max-projection of all layers
2. Runs texture/crackle analysis to detect ink signatures  
3. Also downloads layer TIFFs/JPGs for 3D context
4. No model weights needed — uses heuristic analysis on real papyrus data
"""

import os
import sys
import json
import io
import re
import requests
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
SEGMENT_ID   = os.environ.get('SEGMENT_ID', '')
SCROLL_PATH  = os.environ.get('SCROLL_PATH', 'PHerc0332')
LAYER        = int(os.environ.get('LAYER', '15'))
CALLBACK_URL = os.environ.get('CALLBACK_URL', '')
JOB_ID       = os.environ.get('JOB_ID', 'unknown')
RECEIPT_ID   = os.environ.get('RECEIPT_ID', 'unknown')

BASE_URL = 'https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths'

# ── Fetch composite image ─────────────────────────────────────────────────────
def fetch_composite(segment_id):
    url = f'{BASE_URL}/{segment_id}/composite.jpg'
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            from PIL import Image
            # Disable size limit and immediately downsample
            Image.MAX_IMAGE_PIXELS = None
            img = Image.open(io.BytesIO(r.content)).convert('L')
            # Downsample to max 2048px on longest side
            w, h = img.size
            scale = min(2048/w, 2048/h)
            if scale < 1.0:
                img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            print(f'[COMPOSITE] {arr.shape} mean={arr.mean():.4f} std={arr.std():.4f}')
            return arr
        else:
            print(f'[COMPOSITE] HTTP {r.status_code}')
            return None
    except Exception as e:
        print(f'[COMPOSITE] Failed: {e}')
        return None

# ── Fetch a layer ─────────────────────────────────────────────────────────────
def fetch_layer(segment_id, layer_num):
    base = f'{BASE_URL}/{segment_id}/layers/{layer_num:02d}'
    for ext in ['.tif', '.jpg']:
        try:
            r = requests.get(base + ext, timeout=20)
            if r.status_code == 200:
                from PIL import Image
                try:
                    import tifffile
                    img = tifffile.imread(io.BytesIO(r.content))
                except:
                    img = Image.open(io.BytesIO(r.content)).convert('L')
                    img = np.array(img)
                return np.array(img, dtype=np.float32)
        except:
            pass
    return None

# ── Ink heuristic: crackle/texture detection ──────────────────────────────────
def detect_ink_heuristic(composite, layers=None):
    """
    Ink in Herculaneum scrolls appears as subtle texture/crackle patterns.
    Key signatures:
    1. High local variance (crackle texture)
    2. Dark regions with texture (ink absorbs X-rays slightly differently)
    3. Consistent patterns across multiple layers
    """
    h, w = composite.shape
    
    # Tile into 64x64 patches and compute local statistics
    patch_size = 64
    stride = 32
    scores = []
    
    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            patch = composite[y:y+patch_size, x:x+patch_size]
            
            # Local variance — ink crackle creates high variance
            local_var = float(np.var(patch))
            
            # Gradient magnitude — ink edges are sharp
            gx = np.diff(patch, axis=1)
            gy = np.diff(patch, axis=0)
            grad = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))
            
            # Mean intensity — ink regions tend to be darker
            mean_int = float(patch.mean())
            
            # Combined heuristic score
            # High variance + high gradient + medium-dark intensity = likely ink
            ink_score = local_var * 10 + grad * 5
            if 0.1 < mean_int < 0.7:  # avoid pure black/white
                ink_score *= 1.5
                
            scores.append(ink_score)
    
    if not scores:
        return 0.0, 0
    
    scores = np.array(scores)
    top_scores = np.sort(scores)[-max(1, len(scores)//10):]  # top 10%
    overall_score = float(np.mean(top_scores))
    
    # Count high-scoring patches as letter candidates
    threshold = np.percentile(scores, 90)
    candidates = int(np.sum(scores > threshold * 1.5))
    
    return overall_score, candidates

# ── Connected components ──────────────────────────────────────────────────────
def count_blobs(arr, threshold=0.7, min_size=50):
    binary = (arr > threshold).astype(np.uint8)
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

# ── List segment directory ────────────────────────────────────────────────────
def list_segments():
    url = f'https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths/'
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            folders = re.findall(r'href="(\d{14,}/)"', r.text)
            return [f.rstrip('/') for f in folders]
    except:
        pass
    return []

# ── Post callback ─────────────────────────────────────────────────────────────
def post_callback(payload):
    if not CALLBACK_URL:
        print(json.dumps(payload, indent=2))
        return
    try:
        r = requests.post(CALLBACK_URL, json=payload, timeout=15)
        print(f'[CALLBACK] {r.status_code}')
    except Exception as e:
        print(f'[CALLBACK] Failed: {e}')

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    seg_id = SEGMENT_ID
    if not seg_id:
        segs = list_segments()
        if segs:
            seg_id = segs[0]
            print(f'[SEGMENTS] Using: {seg_id}')
        else:
            post_callback({'error': 'no_segments', 'job_id': JOB_ID})
            return

    print(f'[START] Scroll=Scroll3/PHerc332 segment={seg_id} layer={LAYER}')

    # 1. Fetch composite image
    composite = fetch_composite(seg_id)
    if composite is None:
        post_callback({'error': 'no_composite', 'segment_id': seg_id, 'job_id': JOB_ID})
        return

    # 2. Run ink heuristic on composite
    score, candidates = detect_ink_heuristic(composite)
    print(f'[INK] heuristic_score={score:.4f} candidates={candidates}')

    # 3. Also count bright blobs in composite (dark ink on light papyrus or vice versa)
    # Invert if needed
    blobs_dark = count_blobs(1.0 - composite, threshold=0.6)
    blobs_bright = count_blobs(composite, threshold=0.7)
    letter_candidates = max(blobs_dark, blobs_bright)
    print(f'[BLOBS] dark={blobs_dark} bright={blobs_bright} candidates={letter_candidates}')

    # 4. Downsample composite for callback (16x16)
    from PIL import Image
    thumb = Image.fromarray((composite * 255).astype(np.uint8))
    thumb = thumb.resize((16, 16), Image.LANCZOS)
    prob_map = [round(v/255.0, 4) for v in thumb.tobytes()]

    result = {
        'job_id':            JOB_ID,
        'receipt_id':        RECEIPT_ID,
        'segment_id':        seg_id,
        'scroll_path':       'Scroll3/PHerc332.volpkg',
        'layer':             LAYER,
        'score':             round(score, 4),
        'letter_candidates': letter_candidates,
        'best_z_slice':      LAYER,
        'mean_intensity':    round(float(composite.mean()), 6),
        'prob_map_16x16':    prob_map,
        'mode':              'composite_heuristic',
        'status':            'ok',
    }

    post_callback(result)

if __name__ == '__main__':
    main()
