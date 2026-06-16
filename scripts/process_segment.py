#!/usr/bin/env python3
"""
Vesuvius Scroll 3 Segment Processor — TrustKernel compute layer
Runs on GitHub Actions free tier. No model weights required.

Strategy:
1. Discover what files exist in the segment (composite.jpg, layers/*.tif, layers/*.jpg)
2. Load whatever is available — composite preferred, layer fallback
3. Run crackle/texture heuristic to detect ink signatures
4. Post results back to TrustKernel kernel

Segment data at: https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths/
"""

import os, sys, io, re, json, requests
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # disable decompression bomb check

# ── Config ────────────────────────────────────────────────────────────────────
SEGMENT_ID   = os.environ.get('SEGMENT_ID', '').strip()
LAYER        = int(os.environ.get('LAYER', '15'))
CALLBACK_URL = os.environ.get('CALLBACK_URL', '')
JOB_ID       = os.environ.get('JOB_ID', 'unknown')
RECEIPT_ID   = os.environ.get('RECEIPT_ID', 'unknown')

BASE = 'https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths'

# Known good segments with their file types
KNOWN_SEGMENTS = {
    '20231030220150': 'unknown',
    '20231031231220': 'unknown',
    '20240618142020': 'jpg',
    '20240618142021': 'jpg',
    '20240618142022': 'jpg',
    '20240712064330': 'tif',
    '20240712071520': 'tif',
    '20240712074250': 'tif',
    '20240715203740': 'unknown',
    '20240716140050': 'jpg',
    '20240716140051': 'jpg',
    '20240716140052': 'jpg',
}

# ── HTTP fetch with retry ─────────────────────────────────────────────────────
def fetch_url(url, timeout=45):
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            return r
        except Exception as e:
            if attempt == 2:
                print(f'[FETCH] Failed after 3 attempts: {url} — {e}')
            continue
    return None

# ── Discover segment files ────────────────────────────────────────────────────
def discover_segment(seg_id):
    """Returns dict with keys: has_composite, layer_ext, num_layers"""
    info = {'has_composite': False, 'layer_ext': None, 'num_layers': 0}
    
    # Check layers directory
    r = fetch_url(f'{BASE}/{seg_id}/layers/')
    if r and r.status_code == 200:
        files = re.findall(r'href="(\d+\.[a-z]+)"', r.text)
        if files:
            # Get extension from first file
            ext = '.' + files[0].split('.')[-1]
            info['layer_ext'] = ext
            info['num_layers'] = len(files)
            print(f'[DISCOVER] layers: {len(files)} files, ext={ext}')
    
    # Check for composite
    r = fetch_url(f'{BASE}/{seg_id}/composite.jpg', timeout=10)
    if r and r.status_code == 200 and len(r.content) > 10000:
        info['has_composite'] = True
        print(f'[DISCOVER] composite.jpg: {len(r.content)//1024}KB')
    
    return info

# ── Load image from bytes ─────────────────────────────────────────────────────
def load_image(content, ext='.jpg'):
    try:
        if ext == '.tif' or ext == '.tiff':
            import tifffile
            img = tifffile.imread(io.BytesIO(content))
            arr = np.array(img, dtype=np.float32)
        else:
            img = Image.open(io.BytesIO(content)).convert('L')
            arr = np.array(img, dtype=np.float32)
        
        # Handle multi-channel
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        
        # Normalize to 0-1
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        
        return arr
    except Exception as e:
        print(f'[LOAD] Error: {e}')
        return None

# ── Fetch composite (downsampled) ─────────────────────────────────────────────
def fetch_composite(seg_id):
    r = fetch_url(f'{BASE}/{seg_id}/composite.jpg', timeout=60)
    if not r or r.status_code != 200:
        return None
    try:
        img = Image.open(io.BytesIO(r.content)).convert('L')
        # Downsample to 1024px max
        w, h = img.size
        scale = min(1024/w, 1024/h, 1.0)
        if scale < 1.0:
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        print(f'[COMPOSITE] shape={arr.shape} mean={arr.mean():.4f} std={arr.std():.4f}')
        return arr
    except Exception as e:
        print(f'[COMPOSITE] Failed: {e}')
        return None

# ── Fetch a single layer ──────────────────────────────────────────────────────
def fetch_layer(seg_id, layer_num, ext=None):
    exts = [ext] if ext else ['.tif', '.jpg']
    for e in exts:
        url = f'{BASE}/{seg_id}/layers/{layer_num:02d}{e}'
        r = fetch_url(url, timeout=30)
        if r and r.status_code == 200:
            arr = load_image(r.content, e)
            if arr is not None:
                print(f'[LAYER] {layer_num:02d}{e} shape={arr.shape} mean={arr.mean():.4f}')
                return arr
    return None

# ── Fetch multiple layers and take mean ───────────────────────────────────────
def fetch_layers_mean(seg_id, ext, num_layers):
    """Fetch up to 5 evenly spaced layers and average them"""
    if num_layers == 0:
        return None
    
    # Pick up to 5 evenly spaced layer indices
    indices = np.linspace(0, min(num_layers-1, 29), min(5, num_layers), dtype=int)
    layers = []
    target_shape = None
    
    for idx in indices:
        arr = fetch_layer(seg_id, idx, ext)
        if arr is not None:
            if target_shape is None:
                target_shape = arr.shape
            if arr.shape == target_shape:
                layers.append(arr)
    
    if not layers:
        return None
    
    result = np.mean(layers, axis=0)
    print(f'[LAYERS] Averaged {len(layers)} layers, shape={result.shape}')
    return result

# ── Ink detection heuristic ───────────────────────────────────────────────────
def detect_ink(image):
    """
    Ink in Herculaneum scrolls creates subtle texture/crackle patterns.
    We use local variance and gradient analysis.
    Returns (score, letter_candidates)
    """
    h, w = image.shape
    patch_size = min(64, h//4, w//4)
    if patch_size < 8:
        return 0.0, 0
    
    stride = patch_size // 2
    patch_scores = []
    
    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            p = image[y:y+patch_size, x:x+patch_size]
            var = float(np.var(p))
            gx = np.diff(p, axis=1)
            gy = np.diff(p, axis=0)
            grad = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))
            mean = float(p.mean())
            
            # Ink signature: high variance, high gradient, medium intensity
            score = var * 10 + grad * 5
            if 0.05 < mean < 0.85:
                score *= 1.3
            patch_scores.append(score)
    
    if not patch_scores:
        return 0.0, 0
    
    arr = np.array(patch_scores)
    top10 = np.sort(arr)[-max(1, len(arr)//10):]
    overall = float(np.mean(top10))
    
    # Letter candidates: patches significantly above median
    median = float(np.median(arr))
    candidates = int(np.sum(arr > median * 3))
    
    return overall, candidates

# ── Post callback ─────────────────────────────────────────────────────────────
def post_callback(payload):
    if not CALLBACK_URL:
        print(json.dumps(payload, indent=2))
        return
    try:
        r = requests.post(CALLBACK_URL, json=payload, timeout=15)
        print(f'[CALLBACK] {r.status_code}')
    except Exception as e:
        print(f'[CALLBACK] error: {e}')

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Pick segment
    seg_id = SEGMENT_ID
    if not seg_id:
        # Pick random known segment
        import random
        seg_id = random.choice(list(KNOWN_SEGMENTS.keys()))
        print(f'[SEG] No segment specified, using: {seg_id}')
    
    print(f'[START] segment={seg_id} layer={LAYER}')
    
    # Discover what's available
    info = discover_segment(seg_id)
    
    # Load data — composite preferred, layer fallback
    image = None
    source = 'none'
    
    if info['has_composite']:
        image = fetch_composite(seg_id)
        source = 'composite'
    
    if image is None and info['layer_ext'] and info['num_layers'] > 0:
        image = fetch_layers_mean(seg_id, info['layer_ext'], info['num_layers'])
        source = 'layers'
    
    if image is None:
        # Last resort: try fetching specific layers directly
        for l in [15, 14, 16, 13, 17, 10, 20, 5, 25]:
            image = fetch_layer(seg_id, l)
            if image is not None:
                source = f'layer_{l}'
                break
    
    if image is None:
        print('[ERROR] Could not load any data from segment')
        post_callback({
            'job_id': JOB_ID, 'receipt_id': RECEIPT_ID,
            'segment_id': seg_id, 'score': 0.0,
            'letter_candidates': 0, 'status': 'no_data',
            'mode': 'segment_surface'
        })
        return
    
    # Run ink detection
    score, candidates = detect_ink(image)
    mean_intensity = float(image.mean())
    print(f'[INK] score={score:.4f} candidates={candidates} mean={mean_intensity:.4f} source={source}')
    
    # Thumbnail for callback
    thumb = Image.fromarray((image * 255).astype(np.uint8))
    thumb = thumb.resize((16, 16), Image.LANCZOS)
    prob_map = [round(v/255.0, 4) for v in thumb.tobytes()]
    
    post_callback({
        'job_id':            JOB_ID,
        'receipt_id':        RECEIPT_ID,
        'segment_id':        seg_id,
        'scroll_path':       'Scroll3/PHerc332.volpkg',
        'layer':             LAYER,
        'score':             round(score, 4),
        'letter_candidates': candidates,
        'best_z_slice':      LAYER,
        'mean_intensity':    round(mean_intensity, 6),
        'prob_map_16x16':    prob_map,
        'mode':              f'composite_heuristic_{source}',
        'status':            'ok',
    })

if __name__ == '__main__':
    main()
