#!/usr/bin/env python3
"""
Vesuvius Scroll 3 Segment Processor — TrustKernel compute layer
Runs on GitHub Actions free tier. No model weights needed.

Priority order:
1. Community ink predictions (layers-overlay/) — Sean Johnson's actual output
2. Composite image (composite.jpg) — max-projection surface render
3. Surface layers (layers/*.tif or *.jpg) — raw CT surface data
"""

import os, io, re, json, requests
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SEGMENT_ID   = os.environ.get('SEGMENT_ID', '').strip()
LAYER        = int(os.environ.get('LAYER', '15'))
CALLBACK_URL = os.environ.get('CALLBACK_URL', '')
JOB_ID       = os.environ.get('JOB_ID', 'unknown')
RECEIPT_ID   = os.environ.get('RECEIPT_ID', 'unknown')

BASE = 'https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths'

KNOWN_SEGMENTS = [
    '20231030220150', '20231031231220',
    '20240618142020', '20240618142021', '20240618142022',
    '20240712064330', '20240712071520', '20240712074250',
    '20240715203740', '20240716140050', '20240716140051', '20240716140052',
]

def fetch_url(url, timeout=45):
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            return r
        except Exception as e:
            if attempt == 2:
                print(f'[FETCH] Failed: {url} — {e}')
    return None

def load_image(content, ext='.jpg'):
    try:
        if ext in ('.tif', '.tiff'):
            import tifffile
            img = tifffile.imread(io.BytesIO(content))
        else:
            img = Image.open(io.BytesIO(content)).convert('L')
            img = np.array(img)
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        return arr
    except Exception as e:
        print(f'[LOAD] Error: {e}')
        return None

def fetch_community_prediction(seg_id):
    """Download Sean Johnson's layers-overlay predictions if available"""
    overlay_url = f'{BASE}/{seg_id}/layers-overlay/'
    r = fetch_url(overlay_url, timeout=10)
    if not r or r.status_code != 200:
        return None, None
    
    print(f'[COMMUNITY] Found predictions at: {overlay_url}')
    files = re.findall(r'href="([^"]+\.(?:png|jpg|tif)[^"]*)"', r.text)
    files = [f.strip('/') for f in files if not f.startswith('?')]
    print(f'[COMMUNITY] Files: {files[:10]}')
    
    # Download first available prediction file
    for fname in files[:5]:
        ext = '.' + fname.split('.')[-1]
        img_url = overlay_url.rstrip('/') + '/' + fname
        img_r = fetch_url(img_url, timeout=30)
        if img_r and img_r.status_code == 200:
            arr = load_image(img_r.content, ext)
            if arr is not None:
                print(f'[COMMUNITY] Loaded: {fname} shape={arr.shape} mean={arr.mean():.4f}')
                return arr, fname
    return None, None

def fetch_composite(seg_id):
    r = fetch_url(f'{BASE}/{seg_id}/composite.jpg', timeout=60)
    if not r or r.status_code != 200:
        return None
    try:
        img = Image.open(io.BytesIO(r.content)).convert('L')
        w, h = img.size
        scale = min(1024/w, 1024/h, 1.0)
        if scale < 1.0:
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        print(f'[COMPOSITE] shape={arr.shape} mean={arr.mean():.4f}')
        return arr
    except Exception as e:
        print(f'[COMPOSITE] Error: {e}')
        return None

def discover_segment(seg_id):
    info = {'layer_ext': None, 'num_layers': 0, 'has_composite': False}
    r = fetch_url(f'{BASE}/{seg_id}/layers/')
    if r and r.status_code == 200:
        files = re.findall(r'href="(\d+\.[a-z]+)"', r.text)
        if files:
            info['layer_ext'] = '.' + files[0].split('.')[-1]
            info['num_layers'] = len(files)
            print(f'[DISCOVER] {len(files)} layers, ext={info["layer_ext"]}')
    r = fetch_url(f'{BASE}/{seg_id}/composite.jpg', timeout=10)
    if r and r.status_code == 200 and len(r.content) > 10000:
        info['has_composite'] = True
        print(f'[DISCOVER] composite.jpg: {len(r.content)//1024}KB')
    return info

def fetch_layers(seg_id, ext, num_layers, max_layers=16):
    max_idx = min(num_layers - 1, 64)
    indices = np.linspace(0, max_idx, min(max_layers, num_layers), dtype=int)
    layers = []
    target_shape = None
    for idx in indices:
        for fmt in [f'{idx:02d}', f'{idx:03d}', str(idx)]:
            url = f'{BASE}/{seg_id}/layers/{fmt}{ext}'
            r = fetch_url(url, timeout=30)
            if r and r.status_code == 200:
                arr = load_image(r.content, ext)
                if arr is not None:
                    if target_shape is None:
                        target_shape = arr.shape
                    if arr.shape == target_shape:
                        layers.append(arr)
                        print(f'[LAYER] {fmt}{ext} {arr.shape} mean={arr.mean():.4f}')
                    break
    if layers:
        print(f'[LAYERS] {len(layers)} layers loaded')
    return layers

def heuristic_ink(image):
    h, w = image.shape
    patch_size = min(64, h//4, w//4)
    if patch_size < 8:
        return 0.0, 0
    stride = patch_size // 2
    scores = []
    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            p = image[y:y+patch_size, x:x+patch_size]
            var = float(np.var(p))
            gx = np.diff(p, axis=1)
            gy = np.diff(p, axis=0)
            grad = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))
            mean = float(p.mean())
            score = var * 10 + grad * 5
            if 0.05 < mean < 0.85:
                score *= 1.3
            scores.append(score)
    if not scores:
        return 0.0, 0
    arr = np.array(scores)
    top10 = np.sort(arr)[-max(1, len(arr)//10):]
    overall = float(np.mean(top10))
    median = float(np.median(arr))
    candidates = int(np.sum(arr > median * 3))
    return overall, candidates

def post_callback(payload):
    if not CALLBACK_URL:
        print(json.dumps(payload, indent=2))
        return
    try:
        r = requests.post(CALLBACK_URL, json=payload, timeout=15)
        print(f'[CALLBACK] {r.status_code}')
    except Exception as e:
        print(f'[CALLBACK] error: {e}')

def main():
    import random
    seg_id = SEGMENT_ID or random.choice(KNOWN_SEGMENTS)
    print(f'[START] segment={seg_id} layer={LAYER}')

    # Priority 1: Community ink predictions (layers-overlay/)
    community_image, community_file = fetch_community_prediction(seg_id)
    
    # Discover segment structure
    info = discover_segment(seg_id)

    # Priority 2: Composite image
    composite = None
    if info['has_composite']:
        composite = fetch_composite(seg_id)

    # Priority 3: Surface layers
    layers = []
    if info['layer_ext'] and info['num_layers'] > 0:
        layers = fetch_layers(seg_id, info['layer_ext'], info['num_layers'])

    # Select best image source
    if community_image is not None:
        image = community_image
        source = f'community_overlay:{community_file}'
    elif composite is not None:
        image = composite
        source = 'composite'
    elif layers:
        image = np.mean(layers, axis=0)
        source = 'layers_mean'
    else:
        print('[ERROR] No data loaded')
        post_callback({'job_id': JOB_ID, 'segment_id': seg_id,
                       'score': 0.0, 'letter_candidates': 0,
                       'status': 'no_data', 'mode': 'segment_surface'})
        return

    # Run ink heuristic
    score, candidates = heuristic_ink(image)
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
        'layer':             LAYER,
        'score':             round(score, 4),
        'letter_candidates': candidates,
        'best_z_slice':      LAYER,
        'mean_intensity':    round(mean_intensity, 6),
        'prob_map_16x16':    prob_map,
        'mode':              f'heuristic_{source}',
        'status':            'ok',
    })

if __name__ == '__main__':
    main()
