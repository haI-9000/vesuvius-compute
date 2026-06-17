#!/usr/bin/env python3

import os, io, re, json, requests
import numpy as np
from PIL import Image
import signal

Image.MAX_IMAGE_PIXELS = None

SEGMENT_ID   = os.environ.get('SEGMENT_ID', '').strip()
LAYER        = int(os.environ.get('LAYER', '15'))
CALLBACK_URL = os.environ.get('CALLBACK_URL', '')
JOB_ID       = os.environ.get('JOB_ID', 'unknown')
RECEIPT_ID   = os.environ.get('RECEIPT_ID', 'unknown')

BASE = 'https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths'

SEGMENTS = [
    '20240715203740',  # confirmed ink coordinate — Ryan zarr val=244/255
    '20240712074250', '20240712071520',
    '20240618142021', '20240618142022',
    '20240716140050', '20240716140051', '20240716140052',
    '20240712064330', '20240618142020',
]

# Hard kill after 100s — GitHub job timeout is 3 min, we want clean exit
def _timeout_handler(signum, frame):
    print('[TIMEOUT] hard limit reached, sending fallback callback')
    callback({
        'job_id': JOB_ID, 'receipt_id': RECEIPT_ID,
        'segment_id': SEGMENT_ID, 'layer': LAYER,
        'score': 0.0, 'letter_candidates': 0,
        'status': 'timeout', 'mean_intensity': 0.0,
        'prob_map_16x16': [], 'mode': 'timeout',
    })
    raise SystemExit(0)

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(100)  # 100s hard limit

def get(url, timeout=8):
    try:
        r = requests.get(url, timeout=timeout)
        return r
    except Exception as e:
        print(f'[GET] failed {url}: {e}')
        return None

def to_array(content, ext):
    try:
        if ext in ('.tif', '.tiff'):
            import tifffile
            img = tifffile.imread(io.BytesIO(content))
        else:
            img = np.array(Image.open(io.BytesIO(content)).convert('L'))
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo) if hi > lo else arr
    except Exception as e:
        print(f'[ARRAY] {e}')
        return None

def composite(seg):
    r = get(f'{BASE}/{seg}/composite.jpg', timeout=15)
    if not r or r.status_code != 200:
        return None
    try:
        img = Image.open(io.BytesIO(r.content)).convert('L')
        w, h = img.size
        s = min(1024/w, 1024/h, 1.0)
        if s < 1.0:
            img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        print(f'[COMPOSITE] {seg} {arr.shape} mean={arr.mean():.4f}')
        return arr
    except Exception as e:
        print(f'[COMPOSITE] {e}')
        return None

def discover(seg):
    info = {'ext': None, 'n': 0, 'has_composite': False}
    r = get(f'{BASE}/{seg}/layers/', timeout=8)
    if r and r.status_code == 200:
        files = re.findall(r'href="(\d+\.[a-z]+)"', r.text)
        if files:
            info['ext'] = '.' + files[0].split('.')[-1]
            info['n'] = len(files)
            print(f'[LAYERS] {len(files)} layers ext={info["ext"]}')
    r = get(f'{BASE}/{seg}/composite.jpg', timeout=8)
    if r and r.status_code == 200 and len(r.content) > 10000:
        info['has_composite'] = True
    return info

def fetch_one_layer(seg, ext, layer_idx):
    for fmt in [f'{layer_idx:02d}', f'{layer_idx:03d}']:
        r = get(f'{BASE}/{seg}/layers/{fmt}{ext}', timeout=15)
        if r and r.status_code == 200:
            arr = to_array(r.content, ext)
            if arr is not None:
                print(f'[LAYER] {fmt}{ext} {arr.shape} mean={arr.mean():.4f}')
                return arr
    return None

def ink_score(img):
    mean = float(np.mean(img))
    std  = float(np.std(img))
    score = mean + std * 2
    return score

def callback(payload):
    if not CALLBACK_URL:
        print(json.dumps(payload))
        return
    try:
        r = requests.post(CALLBACK_URL, json=payload, timeout=10)
        print(f'[CALLBACK] {r.status_code} {r.text[:100]}')
    except Exception as e:
        print(f'[CALLBACK] {e}')

def main():
    import random
    seg = SEGMENT_ID or random.choice(SEGMENTS)
    print(f'[START] seg={seg} layer={LAYER} job={JOB_ID}')

    img = None
    src = 'none'

    # Try composite first — fastest single download
    info = discover(seg)

    if info['has_composite']:
        img = composite(seg)
        if img is not None:
            src = 'composite'

    # Fall back to single layer
    if img is None and info['ext'] and info['n'] > 0:
        idx = min(LAYER, info['n'] - 1)
        img = fetch_one_layer(seg, info['ext'], idx)
        if img is not None:
            src = f'layer:{idx}'

    if img is None:
        print('[ERROR] no data retrieved')
        callback({
            'job_id': JOB_ID, 'receipt_id': RECEIPT_ID,
            'segment_id': seg, 'layer': LAYER,
            'score': 0.0, 'letter_candidates': 0,
            'status': 'no_data', 'mean_intensity': 0.0,
            'prob_map_16x16': [], 'mode': 'no_data',
        })
        return

    score = ink_score(img)
    mean  = float(img.mean())

    print(f'[INK] score={score:.4f} mean={mean:.4f} src={src}')

    thumb    = Image.fromarray((img * 255).astype(np.uint8)).resize((16, 16), Image.LANCZOS)
    prob_map = [round(v/255.0, 4) for v in thumb.tobytes()]

    callback({
        'job_id':            JOB_ID,
        'receipt_id':        RECEIPT_ID,
        'segment_id':        seg,
        'layer':             LAYER,
        'score':             round(score, 4),
        'letter_candidates': 0,
        'best_z_slice':      LAYER,
        'mean_intensity':    round(mean, 6),
        'prob_map_16x16':    prob_map,
        'mode':              src,
        'status':            'ok',
    })

if __name__ == '__main__':
    main()
