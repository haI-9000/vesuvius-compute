#!/usr/bin/env python3

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

SEGMENTS = [
    '20231030220150', '20231031231220',
    '20240618142020', '20240618142021', '20240618142022',
    '20240712064330', '20240712071520', '20240712074250',
    '20240715203740', '20240716140050', '20240716140051', '20240716140052',
]

def get(url, timeout=10):
    for _ in range(2):
        try:
            r = requests.get(url, timeout=timeout)
            return r
        except:
            pass
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
    except:
        return None

def overlay(seg):
    r = get(f'{BASE}/{seg}/layers-overlay/', timeout=5)
    if not r or r.status_code != 200:
        return None, None
    files = [f.strip('/') for f in re.findall(r'href="(\d[^"]+\.(?:png|jpg|tif))"', r.text)]
    for f in files[:3]:
        ext = '.' + f.rsplit('.', 1)[-1]
        r2 = get(f'{BASE}/{seg}/layers-overlay/{f}', timeout=15)
        if r2 and r2.status_code == 200:
            arr = to_array(r2.content, ext)
            if arr is not None:
                return arr, f
    return None, None

def composite(seg):
    r = get(f'{BASE}/{seg}/composite.jpg', timeout=20)
    if not r or r.status_code != 200:
        return None
    try:
        img = Image.open(io.BytesIO(r.content)).convert('L')
        w, h = img.size
        s = min(1024/w, 1024/h, 1.0)
        if s < 1.0:
            img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr
    except:
        return None

def discover(seg):
    info = {'ext': None, 'n': 0, 'has_composite': False}
    r = get(f'{BASE}/{seg}/layers/')
    if r and r.status_code == 200:
        files = re.findall(r'href="(\d+\.[a-z]+)"', r.text)
        if files:
            info['ext'] = '.' + files[0].split('.')[-1]
            info['n'] = len(files)
    r = get(f'{BASE}/{seg}/composite.jpg', timeout=5)
    if r and r.status_code == 200 and len(r.content) > 10000:
        info['has_composite'] = True
    return info

def fetch_layers(seg, ext, n, max_n=16):
    indices = np.linspace(0, min(n-1, 64), min(max_n, n), dtype=int)
    layers, shape = [], None
    for i in indices:
        for fmt in [f'{i:02d}', f'{i:03d}']:
            r = get(f'{BASE}/{seg}/layers/{fmt}{ext}', timeout=15)
            if r and r.status_code == 200:
                arr = to_array(r.content, ext)
                if arr is not None:
                    if shape is None: shape = arr.shape
                    if arr.shape == shape:
                        layers.append(arr)
                    break
    return layers

def ink_score(img):
    h, w = img.shape
    ps = min(64, h//4, w//4)
    if ps < 8: return 0.0, 0
    st = ps // 2
    scores = []
    for y in range(0, h - ps, st):
        for x in range(0, w - ps, st):
            p = img[y:y+ps, x:x+ps]
            v = float(np.var(p))
            g = float(np.mean(np.abs(np.diff(p, axis=1))) + np.mean(np.abs(np.diff(p, axis=0))))
            s = v * 10 + g * 5
            if 0.05 < p.mean() < 0.85:
                s *= 1.3
            scores.append(s)
    if not scores: return 0.0, 0
    a = np.array(scores)
    score = float(np.mean(np.sort(a)[-max(1, len(a)//10):]))
    cands = int(np.sum(a > np.median(a) * 3))
    return score, cands

def callback(payload):
    if not CALLBACK_URL:
        print(json.dumps(payload))
        return
    try:
        r = requests.post(CALLBACK_URL, json=payload, timeout=10)
        print(f'[CALLBACK] {r.status_code}')
    except Exception as e:
        print(f'[CALLBACK] {e}')

def main():
    import random
    seg = SEGMENT_ID or random.choice(SEGMENTS)
    print(f'[START] {seg} layer={LAYER}')

    ov_img, ov_file = overlay(seg)
    info = discover(seg)
    comp = composite(seg) if info['has_composite'] else None
    layers = fetch_layers(seg, info['ext'], info['n']) if info['ext'] else []

    if ov_img is not None:
        img, src = ov_img, f'overlay:{ov_file}'
    elif comp is not None:
        img, src = comp, 'composite'
    elif layers:
        img, src = np.mean(layers, axis=0), 'layers'
    else:
        callback({'job_id': JOB_ID, 'segment_id': seg, 'score': 0.0,
                  'letter_candidates': 0, 'status': 'no_data'})
        return

    score, cands = ink_score(img)
    mean = float(img.mean())

    callback({
        'job_id':            JOB_ID,
        'receipt_id':        RECEIPT_ID,
        'segment_id':        seg,
        'layer':             LAYER,
        'score':             round(score, 4),
        'letter_candidates': cands,
        'best_z_slice':      LAYER,
        'mean_intensity':    round(mean, 6),
        'mode':              src,
        'status':            'ok',
    })

if __name__ == '__main__':
    main()
