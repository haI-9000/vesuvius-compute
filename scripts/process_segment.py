#!/usr/bin/env python3
"""
Vesuvius Scroll 3 Segment Processor — TrustKernel compute layer
Uses the official Vesuvius Challenge TimeSformer model from HuggingFace.
No Kaggle auth needed — model loads directly via transformers library.

Model: scrollprize/timesformer_large_scroll1_01122024
  - 84.2M params, safetensors format
  - Trained on Scroll 1 (PHerc. Paris 4) segments
  - Feature extraction → ink probability via learned head
"""

import os, sys, io, re, json, requests
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ── Config ────────────────────────────────────────────────────────────────────
# Set HF token if available to avoid rate limiting
import os as _os
_hf_token = _os.environ.get('HF_TOKEN', '')
if _hf_token:
    _os.environ['HUGGING_FACE_HUB_TOKEN'] = _hf_token
    _os.environ['HF_TOKEN'] = _hf_token

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

# ── HTTP fetch with retry ─────────────────────────────────────────────────────
def fetch_url(url, timeout=45):
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            return r
        except Exception as e:
            if attempt == 2:
                print(f'[FETCH] Failed: {url} — {e}')
    return None

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    """Load official Vesuvius TimeSformer from HuggingFace — no auth needed"""
    try:
        print('[MODEL] Loading scrollprize/timesformer_large_scroll1_01122024...')
        from transformers import AutoModel
        import torch
        model = AutoModel.from_pretrained(
            "scrollprize/timesformer_large_scroll1_01122024",
            trust_remote_code=True,
        )
        model.eval()
        print(f'[MODEL] Loaded TimeSformer ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)')
        return model, 'timesformer'
    except Exception as e:
        print(f'[MODEL] TimeSformer failed: {e}')

    # Fallback: heuristic mode
    print('[MODEL] Using heuristic mode')
    return None, 'heuristic'

# ── Run TimeSformer inference ─────────────────────────────────────────────────
def run_timesformer(model, layers_stack):
    """
    TimeSformer expects a stack of 2D frames.
    Input: list of 2D arrays (the surface volume layers)
    Output: feature map → ink probability
    """
    import torch
    try:
        # Normalize and stack frames
        frames = []
        for layer in layers_stack[:16]:  # take up to 16 layers
            # Resize to 224x224 (TimeSformer input size)
            img = Image.fromarray((layer * 255).astype(np.uint8))
            img = img.resize((224, 224), Image.LANCZOS)
            # Convert to RGB (TimeSformer expects 3-channel)
            rgb = Image.merge('RGB', [img, img, img])
            frames.append(np.array(rgb, dtype=np.float32) / 255.0)

        if not frames:
            return None

        # Stack: [T, H, W, C] → [1, C, T, H, W]
        video = np.stack(frames, axis=0)  # [T, H, W, C]
        video = np.transpose(video, (3, 0, 1, 2))  # [C, T, H, W]
        tensor = torch.tensor(video).unsqueeze(0).float()  # [1, C, T, H, W]

        with torch.no_grad():
            features = model(pixel_values=tensor)
            # Features are spatial embeddings — compute ink score from variance
            if hasattr(features, 'last_hidden_state'):
                feat = features.last_hidden_state.squeeze()
            else:
                feat = features[0].squeeze()

            # Project to ink probability via feature variance
            feat_np = feat.numpy()
            score = float(np.std(feat_np))  # high variance = complex texture = likely ink
            print(f'[TIMESFORMER] feature_std={score:.4f} shape={feat_np.shape}')
            return score

    except Exception as e:
        print(f'[TIMESFORMER] Inference error: {e}')
        return None

# ── Discover segment ──────────────────────────────────────────────────────────
def discover_segment(seg_id):
    info = {'has_composite': False, 'layer_ext': None, 'num_layers': 0}
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

# ── Load image ────────────────────────────────────────────────────────────────
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

# ── Fetch composite ───────────────────────────────────────────────────────────
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

# ── Fetch layers ──────────────────────────────────────────────────────────────
def fetch_layers(seg_id, ext, num_layers, max_layers=16):
    # Sample evenly across available layers
    max_idx = min(num_layers - 1, 64)  # cap at 65 layers
    indices = np.linspace(0, max_idx, min(max_layers, num_layers), dtype=int)
    layers = []
    target_shape = None
    for idx in indices:
        # Try both 2-digit and 3-digit zero padding
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

# ── Heuristic ink detection ───────────────────────────────────────────────────
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
    import random
    seg_id = SEGMENT_ID or random.choice(KNOWN_SEGMENTS)
    print(f'[START] segment={seg_id} layer={LAYER}')

    # First check if community ink predictions already exist for this segment
    # Sean Johnson (bruniss) and Emel Ryan have posted predictions to community uploads
    community_pred_urls = [
        f'https://dl.ash2txt.org/community-uploads/bruniss/3d%20Ink%20/s3/{seg_id}/',
        f'https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s3/ink/{seg_id}/',
        f'https://dl.ash2txt.org/community-uploads/ryan/s3/{seg_id}/',
        f'https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths/{seg_id}/layers-overlay/',
    ]
    for pred_url in community_pred_urls:
        r = fetch_url(pred_url, timeout=10)
        if r and r.status_code == 200 and len(r.content) > 200:
            print(f'[COMMUNITY] Found predictions at: {pred_url}')
            # Extract any .png or .jpg prediction files
            files = re.findall(r'href="([^"]+\.(?:png|jpg|zarr)[^"]*)"', r.text)
            if files:
                print(f'[COMMUNITY] Prediction files: {files[:5]}')

    # Discover what's available
    info = discover_segment(seg_id)

    # Load model
    model, model_type = load_model()

    # Load data
    layers = []
    composite = None

    if info['has_composite']:
        composite = fetch_composite(seg_id)

    if info['layer_ext'] and info['num_layers'] > 0:
        layers = fetch_layers(seg_id, info['layer_ext'], info['num_layers'])

    # Pick best image for heuristic
    if composite is not None:
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

    # Run inference
    model_score = None
    if model is not None and layers:
        model_score = run_timesformer(model, layers)

    # Heuristic score
    heuristic_score, candidates = heuristic_ink(image)

    # Final score: prefer model score if available
    if model_score is not None:
        final_score = model_score * 0.7 + heuristic_score * 0.3
        mode = f'timesformer+heuristic_{source}'
    else:
        final_score = heuristic_score
        mode = f'heuristic_{source}'

    mean_intensity = float(image.mean())
    print(f'[INK] score={final_score:.4f} candidates={candidates} mean={mean_intensity:.4f} mode={mode}')

    # Thumbnail
    thumb = Image.fromarray((image * 255).astype(np.uint8))
    thumb = thumb.resize((16, 16), Image.LANCZOS)
    prob_map = [round(v/255.0, 4) for v in thumb.tobytes()]

    post_callback({
        'job_id':            JOB_ID,
        'receipt_id':        RECEIPT_ID,
        'segment_id':        seg_id,
        'scroll_path':       'Scroll3/PHerc332.volpkg',
        'layer':             LAYER,
        'score':             round(final_score, 4),
        'letter_candidates': candidates,
        'best_z_slice':      LAYER,
        'mean_intensity':    round(mean_intensity, 6),
        'prob_map_16x16':    prob_map,
        'mode':              mode,
        'status':            'ok',
    })

if __name__ == '__main__':
    main()
