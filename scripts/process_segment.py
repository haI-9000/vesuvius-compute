#!/usr/bin/env python3
import os, sys, io, re, json, requests
import numpy as np
from PIL import Image
import cv2
import random
import time

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
            time.sleep(1)
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

def load_kaggle_model():
    try:
        print('[KAGGLE] Loading model...')
        import torch
        try:
            from kaggle_model import Net
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from kaggle_model import Net

        weight_paths = [
            '/kaggle/input/ink-01-weight/fix-rot-00005904.model.pth',
            '/tmp/fix-rot-00005904.model.pth',
            'fix-rot-00005904.model.pth',
        ]
        weights_path = None
        for path in weight_paths:
            if os.path.exists(path):
                weights_path = path
                break

        if weights_path is None:
            try:
                import kagglehub
                path = kagglehub.model_download('ryches/vesuvius-ink-detection-1st-place/fix-rot-00005904')
                if os.path.exists(path):
                    weights_path = path
            except Exception:
                pass

        if weights_path is None:
            print('[KAGGLE] Weights not found')
            return None

        print(f'[KAGGLE] Loading weights from {weights_path}')
        model = Net()
        state = torch.load(weights_path, map_location='cpu')
        state_dict = state.get('state_dict', state)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        print('[KAGGLE] Model loaded')
        return model
    except Exception as e:
        print(f'[KAGGLE] Error: {e}')
        return None

def run_kaggle_inference(model, layers_stack):
    try:
        import torch
        from skimage.transform import resize

        depth = 16
        if len(layers_stack) < depth:
            return None, None

        indices = np.linspace(0, len(layers_stack)-1, depth, dtype=int)
        selected = [layers_stack[i] for i in indices]

        resized = []
        for layer in selected:
            layer_norm = (layer - layer.min()) / (layer.max() - layer.min() + 1e-7)
            resized.append(resize(layer_norm, (384, 384), preserve_range=True, mode='reflect'))

        volume = np.stack(resized, axis=0).astype(np.float32)
        input_tensor = torch.tensor(volume).unsqueeze(0)

        with torch.no_grad():
            output = model({'volume': input_tensor})
            prob_map = output['ink'].squeeze().cpu().numpy()

        threshold = 0.5
        binary = (prob_map > threshold).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        letter_candidates = sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 20)
        high_conf = prob_map[prob_map > 0.8]
        score = float(high_conf.mean() * 100) if len(high_conf) > 0 else 0.0

        print(f'[KAGGLE] score={score:.2f}, candidates={letter_candidates}')
        return score, letter_candidates
    except Exception as e:
        print(f'[KAGGLE] Inference error: {e}')
        return None, None

def load_timesformer():
    try:
        print('[TIMESFORMER] Loading...')
        try:
            import timesformer_pytorch
        except ImportError:
            print('[TIMESFORMER] timesformer_pytorch not installed')
            return None

        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            "scrollprize/timesformer_large_scroll1_01122024",
            trust_remote_code=True,
        )
        model.eval()
        return model
    except Exception as e:
        print(f'[TIMESFORMER] Error: {e}')
        return None

def run_timesformer(model, layers_stack):
    try:
        import torch
        frames = []
        for layer in layers_stack[:16]:
            img = Image.fromarray((layer * 255).astype(np.uint8))
            img = img.resize((224, 224), Image.LANCZOS)
            rgb = Image.merge('RGB', [img, img, img])
            frames.append(np.array(rgb, dtype=np.float32) / 255.0)

        if not frames:
            return None

        video = np.stack(frames, axis=0)
        video = np.transpose(video, (3, 0, 1, 2))
        tensor = torch.tensor(video).unsqueeze(0).float()

        with torch.no_grad():
            features = model(pixel_values=tensor)
            if hasattr(features, 'last_hidden_state'):
                feat = features.last_hidden_state.squeeze()
            else:
                feat = features[0].squeeze()
            feat_np = feat.numpy()
            score = float(np.std(feat_np))
            print(f'[TIMESFORMER] score={score:.4f}')
            return score
    except Exception as e:
        print(f'[TIMESFORMER] Inference error: {e}')
        return None

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
    seg_id = SEGMENT_ID or random.choice(KNOWN_SEGMENTS)
    print(f'[START] segment={seg_id} layer={LAYER}')

    community_pred_urls = [
        f'https://dl.ash2txt.org/community-uploads/bruniss/3d%20Ink%20/s3/{seg_id}/',
        f'https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s3/ink/{seg_id}/',
        f'https://dl.ash2txt.org/community-uploads/ryan/s3/{seg_id}/',
        f'https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths/{seg_id}/layers-overlay/',
    ]
    for pred_url in community_pred_urls:
        r = fetch_url(pred_url, timeout=10)
        if r and r.status_code == 200 and len(r.content) > 200:
            print(f'[COMMUNITY] Predictions: {pred_url}')

    info = discover_segment(seg_id)
    layers = []
    composite = None

    if info['has_composite']:
        composite = fetch_composite(seg_id)

    if info['layer_ext'] and info['num_layers'] > 0:
        layers = fetch_layers(seg_id, info['layer_ext'], info['num_layers'])

    if composite is not None:
        image = composite
        source = 'composite'
    elif layers:
        image = np.mean(layers, axis=0)
        source = 'layers_mean'
    else:
        print('[ERROR] No data')
        post_callback({
            'job_id': JOB_ID,
            'receipt_id': RECEIPT_ID,
            'segment_id': seg_id,
            'score': 0.0,
            'letter_candidates': 0,
            'status': 'no_data',
            'mode': 'segment_surface'
        })
        return

    kaggle_model = load_kaggle_model()
    timesformer_model = load_timesformer()

    results = {}

    if kaggle_model is not None and layers:
        kaggle_score, kaggle_candidates = run_kaggle_inference(kaggle_model, layers)
        if kaggle_score is not None:
            results['kaggle'] = {'score': kaggle_score, 'candidates': kaggle_candidates}

    if timesformer_model is not None and layers:
        ts_score = run_timesformer(timesformer_model, layers)
        if ts_score is not None:
            results['timesformer'] = ts_score

    heuristic_score, heuristic_candidates = heuristic_ink(image)
    results['heuristic'] = {'score': heuristic_score, 'candidates': heuristic_candidates}

    if 'kaggle' in results:
        final_score = results['kaggle']['score']
        candidates = results['kaggle']['candidates']
        mode = 'kaggle_1st_place'
    elif 'timesformer' in results:
        final_score = results['timesformer'] * 0.7 + results['heuristic']['score'] * 0.3
        candidates = results['heuristic']['candidates']
        mode = 'timesformer+heuristic'
    else:
        final_score = results['heuristic']['score']
        candidates = results['heuristic']['candidates']
        mode = 'heuristic_only'

    mean_intensity = float(image.mean())
    print(f'[INK] score={final_score:.4f} candidates={candidates} mean={mean_intensity:.4f} mode={mode}')

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
