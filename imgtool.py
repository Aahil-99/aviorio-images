"""Aviorio image tool. Free, open-source models only, CPU is fine.
  imgtool.py fetch   plan.json SRC         download every image in the plan
  imgtool.py audit   SRC OUT               metrics + near-duplicate groups + numbered contact sheet
  imgtool.py process plan.json SRC OUT     run each image's action, write final files + preview.jpg
Actions: keep | cut | studio | upscale | detext | detext+upscale
Models (all commercial-use licenses): BiRefNet-general via rembg (MIT), Real-ESRGAN x2plus and x4plus (BSD-3),
LaMa big-lama (Apache-2.0), RapidOCR (Apache-2.0)."""
import sys, os, json, time, glob, urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

MODELS = os.environ.get('MODELS', os.path.expanduser('~/models'))
ESRGAN = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
ESRGAN4 = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
LAMA = 'https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt'
SIZE = 2000  # output canvas for studio heroes, above Google's 1500 px advice

def model(url):
    os.makedirs(MODELS, exist_ok=True); p = os.path.join(MODELS, url.rsplit('/', 1)[1])
    if not os.path.exists(p): urllib.request.urlretrieve(url, p)
    return p

def font(n):
    for f in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']:
        if os.path.exists(f): return ImageFont.truetype(f, n)
    return ImageFont.load_default()

def load(path): return Image.open(path).convert('RGB')

def find(src, key):
    m = glob.glob(os.path.join(src, key + '.*')); return m[0] if m else None

# ---------- audit ----------
def sharpness(im):
    import cv2
    g = np.asarray(im.convert('L').resize((512, 512)), np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())

def dhash(im, n=16):
    a = np.asarray(im.convert('L').resize((n + 1, n)), np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()

def white_border(im):
    a = np.asarray(im.resize((200, 200)), np.int16)
    edge = np.concatenate([a[:6].reshape(-1, 3), a[-6:].reshape(-1, 3), a[:, :6].reshape(-1, 3), a[:, -6:].reshape(-1, 3)])
    return float((edge.min(1) > 235).mean())

def audit(src, out):
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR(); os.makedirs(out, exist_ok=True)
    files = sorted(f for f in glob.glob(os.path.join(src, '*')) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))
    rows, hashes = [], []
    for f in files:
        im = load(f); r, _ = ocr(np.asarray(im.resize((min(im.width, 1000), int(im.height * min(im.width, 1000) / im.width))))[:, :, ::-1])
        words = [t for _, t, _ in (r or [])]
        rows.append({'key': os.path.splitext(os.path.basename(f))[0], 'w': im.width, 'h': im.height,
                     'sharpness': round(sharpness(im)), 'white_border': round(white_border(im), 2),
                     'text_boxes': len(words), 'text': ' | '.join(words)[:160]})
        hashes.append(dhash(im))
    group = list(range(len(rows)))
    for i in range(len(rows)):
        for j in range(i):
            if (hashes[i] != hashes[j]).sum() <= 40: group[i] = group[j]; break
    for i, r in enumerate(rows): r['dup_group'] = group[i]
    json.dump(rows, open(os.path.join(out, 'audit.json'), 'w'), indent=1)
    T, cols = 320, 5; n = len(rows); sheet = Image.new('RGB', (cols * T, ((n + cols - 1) // cols) * (T + 46)), 'white')
    d = ImageDraw.Draw(sheet); F = font(16)
    for i, f in enumerate(files):
        im = load(f); im.thumbnail((T - 8, T - 8)); x, y = (i % cols) * T, (i // cols) * (T + 46)
        sheet.paste(im, (x + (T - im.width) // 2, y + 4)); r = rows[i]
        d.text((x + 6, y + T), f"#{i+1} {r['key'][:14]} {r['w']}px", fill='black', font=F)
        d.text((x + 6, y + T + 20), f"sharp {r['sharpness']} dup {r['dup_group']+1} txt {r['text_boxes']}", fill='black', font=F)
    sheet.save(os.path.join(out, 'contact_sheet.jpg'), quality=85)
    print(json.dumps(rows, indent=1))

# ---------- actions ----------
_rembg = None
def studio(im, top=(250, 249, 247), bottom=(232, 229, 225)):
    global _rembg
    from rembg import remove, new_session
    _rembg = _rembg or new_session('birefnet-general')
    cut = remove(im, session=_rembg, post_process_mask=True); cut = cut.crop(cut.getbbox())
    s = (SIZE * 0.80) / max(cut.size); cut = cut.resize((round(cut.width * s), round(cut.height * s)), Image.LANCZOS)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]; t = (yy / SIZE)[:, :, None]
    bg = np.array(top) * (1 - t) + np.array(bottom) * t
    r = np.sqrt(((xx - SIZE / 2) / (SIZE * .75)) ** 2 + ((yy - SIZE * .42) / (SIZE * .75)) ** 2)
    bg = bg * (1 - 0.06 * np.clip(r, 0, 1)[:, :, None])
    x = (SIZE - cut.width) // 2; y = int(SIZE * 0.88) - cut.height
    a = np.asarray(cut)[:, :, 3] / 255.0; rows = np.where(a.max(1) > .5)[0]; base = y + rows.max()
    cols = np.where(a[rows.max() - int(cut.height * .08):rows.max()].max(0) > .5)[0]; c0, c1 = x + cols.min(), x + cols.max()
    ell = np.clip(1 - (((xx - (c0 + c1) / 2) / ((c1 - c0) / 2 + 30)) ** 2 + ((yy - base - 6) / 22) ** 2), 0, 1)
    ell = np.asarray(Image.fromarray((ell * 255).astype('uint8')).filter(ImageFilter.GaussianBlur(22))) / 255.0
    bg = bg * (1 - np.clip(.5 * ell, 0, .5))[:, :, None]
    reg = bg[y:y + cut.height, x:x + cut.width]
    bg[y:y + cut.height, x:x + cut.width] = reg * (1 - a[:, :, None]) + np.asarray(cut)[:, :, :3] * a[:, :, None]
    return Image.fromarray(np.clip(bg, 0, 255).astype('uint8'))

_esr = {}
def upscale(im):
    """Free AI upscale. Short side under 800 px: 4x. Under 1500 px: 2x. Otherwise unchanged."""
    import torch; from spandrel import ModelLoader
    short = min(im.size)
    if short >= 1500: return im
    k = 4 if short < 800 else 2
    if k not in _esr: _esr[k] = ModelLoader().load_from_file(model(ESRGAN4 if k == 4 else ESRGAN)).eval()
    net = _esr[k]
    x = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None]
    H, W = x.shape[2:]; out = torch.zeros(1, 3, H * k, W * k); T, p = 200, 10
    with torch.no_grad():
        for y0 in range(0, H, T):
            for x0 in range(0, W, T):
                ya, xa, yb, xb = max(0, y0 - p), max(0, x0 - p), min(H, y0 + T + p), min(W, x0 + T + p)
                o = net(x[:, :, ya:yb, xa:xb]); h, w = min(T, H - y0), min(T, W - x0)
                out[:, :, y0 * k:(y0 + h) * k, x0 * k:(x0 + w) * k] = o[:, :, (y0 - ya) * k:(y0 - ya + h) * k, (x0 - xa) * k:(x0 - xa + w) * k]
    res = Image.fromarray((out[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype('uint8'))
    if max(res.size) > 3000: res.thumbnail((3000, 3000), Image.LANCZOS)
    return res

_ocr = _lama = None
def detext(im, boxes=None):
    """Remove text found by OCR. boxes: optional [[x0,y0,x1,y1],...] to limit removal to those areas."""
    global _ocr, _lama
    import cv2, torch; from rapidocr_onnxruntime import RapidOCR
    _ocr = _ocr or RapidOCR(); r, _ = _ocr(np.asarray(im)[:, :, ::-1])
    mask = np.zeros((im.height, im.width), np.uint8)
    for box, txt, _c in (r or []):
        b = np.array(box, np.int32); cx, cy = b[:, 0].mean(), b[:, 1].mean()
        if boxes and not any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in boxes): continue
        cv2.fillPoly(mask, [b], 255); print('  removing text:', txt)
    if not mask.any(): return im
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    _lama = _lama or torch.jit.load(model(LAMA), map_location='cpu').eval()
    H, W = im.height, im.width; ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    img = np.pad(np.asarray(im, np.float32) / 255, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    mk = np.pad((mask > 0).astype(np.float32), ((0, ph), (0, pw)))
    with torch.no_grad(): o = _lama(torch.from_numpy(img).permute(2, 0, 1)[None], torch.from_numpy(mk)[None, None])
    return Image.fromarray((o[0].permute(1, 2, 0).numpy()[:H, :W] * 255).clip(0, 255).astype('uint8'))

def zoom(before, after, path, key):
    """Side by side close-up of the busiest part of the photo, the way a shopper sees it when zooming:
    left is the current photo enlarged as a browser would, right is the new one at the same spot."""
    import cv2
    g = cv2.Laplacian(np.asarray(before.convert('L'), np.float32), cv2.CV_32F) ** 2
    c = max(64, min(before.size) // 4); best, bx, by = -1, 0, 0
    for y in range(0, before.height - c + 1, c // 2):
        for x in range(0, before.width - c + 1, c // 2):
            v = g[y:y + c, x:x + c].mean()
            if v > best: best, bx, by = v, x, y
    k = after.width / before.width; S = 640
    a = before.crop((bx, by, bx + c, by + c)).resize((S, S), Image.BICUBIC)
    b = after.crop((round(bx * k), round(by * k), round((bx + c) * k), round((by + c) * k))).resize((S, S), Image.LANCZOS)
    sheet = Image.new('RGB', (2 * S + 30, S + 60), 'white'); d = ImageDraw.Draw(sheet); F = font(24)
    d.text((10, 14), f"{key} now ({before.width}px), zoomed", fill='black', font=F)
    d.text((S + 20, 14), f"{key} after ({after.width}px), same spot", fill='black', font=F)
    sheet.paste(a, (10, 50)); sheet.paste(b, (S + 20, 50)); sheet.save(path, quality=90)

# Aviorio brand palette, from the live theme (Aviorio V2) settings: accent 1, accent 2, background.
BRAND = {'espresso': '422D23', 'bronze': '9A7646', 'cream': 'F2ECE2'}
INDOOR_STYLE = (' The room must clearly show the Aviorio brand palette: walls painted deep espresso brown (#422D23),'
                ' or warm cream (#F2ECE2) walls with espresso wood paneling; espresso and cream furniture and textiles;'
                ' brushed bronze (#9A7646) accents such as a side table, picture frames or a vase. Warm, cozy, upscale,'
                ' soft warm evening light, photorealistic interior design photography.')

NO_EXTRA = (' The scene must not contain any other lamp, light fixture, pole, stand or copy of the product,'
            ' so the real product is the only one in the picture.')
def photoroom(path, it):
    """Paid option ($0.10 per image on Photoroom's Plus plan, free and watermarked with a sandbox_ key).
    Runs only when a plan asks for it and the repo secret PHOTOROOM_API_KEY is set. Never called by default."""
    import requests, io
    key = os.environ.get('PHOTOROOM_API_KEY', '')
    if not key: raise SystemExit('PHOTOROOM_API_KEY repo secret is not set')
    if it.get('sandbox_only') and not key.startswith('sandbox_'): raise SystemExit('Plan is marked sandbox_only but the saved key is a live (paid) key, so nothing was sent')
    data = {'outputSize': it.get('size', '2000x2000'), 'padding': str(it.get('padding', 0.3 if it.get('prompt') else 0.1)),
            'lighting.mode': it.get('lighting', 'ai.auto'), 'shadow.mode': it.get('shadow', 'ai.soft')}
    # Brand rule: indoor products get the espresso and cream palette in the scene; outdoor scenes stay natural.
    indoor = it.get('setting', 'indoor') == 'indoor'
    if it.get('prompt'): data['background.prompt'] = it['prompt'] + (INDOOR_STYLE if indoor else '') + NO_EXTRA
    else: data['background.color'] = it.get('bg', BRAND['cream'] if indoor else 'F7F7F7')
    r = requests.post('https://image-api.photoroom.com/v2/edit', headers={'x-api-key': key},
                      files={'imageFile': open(path, 'rb')}, data=data, timeout=180)
    if r.status_code != 200: raise SystemExit(f'Photoroom error {r.status_code}: {r.text[:300]}')
    print('  photoroom', 'sandbox (watermarked)' if key.startswith('sandbox_') else 'live ($0.10)')
    return Image.open(io.BytesIO(r.content)).convert('RGB')

def process(plan_path, src, out):
    plan = json.load(open(plan_path)); os.makedirs(out, exist_ok=True); pairs = []
    for it in plan['images']:
        act = it['action']; f = find(src, it['key'])
        if act == 'cut' or not f: continue
        t = time.time(); im = load(f); res = im
        for step in act.split('+'):
            if step == 'studio': res = studio(res)
            elif step == 'studio_brand': res = studio(res, (242, 236, 226), (214, 199, 178))  # Aviorio cream backdrop
            elif step == 'photoroom': res = photoroom(f, it)
            elif step in ('upscale', 'sharpen'): res = upscale(res)
            elif step == 'detext': res = detext(res, it.get('text_boxes'))
        name = f"{it['position']:02d}-{it['role']}-{it['key']}.jpg"
        res.save(os.path.join(out, name), quality=92); it['output'] = name
        pairs.append((it['position'], im, res, f"{it['position']}. {it['role']}: {act}")); print(name, act, res.size, round(time.time() - t, 1), 's')
        if 'upscale' in act or 'sharpen' in act: zoom(im, res, os.path.join(out, f"zoom-{it['position']:02d}-{it['key']}.jpg"), it['key'])
    json.dump(plan, open(os.path.join(out, 'plan_result.json'), 'w'), indent=1)
    R = 420; F = font(22); pairs.sort(key=lambda p: p[0]); pairs = [p[1:] for p in pairs]
    sheet = Image.new('RGB', (2 * R + 40, len(pairs) * (R + 40)), 'white'); d = ImageDraw.Draw(sheet)
    for i, (a, b, lab) in enumerate(pairs):
        y = i * (R + 40); d.text((10, y + 6), lab, fill='black', font=F)
        for k, im in enumerate((a, b)):
            im = im.copy(); im.thumbnail((R, R)); sheet.paste(im, (10 + k * (R + 20), y + 36))
    sheet.save(os.path.join(out, 'preview.jpg'), quality=85)

def fetch(plan_path, src):
    os.makedirs(src, exist_ok=True)
    for it in json.load(open(plan_path))['images']:
        ext = '.png' if '.png' in it['url'].split('?')[0].lower() else '.jpg'
        req = urllib.request.Request(it['url'], headers={'User-Agent': 'Mozilla/5.0'})
        open(os.path.join(src, it['key'] + ext), 'wb').write(urllib.request.urlopen(req).read())

if __name__ == '__main__':
    cmd = sys.argv[1]
    {'audit': lambda: audit(sys.argv[2], sys.argv[3]), 'process': lambda: process(*sys.argv[2:5]),
     'fetch': lambda: fetch(sys.argv[2], sys.argv[3])}[cmd]()
