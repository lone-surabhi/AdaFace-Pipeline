import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src import net
from src.align import get_aligned_face

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CKPTS = {
    "ir_101": os.path.join(BASE_DIR, "pretrained", "adaface_ir101_ms1mv2.ckpt"),
}

def load_adaface(arch="ir_101", ckpt_path=None, device=None):
    assert arch in CKPTS or ckpt_path, f"Unknown arch {arch} and no ckpt provided."
    ckpt_path = ckpt_path or CKPTS[arch]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = net.build_model(arch).to(device).eval()
    print(f"[load] backbone={arch}  ckpt={ckpt_path}  device={device}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    clean = {}
    for k, v in state.items():
        if k.startswith("model."):   k = k[6:]
        if k.startswith("module."):  k = k[7:]
        k = k.replace(".res_layer.", ".res.")
        clean[k] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        print("[warn] missing keys (first 10):", missing[:10])
    if unexpected:
        print("[warn] unexpected keys (first 10):", unexpected[:10])
    return model, device

def to_input_tensor(pil_rgb_112):
    arr = np.array(pil_rgb_112).astype(np.float32) / 255.0  # HxWx3 RGB [0,1]
    bgr = arr[:, :, ::-1]                                   # RGB->BGR
    bgr = (bgr - 0.5) / 0.5
    x = torch.from_numpy(bgr).permute(2, 0, 1).unsqueeze(0) # 1x3x112x112
    return x.float()

@torch.inference_mode()
def embed_path(model, device, image_path):
    pil = Image.open(image_path).convert("RGB")
    aligned = get_aligned_face(rgb_pil_image=pil)
    if aligned is None:
        return None
    x = to_input_tensor(aligned).to(device)
    f, _ = model(x)                 
    f = F.normalize(f, p=2, dim=1) 
    return f[0].cpu().numpy()

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float((a * b).sum())

def compare_two(model, device, p1, p2):
    f1 = embed_path(model, device, p1)
    f2 = embed_path(model, device, p2)
    if f1 is None or f2 is None:
        return None
    return cosine(f1, f2)

def main():
    ap = argparse.ArgumentParser(description="AdaFace pipeline: align -> embed -> cosine similarity")
    ap.add_argument("path1", help="image file OR folder")
    ap.add_argument("path2", help="image file OR folder")
    ap.add_argument("--arch", default="ir_101", choices=list(CKPTS.keys()))
    ap.add_argument("--ckpt", default=None, help="override checkpoint path")
    args = ap.parse_args()

    model, device = load_adaface(args.arch, args.ckpt)

    p1, p2 = args.path1, args.path2
    if os.path.isfile(p1) and os.path.isfile(p2):
        sim = compare_two(model, device, p1, p2)
        if sim is None:
            print("[error] face not detected in one of the images.")
        else:
            print(f"{os.path.basename(p1)} vs {os.path.basename(p2)}  cosine: {sim:.4f}")
    elif os.path.isdir(p1) and os.path.isdir(p2):
        names1 = sorted([f for f in os.listdir(p1) if os.path.isfile(os.path.join(p1, f))])
        names2 = sorted([f for f in os.listdir(p2) if os.path.isfile(os.path.join(p2, f))])
        n = min(len(names1), len(names2))
        if n == 0:
            print("[error] one of the folders is empty.")
            return
        for i in range(n):
            a = os.path.join(p1, names1[i])
            b = os.path.join(p2, names2[i])
            sim = compare_two(model, device, a, b)
            tag = f"{names1[i]} vs {names2[i]}"
            if sim is None:
                print(f"{tag}: [no face]")
            else:
                print(f"{tag}: {sim:.4f}")
    else:
        print("[error]")

if __name__ == "__main__":
    main()