import os, argparse, numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc

from src import net
from src.align import get_aligned_face  

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model(arch="ir_101", ckpt_path=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if ckpt_path is None:
        ckpt_path = os.path.join(BASE_DIR, "pretrained", "adaface_ir101_ms1mv2.ckpt")
    model = net.build_model(arch).to(device).eval()
    print(f"[load] backbone={arch}  ckpt={ckpt_path}  device={device}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    clean = {}
    for k, v in state.items():
        if k.startswith("model."):   k = k[6:]
        if k.startswith("module."):  k = k[7:]
        k = k.replace(".res_layer.", ".res.")
        k = k.replace(".shortcut_layer.", ".shortcut")
        k = k.replace(".shortcut0.", ".shortcut.0.")
        k = k.replace(".shortcut1.", ".shortcut.1.")
        clean[k] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:    print("[warn] missing keys (first 10):", missing[:10])
    if unexpected: print("[warn] unexpected keys (first 10):", unexpected[:10])
    return model, device

def to_input_tensor(pil_rgb_112):
    arr = np.array(pil_rgb_112).astype(np.float32) / 255.0
    bgr = arr[:, :, ::-1]       # RGB->BGR
    bgr = (bgr - 0.5) / 0.5
    return torch.from_numpy(bgr).permute(2,0,1).unsqueeze(0).float()

@torch.inference_mode()
def embed_pre_aligned(model, device, pil_rgb_112):
    x = to_input_tensor(pil_rgb_112).to(device)
    f, _ = model(x)
    f = F.normalize(f, p=2, dim=1)
    return f[0].cpu().numpy()

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float((a * b).sum())

def pick_threshold(scores, labels):
    ths = np.linspace(-1, 1, 4001)
    best_acc, best_thr = 0.0, 0.0
    for t in ths:
        acc = ((scores > t).astype(int) == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, t
    return best_thr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="Folder that contains the relative paths from the pairs file (e.g., .../data/agedb30)")
    ap.add_argument("--pairs", required=True,
                    help="Path to your annotation file, e.g., agedb_30_112x112/agedb_30_ann.txt")
    ap.add_argument("--arch", default="ir_101", choices=["ir_101"])
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    model, device = load_model(args.arch, args.ckpt)

    pairs = []
    with open(args.pairs, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) != 3: continue
            lab = int(parts[0])
            rel1, rel2 = parts[1], parts[2]
            p1 = os.path.join(args.data_root, rel1)
            p2 = os.path.join(args.data_root, rel2)
            pairs.append((p1, p2, lab))

    print(f"[info] loaded pairs: {len(pairs)}")
    if len(pairs) == 0:
        print("[error] No pairs loaded. Check --data_root and --pairs paths.")
        return

    scores, labs = [], []
    missing_files = 0
    ok = 0

    for p1, p2, lab in tqdm(pairs, desc="AgeDB eval (pre-aligned)"):
        if not (os.path.exists(p1) and os.path.exists(p2)):
            missing_files += 1
            continue

        f1 = embed_pre_aligned(model, device, Image.open(p1).convert("RGB"))
        f2 = embed_pre_aligned(model, device, Image.open(p2).convert("RGB"))
        s = cosine(f1, f2)
        scores.append(s); labs.append(lab)
        ok += 1

    print(f"[stats] total pairs: {len(pairs)}, used: {ok}, missing_files: {missing_files}")

    scores = np.array(scores); labs = np.array(labs, dtype=int)
    if len(scores) == 0:
        print("[error] No valid pairs processed. Double-check the relative paths in the pairs file.")
        return

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accs, aucs = [], []
    for tr, te in kf.split(scores):
        thr = pick_threshold(scores[tr], labs[tr])
        acc = ((scores[te] > thr).astype(int) == labs[te]).mean()
        fpr, tpr, _ = roc_curve(labs[te], scores[te])
        accs.append(acc); aucs.append(auc(fpr, tpr))

    print(f"AgeDB-30 10-fold ACC: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}% | AUC: {np.mean(aucs):.4f}")

if __name__ == "__main__":
    main()