from PIL import Image
import torch
from facenet_pytorch import MTCNN

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_mtcnn = MTCNN(image_size=112, post_process=True, keep_all=False, device=_DEVICE)

def get_aligned_face(image_path=None, rgb_pil_image=None):
    if rgb_pil_image is None:
        if image_path is None:
            raise ValueError("Provide either image_path or rgb_pil_image.")
        img = Image.open(image_path).convert('RGB')
    else:
        if not isinstance(rgb_pil_image, Image.Image):
            raise TypeError("rgb_pil_image must be a PIL.Image")
        img = rgb_pil_image

    with torch.inference_mode():
        t = _mtcnn(img)  
    if t is None:
        return None

    arr = (t.permute(1, 2, 0).clamp(-1, 1).add(1).mul(127.5).byte().cpu().numpy())
    return Image.fromarray(arr, mode="RGB")