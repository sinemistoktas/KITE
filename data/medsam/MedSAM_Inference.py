import numpy as np
import torch
import torch.nn.functional as F
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from skimage import transform, io
from PIL import Image
from io import BytesIO
import sys, os
sys.path.append(os.path.dirname(__file__))  # ✅ makes segment_anything importable
from segment_anything import sam_model_registry

# Configuration
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'model', 'medsam_vit_b.pth')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MedSAM model once
medsam_model = sam_model_registry["vit_b"](checkpoint=CHECKPOINT_PATH).to(DEVICE)
medsam_model.eval()

@torch.no_grad()
def run_medsam(image: np.ndarray, box: list) -> BytesIO:
    """
    Run MedSAM segmentation on a given image with a bounding box prompt.

    Args:
        image (np.ndarray): Input image (H, W, 3) or (H, W).
        box (list): Bounding box in format [x0, y0, x1, y1] in original image scale.

    Returns:
        BytesIO: PNG image of segmentation mask.
    """
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=-1)
    H, W, _ = image.shape

    # Preprocess image
    img_1024 = transform.resize(
        image, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), 1e-8, None)
    img_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    image_embedding = medsam_model.image_encoder(img_tensor)

    # Normalize bounding box
    box_np = np.array([box])
    box_1024 = box_np / np.array([W, H, W, H]) * 1024
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=DEVICE)[:, None, :]

    sparse_emb, dense_emb = medsam_model.prompt_encoder(points=None, boxes=box_torch, masks=None)
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
    )

    pred = torch.sigmoid(low_res_logits)
    pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
    mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    # Convert to PNG in-memory
    output_image = Image.fromarray(mask)
    buffer = BytesIO()
    output_image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer
