import numpy as np

def _load_image(str_path, resize):
    """Load and preprocess the quadcopter image."""
    from PIL import Image
    
    img = Image.open(str_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize(resize, Image.Resampling.LANCZOS)

    return np.array(img, dtype=np.uint8)

def _draw_rotated_image(canvas, image, position=(0, 0), rotation=0, pixels_per_meter=1):
    """Draw a rotated image on the canvas at the specified position."""
    from PIL import Image
    
    pil_image = Image.fromarray(image)
    rotated_image = pil_image.rotate(np.degrees(rotation), expand=False, resample=Image.Resampling.BICUBIC)
    rotated_array = np.array(rotated_image, dtype=np.uint8)
    
    img_h, img_w = rotated_array.shape[:2]
    
    px = int(position[0] * pixels_per_meter + canvas.shape[1] // 2)
    py = int(canvas.shape[0] // 2 - position[1] * pixels_per_meter)
    
    x1 = max(0, px - img_w // 2)
    y1 = max(0, py - img_h // 2)
    x2 = min(canvas.shape[1], px + img_w // 2)
    y2 = min(canvas.shape[0], py + img_h // 2)
    
    # Bounds check
    if x1 >= x2 or y1 >= y2:
        # Image is completely outside canvas bounds
        return canvas
    
    img_x1 = max(0, img_w // 2 - px)
    img_y1 = max(0, img_h // 2 - py)
    img_x2 = img_x1 + (x2 - x1)
    img_y2 = img_y1 + (y2 - y1)
    
    # Blend image onto canvas TODO: Fix
    if rotated_array.shape[2] == 4:  # RGBA
        alpha = rotated_array[:, :, 3] / 255.0
        print(alpha)
        for c in range(3):
            canvas[y1:y2, x1:x2, c] = (
                canvas[y1:y2, x1:x2, c] * (1 - alpha[img_y1:img_y2, img_x1:img_x2]) +
                rotated_array[img_y1:img_y2, img_x1:img_x2, c] * alpha[img_y1:img_y2, img_x1:img_x2]
            ).astype(np.uint8)
    else:  # RGB
        canvas[y1:y2, x1:x2] = rotated_array[img_y1:img_y2, img_x1:img_x2]
    
    return canvas
