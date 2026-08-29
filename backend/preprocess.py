import base64
import io
import numpy as np
from PIL import Image

def preprocess_canvas_image(base64_str: str) -> np.ndarray:
    """
    Preprocess raw base64 PNG canvas image to standard 28x28 MNIST format:
    1. Decode base64 PNG image string.
    2. Convert to grayscale ('L' mode).
    3. Invert colors (MNIST expects white digit on black background).
    4. Find stroke bounding box using NumPy array indexing.
    5. Resize preserving aspect ratio to fit inside 20x20 frame using PIL BILINEAR.
    6. Center resized stroke inside 28x28 array frame with padding.
    7. Normalize pixel values to [0.0, 1.0].
    8. Reshape to Keras batch format: (1, 28, 28, 1).
    """
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]

    image_bytes = base64.b64decode(base64_str)
    pil_img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img_arr = np.array(pil_img)

    # Invert image: Canvas has white background (255) and black stroke (0).
    # MNIST expects black background (0) and white digit stroke (255).
    inverted = 255 - img_arr

    # Threshold noise
    stroke_mask = inverted > 30
    y_indices, x_indices = np.where(stroke_mask)

    if len(y_indices) == 0 or len(x_indices) == 0:
        # Return blank 28x28 array if canvas is empty
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))
    min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))

    # Crop digit bounding box
    cropped = inverted[min_y:max_y+1, min_x:max_x+1]
    cropped_img = Image.fromarray(cropped)

    w, h = cropped_img.size
    max_dim = max(w, h)
    scale = 20.0 / max_dim
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Resize preserving aspect ratio
    resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_arr = np.array(resized_img, dtype=np.float32)

    # Create 28x28 black canvas array
    canvas_28 = np.zeros((28, 28), dtype=np.float32)

    # Center resized stroke inside 28x28 frame
    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2
    canvas_28[start_y:start_y+new_h, start_x:start_x+new_w] = resized_arr

    # Normalize pixels to range [0.0, 1.0]
    normalized = canvas_28 / 255.0

    # Reshape to Keras batch format (1, 28, 28, 1)
    input_tensor = np.expand_dims(normalized, axis=(0, -1)).astype(np.float32)
    return input_tensor
