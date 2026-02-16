from google.colab import files
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

uploaded = files.upload() 
filename = list(uploaded.keys())[0]
img = Image.open(filename)

img = np.array(img)

def convolve(image, kernel):
    kh, kw = kernel.shape
    h, w = image.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    result = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region * kernel)
    return result

# Step 3: Define filters
blur_kernel = np.ones((3,3)) / 9.0   # blur
edge_kernel = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]])  # edge detection
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])  # sharpen

# Step 4: Apply filters channel-wise (if color)
if img.ndim == 3:  # RGB image
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    # Blur
    blurred_r = convolve(r, blur_kernel)
    blurred_g = convolve(g, blur_kernel)
    blurred_b = convolve(b, blur_kernel)
    blurred = np.stack([blurred_r, blurred_g, blurred_b], axis=-1)

    # Edges
    edges_r = convolve(r, edge_kernel)
    edges_g = convolve(g, edge_kernel)
    edges_b = convolve(b, edge_kernel)
    edges = np.stack([edges_r, edges_g, edges_b], axis=-1)

    # Sharpen
    sharp_r = convolve(r, sharpen_kernel)
    sharp_g = convolve(g, sharpen_kernel)
    sharp_b = convolve(b, sharpen_kernel)
    sharpened = np.stack([sharp_r, sharp_g, sharp_b], axis=-1)
else:
    blurred = convolve(img, blur_kernel)
    edges = convolve(img, edge_kernel)
    sharpened = convolve(img, sharpen_kernel)
plt.figure(figsize=(14, 8))

plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(img)

plt.subplot(1, 4, 2)
plt.title("Blurred")
plt.imshow(np.clip(blurred, 0, 255).astype(np.uint8))

plt.subplot(1, 4, 3)
plt.title("Edges")
plt.imshow(np.clip(edges, 0, 255).astype(np.uint8))

plt.subplot(1, 4, 4)
plt.title("Sharpened")
plt.imshow(np.clip(sharpened, 0, 255).astype(np.uint8))

plt.show()







