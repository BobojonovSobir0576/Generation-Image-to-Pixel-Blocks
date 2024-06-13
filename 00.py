import numpy as np
import matplotlib.pyplot as plt

def pixelate_rgb(img, window):
    n, m, _ = img.shape
    n, m = n - n % window, m - m % window
    img1 = np.zeros((n, m, 3))
    for x in range(0, n, window):
        for y in range(0, m, window):
            img1[x:x+window, y:y+window] = img[x:x+window, y:y+window].mean(axis=(0, 1))
    return img1

img = plt.imread('pic-main_DcUXjJE.jpg')

pixelated_image_5 = pixelate_rgb(img, 5)
pixelated_image_10 = pixelate_rgb(img, 10)
pixelated_image_20 = pixelate_rgb(img, 20)
pixelated_image_30 = pixelate_rgb(img, 30)

# Normalize the images to ensure RGB values are in the range [0, 1]
pixelated_image_5 = np.clip(pixelated_image_5, 0, 1)
pixelated_image_10 = np.clip(pixelated_image_10, 0, 1)
pixelated_image_20 = np.clip(pixelated_image_20, 0, 1)
pixelated_image_30 = np.clip(pixelated_image_30, 0, 1)

# Save pixelated images as JPEG or PNG files
plt.imsave('pixelated_image_5.jpg', pixelated_image_5)
plt.imsave('pixelated_image_10.jpg', pixelated_image_10)
plt.imsave('pixelated_image_20.jpg', pixelated_image_20)
plt.imsave('pixelated_image_30.jpg', pixelated_image_30)

print("Pixelated images saved as JPEG files.")
