import matplotlib.pyplot as plt
import cv2

# Maskeyi yükle (grayscale olarak)
mask = cv2.imread('data/duke_original/lesion/Subject_08_19.png', cv2.IMREAD_GRAYSCALE)

# Kontrastı artırmak için histogram eşitleme
enhanced = cv2.equalizeHist(mask)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Original Mask")
plt.imshow(mask, cmap='gray')
plt.subplot(1,2,2)
plt.title("Increased Contrast")
plt.imshow(enhanced, cmap='gray')
plt.show()
