import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, data, color

image = color.rgb2gray(data.astronaut())

lbp = feature.local_binary_pattern(image, P=8, R=1)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(lbp, cmap='gray')
ax[1].set_title('Local Binary Patterns (LBP)')
plt.show()

hist, _ = np.histogram(lbp, bins=np.arange(0, lbp.max() + 2), density=True)
plt.bar(range(len(hist)), hist, width=0.8, align='center')
plt.title('Histogram of Local Binary Patterns (HLBP)')
plt.xlabel('Pattern')
plt.ylabel('Normalized Frequency')
plt.show()
