import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ref = Image.open("Reference.jpg")
reference = np.array(ref)
im = Image.open("Source.jpg")
image = np.array(im)


def _match_cumulative_cdf(source, template):
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size
    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)


def match(image, reference):
    matched = np.empty(image.shape, dtype=image.dtype)
    for channel in range(image.shape[-1]):
        matched_channel = _match_cumulative_cdf(image[..., channel],
                                                reference[..., channel])
        matched[..., channel] = matched_channel
    return matched


matched = match(image, reference)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)
for aa in (ax1, ax2, ax3):
    aa.set_axis_off()
ax1.imshow(image)
ax1.set_title('Source')
ax2.imshow(reference)
ax2.set_title('Reference')
ax3.imshow(matched)
ax3.set_title('Matched')

plt.tight_layout()
plt.show()