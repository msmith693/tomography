import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import (
    exposure,
    feature,
    filters,
    io,
    measure,
    morphology,
    restoration,
    segmentation,
    transform,
    util,
)
import napari

nuclei = io.imread("images/cells.tif")
membranes = io.imread("images/cells_membrane.tif")

print("shape: {}".format(nuclei.shape))
print("dtype: {}".format(nuclei.dtype))
print("range: ({}, {})".format(np.min(nuclei), np.max(nuclei)))

# original spacing reported by microscope in micrometres
# here, the 0.29 micrometre distance is the distance between image slices
original_spacing = np.array([0.29, 0.065, 0.065])

# each slice was downsampled by 4x to make data smaller
rescaled_spacing = original_spacing * [1, 4, 4]

# normalising the colun row spacng so pixels are a distance of 1 apart
spacing = rescaled_spacing / rescaled_spacing[2]


# Helper function for plotting histograms.
def plot_hist(ax, data, title=None):
    ax.hist(data.ravel(), bins=256)  # 256 bins for 256 intensity vals
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    if title:
        ax.set_title(title)


equalized = exposure.equalize_hist(nuclei)  # redistributes image intensities

# fig, ((a, b), (c, d)) = plt.subplots(nrows=2, ncols=2)

# plot_hist(a, nuclei, title="Original")
# plot_hist(b, equalized, title="Histogram equalization")

# cdf, bins = exposure.cumulative_distribution(nuclei.ravel())
# c.plot(bins, cdf, "r")
# c.set_title("Original CDF")

# cdf, bins = exposure.cumulative_distribution(equalized.ravel())  #cumulative intensity values along x axis
# d.plot(bins, cdf, "r")
# d.set_title("Histogram equalization CDF");

# fig.tight_layout()
# plt.show()

# removing salt and pepper noise (extrema intensity vals)
vmin, vmax = np.quantile(nuclei, q=(0.005, 0.995))

stretched = exposure.rescale_intensity(
    nuclei, in_range=(vmin, vmax), out_range=np.float32
)

# edge detection filter
edges = filters.sobel(nuclei)

# thresholding (image segmentation)
denoised = ndi.median_filter(nuclei, size=3)  # picks out median value in 3 pixel window
li_thresholded = denoised > filters.threshold_li(
    denoised
)  # takes every array value in denoised that is larger than threshold val and 0 intensity for other vals

filled = ndi.binary_fill_holes(li_thresholded)  # fill thresholding holes

# morphology operations to remove small holes or bright spots
width = 20  # min width of accepted (non removed) holes or bright objects

remove_holes = morphology.remove_small_holes(
    filled,
    area_threshold=width
    ** 3,  # approximated as cube - we are still dealing with 3d image array!
)

remove_objects = morphology.remove_small_objects(  # remove further objects (beright spots) with morphology
    remove_holes, min_size=width**3
)

# image segmentation and labeling
labels = measure.label(remove_objects)

# Display calculated BAD marker location for watershed segmentation
transformed = ndi.distance_transform_edt(remove_objects, sampling=spacing)
maxima = morphology.local_maxima(transformed)

viewer = napari.Viewer()
# viewer.add_image(nuclei, contrast_limits=[0, 1],
#                            scale=spacing)
# viewer.add_image(equalized, contrast_limits=[0, 1], name='histeq')
# viewer.add_image(stretched, contrast_limits=[0, 1], name='stretched')
# #blends edges with nuclei image additively (add intensities)
viewer.add_image(nuclei, blending="additive", colormap="green", name="nuclei")
viewer.add_image(
    edges, blending="additive", colormap="magenta", contrast_limits=[0, 1], name="edges"
)
# viewer.add_image(denoised, contrast_limits=[0, 1], name='denoised')
viewer.add_image(li_thresholded, name="thresholded", opacity=0.3)
viewer.add_image(filled, name="filled", opacity=0.3)
viewer.add_image(remove_objects, name="cleaned", opacity=0.3)
viewer.add_labels(labels, name="labels")
viewer.add_points(np.transpose(np.nonzero(maxima)), name="bad points")

# cleans bad points layer (could just delete it and recreate new area, or hide it in gui)
viewer.layers["bad points"].visible = False
points = viewer.add_points(name="interactive points", ndim=3)
points.mode = "add"
# nuclei locations
points.data = np.array(
    [
        [30.0, 14.2598685, 27.7741219],
        [30.0, 30.10416663, 81.36513029],
        [30.0, 13.32785096, 144.27631406],
        [30.0, 46.8804823, 191.80920846],
        [30.0, 43.15241215, 211.84758551],
        [30.0, 94.87938547, 160.12061219],
        [30.0, 72.97697335, 112.58771779],
        [30.0, 138.21820096, 189.01315585],
        [30.0, 144.74232372, 242.60416424],
        [30.0, 98.14144685, 251.92433962],
        [30.0, 153.59649032, 112.58771779],
        [30.0, 134.49013081, 40.35635865],
        [30.0, 182.95504275, 48.74451649],
        [30.0, 216.04166532, 80.89912152],
        [30.0, 235.14802483, 130.296051],
        [30.0, 196.00328827, 169.44078757],
        [30.0, 245.86622651, 202.06140137],
        [30.0, 213.71162148, 250.52631331],
        [28.0, 87.42324517, 52.00657787],
    ],
    dtype=float,
)
marker_locations = points.data

markers = np.zeros(nuclei.shape, dtype=np.uint32)
marker_indices = tuple(np.round(marker_locations).astype(int).T)
markers[marker_indices] = np.arange(len(marker_locations)) + 1
markers_big = morphology.dilation(markers, morphology.ball(5))  # increase marker size

segmented = segmentation.watershed(edges, markers_big, mask=remove_objects)

viewer.add_labels(segmented, name="segmented")
napari.run()

