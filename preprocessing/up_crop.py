def get_crop(imageshape, person_center):
    crop = 513
    """Crop the image to the given maximum size. Use the person center."""
    if imageshape[0] > crop:
        crop_y = _np.array([int(_np.floor(person_center[1]) - _np.floor(crop / 2.)),
                           int(_np.floor(person_center[1]) + _np.ceil(crop / 2.))],
                           dtype='int')
        remaining_region_size = [crop_y[0], imageshape[0] - crop_y[1]]
        if remaining_region_size[0] < remaining_region_size[1]:
            if remaining_region_size[0] < 0 and remaining_region_size[1] > 0:
                crop_y += min(remaining_region_size[1], -remaining_region_size[0])
        else:
            if remaining_region_size[1] < 0 and remaining_region_size[0] > 0:
                crop_y -= min(remaining_region_size[0], -remaining_region_size[1])
        assert crop_y[1] - crop_y[0] == crop
    else:
        crop_y = [0, imageshape[0]]
    if imageshape[1] > crop:
        crop_x = _np.array([int(_np.floor(person_center[0]) - _np.floor(crop / 2.)),
                           int(_np.floor(person_center[0]) + _np.ceil(crop / 2.))],
                           dtype='int')
        remaining_region_size = [crop_x[0], imageshape[1] - crop_x[1]]
        if remaining_region_size[0] < remaining_region_size[1]:
            if remaining_region_size[0] < 0 and remaining_region_size[1] > 0:
                crop_x += min(remaining_region_size[1], -remaining_region_size[0])
        else:
            if remaining_region_size[1] < 0 and remaining_region_size[0] > 0:
                crop_x -= min(remaining_region_size[0], -remaining_region_size[1])
        assert crop_x[1] - crop_x[0] == crop
    else:
        crop_x = [0, imageshape[1]]
    return crop_y, crop_x


import sys
import numpy as _np

height = int(sys.argv[1])
width = int(sys.argv[2])
# person_center floor'ed computed as np.mean(joints[:2, joints[2, :] == 1], axis=1) * norm_factor
person_center_height = int(sys.argv[3])
person_center_width = int(sys.argv[4])

imageshape = _np.array((height, width))
person_center = _np.array((person_center_height, person_center_width))

crop_y, crop_x = get_crop(imageshape, person_center)

print(crop_y[0], crop_y[1], crop_x[0], crop_x[1])

# Use this crop_y and crop_x as follows:

# image = image[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1], :]

# https://github.com/classner/up/blob/master/segmentation/tools/create_dataset.py
# https://github.com/classner/up/blob/master/up_tools/model.py
