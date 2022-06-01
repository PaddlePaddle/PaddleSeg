import paddle as torch
import paddle
import paddle.nn.functional as F

import math

def dense_patch_slices(image_size, patch_size, scan_interval):
    """
    Enumerate all slices defining 2D/3D patches of size `patch_size` from an `image_size` input image.

    Args:
        image_size (tuple of int): dimensions of image to iterate over
        patch_size (tuple of int): size of patches to generate slices
        scan_interval (tuple of int): dense patch sampling interval

    Returns:
        a list of slice objects defining each patch
    """
    num_spatial_dims = len(image_size)
    if num_spatial_dims not in (2, 3):
        raise ValueError('image_size should has 2 or 3 elements')
    patch_size =  patch_size
    scan_interval = scan_interval

    scan_num = [int(math.ceil(float(image_size[i]) / scan_interval[i])) if scan_interval[i] != 0 else 1
                for i in range(num_spatial_dims)]
    slices = []
    if num_spatial_dims == 3:
        for i in range(scan_num[0]):
            start_i = i * scan_interval[0]
            start_i -= max(start_i + patch_size[0] - image_size[0], 0)
            slice_i = slice(start_i, start_i + patch_size[0])

            for j in range(scan_num[1]):
                start_j = j * scan_interval[1]
                start_j -= max(start_j + patch_size[1] - image_size[1], 0)
                slice_j = slice(start_j, start_j + patch_size[1])

                for k in range(0, scan_num[2]):
                    start_k = k * scan_interval[2]
                    start_k -= max(start_k + patch_size[2] - image_size[2], 0)
                    slice_k = slice(start_k, start_k + patch_size[2])
                    slices.append((slice_i, slice_j, slice_k))
    else:
        for i in range(scan_num[0]):
            start_i = i * scan_interval[0]
            start_i -= max(start_i + patch_size[0] - image_size[0], 0)
            slice_i = slice(start_i, start_i + patch_size[0])

            for j in range(scan_num[1]):
                start_j = j * scan_interval[1]
                start_j -= max(start_j + patch_size[1] - image_size[1], 0)
                slice_j = slice(start_j, start_j + patch_size[1])
                slices.append((slice_i, slice_j))
    return slices



def sliding_window_inference(inputs, roi_size, sw_batch_size, predictor):
    """Use SlidingWindow method to execute inference.

    Args:
        inputs (torch Tensor): input image to be processed (assuming NCHW[D])
        roi_size (list, tuple): the window size to execute SlidingWindow inference.
        sw_batch_size (int): the batch size to run window slices.
        predictor (Callable): given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
            should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
            where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.

    Note:
        must be channel first, support both 2D and 3D.
        input data must have batch dim.
        execute on 1 image/per inference, run a batch of window slices of 1 input image.
    """
    num_spatial_dims = len(inputs.shape) - 2
    assert len(roi_size) == num_spatial_dims, 'roi_size {} does not match input dims.'.format(roi_size)

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    # TODO: Enable batch sizes > 1 in future
    if batch_size > 1:
        raise NotImplementedError

    original_image_size = [image_size[i] for i in range(num_spatial_dims)]
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = [i for k in range(len(inputs.shape) - 1, 1, -1) for i in (0, max(roi_size[k - 2] - inputs.shape[k], 0))]
    inputs = F.pad(inputs, pad=pad_size, mode='constant', value=0,data_format="NDHWC")

    # TODO: interval from user's specification
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)

    slice_batches = []
    for slice_index in range(0, len(slices), sw_batch_size):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))
        input_slices = []
        for curr_index in slice_index_range:
            if num_spatial_dims == 3:
                slice_i, slice_j, slice_k = slices[curr_index]
                input_slices.append(inputs[0, :, slice_i, slice_j, slice_k])
            else:
                slice_i, slice_j = slices[curr_index]
                input_slices.append(inputs[0, :, slice_i, slice_j])
        slice_batches.append(torch.stack(input_slices))

    # Perform predictions
    output_rois = list()
    for data in slice_batches:
        seg_prob = predictor(data)  # batched patch segmentation

        output_rois.append(seg_prob[0].numpy())

    # stitching output image

    output_classes = output_rois[0].shape[1]
    output_shape = [batch_size, output_classes] + list(image_size)

    # allocate memory to store the full output and the count for overlapping parts
    output_image = torch.zeros(output_shape, dtype=torch.float32).numpy()
    count_map = torch.zeros(output_shape, dtype=torch.float32).numpy()


    for window_id, slice_index in enumerate(range(0, len(slices), sw_batch_size)):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))
        # store the result in the proper location of the full output
        for curr_index in slice_index_range:
            if num_spatial_dims == 3:
                slice_i, slice_j, slice_k = slices[curr_index]
                ors=output_rois[window_id][curr_index - slice_index, :]


                output_image[0, :, slice_i, slice_j, slice_k] += ors


                count_map[0, :, slice_i, slice_j, slice_k] += 1.
            else:
                slice_i, slice_j = slices[curr_index]
                output_image[0, :, slice_i, slice_j] += output_rois[window_id][curr_index - slice_index, :]
                count_map[0, :, slice_i, slice_j] += 1.

    # account for any overlapping sections
    output_image /= count_map

    output_image=paddle.to_tensor(output_image) 


    if num_spatial_dims == 3:
        return (output_image[..., :original_image_size[0], :original_image_size[1], :original_image_size[2]],)
    return (output_image[..., :original_image_size[0], :original_image_size[1]] ,) # 2D



def _get_scan_interval(image_size, roi_size, num_spatial_dims):
    assert (len(image_size) == num_spatial_dims), 'image coord different from spatial dims.'
    assert (len(roi_size) == num_spatial_dims), 'roi coord different from spatial dims.'

    scan_interval = [1 for _ in range(num_spatial_dims)]
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval[i] = int(roi_size[i])
        else:
            # this means that it's r-16 (if r>=64) and r*0.75 (if r<=64)
            scan_interval[i] = int(max(roi_size[i] - 16, roi_size[i] * 0.75))
    return tuple(scan_interval)