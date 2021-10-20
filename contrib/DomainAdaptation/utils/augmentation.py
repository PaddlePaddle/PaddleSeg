import numpy as np
import cv2
import paddle
import albumentations as al
import paddleseg.transforms as trans


def get_augmentation():
    return al.Compose([
        al.RandomResizedCrop(512, 512, scale=(0.2, 1.)),
        al.Compose([
            al.RandomBrightnessContrast(p=1),
            al.HueSaturationValue(p=1),
        ],
                   p=0.8),
        al.ToGray(p=0.2),
        al.GaussianBlur(5, p=0.5),
    ])


# def get_augmentation_paddle():
#     return trans.Compose([
#         trans.RandomScaleAspect(
#             min_scale=0.2,
#             aspect_ratio=0.75),  # not to 512, 512, but the origin size
#         trans.RandomDistort(
#             brightness_range=0.2,
#             brightness_prob=0.8,
#             contrast_range=0.2,
#             contrast_prob=0.8,
#             hue_range=20,
#             hue_prob=0.8,
#             saturation_prob=0.8,
#             saturation_range=0.2),  # skip to gray
#         trans.RandomBlur(prob=0.5)
#     ]  # maxkernel should be 5
#                          )


def augment(images, labels, aug, logger):
    """Augments both image and label. Assumes input is a tensor with
       a batch dimension and values normalized to N(0,1)."""
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434),
                        dtype=np.float32)
    # Transform label shape: B, C, W, H ==> B, W, H, C
    labels_are_3d = (len(labels.shape) == 4)
    # logger.add("olabel", labels.cpu().detach().numpy())
    if labels_are_3d:
        labels = labels.permute(0, 2, 3, 1)

    # Transform each image independently. This is slow, but whatever.
    aug_images, aug_labels = [], []
    for image, label in zip(images, labels):

        # Step 1: Undo normalization transformation, convert to numpy
        image = cv2.cvtColor(
            image.numpy().transpose(1, 2, 0) + IMG_MEAN,
            cv2.COLOR_BGR2RGB).astype(
                np.uint8)  # todo: check shape cmp with pixmatch
        label = label.numpy()  # convert to np
        label = label.astype('int64')
        # logger.add("uimage", image)
        # logger.add("ulabel", label)

        # Step 2: Perform transformations on numpy images
        data = aug(image=image, mask=label)
        image, label = data['image'], data['mask']
        # logger.add("augimage", image)
        # logger.add("auglabel", label)

        # Step 3: Convert back to PyTorch tensors
        image = paddle.to_tensor((cv2.cvtColor(
            image.astype(np.float32), cv2.COLOR_RGB2BGR) - IMG_MEAN).transpose(
                2, 0, 1))
        # label = np.where(label==-1, 255, label)
        label = paddle.to_tensor(label)
        if not labels_are_3d:
            label = label.astype('int64')
        # logger.add("tensorimage", image.cpu().detach().numpy())
        # logger.add("tensorlabel", label.cpu().detach().numpy())

        # Add to list
        aug_images.append(image)
        aug_labels.append(label)

    # Stack
    images = paddle.stack(aug_images, axis=0)
    labels = paddle.stack(aug_labels, axis=0)

    # Transform label shape back: B, W, H, C ==> B, C, W, H
    if labels_are_3d:
        labels = labels.permute(0, 3, 1, 2)
    return images, labels
