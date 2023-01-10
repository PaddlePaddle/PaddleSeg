English | [简体中文](encoding_protocol_cn.md)

# Encoding Protocol

In this document, we introduce the protocol used by this toolkit to encode panoptic segmentation labels. 

Basically, every single pixel in a panoptic segmentation map (e.g. the reference labels or the model prediction) has the value `pan_id`. Given the number of classes `c` and the maximum number of instances `n`, `pan_id` ranges from `0` to `c * label_divisor + n`. 

For thing classes, there is:

```plain
pan_id = (cat_id + 1) * label_divisor + ins_id
```

where `cat_id` is the category ID (starts from `0`) and `ins_id` is the instance ID (starts from `1`).

For stuff classes, there is:

```plain
pan_id = (cat_id + 1) * label_divisor
```

For unknown labels, `pan_id` is equal to `0`.

`label_divisor` is a pre-defined constant. The default value of `label_divisor` is `1000`.