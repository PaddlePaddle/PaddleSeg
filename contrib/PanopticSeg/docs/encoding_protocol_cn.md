[English](encoding_protocol_en.md) | 简体中文

# 全景分割标签编码规则

本文档介绍本工具箱对全景分割标签的编码规则。

将全景分割图（例如参考的标签图或者模型预测输出）中的像素值记作 `pan_id`。假设共有 `c` 个类别，最大的对象个数是 `n`，那么 `pan_id` 的取值范围在 `0` 到 `c * label_divisor + n` 之间（左闭右闭）。

对于 thing 类别，`pan_id` 的计算方式如下：

```plain
pan_id = (cat_id + 1) * label_divisor + ins_id
```

其中，`cat_id` 是类别的 ID（从 `0` 开始计数），而 `ins_id` 是实例的 ID（从 `1` 开始计数）。

对于 stuff 类别，`pan_id` 的计算方式如下：

```plain
pan_id = (cat_id + 1) * label_divisor
```

对于未知的像素（无法确定结果或未标注），`pan_id` 取值为 `0`。

`label_divisor` 是一个预设的常数，其默认值为 `1000`。
