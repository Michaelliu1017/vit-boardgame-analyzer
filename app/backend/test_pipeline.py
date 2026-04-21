# test_pipeline.py
import torch
import torch.nn as nn
import timm
import numpy as np
import cv2
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from torchvision.ops import nms
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from collections import Counter

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# ── 加载模型 ──────────────────────────────────────────
print("加载 OWL-ViT...")
owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
owl_model     = OwlViTForObjectDetection.from_pretrained(
    "google/owlvit-large-patch14"
).to(device)
owl_model.eval()

print("加载 ViT 分类器...")
vit_ckpt    = torch.load("vit_classifier.pth",
                          map_location=device, weights_only=False)
vit_model   = timm.create_model(
    'vit_small_patch16_224', pretrained=False,
    num_classes=len(vit_ckpt['class_names'])
).to(device)
vit_model.load_state_dict(vit_ckpt['model_state'])
vit_model.eval()
class_names = vit_ckpt['class_names']
print(f"类别: {class_names}")

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

OWL_TEXTS = [[
    "plastic military figurine toy",
    "plastic artillery toy",
    "plastic tank toy",
    "plastic airplane toy",
    "plastic warship toy",
    "plastic anti air gun toy",
]]

# ── 读取图片 ──────────────────────────────────────────
image_path = sys.argv[1] if len(sys.argv) > 1 else "test/t5.jpg"
image      = Image.open(image_path).convert('RGB')
print(f"\n图片: {image_path}  尺寸: {image.size}")


# ════════════════════════════════════════════════════════
# 颜色判断阵营
# ════════════════════════════════════════════════════════
def get_faction_by_color(image, box):
    x1, y1, x2, y2 = [int(b.item()) for b in box]
    crop    = np.array(image.crop((x1, y1, x2, y2)))
    img_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    mask_jp = cv2.inRange(img_hsv, np.array([5,100,100]), np.array([20,255,255]))
    mask_us = cv2.inRange(img_hsv, np.array([25,40,40]),  np.array([45,255,255]))
    jp = int(mask_jp.sum())
    us = int(mask_us.sum())
    if jp == 0 and us == 0:
        return "unknown", jp, us
    return ("JP" if jp >= us else "US"), jp, us


# ════════════════════════════════════════════════════════
# 过滤包含其他框的大框
# ════════════════════════════════════════════════════════
def filter_containing_boxes(boxes, scores, labels, overlap_threshold=0.8):
    if len(boxes) == 0:
        return boxes, scores, labels

    keep = list(range(len(boxes)))

    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j or i not in keep or j not in keep:
                continue

            xi1 = max(boxes[i][0], boxes[j][0])
            yi1 = max(boxes[i][1], boxes[j][1])
            xi2 = min(boxes[i][2], boxes[j][2])
            yi2 = min(boxes[i][3], boxes[j][3])

            inter_w = max(0, xi2 - xi1)
            inter_h = max(0, yi2 - yi1)
            inter   = inter_w * inter_h

            area_j = ((boxes[j][2] - boxes[j][0]) *
                      (boxes[j][3] - boxes[j][1]))

            if area_j > 0 and inter / area_j > overlap_threshold:
                if i in keep:
                    keep.remove(i)

    keep = torch.tensor(keep)
    return boxes[keep], scores[keep], labels[keep]


# ════════════════════════════════════════════════════════
# Step 1: OWL-ViT 检测
# ════════════════════════════════════════════════════════
inputs = owl_processor(
    text=OWL_TEXTS, images=image, return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = owl_model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])
results      = owl_processor.post_process_grounded_object_detection(
    outputs, threshold=0.05, target_sizes=target_sizes
)[0]

boxes  = results["boxes"].cpu()
scores = results["scores"].cpu()
labels = results["labels"].cpu()

if len(boxes) > 0:
    # NMS
    keep   = nms(boxes, scores, iou_threshold=0.2)
    boxes  = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # 面积和长宽比过滤
    img_w, img_h = image.size
    img_area     = img_w * img_h
    valid = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        box_w        = x2 - x1
        box_h        = y2 - y1
        box_area     = box_w * box_h
        box_ratio    = box_area / img_area
        aspect_ratio = max(box_w, box_h) / (min(box_w, box_h) + 1e-6)
        # 0.005                0.3                    3.0
        if 0.001 < box_ratio < 0.4 and aspect_ratio < 4.0:
            valid.append(i)

    if valid:
        valid  = torch.tensor(valid)
        boxes  = boxes[valid]
        scores = scores[valid]
        labels = labels[valid]

    # 过滤包含其他框的大框
    boxes, scores, labels = filter_containing_boxes(boxes, scores, labels)

print(f"OWL-ViT 检测到 {len(boxes)} 个棋子")


# ════════════════════════════════════════════════════════
# Step 2: 颜色判断阵营
# ════════════════════════════════════════════════════════
factions    = []
color_stats = []
for box in boxes:
    faction, jp_px, us_px = get_faction_by_color(image, box)
    factions.append(faction)
    color_stats.append((jp_px, us_px))


# ════════════════════════════════════════════════════════
# Step 3: ViT 分类
# ════════════════════════════════════════════════════════
crops       = []
predictions = []
confidences = []

for box in boxes:
    x1, y1, x2, y2 = [int(b.item()) for b in box]
    crop   = image.crop((x1, y1, x2, y2))
    crops.append(crop)

    tensor = val_tf(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = vit_model(tensor)[0]
        probs  = torch.softmax(logits, dim=0)
        pred   = probs.argmax().item()
        conf   = probs[pred].item()

    predictions.append(class_names[pred])
    confidences.append(conf)


# ════════════════════════════════════════════════════════
# 可视化1: 原图 + 检测框
# ════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, figsize=(14, 10))
ax.imshow(image)

for i, (box, score, faction, pred) in enumerate(
        zip(boxes, scores, factions, predictions)):
    x1, y1, x2, y2 = [b.item() for b in box]
    color = 'orange' if faction == "JP" else 'lime'
    rect  = patches.Rectangle(
        (x1, y1), x2-x1, y2-y1,
        linewidth=2, edgecolor=color, facecolor='none'
    )
    ax.add_patch(rect)
    ax.text(
        x1, y1-8,
        f"#{i+1} [{faction}] {pred}",
        color=color, fontsize=9,
        bbox=dict(facecolor='black', alpha=0.7, pad=2)
    )

ax.set_title(f"OWL-ViT 检测结果 ({len(boxes)} 个棋子)", fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.savefig('result_detection.jpg', dpi=150, bbox_inches='tight')
plt.show()
print("检测结果保存到 result_detection.jpg")


# ════════════════════════════════════════════════════════
# 可视化2: 每个裁剪图 + ViT 分类结果
# ════════════════════════════════════════════════════════
n     = len(crops)
ncols = 4
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols,
                          figsize=(ncols * 3, nrows * 3.5))
axes = axes.flatten() if n > 1 else [axes]

for i, (crop, pred, conf, faction, (jp_px, us_px)) in enumerate(
        zip(crops, predictions, confidences, factions, color_stats)):

    axes[i].imshow(crop)

    # 绿色=颜色阵营和ViT预测一致，红色=不一致
    title_color = 'green' if (
        (pred.startswith("JP") and faction == "JP") or
        (pred.startswith("US") and faction == "US")
    ) else 'red'

    axes[i].set_title(
        f"#{i+1} {pred}\n"
        f"conf={conf:.2f}  [{faction}]\n"
        f"JP px:{jp_px}  US px:{us_px}",
        fontsize=8, color=title_color
    )
    axes[i].axis('off')

for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.suptitle("裁剪图 + ViT 分类\n绿=阵营一致  红=阵营不一致", fontsize=12)
plt.tight_layout()
plt.savefig('result_crops.jpg', dpi=150, bbox_inches='tight')
plt.show()
print("裁剪结果保存到 result_crops.jpg")


# ════════════════════════════════════════════════════════
# 打印详细统计
# ════════════════════════════════════════════════════════
print("\n══ 详细识别结果 ══")
print(f"{'#':<4} {'OWL label':<28} {'颜色阵营':<8} "
      f"{'ViT预测':<18} {'置信度':<8} {'一致?'}")
print("─" * 80)

for i, (box, score, label, faction, pred, conf) in enumerate(
        zip(boxes, scores, labels, factions, predictions, confidences)):
    owl_label  = OWL_TEXTS[0][label.item()]
    consistent = "✓" if (
        (pred.startswith("JP") and faction == "JP") or
        (pred.startswith("US") and faction == "US")
    ) else "✗ ← 注意"
    print(f"{i+1:<4} {owl_label:<28} {faction:<8} "
          f"{pred:<18} {conf:.3f}    {consistent}")

print("\n── ViT 分类汇总 ──")
counts = Counter(predictions)
for cls, cnt in sorted(counts.items()):
    faction = "JP进攻" if cls.startswith("JP") else "US防守"
    print(f"  {cls:<18} × {cnt}  ({faction})")

print("\n── 颜色阵营汇总 ──")
jp_total = sum(1 for f in factions if f == "JP")
us_total = sum(1 for f in factions if f == "US")
uk_total = sum(1 for f in factions if f == "unknown")
print(f"  JP (橙色):   {jp_total}")
print(f"  US (黄绿色): {us_total}")
if uk_total:
    print(f"  unknown:     {uk_total}")

