# test_pipeline.py
import torch, timm, cv2, sys, numpy as np
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from torchvision.ops import nms
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

# ════════════════════════════════════════════════════════
# global para
# ════════════════════════════════════════════════════════
IMAGE_PATH       = "test/t0.jpg"
                                                                # prev para set
OWL_THRESHOLD    = 0.05   # OWL-ViT segmentation confidence             # 0.05
NMS_IOU          = 0.1    # NMS overlap value，越低保留框越少              #  0.2 0.1
BOX_AREA_MIN     = 0.001  # Area filter lower bound                      # 0.0035  0.001
BOX_AREA_MAX     = 0.4    # Area filter upper bound                       # 0.4
ASPECT_RATIO_MAX = 10.0   # width & height filtering                      # 10
OVERLAP_THRESH   = 0.8    # Detection frame threshold，j有80%在i内则删i     #0.8
VIT_CONF_THRESH  = 0.7    # ViT classifier confidence level               # 0.7

# HSV color range
JP_HSV_LOW  = (5,  100, 100)   # orange 1
JP_HSV_HIGH = (20, 255, 255)   # orange 2
US_HSV_LOW  = (25, 40,  40)    # olive 1
US_HSV_HIGH = (45, 255, 255)   # olive 2

OWL_TEXTS = [[
    "plastic military figurine toy",
    "plastic vehicle toy",
    "plastic artillery toy",
    "plastic tank toy",
    "plastic airplane toy",
    "plastic warship toy",
    "plastic anti air gun toy",
]]

# ════════════════════════════════════════════════════════
# 加载模型
# ════════════════════════════════════════════════════════
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
owl_model     = OwlViTForObjectDetection.from_pretrained(
    "google/owlvit-large-patch14").to(device)
owl_model.eval()

vit_ckpt  = torch.load("vit_classifier.pth", map_location=device, weights_only=False)
vit_model = timm.create_model('vit_small_patch16_224', pretrained=False,
                               num_classes=len(vit_ckpt['class_names'])).to(device)
vit_model.load_state_dict(vit_ckpt['model_state'])
vit_model.eval()
CLASS_NAMES = vit_ckpt['class_names']
print(f"Classes: {CLASS_NAMES}")

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ════════════════════════════════════════════════════════
# 读取图片
# ════════════════════════════════════════════════════════
image_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
image      = Image.open(image_path).convert('RGB')
print(f"Image: {image_path}  Size: {image.size}")


# ════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════
def get_faction_by_color(image, box):
    x1,y1,x2,y2 = [int(b.item()) for b in box]
    crop    = np.array(image.crop((x1,y1,x2,y2)))
    hsv     = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    jp = int(cv2.inRange(hsv, np.array(JP_HSV_LOW),  np.array(JP_HSV_HIGH)).sum())
    us = int(cv2.inRange(hsv, np.array(US_HSV_LOW),  np.array(US_HSV_HIGH)).sum())
    if jp == 0 and us == 0: return "unknown", jp, us
    return ("JP" if jp >= us else "US"), jp, us


def filter_containing_boxes(boxes, scores, labels):
    if len(boxes) == 0: return boxes, scores, labels
    keep = list(range(len(boxes)))
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j or i not in keep or j not in keep: continue
            inter = (max(0, min(boxes[i][2],boxes[j][2]) - max(boxes[i][0],boxes[j][0])) *
                     max(0, min(boxes[i][3],boxes[j][3]) - max(boxes[i][1],boxes[j][1])))
            area_j = (boxes[j][2]-boxes[j][0]) * (boxes[j][3]-boxes[j][1])
            if area_j > 0 and inter / area_j > OVERLAP_THRESH:
                if i in keep: keep.remove(i)
    k = torch.tensor(keep)
    return boxes[k], scores[k], labels[k]


# ════════════════════════════════════════════════════════
# Step 1: OWL-ViT 检测
# ════════════════════════════════════════════════════════
inputs = owl_processor(text=OWL_TEXTS, images=image, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = owl_model(**inputs)

results = owl_processor.post_process_grounded_object_detection(
    outputs, threshold=OWL_THRESHOLD,
    target_sizes=torch.tensor([image.size[::-1]])
)[0]

boxes  = results["boxes"].cpu()
scores = results["scores"].cpu()
labels = results["labels"].cpu()
print(f"\n[Step 1] Raw detections: {len(boxes)}")

if len(boxes) > 0:
    keep   = nms(boxes, scores, iou_threshold=NMS_IOU)
    boxes  = boxes[keep]; scores = scores[keep]; labels = labels[keep]
    print(f"         After NMS: {len(boxes)}")

    img_area = image.size[0] * image.size[1]
    valid = [i for i, box in enumerate(boxes)
             if (BOX_AREA_MIN < (box[2]-box[0])*(box[3]-box[1])/img_area < BOX_AREA_MAX
                 and max(box[2]-box[0], box[3]-box[1]) /
                     (min(box[2]-box[0], box[3]-box[1]) + 1e-6) < ASPECT_RATIO_MAX)]
    if valid:
        v = torch.tensor(valid)
        boxes = boxes[v]; scores = scores[v]; labels = labels[v]
    print(f"         After area filter: {len(boxes)}")

    boxes, scores, labels = filter_containing_boxes(boxes, scores, labels)
    print(f"         After overlap filter: {len(boxes)}")


# ════════════════════════════════════════════════════════
# Step 2: ViT 分类 + 置信度过滤
# ════════════════════════════════════════════════════════
crops = []; predictions = []; confidences = []
factions = []; color_stats = []; valid_boxes = []

for box in boxes:
    x1,y1,x2,y2 = [int(b.item()) for b in box]
    crop   = image.crop((x1,y1,x2,y2))
    tensor = val_tf(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(vit_model(tensor)[0], dim=0)
        pred  = probs.argmax().item()
        conf  = probs[pred].item()

    if conf < VIT_CONF_THRESH:
        print(f"  [Drop] conf={conf:.3f} → {CLASS_NAMES[pred]}")
        continue

    faction, jp_px, us_px = get_faction_by_color(image, box)
    crops.append(crop); predictions.append(CLASS_NAMES[pred])
    confidences.append(conf); factions.append(faction)
    color_stats.append((jp_px, us_px)); valid_boxes.append(box)

print(f"\n[Step 2] After confidence filter: {len(valid_boxes)} units")


# ════════════════════════════════════════════════════════
# 可视化1: 原图 + 检测框
# ════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, figsize=(14, 10))
ax.imshow(image)
for i, (box, faction, pred) in enumerate(zip(valid_boxes, factions, predictions)):
    x1,y1,x2,y2 = [b.item() for b in box]
    color = 'orange' if faction == "JP" else 'lime'
    ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                 linewidth=2, edgecolor=color, facecolor='none'))
    ax.text(x1, y1-8, f"#{i+1} [{faction}] {pred}", color=color, fontsize=9,
            bbox=dict(facecolor='black', alpha=0.7, pad=2))
ax.set_title(f"{len(valid_boxes)} units detected  (conf≥{VIT_CONF_THRESH})", fontsize=14)
ax.axis('off'); plt.tight_layout()
plt.savefig('result_detection.jpg', dpi=150, bbox_inches='tight')
plt.show()

# ════════════════════════════════════════════════════════
# 可视化2: 裁剪图 + ViT 结果
# ════════════════════════════════════════════════════════
n = len(crops)
if n > 0:
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3.5))
    axes = axes.flatten() if n > 1 else [axes]
    for i, (crop, pred, conf, faction, (jp_px, us_px)) in enumerate(
            zip(crops, predictions, confidences, factions, color_stats)):
        axes[i].imshow(crop)
        axes[i].set_title(
            f"#{i+1} {pred}\nconf={conf:.2f}  [{faction}]\n"
            f"JP:{jp_px}  US:{us_px}",
            fontsize=8, color='green' if faction in ("JP","US") else 'red')
        axes[i].axis('off')
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.suptitle(f"Crops + ViT  (conf≥{VIT_CONF_THRESH})", fontsize=12)
    plt.tight_layout()
    plt.savefig('result_crops.jpg', dpi=150, bbox_inches='tight')
    plt.show()

# ════════════════════════════════════════════════════════
# 统计
# ════════════════════════════════════════════════════════
print(f"\n── Summary ──")
print(f"  JP: {sum(1 for f in factions if f=='JP')}")
print(f"  US: {sum(1 for f in factions if f=='US')}")
if any(f == 'unknown' for f in factions):
    print(f"  Unknown: {sum(1 for f in factions if f=='unknown')}")
print(f"\n── Unit types ──")
for cls, cnt in Counter(predictions).most_common():
    print(f"  {cls:<14} × {cnt}")