import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from PIL import Image
import torchvision.transforms as T
import os

DEVICE = "cpu"

CLASSES = ["__background__", "opacity", "nodule", "consolidation", "effusion"]

def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES))

    model_path = os.path.join("models", "fasterrcnn_best.pth")
    checkpoint = torch.load(model_path, map_location=DEVICE)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

transform = T.Compose([T.ToTensor()])

def predict(image_path, score_thresh=0.2, iou_thresh=0.3, top_k=3):
    img = Image.open(image_path).convert("L").convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    boxes = outputs["boxes"]
    labels = outputs["labels"]
    scores = outputs["scores"]

    # filter by score threshold
    keep = scores >= score_thresh
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    if len(boxes) > 0:
        keep_idx = nms(boxes, scores, iou_thresh)
        boxes, labels, scores = boxes[keep_idx], labels[keep_idx], scores[keep_idx]

        # keep only top_k boxes
        if len(scores) > top_k:
            top_scores, top_idx = torch.topk(scores, top_k)
            boxes, labels, scores = boxes[top_idx], labels[top_idx], scores[top_idx]

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        detections.append({
            "box": box.tolist(),
            "label": CLASSES[label.item()],
            "score": float(score)
        })

    return detections
