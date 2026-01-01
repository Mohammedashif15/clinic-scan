from PIL import Image, ImageDraw

def draw_boxes(image_path, detections, output_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for det in detections:
        box = [int(b) for b in det["box"]]
        label = det["label"]
        score = det["score"]

        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], max(0, box[1]-15)), f"{label} {score:.2f}", fill="red")

    img.save(output_path)
