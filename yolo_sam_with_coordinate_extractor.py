import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import base64
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, label, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(box[0], box[1], label, size=15, color='purple', horizontalalignment='left', verticalalignment='bottom')

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)
    return image.permute(2, 0, 1).contiguous()

def mask_to_polygon(mask):
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    elif mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    polygons = [[[float(point[0][0]), float(point[0][1])] for point in contour] for contour in contours if len(contour) > 2]
    return polygons


def save_polygons_to_json(image, polygons, labels, filename, img_name):
    image_height, image_width = image.shape[:2]
    image_path = f'./images/{img_name}'
    image_data = encode_image(image_path)

    data = {
        "version": "4.5.7",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "points": [point for point in polygon[0]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            } for label, polygon in zip(labels, polygons) if polygon
        ],
        # "lineColor": [0, 255, 0, 128],
        # "fillColor": [255, 0, 0, 128],
        "imagePath": image_path.split('/')[-1],
        "imageData": image_data,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
def run(img, img_name, class_names):
    model = YOLO('models/best.pt')
    results = model.predict(source=img, save=True)
    input_boxes = []
    label_list = []

    for result in results:
        boxes = result.boxes
        input_boxes.append(boxes.xyxy)
        cls = boxes.cls
        output_index = cls.int()

    for i in output_index:
        label_list.append(class_names[i])

    sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(img)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    batched_input = [{
        'image': prepare_image(img, resize_transform, sam),
        'boxes': resize_transform.apply_boxes_torch(input_boxes[0], img.shape[:2]),
        'original_size': img.shape[:2]
    }]
    batched_output = sam(batched_input, multimask_output=False)

    fig, ax = plt.subplots(1, figsize=(20, 20))
    ax.imshow(img[..., ::-1])

    polygons_list = []
    for mask in batched_output[0]['masks']:
        polygon = mask_to_polygon(mask.cpu().numpy())
        polygons_list.append(polygon)
        show_mask(mask.cpu().numpy(), ax, random_color=True)

    json_filename = f'./output/{img_name.split(".")[0]}.json'
    save_polygons_to_json(img, polygons_list, label_list, json_filename, img_name)

    for box, label in zip(input_boxes[0], label_list):
        show_box(box.cpu().numpy(), label, ax)

    if not os.path.exists('./output'):
        os.makedirs('./output')
    plt.savefig(f'./output/{img_name}')

def main():
    class_names = ['aphid', 'tabaci_immature', 'tabaci_mature']
    img_list = [x for x in os.listdir('./images/') if x.endswith('png') or x.endswith('jpg') or x.endswith('JPG')]
    for img_name in img_list:
        img = cv2.imread(f'./images/{img_name}')
        run(img, img_name, class_names)

if __name__=='__main__':
    main()
