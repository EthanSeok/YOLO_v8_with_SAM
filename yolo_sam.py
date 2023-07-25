import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide


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


def main():
    model = YOLO('./notebooks/best.pt')
    img_name = '20220824-171308.png'
    img = cv2.imread('{img_name}')

    results = model.predict(source=img, save=True)
    print(results)

    class_names = ['aphid', 'tabaci_immature', 'tabaci_mature']
    label_list = []
    input_boxes = []

    for result in results:
        boxes = result.boxes
        input_boxes.append(boxes.xyxy)
        cls = boxes.cls
        output_index = cls.int()

    # print(output_index)

    for i in output_index:
        # class_name = class_names[output_index[i]]
        class_name = class_names[i]
        label_list.append(class_name)

    sam_checkpoint = "./model/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    predictor.set_image(img)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    batched_input = [
        {
            'image': prepare_image(img, resize_transform, sam),
            'boxes': resize_transform.apply_boxes_torch(input_boxes[0], img.shape[:2]),
            'original_size': img.shape[:2]
        }
    ]
    batched_output = sam(batched_input, multimask_output=False)
    # print(batched_output[0].keys())

    fig, ax = plt.subplots(1, figsize=(20, 20))

    ax.imshow(img[..., ::-1])
    for mask in batched_output[0]['masks']:
        show_mask(mask.cpu().numpy(), ax, random_color=True)

    for box, label in zip(input_boxes[0], label_list):
        show_box(box.cpu().numpy(), label, ax)

    if not os.path.exists('./output'):
        os.makedirs('./output')

    plt.savefig(f'./output/{img_name}')

if __name__=='__main__':
    main()
