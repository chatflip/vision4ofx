import time
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models


def load_model(application):
    if application == 'detect':
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif application == 'mask':
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    elif application == 'keypoint':
        model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    return model.eval()


def decode(image, outputs, application):
    if application == 'detect':
        return decode_detect(image, outputs)
    elif application == 'mask':
        return decode_mask(image, outputs)
    elif application == 'keypoint':
        return decode_keypoint(image, outputs)


def decode_detect(image, output):
    boxes = output['boxes'].cpu()
    labels = output['labels'].cpu()
    scores = output['scores'].cpu()
    num_target = np.sum(np.where(scores > 0.7, True, False))

    for i in range(num_target):
        if labels[i] != 1:
            continue
        bbox = boxes[i]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 10)
    return image


def decode_mask(image, output):
    labels = output['labels'].cpu()
    scores = output['scores'].cpu()
    masks = output['masks'].cpu()
    num_target = np.sum(np.where(scores > 0.7, True, False))

    height, width = image.shape[:2]
    all_mask = np.zeros((height, width, 1), dtype=np.float)

    for i in range(num_target):
        if labels[i] != 1:
            continue
        mask = masks[i].numpy()
        all_mask[:, :, 0] += mask[0, :, :]

    image = np.array(image, dtype=np.float)
    image *= np.clip(all_mask, None, 1.0)
    image = np.array(image, dtype=np.uint8)

    return image


def decode_keypoint(image, output):
    pose_chain = [
        (0, 1), (1, 3), (0, 2), (2, 4),
        (0, 5), (5, 7), (7, 9), (5, 11),
        (11, 13), (13, 15), (0, 6), (6, 8),
        (8, 10), (6, 12), (12, 14), (14, 16)
    ]

    image = np.array(image, dtype=np.uint8)

    labels = output['labels'].cpu()
    scores = output['scores'].cpu()
    keypoints = output['keypoints'].cpu()
    num_keypoint = keypoints.shape[1]
    num_target = np.sum(np.where(scores > 0.7, True, False))

    for i in range(num_target):
        if labels[i] == 1:
            keypoint = keypoints[i]
            for j in range(num_keypoint):
                point = keypoint[j]
                x, y = int(point[0]), int(point[1])
                image = cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
            for chain in pose_chain:
                start = int(keypoint[chain[0], 0]), int(keypoint[chain[0], 1])
                end = int(keypoint[chain[1], 0]), int(keypoint[chain[1], 1])
                image = cv2.line(image, start, end, (255, 255, 255), 2)
    return image


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()

    selected_model = 'detect'
    model = load_model(selected_model).to(device)

    cap_prop = {
        'fullHD': {
            'height': 1080,
            'width': 1920,
            'fps': 30
        },
        'HD': {
            'height': 720,
            'width': 1280,
            'fps': 60
        }
    }

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_prop['HD']['height'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_prop['HD']['width'])
    cap.set(cv2.CAP_PROP_FPS, cap_prop['HD']['fps'])

    while(True):
        ret, frame = cap.read()
        start_time = time.perf_counter()



        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)[0]
        frame = decode(frame, outputs, selected_model)
        cv2.imshow('frame', frame)

        end_time = time.perf_counter()
        inference_time = end_time - start_time

        sys.stdout.write('\r{} fps: {:5.2f}'.format(
            selected_model, 1.0 / inference_time))
        sys.stdout.flush()

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('1'):
            selected_model = 'detect'
            model = load_model(selected_model).to(device)
        elif key == ord('2'):
            selected_model = 'mask'
            model = load_model(selected_model).to(device)
        elif key == ord('3'):
            selected_model = 'keypoint'
            model = load_model(selected_model).to(device)

    cap.release()
    cv2.destroyAllWindows()
