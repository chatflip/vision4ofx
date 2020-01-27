import argparse
import time

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
    num_target = np.sum(np.where(scores > 0.97, True, False))

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


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='input.mp4')
    parser.add_argument('--output', type=str, default='output.mp4')
    parser.add_argument('--mode', type=str, default='keypoint')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()
    selected_model = args.mode
    model = load_model(selected_model).to(device)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    cap = cv2.VideoCapture(args.input)
    out = cv2.VideoWriter(args.output, fourcc,
                           cap.get(cv2.CAP_PROP_FPS),
                           (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    start_time = time.perf_counter()
    cnt = 0
    while(True):
            try:
                if cnt % 10 == 0:
                    print('{} / {}'.format(cnt,
                        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
                ret, frame = cap.read()
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_tensor = transform(image_rgb).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(image_tensor)[0]
                frame = decode(frame, outputs, selected_model)
                out.write(frame)
                cnt += 1
            except:
                cap.release()
                out.release()
                break

    cv2.destroyAllWindows()
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    print('{} elapsed time: {:5.2f}s'.format(
        selected_model, inference_time))
