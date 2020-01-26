import os

import torch
import torchvision

if __name__ == '__main__':
    if not os.path.exists('weight'):
        os.makedirs('weight')

    detect_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    detect_script = torch.jit.script(detect_model)
    detect_script.eval()
    detect_script.save('weight/fasterrcnn_resnet50_fpn.pt')

    mask_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    mask_script = torch.jit.script(mask_model)
    mask_script.eval()
    mask_script.save('weight/maskrcnn_resnet50_fpn.pt')

    keypoint_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    keypoint_script = torch.jit.script(keypoint_model)
    keypoint_script.eval()
    keypoint_script.save('weight/keypointrcnn_resnet50_fpn.pt')
