import os
from .DBNet.DBNet import DBNet
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict

import cv2
import numpy as np
from .utils import resize_aspect_ratio, normalize_mean_variance, get_det_boxes, adjust_result_coordinates, OCRFA


def copy_state_dict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def test_net(canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, poly, device,
             estimate_num_chars=False):
    if isinstance(image, np.ndarray) and len(image.shape) == 4:  # image is batch of np arrays
        image_arrs = image
    else:  # image is single numpy array
        image_arrs = [image]

    img_resized_list = []
    # resize
    for img in image_arrs:
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, canvas_size,
                                                                      interpolation=cv2.INTER_LINEAR,
                                                                      mag_ratio=mag_ratio)
        img_resized_list.append(img_resized)
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing
    x = [np.transpose(normalize_mean_variance(n_img), (2, 0, 1))
         for n_img in img_resized_list]
    x = torch.from_numpy(np.array(x))
    x = x.to(device)

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    boxes_list, polys_list = [], []
    for out in y:
        # make score and link map
        score_text = out[:, :, 0].cpu().data.numpy()
        score_link = out[:, :, 1].cpu().data.numpy()

        # Post-processing
        boxes, polys, mapper = get_det_boxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly, estimate_num_chars)

        # coordinate adjustment
        boxes = adjust_result_coordinates(boxes, ratio_w, ratio_h)
        polys = adjust_result_coordinates(polys, ratio_w, ratio_h)
        if estimate_num_chars:
            boxes = list(boxes)
            polys = list(polys)
        for k in range(len(polys)):
            if estimate_num_chars:
                boxes[k] = (boxes[k], mapper[k])
            if polys[k] is None:
                polys[k] = boxes[k]
        boxes_list.append(boxes)
        polys_list.append(polys)

    return boxes_list, polys_list


def get_detector(trained_model, device='cpu', quantize=True, cudnn_benchmark=False):
    net = OCRFA()

    if device == 'cpu':
        net.load_state_dict(copy_state_dict(torch.load(trained_model, map_location=device)))
        if quantize:
            try:
                torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        net.load_state_dict(copy_state_dict(torch.load(trained_model, map_location=device)))
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = cudnn_benchmark

    net.eval()
    return net


def get_textbox(detector, image, canvas_size, mag_ratio, text_threshold, link_threshold, low_text, poly, device,
                optimal_num_chars=None, **kwargs):
    result = []
    estimate_num_chars = optimal_num_chars is not None
    bboxes_list, polys_list = test_net(canvas_size, mag_ratio, detector,
                                       image, text_threshold,
                                       link_threshold, low_text, poly,
                                       device, estimate_num_chars)
    if estimate_num_chars:
        polys_list = [[p for p, _ in sorted(polys, key=lambda x: abs(optimal_num_chars - x[1]))]
                      for polys in polys_list]

    for polys in polys_list:
        single_img_result = []
        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            single_img_result.append(poly)
        result.append(single_img_result)

    return result


def test_net_db(
        image,
        detector,
        threshold=0.2,
        bbox_min_score=0.2,
        bbox_min_size=3,
        max_candidates=0,
        canvas_size=None,
        poly=False,
        device='cpu'
):
    if isinstance(image, np.ndarray) and len(image.shape) == 4:
        image_arrs = image
    else:
        image_arrs = [image]

    # resize
    images, original_shapes = zip(*[detector.resize_image(img, canvas_size) for img in image_arrs])
    # preprocessing
    images = [np.transpose(detector.normalize_image(n_img), (2, 0, 1)) for n_img in images]
    image_tensor = torch.from_numpy(np.array(images)).to(device)
    # forward pass
    with torch.no_grad():
        hmap = detector.image2hmap(image_tensor.to(device))
        bboxes, _ = detector.hmap2bbox(
            image_tensor,
            original_shapes,
            hmap,
            text_threshold=threshold,
            bbox_min_score=bbox_min_score,
            bbox_min_size=bbox_min_size,
            max_candidates=max_candidates,
            as_polygon=False)
        if poly:
            polys, _ = detector.hmap2bbox(
                image_tensor,
                original_shapes,
                hmap,
                text_threshold=threshold,
                bbox_min_score=bbox_min_score,
                bbox_min_size=bbox_min_size,
                max_candidates=max_candidates,
                as_polygon=True)
        else:
            polys = bboxes

    return bboxes, polys


def get_detector_db(trained_model, backbone='resnet18', device='cpu', quantize=True, cudnn_benchmark=False):
    dbnet = DBNet(initialize_model=False,
                  dynamic_import_relative_path=os.path.join("ocrfa", "DBNet"),
                  device=device,
                  verbose=0)
    if backbone not in ['resnet18', 'resnet50']:
        raise ValueError("Invalid backbone. Options are 'resnet18' or 'resnet50'.")
    dbnet.initialize_model(dbnet.configs[backbone]['model'],
                           trained_model)
    if torch.device(device).type == 'cpu':
        if quantize:
            try:
                torch.quantization.quantize_dynamic(dbnet, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        dbnet.model = torch.nn.DataParallel(dbnet.model).to(device)
        cudnn.benchmark = cudnn_benchmark

    dbnet.model.eval()

    return dbnet


def get_textbox_db(detector,
                   image,
                   canvas_size=None,
                   poly=False,
                   threshold=0.2,
                   bbox_min_score=0.2,
                   bbox_min_size=3,
                   max_candidates=0,
                   device='cpu',
                   **kwargs
                   ):
    if torch.device(device).type != detector.device:
        raise RuntimeError(' '.join([
            "DBNet detector is initialized with {} device, but detection routine",
            "is called with device = {}.",
            "To use this detector both have to be the same."
        ]).format(detector.device, device))

    _, polys_list = test_net_db(
        image,
        detector,
        threshold=threshold,
        bbox_min_score=bbox_min_score,
        bbox_min_size=bbox_min_size,
        max_candidates=max_candidates,
        canvas_size=canvas_size,
        poly=poly,
        device=device
    )

    return [[np.array(box).astype(np.int32).reshape((-1)) for box in polys] for polys in polys_list]
