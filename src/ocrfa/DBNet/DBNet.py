# https://github.com/MhLiao/DB
import os
import math
import yaml
from shapely.geometry import Polygon
import PIL.Image
import numpy as np
import cv2
import pyclipper
import torch

from .model.constructor import Configurable


class DBNet:
    def __init__(
            self,
            backbone="resnet18",
            weight_dir=None,
            weight_name='pretrained',
            initialize_model=True,
            dynamic_import_relative_path=None,
            device='cuda',
            **kwargs
    ):

        self.device = device

        config_path = os.path.join(os.path.dirname(__file__), "configs", "DBNet_inference.yaml")
        with open(config_path, 'r') as fid:
            self.configs = yaml.safe_load(fid)

        if dynamic_import_relative_path is not None:
            self.configs = self.set_relative_import_path(self.configs, dynamic_import_relative_path)

        if backbone in self.configs.keys():
            self.backbone = backbone
        else:
            raise ValueError("Invalid backbone. Current support backbone are {}.".format(",".join(self.configs.keys())))

        if weight_dir is not None:
            self.weight_dir = weight_dir
        else:
            self.weight_dir = os.path.join(os.path.dirname(__file__), 'weights')

        if initialize_model:
            if weight_name in self.configs[backbone]['weight'].keys():
                weight_path = os.path.join(self.weight_dir, self.configs[backbone]['weight'][weight_name])
                error_message = "A weight with a name {} is found in DBNet_inference.yaml but cannot be find file: {}."
            else:
                weight_path = os.path.join(self.weight_dir, weight_name)
                error_message = (
                    "A weight with a name {} is not found in"
                    " DBNet_inference.yaml and cannot be find file: {}."
                )

            if not os.path.isfile(weight_path):
                raise FileNotFoundError(error_message.format(weight_name, weight_path))

            self.initialize_model(self.configs[backbone]['model'], weight_path)

        else:
            self.model = None

        self.BGR_MEAN = np.array(self.configs['BGR_MEAN'])
        self.min_detection_size = self.configs['min_detection_size']
        self.max_detection_size = self.configs['max_detection_size']

    def set_relative_import_path(self, configs, dynamic_import_relative_path):

        assert dynamic_import_relative_path is not None
        prefices = dynamic_import_relative_path.split(os.sep)
        for key, value in configs.items():
            if key == 'class':
                configs.update({key: ".".join(prefices + value.split("."))})
            else:
                if isinstance(value, dict):
                    value = self.set_relative_import_path(value, dynamic_import_relative_path)
                else:
                    pass
        return configs

    def load_weight(self, weight_path):

        if self.model is None:
            raise RuntimeError("model has not yet been constructed.")
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device), strict=False)
        self.model.eval()

    def construct_model(self, config):

        self.model = Configurable.construct_class_from_config(config).structure.builder.build(self.device)

    def initialize_model(self, model_config, weight_path):

        self.construct_model(model_config)
        self.load_weight(weight_path)
        if isinstance(self.model.model, torch.nn.DataParallel) and self.device == 'cpu':
            self.model.model = self.model.model.module.to(self.device)

    def get_cv2_image(self, image):
        if isinstance(image, str):
            if os.path.isfile(image):
                image = cv2.imread(image, cv2.IMREAD_COLOR).astype('float32')
            else:
                raise FileNotFoundError("Cannot find {}".format(image))
        elif isinstance(image, np.ndarray):
            image = image.astype('float32')
        elif isinstance(image, PIL.Image.Image):
            image = np.asarray(image)[:, :, ::-1]
        else:
            raise TypeError("Unsupport image format. Only path-to-file, opencv BGR image, and PIL image are supported.")

        return image

    def resize_image(self, img, detection_size=None):

        height, width, _ = img.shape
        if detection_size is None:
            detection_size = max(self.min_detection_size, min(height, width, self.max_detection_size))

        if height < width:
            new_height = int(math.ceil(detection_size / 32) * 32)
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = int(math.ceil(detection_size / 32) * 32)
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))

        return resized_img, (height, width)

    def image_array2tensor(self, image):

        return torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)

    def normalize_image(self, image):

        return (image - self.BGR_MEAN) / 255.0

    def load_image(self, image_path, detection_size=0):
        img = self.get_cv2_image(image_path)
        img, original_shape = self.resize_image(img, detection_size=detection_size)
        img = self.normalize_image(img)
        img = self.image_array2tensor(img)

        return img, original_shape

    def load_images(self, images, detection_size=None):

        images, original_shapes = zip(*[self.load_image(image, detection_size=detection_size)
                                        for image in images])
        return torch.cat(images, dim=0), original_shapes

    def hmap2bbox(
            self,
            image_tensor,
            original_shapes,
            hmap,
            text_threshold=0.2,
            bbox_min_score=0.2,
            bbox_min_size=3,
            max_candidates=0,
            as_polygon=False
    ):

        segmentation = self.binarize(hmap, threshold=text_threshold)
        boxes_batch = []
        scores_batch = []
        for batch_index in range(image_tensor.size(0)):
            height, width = original_shapes[batch_index]
            if as_polygon:
                boxes, scores = self.polygons_from_bitmap(
                    hmap[batch_index],
                    segmentation[batch_index],
                    width,
                    height,
                    bbox_min_score=bbox_min_score,
                    bbox_min_size=bbox_min_size,
                    max_candidates=max_candidates
                )
            else:
                boxes, scores = self.boxes_from_bitmap(
                    hmap[batch_index],
                    segmentation[batch_index],
                    width,
                    height,
                    bbox_min_score=bbox_min_score,
                    bbox_min_size=bbox_min_size,
                    max_candidates=max_candidates
                )

            boxes_batch.append(boxes)
            scores_batch.append(scores)

        boxes_batch, scores_batch = zip(*[zip(*[(box, score)
                                                for (box, score) in zip(boxes, scores) if score > 0]
                                              ) if any(scores > 0) else [(), ()]
                                          for (boxes, scores) in zip(boxes_batch, scores_batch)]
                                        )

        return boxes_batch, scores_batch

    @staticmethod
    def binarize(tensor, threshold):

        return tensor > threshold

    def polygons_from_bitmap(
            self,
            hmap,
            segmentation,
            dest_width,
            dest_height,
            bbox_min_score=0.2,
            bbox_min_size=3,
            max_candidates=0
    ):

        assert segmentation.size(0) == 1
        bitmap = segmentation.cpu().numpy()[0]  # The first channel
        hmap = hmap.cpu().detach().numpy()[0]
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if max_candidates > 0:
            contours = contours[:max_candidates]

        for contour in contours:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = self.box_score_fast(hmap, points.reshape(-1, 2))  # type:ignore
            if score < bbox_min_score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=2.0)
                if len(box) > 1:
                    continue

            else:
                continue

            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < bbox_min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)

        return boxes, scores

    def boxes_from_bitmap(
            self,
            hmap,
            segmentation,
            dest_width,
            dest_height,
            bbox_min_score=0.2,
            bbox_min_size=3,
            max_candidates=0
    ):

        assert segmentation.size(0) == 1
        bitmap = segmentation.cpu().numpy()[0]  # The first channel
        hmap = hmap.cpu().detach().numpy()[0]
        height, width = bitmap.shape
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if max_candidates > 0:
            num_contours = min(len(contours), max_candidates)
        else:
            num_contours = len(contours)

        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < bbox_min_size:
                continue

            points = np.array(points)
            score = self.box_score_fast(hmap, points.reshape(-1, 2))  # type:ignore
            if score < bbox_min_score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < bbox_min_size + 2:
                continue

            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score

        return boxes.tolist(), scores

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))

        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]

        return box, min(bounding_box[1])

    @staticmethod
    def box_score_fast(hmap, box_):

        h, w = hmap.shape[:2]
        box = box_.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)  # type:ignore

        return cv2.mean(hmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def image2hmap(self, image_tensor):
        return self.model.forward(image_tensor, training=False)

    def inference(
            self,
            image,
            text_threshold=0.2,
            bbox_min_score=0.2,
            bbox_min_size=3,
            max_candidates=0,
            detection_size=None,
            as_polygon=False,
            return_scores=False
    ):

        if not isinstance(image, list):
            image = [image]

        image_tensor, original_shapes = self.load_images(image, detection_size=detection_size)
        with torch.no_grad():
            hmap = self.image2hmap(image_tensor)
            batch_boxes, batch_scores = self.hmap2bbox(
                image_tensor,
                original_shapes,
                hmap,
                text_threshold=text_threshold,
                bbox_min_score=bbox_min_score,
                bbox_min_size=bbox_min_size,
                max_candidates=max_candidates,
                as_polygon=as_polygon
            )

        if return_scores:
            return batch_boxes, batch_scores
        else:
            return batch_boxes
