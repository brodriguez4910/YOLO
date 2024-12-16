import os
import cv2
import numpy as np


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def non_max_suppression(prediction, conf_thres=0.25, nms_thres=0.45, agnostic=False):
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    output = [np.zeros((0, 6))] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        if not x.shape[0]:
            continue

        # # Compute conf
        # if nc == 1:
        #     x[:, 5:] = x[
        #         :, 4:5
        #     ]  # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
        #     # so there is no need to multiplicate.
        # else:
        #     x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # Detections matrix nx6 (xyxy, conf, cls)
        conf = np.max(x[:, 4:], axis=1)
        j = np.argmax(x[:, 4:], axis=1)
        re = np.array(conf.reshape(-1) > conf_thres)
        conf = conf.reshape(-1, 1)
        j = j.reshape(-1, 1)
        x = np.concatenate((box, conf, j), axis=1)[re]
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        # Batched NMS
        c = x[:, 4:5] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4], x[:, 4]  # boxes (offset by class), scores
        # converted to list for opencv nms
        boxes = boxes.tolist()
        scores = scores.tolist()
        # i = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, nms_thres)
        output[xi] = x[cv2.dnn.NMSBoxes(boxes, scores, conf_thres, nms_thres)]
    return output


class Model(object):
    def __init__(self, model_directory, models_config):
        self._net = cv2.dnn.readNet(
            os.path.join(model_directory, models_config["model_file_name"])
        )
        self._rgb = models_config["rgb"]
        self._normalized_input = (
            1.0 / 255.0 if models_config["normalized_input"] else 1.0
        )
        self._image_size = (
            models_config["input_img_size"]["height"],
            models_config["input_img_size"]["width"],
        )
        self.confThreshold = 0.1
        self.nmsThreshold = 0.3

    def run(self, img):
        img_orig_shape = img.shape
        img = cv2.dnn.blobFromImage(
            img,
            self._normalized_input,
            self._image_size,
            swapRB=self._rgb,
            crop=False,
        )

        self._net.setInput(img)
        outputs = self._net.forward(self._net.getUnconnectedOutLayersNames())
        # Prepare output array
        outputs = np.transpose(outputs[0], (0, 2, 1))

        predictions = non_max_suppression(
            outputs, self.confThreshold, self.nmsThreshold
        )[0]

        final_output_boxes = []
        for pred in predictions:
            if pred[4] >= self.confThreshold:
                final_output_boxes.append(
                    np.array(
                        [
                            pred[5],
                            pred[4],
                            int(pred[0] / self._image_size[1] * img_orig_shape[1]),
                            int(pred[1] / self._image_size[0] * img_orig_shape[0]),
                            int(pred[2] / self._image_size[1] * img_orig_shape[1]),
                            int(pred[3] / self._image_size[0] * img_orig_shape[0]),
                        ]
                    )
                )

        return np.array(final_output_boxes)


# Load onnx model in opencv dnn, glob images from "/auto/shared/client_data/natures_touch/blueberries_golden_test"
# and run inference on each image, run nms, save annotated images to disk
import glob

model = Model(
    "runs/detect/train4/weights",
    {
        "model_file_name": "best.onnx",
        "rgb": True,
        "normalized_input": True,
        "input_img_size": {"height": 1024, "width": 1024},
    },
)
images = glob.glob(
    "/auto/shared/client_data/natures_touch/blueberries_golden_test/*.png"
)

for image in images:
    img = cv2.imread(image)
    detections = model.run(img)
    for detection in detections:
        cv2.rectangle(
            img,
            (int(detection[2]), int(detection[3])),
            (int(detection[4]), int(detection[5])),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img,
            str(detection[1]),
            (int(detection[2]), int(detection[3])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    filename = os.path.basename(image).replace(".png", "")
    cv2.imwrite(f"{filename}_annotated.png", img)
