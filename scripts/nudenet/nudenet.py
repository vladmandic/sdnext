#!/bin/env python

import os
import sys
import math
import time
import logging
import cv2
import numpy as np
from PIL import Image


log = logging.getLogger("sd")
session = None
detector = None
default_overlay = os.path.join(os.path.dirname(__file__), 'censored.png')
labels = [
    "female-private-area",
    "female-face",
    "buttocks-bare",
    "female-breast-bare",
    "female-vagina",
    "male-breast-bare",
    "anus-bare",
    "feet-bare",
    "belly",
    "feet",
    "armpits",
    "armpits-bare",
    "male-face",
    "belly-bare",
    "male-penis",
    "anus-area",
    "female-breast",
    "buttocks",
]
nsfw = [
    "buttocks-bare",
    "female-breast-bare",
    "anus-bare",
    "female-vagina",
    "male-penis",
]


class NudeResult:
    output: None
    censor: list = []
    detections: list = []
    censored: list = []


class NudeDetector:
    def __init__(self, providers=None, model=None):
        import onnxruntime
        import huggingface_hub as hf
        from onnxruntime.capi import _pybind_state as C
        from modules import shared

        global session # pylint: disable=global-statement
        self.model_path = model or hf.hf_hub_download(
            repo_id='vladmandic/nudenet',
            filename='nudenet.onnx',
            cache_dir=shared.opts.hfcache_dir,
        )
        if session is None:
            log.info(f'NudeNet load: model="{self.model_path}" providers={providers}')
            session = onnxruntime.InferenceSession(self.model_path, providers=C.get_available_providers() if not providers else providers) # pylint: disable=no-member
        model_inputs = session.get_inputs()
        self.input_width = model_inputs[0].shape[2] # 320
        self.input_height = model_inputs[0].shape[3] # 320
        self.input_name = model_inputs[0].name


    def read_image(self, image, target_size=320):
        if type(image) == str:
            img = cv2.imread(image)
        else:
            img = image
        img_height, img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aspect = img_width / img_height
        if img_height > img_width:
            new_height = target_size
            new_width = int(round(target_size * aspect))
        else:
            new_width = target_size
            new_height = int(round(target_size / aspect))
        resize_factor = math.sqrt((img_width**2 + img_height**2) / (new_width**2 + new_height**2))
        img = cv2.resize(img, (new_width, new_height))
        pad_x = target_size - new_width
        pad_y = target_size - new_height
        pad_top, pad_bottom = [int(i) for i in np.floor([pad_y, pad_y]) / 2]
        pad_left, pad_right = [int(i) for i in np.floor([pad_x, pad_x]) / 2]
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img = cv2.resize(img, (target_size, target_size))
        image_data = img.astype("float32") / 255.0  # normalize
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0)
        return image_data, resize_factor, pad_left, pad_top

    def postprocess(self, output, resize_factor, pad_left, pad_top, min_score):
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= min_score:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left = int(round((x - w * 0.5 - pad_left) * resize_factor))
                top = int(round((y - h * 0.5 - pad_top) * resize_factor))
                width = int(round(w * resize_factor))
                height = int(round(h * resize_factor))
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
        res = []
        for i in indices: # pylint: disable=not-an-iterable
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            res.append({"label": labels[class_id], "id": class_id, "score": round(float(score), 2), "box": box})
        return res

    def pixelate(self, image, blocks=3):
        (h, w) = image.shape[:2] # divide the input image into NxN blocks
        xSteps = np.linspace(0, w, blocks + 1, dtype="int")
        ySteps = np.linspace(0, h, blocks + 1, dtype="int")
        for i in range(1, len(ySteps)):
            for j in range(1, len(xSteps)):
                startX = xSteps[j - 1]
                startY = ySteps[i - 1]
                endX = xSteps[j]
                endY = ySteps[i]
                roi = image[startY:endY, startX:endX]
                (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
                cv2.rectangle(image, (startX, startY), (endX, endY), (B, G, R), -1)
        return image

    def overlay(self, background, foreground, x_offset=None, y_offset=None):
        bg_h, bg_w, bg_channels = background.shape
        fg_h, fg_w, fg_channels = foreground.shape
        if bg_channels != 3:
            log.error(f'NudeNet input image: channels={bg_channels} must be RGB')
            return background
        if fg_channels < 4: # make sure that overlay is rgba
            log.warning('NudeNet overlay image does not have alpha channel')
            foreground = cv2.cvtColor(foreground, cv2.COLOR_RGB2RGBA)
            foreground[:, :, 3] = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
            fg_h, fg_w, fg_channels = foreground.shape
        if x_offset is None: # center by default
            x_offset = (bg_w - fg_w) // 2
        if y_offset is None:
            y_offset = (bg_h - fg_h) // 2
        w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
        h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)
        if w < 1 or h < 1:
            return background
        bg_x = max(0, x_offset) # clip foreground and background images to the overlapping regions
        bg_y = max(0, y_offset)
        fg_x = max(0, x_offset * -1)
        fg_y = max(0, y_offset * -1)
        foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
        background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]
        foreground_colors = foreground[:, :, :3] # separate alpha and color channels from the foreground image
        alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0
        alpha_mask = alpha_mask = alpha_channel[:,:,np.newaxis] # construct an alpha_mask that matches the image shape
        composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask # combine the background with the overlay image weighted by alpha
        background[bg_y:bg_y + h, bg_x:bg_x + w] = composite # overwrite the section of the background image that has been updated
        return background

    def detect(self, image, min_score):
        try:
            preprocessed_image, resize_factor, pad_left, pad_top = self.read_image(image, self.input_width)
            outputs = session.run(None, {self.input_name: preprocessed_image})
            res = self.postprocess(outputs, resize_factor, pad_left, pad_top, min_score)
        except Exception as e:
            log.error(f'NudeNet: {e}')
            return []
        return res

    def censor(self, image, min_score=0.2, censor=None, method='pixelate', blocks=3, overlay=None):
        if type(image) == str:
            image = cv2.imread(image) # input is image path
        else:
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # input is pil image
        nude = NudeResult()
        nude.censor = censor or []
        nude.detections = self.detect(image, min_score)
        nude.censored = [d for d in nude.detections if d["label"] in nude.censor]
        for d in nude.censored:
            box = d["box"]
            x, y, w, h = box[0], box[1], box[2], box[3]
            area = image[y: y+h, x: x+w]
            if method == 'pixelate':
                image[y: y+h, x: x+w] = self.pixelate(area, blocks=blocks)
            elif method == 'blur':
                image[y: y+h, x: x+w] = cv2.blur(area, (blocks, blocks))
            elif method == 'gaussian blur':
                image[y: y+h, x: x+w] = cv2.GaussianBlur(area, (blocks, blocks), 0)
            elif method == 'median blur':
                image[y: y+h, x: x+w] = cv2.medianBlur(area, blocks)
            elif method == 'block':
                image[y: y+h, x: x+w] = (0, 0, 0)
            elif method == 'image':
                if overlay is None or overlay == '':
                    overlay = default_overlay
                if not os.path.exists(overlay):
                    log.error(f'NudeNet overlay image not found: file={overlay}')
                    overlay = default_overlay
                pasty = cv2.imread(overlay, cv2.IMREAD_UNCHANGED)
                pasty = cv2.resize(pasty, (w, h))
                image = self.overlay(image, pasty, x, y)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        nude.output = Image.fromarray(image)
        return nude


def cli():
    global detector # pylint: disable=global-statement
    sys.argv.pop(0)
    if len(sys.argv) == 0:
        log.error('nudenet: no files specified')
    for fn in sys.argv:
        t0 = time.time()
        pil = Image.open(fn)
        if detector is None:
            detector = NudeDetector(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        nudes = detector.censor(image=pil, censor=['female breast bare', 'female genitalia bare'], min_score=0.2, method='pixelate')
        t1 = time.time()
        log.info(vars(nudes))
        f = os.path.splitext(fn)[0] + '_censored.jpg'
        nudes.output.save(f)
        log.info(f'nudenet: input={fn} output={f} time={t1-t0:.2f}s')


if __name__ == "__main__":
    cli()
