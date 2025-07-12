import time
import cv2
import numpy as np
from modules import shared, devices


face_helper = None


def restore(np_image, name, session, strength): # pylint: disable=unused-argument
    t0 = time.time()
    global face_helper # pylint: disable=global-statement
    try:
        from modules.facelib.utils.face_restoration_helper import FaceRestoreHelper
        from modules.facelib.detection.retinaface import retinaface
    except Exception as e:
        shared.log.error(f"FaceRestorer error: {e}")
        return np_image
    if hasattr(retinaface, 'device'):
        retinaface.device = devices.device
    if face_helper is None:
        face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', use_parse=True, device=devices.device)

    np_image = np_image[:, :, ::-1]
    original_resolution = np_image.shape[0:2]
    resolution = session.get_inputs()[0].shape[-2:]

    if face_helper is None or session is None:
        return np_image
    face_helper.clean_all()
    face_helper.read_image(np_image)
    face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
    face_helper.align_warp_face()

    detected_faces = len(face_helper.cropped_faces)
    for cropped_face in face_helper.cropped_faces:
        cropped_face = cv2.resize(cropped_face, resolution, interpolation=cv2.INTER_LANCZOS4)
        cropped_face = cropped_face.astype(np.float16)[:,:,::-1] / 255.0
        cropped_face = cropped_face.transpose((2, 0, 1))
        cropped_face = (cropped_face - 0.5) / 0.5
        cropped_face = np.expand_dims(cropped_face, axis=0).astype(np.float16)
        w = np.array([strength], dtype=np.double)
        if 'codeformer' in name:
            restored_face = session.run(None, {'x':cropped_face, 'w':w})[0][0]
        else:
            restored_face = session.run(None, {'input':cropped_face})[0][0]
        restored_face = (restored_face.transpose(1,2,0).clip(-1,1) + 1) * 0.5
        restored_face = (restored_face * 255)[:,:,::-1]
        restored_face = restored_face.clip(0, 255).astype('uint8')
        face_helper.add_restored_face(restored_face)
    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image()
    restored_img = restored_img[:, :, ::-1]
    if original_resolution != restored_img.shape[0:2]:
        restored_img = cv2.resize(restored_img, (0, 0), fx=original_resolution[1]/restored_img.shape[1], fy=original_resolution[0]/restored_img.shape[0], interpolation=cv2.INTER_LANCZOS4)

    face_helper.clean_all()
    t1 = time.time()
    shared.log.info(f'Detailer: model="{name}" faces={detected_faces} strength={strength} time={t1-t0:.3f}')

    return restored_img
