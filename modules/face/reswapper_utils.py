import cv2
import numpy as np


### https://github.com/somanchiu/ReSwapper/blob/GAN/Image.py

input_std = 255.0
input_mean = 0.0


def get_emap():
    emap = np.load("modules/face/reswapper_emap.npy") # https://github.com/somanchiu/ReSwapper/blob/GAN/emap.npy
    return emap


def postprocess_face(face_tensor):
    face_tensor = face_tensor.squeeze().cpu().detach()
    face_np = (face_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    return face_np

def getBlob(aimg, input_size = (128, 128)):
    blob = cv2.dnn.blobFromImage(aimg, 1.0 / input_std, input_size, (input_mean, input_mean, input_mean), swapRB=True)
    return blob


def getLatent(source_face):
    latent = source_face.normed_embedding.reshape((1,-1))
    emap = get_emap()
    latent = np.dot(latent, emap)
    latent /= np.linalg.norm(latent)
    return latent


def blend_swapped_image(swapped_face, target_image, M):
    h, w = target_image.shape[:2]
    M_inv = cv2.invertAffineTransform(M)
    warped_face = cv2.warpAffine(swapped_face, M_inv, (w, h),borderValue=0.0)
    img_white = np.full((swapped_face.shape[0], swapped_face.shape[1]), 255, dtype=np.float32)
    img_mask = cv2.warpAffine(img_white, M_inv, (w, h), borderValue=0.0)
    img_mask[img_mask > 20] = 255
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    if len(mask_h_inds) > 0 and len(mask_w_inds) > 0:  # safety check
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))
        k = max(mask_size // 10, 10)
        kernel = np.ones((k, k), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        k = max(mask_size // 20, 5)
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    img_mask = img_mask / 255.0
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    result = img_mask * warped_face + (1 - img_mask) * target_image.astype(np.float32)
    result = result.astype(np.uint8)
    return result


def drawKeypoints(image, keypoints, colorBGR, keypointsRadius=2):
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(image, (x, y), radius=keypointsRadius, color=colorBGR, thickness=-1) # BGR format, -1 means filled circle


### https://github.com/somanchiu/ReSwapper/blob/GAN/face_align.py

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)


def estimate_norm(lmk, image_size=112,mode='arcface'): # pylint: disable=unused-argument
    from skimage import transform as trans
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    ratio = float(image_size)/112.0
    diff_x = 0
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M


def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def norm_crop2(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, M


def square_crop(im, S):
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im
    return det_im, scale


def transform(data, center, output_size, scale, rotation):
    from skimage import transform as trans
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i] = new_pt[0:2]
    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    #print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale
    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)
