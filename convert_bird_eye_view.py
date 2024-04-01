import cv2
import numpy as np





def camera_calibrate(objpoints, imgpoints, img):
    """Calibrate camera and undistort an image."""
    # Test undistortion on an image
    img_size = img.shape[0:2]
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return ret, mtx, dist, dst

def  undistort_image(img, mtx, dist):
    # Assuming mtx and dist are the camera calibration matrices
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def perspective_warp(img, M):
    # img_size = (img.shape[1], img.shape[0])
    img_size = (img.shape[0], img.shape[1])

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped
def perspective_transforms(src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

def perspective_unwarp(img, Minv):
    img_size = (img.shape[0], img.shape[1])

    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)

    return unwarped


def calc_warp_points(img_height, img_width, x_center_adj=20):
    imshape = (img_height, img_width)
    xcenter = imshape[1] / 2 + x_center_adj
    xoffset = 0
    xfd = 140
    yf = 140
    src = np.float32([
        (xoffset, imshape[0]),
        (xcenter - xfd, yf),
        (xcenter + xfd, yf),
        (imshape[1] - xoffset, imshape[0])
    ])

    dst = np.float32([
        (xoffset, imshape[1]),
        (xoffset, 0),
        (imshape[0] - xoffset, 0),
        (imshape[0] - xoffset, imshape[1])
    ])

    return src, dst

