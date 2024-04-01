import cv2
import numpy as np
from scipy import signal




def lane_histogram(img, height_start, height_end):
    histogram = np.sum(img[int(height_start):int(height_end),:], axis=0)
    return histogram

#sai
def calc_fit_from_boxes(boxes,old_fit):
    if len(boxes) > 0:
        # flaten and adjust all boxes for the binary images
        xs = np.concatenate([b.nonzerox + b.x_left for b in boxes])
        ys = np.concatenate([b.nonzeroy + b.y_bottom for b in boxes])

        # return the polynominal
        return np.polyfit(ys, xs, 2)
    else:
        return old_fit









def find_lane_windows(window_box, binimg):
    boxes = []
    continue_lane_search = True
    contiguous_box_no_line_count = 0

    while continue_lane_search and window_box.y_top >= 0:
        if window_box.has_line():
            # Kiểm tra và điều chỉnh nếu cửa vượt ra khỏi khung hình
            if window_box.x_left < 0:
                window_box.x_left = 0
                window_box.x_center = window_box.x_left + int(window_box.width/2)
            elif window_box.x_right >= binimg.shape[1]:
                window_box.x_right = binimg.shape[1] - 1
                window_box.x_center = window_box.x_right - int(window_box.width/2)

            boxes.append(window_box)

        window_box = window_box.next_windowbox(binimg)

        if window_box.has_lane():
            if window_box.has_line():
                contiguous_box_no_line_count = 0
            else:
                contiguous_box_no_line_count += 1

                if contiguous_box_no_line_count >= 4:
                    continue_lane_search = False

    return boxes

def lane_peaks(histogram):
    peaks = signal.find_peaks_cwt(histogram, np.arange(1, 150), min_length=100)

    return peaks
def poly_fitx(fity, line_fit):
    fit_linex = line_fit[0]*fity**2 + line_fit[1]*fity + line_fit[2]
    return fit_linex

def calc_curvature(poly, height=140):
    fity = np.linspace(0, height - 1, num=height)
    y_eval = np.max(fity)

    # Define conversions in x and y from pixels space to meters
    # ym_per_pix = 13.5/1280 # meters per pixel in y dimension
    # lane_px_height=130 # manual observation
    lane_px_height = 275  # manual observation
    ym_per_pix = (3. / lane_px_height)  # meters per pixel in y dimension
    # xm_per_pix = 6.5/720 # meters per pixel in x dimension
    lane_px_width = 413
    xm_per_pix = 3.7 / lane_px_width

    def fit_in_m(poly):
        xs = poly_fitx(fity, poly)
        xs = xs[::-1]  # Reverse to match top-to-bottom in y

        return np.polyfit(fity * ym_per_pix, xs * xm_per_pix, 2)

    if poly is None:
        return .0

    poly_cr = fit_in_m(poly)
    curveradm = ((1 + (2 * poly_cr[0] * y_eval * ym_per_pix + poly_cr[1]) ** 2) ** 1.5) / np.absolute(2 * poly_cr[0])

    return curveradm

def fit_window(binimg, poly, margin=20):
    height = binimg.shape[0]
    y = binimg.shape[0]

    nonzero = binimg.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    fity = np.linspace(0, height - 1, height)

    def window_lane(poly):
        return (
                (nonzerox > (poly[0] * (nonzeroy ** 2) + poly[1] * nonzeroy + poly[2] - margin))
                & (nonzerox < (poly[0] * (nonzeroy ** 2) + poly[1] * nonzeroy + poly[2] + margin))
        )

    def fit(lane_inds):
        xs = nonzerox[lane_inds]

        return np.polyfit(fity, xs, 2)

    return fit(window_lane(poly))
def calc_lr_fit_from_polys(binimg, line, margin):
    nonzero = binimg.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    def window_lane(poly):
        if poly is None:
            return None
        return (
            (nonzerox > (poly[0] * (nonzeroy ** 2) + poly[1] * nonzeroy + poly[2] - margin))
            & (nonzerox < (poly[0] * (nonzeroy ** 2) + poly[1] * nonzeroy + poly[2] + margin))
        )

    def window_polyfit(lane_inds):
        if len(lane_inds) == 0:
            return None
        xs = nonzerox[lane_inds]
        ys = nonzeroy[lane_inds]
        # Tránh trường hợp chia cho 0
        if len(ys) == 0:
            return None
        # return the polynominal
        return np.polyfit(ys, xs, 2)

    new_line = window_polyfit(window_lane(line))

    return (new_line)
