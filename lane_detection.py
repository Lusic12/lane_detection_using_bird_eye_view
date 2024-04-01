import cv2
import numpy as np
from collections import deque
from convert_bird_eye_view import perspective_unwarp,calc_warp_points,perspective_transforms,perspective_warp
from img_processing import combined_threshold
from find_lane import poly_fitx,calc_fit_from_boxes,find_lane_windows,lane_peaks,calc_lr_fit_from_polys,lane_histogram



class WindowBox(object):
    def __init__(self, binimg, x_center, y_top, width=100, height=40, mincount=100, lane_found=False):
        self.x_center = x_center
        self.y_top = y_top
        self.width = width
        self.height = height
        self.mincount = mincount
        self.lane_found = lane_found
        self.x_left = self.x_center - int(self.width/2)
        self.x_right = self.x_center + int(self.width/2)
        self.y_bottom = self.y_top - self.height
        self.imgwindow = binimg[self.y_bottom:self.y_top, self.x_left:self.x_right]
        self.nonzeroy = self.imgwindow.nonzero()[0]
        self.nonzerox = self.imgwindow.nonzero()[1]

    def center(self):
        return (self.x_center, int(self.y_top - self.y_bottom) / 2)

    def next_windowbox(self, binimg):
        if self.has_line():
            x_center = int(np.mean(self.nonzerox + self.x_left))
        else:
            x_center = self.x_center

        y_top = self.y_bottom

        if x_center - int(self.width/2) < 0:
            x_center = int(self.width/2)
        elif x_center + int(self.width/2) >= binimg.shape[1]:
            x_center = binimg.shape[1] - int(self.width/2) - 1

        return WindowBox(binimg, x_center, y_top,
                         width=self.width, height=self.height, mincount=self.mincount,
                         lane_found=self.lane_found)

    def _nonzerox_count(self):
        return len(self.nonzerox)

    def has_line(self):
        return (self._nonzerox_count() > self.mincount)

    def has_lane(self):
        if not self.lane_found and self.has_line():
            self.lane_found = True
        return self.lane_found

    def __str__(self):
        return "WindowBox [%.3f, %.3f, %.3f, %.3f]" % (self.x_left,
                                                       self.y_bottom,
                                                       self.x_right,
                                                       self.y_top)


class Line():
    NOTE_POLYNOMIAL_INVALID_MSG = "Invalid polynomial for the line."

    def __init__(self, ploty, poly_fit, binimg):
        self.__ploty = ploty
        self.__poly_fit = poly_fit
        self.__binimg = binimg
        self.__y_bottom = np.min(ploty)
        self.__y_top = np.max(ploty)

        try:
            self.__x_bottom = poly_fitx(self.__y_bottom, self.poly_fit)
            self.__x_top = poly_fitx(self.__y_top, self.poly_fit)
        except TypeError:
            raise ValueError(Line.NOTE_POLYNOMIAL_INVALID_MSG)

    @property
    def xs(self):
        return poly_fitx(self.ploty, self.poly_fit)

    @property
    def ploty(self):
        return self.__ploty

    @property
    def poly_fit(self):
        return self.__poly_fit

    @property
    def binimg(self):
        return self.__binimg

    @property
    def y_bottom(self):
        return self.__y_bottom

    @property
    def y_top(self):
        return self.__y_top

    @property
    def x_bottom(self):
        return self.__x_bottom

    @property
    def x_top(self):
        return self.__x_top

    def __str__(self):
        return "Line( %s, bot:(%d,%d) top:(%d,%d))" % (self.poly_fit,
                                                        self.x_bottom, self.y_bottom,
                                                        self.x_top, self.y_top)


class RoadLine():
    LINE_ISNT_SANE_MSG = "Line didn't pass sanity checks."

    def __init__(self, line, poly_fit, line_history_max=6):
        self.__line_history = deque([])
        self.line_history_max = line_history_max
        self.line = line
        self.poly_fit = poly_fit

    def __str__(self):
        return "RoadLine(%s, poly_fit=%s, mean_fit=%])" % (self.line, self.poly_fit,self.mean_fit)

    @property
    def line(self):
        if len(self.__line_history) > 0:
            return self.__line_history[-1]
        else:
            return None

    @line.setter
    def line(self, line):
        self._queue_to_history(line)

    def _queue_to_history(self, line):
        self.__line_history.append(line)
        if self.line_history_count > self.line_history_max:
            self.__line_history.popleft()

    @property
    def line_history_count(self):
        return len(self.__line_history)

    @property
    def line_fits(self):
        return np.array([line.poly_fit for line in self.__line_history])

    @property
    def mean_fit(self):
        lf = self.line_fits
        nweights = len(lf)

        weights = None
        if nweights > 1:
            if nweights == 2:
                weights = [.20, .80]  # Giảm trọng số của trọng số mới

            else:
                weights = [.50, .50]  # Giảm trọng số của trọng số mới
                if nweights > len(weights):
                    weights = np.pad(weights, (0, nweights - len(weights)), 'constant',
                                     constant_values=(1 - np.sum(weights)) / (nweights - len(weights)))

        return np.average(lf, weights=weights, axis=0)

    @property
    def ploty(self):
        return self.line.ploty

    @property
    def mean_xs(self):
        return poly_fitx(self.ploty, self.mean_fit)


class Lane():
    lane_px_width = 413
    xm_per_pix = 3.7 / lane_px_width

    def __init__(self, img_height, img_width):
        self.__img_height = img_height
        self.__img_width = img_width
        self.__image = None
        self.__undistorted_image = None
        self.__binary_image = None
        self.__warped = None
        self.old_left_poly_fit = None
        self.old_right_poly_fit = None
        warp_src, warp_dst = calc_warp_points(img_height, img_width)
        self.__warp_src = warp_src
        self.__warp_dst = warp_dst
        M, Minv = perspective_transforms(warp_src, warp_dst)
        self.__M = M
        self.__Minv = Minv
        warped_height = img_width
        self.__ploty = np.linspace(0, warped_height - 1, warped_height)
        self.__pts_zero = np.zeros((1, warped_height, 2), dtype=int)
        self.__road_line_left = None
        self.__road_line_right = None

    @property
    def img_height(self):
        return self.__img_height

    @property
    def img_width(self):
        return self.__img_width

    @property
    def pts_zero(self):
        return self.__pts_zero

    @property
    def ploty(self):
        return self.__ploty

    @property
    def undistorted_image(self):
        return self.__undistorted_image

    @property
    def binary_image(self):
        return self.__binary_image

    @property
    def warped(self):
        return self.__warped

    @property
    def road_line_left(self):
        return self.__road_line_left

    @road_line_left.setter
    def road_line_left(self, value):
        self.__road_line_left = value

    @property
    def road_line_right(self):
        return self.__road_line_right

    @road_line_right.setter
    def road_line_right(self, value):
        self.__road_line_right = value

    @property
    def result(self):
        result,distance, text= self._draw_lanes_unwarped()
        return result,distance,text

    @property
    def result_decorated(self):
        return self.result

    @property
    def image(self):
        return self.__image

    @image.setter
    def image(self, image):
        self.__image = image
        undistorted=self.__image
        self.__undistorted_image = image
        self.__binary_image, self.__warped = self._undistort_warp_search(undistorted)

        if self.road_line_left is None and self.road_line_right is None  :
            self._full_histogram_lane_search()
        else:
            self._recalc_road_lines_from_polyfit()

    def _undistort_warp_search(self, undistorted):
        binary_image = combined_threshold(undistorted)
        warped = perspective_warp(binary_image, self.__M)
        cv2.imshow('warped',warped)
        return binary_image, warped

    @property
    def x_center_offset(self):
        lx = self.x_start_left
        rx = self.x_start_right
        xcenter = int(self.warped.shape[1] / 2)
        offset = (rx - xcenter) - (xcenter - lx)
        return self._x_pix_to_m(offset)

    @property
    def x_start_left(self):
        return self.road_line_left.line.x_top

    @property
    def x_start_right(self):
        return self.road_line_right.line.x_top

    def _x_pix_to_m(self, pix):
        return pix * self.xm_per_pix

    def _full_histogram_lane_search(self):
        histogram = self.warped_histogram
        peaks = lane_peaks(histogram)

        if len(peaks) > 1:
            peak_left, *_, peak_right = peaks
            left_line, left_poly_fit = self._road_line_box_search(peak_left, self.old_left_poly_fit)
            right_line, right_poly_fit = self._road_line_box_search(peak_right, self.old_right_poly_fit)

            self.__road_line_left = RoadLine(left_line, left_poly_fit)
            self.__road_line_right = RoadLine(right_line, right_poly_fit)

            self.old_left_poly_fit = left_poly_fit
            self.old_right_poly_fit = right_poly_fit
        else:
            try:
                if  peaks[0] < 120:
                    peak_left = peaks[0] if peaks else 25
                    left_line, left_poly_fit = self._road_line_box_search(peak_left, self.old_left_poly_fit)
                    self.__road_line_left = RoadLine(left_line, left_poly_fit)
                    self.old_left_poly_fit = left_poly_fit
                else:
                    peak_right = peaks[-1] if peaks else 200
                    right_line, right_poly_fit = self._road_line_box_search(peak_right, self.old_right_poly_fit)
                    self.__road_line_right = RoadLine(right_line, right_poly_fit)
                    self.old_right_poly_fit = right_poly_fit
            except Exception as e:
                print("An error occurred:", e)

    @property
    def warped_histogram(self):
        return lane_histogram(self.warped, int(50), 300)

    def _road_line_box_search(self, x_start, old, nwindows=12, width=40):
        ytop = self.warped.shape[0]
        height = int(ytop / nwindows)

        wb = WindowBox(self.warped, x_start, ytop, width=width, height=height)
        boxes = find_lane_windows(wb, self.warped)
        poly_fit = calc_fit_from_boxes(boxes, old)
        line = Line(self.ploty, poly_fit, self.warped)
        return line, poly_fit

    def _recalc_road_lines_from_polyfit(self, margin=5):
        left_fit = self.road_line_left.line.poly_fit
        right_fit = self.road_line_right.line.poly_fit
        try:

            try:
                new_right_fit = calc_lr_fit_from_polys(self.warped, right_fit, margin)

                self.road_line_right.line = self._line_from_fit(new_right_fit)

            except:
                new_left_fit = calc_lr_fit_from_polys(self.warped, left_fit, margin)

                self.road_line_left.line = self._line_from_fit(new_left_fit)


        except:
            self._full_histogram_lane_search()



    def _line_from_fit(self, new_fit):
        if new_fit is None:
            raise ValueError("no polynominal fit")
        line = Line(self.ploty, new_fit, self.warped)
        return line

    def _draw_lanes_unwarped(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        distance = 0
        text = None
        warp_zero = np.zeros_like(self.warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        if self.road_line_left is not None and self.road_line_right is not None:
            mean_xs_left, mean_xs_right = self.road_line_left.mean_xs, self.road_line_right.mean_xs
            pts_left = np.array([np.transpose(np.vstack([mean_xs_left, self.ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([mean_xs_right, self.ploty])))])
            vertex1 = (int(pts_left[0][0][0]), int(pts_left[0][0][1]))
            vertex3 = (int(pts_right[0][-1][0]), int(pts_right[0][-1][1]))

            pts = np.hstack((pts_left, pts_right))

            cv2.fillPoly(color_warp, np.int_([pts]), (0, 120, 255))

            midpoint_x = (vertex3[0] + vertex1[0]) // 2
            midpoint_y = (vertex3[1] + vertex1[1]) // 2
            distance = vertex3[0]-vertex1[0]
            midpoint = (int(midpoint_x), int(midpoint_y))
            cv2.circle(color_warp, (110, 0), 2, (255, 255, 0), -1)
            cv2.circle(color_warp, midpoint, 2, (0, 0, 255), -1)
            text = None

        elif self.road_line_left is not None:
            mean_xs_left = self.road_line_left.mean_xs
            pts_left = np.array([np.transpose(np.vstack([mean_xs_left, self.ploty]))])
            cv2.fillPoly(color_warp, np.int_([pts_left]), (0, 120, 255))
            vertex4 = (int(pts_left[0][-1][0]), int(pts_left[0][-1][1]))
            midpoint_x = vertex4[0]
            midpoint_y = vertex4[1]
            midpoint = (int(midpoint_x), int(midpoint_y))

            cv2.circle(color_warp, midpoint, 20, (255, 0, 0), -1)
            text = f"bam lane trai"
            distance = 0
        elif self.road_line_right is not None:

            mean_xs_right = self.road_line_right.mean_xs
            pts_right = np.array([np.flipud(np.transpose(np.vstack([mean_xs_right, self.ploty])))])
            cv2.fillPoly(color_warp, np.int_([pts_right]), (0, 0, 255))
            vertex2 = (int(pts_right[0][0][0]), int(pts_right[0][0][1]))
            midpoint_x = vertex2[0]
            midpoint_y = vertex2[1]
            midpoint = (int(midpoint_x), int(midpoint_y))


            cv2.circle(color_warp, midpoint, 20, (255, 0, 0), -1)
            text = f"bam lane phai"
            distance = 0

        newwarp = perspective_unwarp(color_warp, self.__Minv)
        result = cv2.addWeighted(self.undistorted_image, 1, newwarp, 0.3, 0)
        # Add text overlay for distance


        return result, distance, text