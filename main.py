import cv2
from find_lane import Lane  # Replace 'your_module_name' with the actual name of your module
from spi_rasptostm32 import SPI
from camera_pi import MyCamera

def pause_video(paused):
    while True:
        key = cv2.waitKey(0)
        if key == ord('p'):
            break

def create_new_lane_if_needed(lane, distance_threshold, distance_threshold_far, distance):
    current_time = cv2.getTickCount()

    if distance < distance_threshold or distance > distance_threshold_far:
        lane.road_line_left = None
        lane.road_line_right = None

def main():
    lane = Lane(img_height=240, img_width=320)  # Replace with actual calibration parameters
    camera = MyCamera()  # Create an instance of the camera

    paused = False  # Flag to control pausing

    while True:
        frame = camera.capture_image()  # Capture frame from the camera

        if frame is None:
            print("Error: Could not read frame from camera.")
            break

        if not paused:
            lane.image = frame
            result, distance = lane.result_decorated
            create_new_lane_if_needed(lane, 150, 160, distance)
            a=(lane.x_center_offset)
            print(a)
            cv2.imshow('Lane Detection', result)

        key = cv2.waitKey(1)

        if key == ord('p'):
            paused = not paused
            if paused:
                print("Video paused. Press 'p' to resume.")
                pause_video(paused)

        elif key == ord('q'):
            break

    camera.close()  # Close the camera

if __name__ == "__main__":
    main()
