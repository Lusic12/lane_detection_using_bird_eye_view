import cv2
from lane_detection import Lane  # Replace 'your_module_name' with the actual name of your module
#from spi_rasptostm32 import SPI
#from camera_pi import MyCamera
from camera_opencv import camera
video_path="/home/lucis/xla.mp4"
def pause_video(paused):
    while True:
        key = cv2.waitKey(0)
        if key == ord('p'):
            break



def main():
    lane = Lane(img_height=240, img_width=320)  # Replace with actual calibration parameters
    #cam pi
    #camera = MyCamera()  # Create an instance of the camera
    player = camera(video_path)

    paused = False  # Flag to control pausing

    while True:
        #using campi
        #frame = camera.capture_image()  # Capture frame from the camera
        frame=player.play()
        lane.image = frame
        result, distance, text = lane.result_decorated
        if text is not None:
            cv2.putText(result, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(result, f"Distance: {distance}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        2,
                        cv2.LINE_AA)

        if distance < 100:
            lane.road_line_left = None
            lane.road_line_right = None

        # Display the result
        cv2.imshow('Lane Detection', result)
        key = cv2.waitKey(1)
        # Pause or resume the video on 'p' key press
        if key == ord('p'):
            paused = not paused
            if paused:
                print("Video paused. Press 'p' to resume.")
                pause_video()

        # Exit the loop if 'q' key is pressed
        elif key == ord('q'):
            break

#using cam pi
#camera.close()  # Close the camera

if __name__ == "__main__":
    main()
