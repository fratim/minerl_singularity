#
# Misc tools for visualizing/monitoring/logging
# agent performance
#
import os

import cv2


class VideoRecorder():
    """
    A simple class for saving frames into a video
    """

    def __init__(self, width, height, save_path, fps=35.0):
        """
        Parameters:
            width, height (int): Width and height of the video to be recorded
            save_path (str): Where video should be stored
            fps (float): The FPS at which video should be stored
        """
        self.save_path = save_path
        self.fps = fps
        self.video_buffer = []
        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_out = cv2.VideoWriter(
            self.save_path + ".avi", fourcc, fps, (width, height)
        )

    def save_frame(self, img):
        """
        Save a frame to video.

        Parameters:
            img (np.ndarray): A (height, width, [channels]) ndarray
                              to be saved. channels is optional or 3,
                              representing either grayscale or RGB image.
        """
        if img.ndim > 2:
            # Flip RGB to BGR
            self.video_out.write(img[..., ::-1])
        else:
            # Grayscale
            self.video_out.write(img)

    def finish_video(self):
        self.video_out.release()

    def __del__(self):
        self.finish_video()
