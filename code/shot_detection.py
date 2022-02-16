
from scenedetect import VideoManager
from scenedetect import SceneManager

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector


class ShotDetector:
    def __init__(self, video_path):
        self.video_path = video_path


    def find_scenes(self, threshold=10.0):
        # Create our video & scene managers, then add the detector.
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=threshold))

        # Improve processing speed by downscaling before processing.
        video_manager.set_downscale_factor()

        # Start the video manager and perform the scene detection.
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        # scene_manager.save-images()
        # Each returned scene is a tuple of the (start, end) timecode.
        return scene_manager.get_scene_list()

def main():
    scene_list = ShotDetector(video_path="videos/vid2.mp4").find_scenes()
    print(scene_list)

if __name__ == "__main__":
    main()
