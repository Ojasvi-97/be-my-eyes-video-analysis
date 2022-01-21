import io
import json
import os
import time
import cv2

from api_calls import APICalls

subscription_key = os.environ["azure_key"]
endpoint = os.environ["azure_endpoint"]


class Analyzer:
    def __init__(self):
        self.final_dict = {}

    def analyze_image(self, image_path):

        image = open(image_path, "rb")

        image_details = APICalls(
            subscription_key=subscription_key,
            endpoint=endpoint
        ).get_features(image=image)
        print('>>', image_details)

    def analyze_video(self, video_path):
        try:
            video = cv2.VideoCapture(video_path)
            count = 0
            while True:
                ret, frame = video.read()
                if ret:
                    name = 'results/frame' + str(count) + '.jpg'
                    image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
                    image_bytes = io.BytesIO(image_bytes)
                    image_details = APICalls(subscription_key=subscription_key,
                                             endpoint=endpoint).get_features(image=image_bytes)
                    print(image_details)
                    time.sleep(0.1)
                    self.final_dict.update({count: image_details})
                    cv2.imwrite(name, frame)
                    count += 1
                else:
                    break
            video.release()
            cv2.destroyAllWindows()

            json_file = open('video_data.json', "w")
            json.dump(self.final_dict, json_file)
            json_file.close()

        except Exception as e:
            print(e)


def main():
    analyzer = Analyzer()
    # analyzer.analyze_video('videos/vid2.mp4')
    analyzer.analyze_image(image_path="images/test.jpg")


if __name__ == "__main__":
    main()
