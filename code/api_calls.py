from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import requests
import time
import json


class APICalls:
    def __init__(self, subscription_key, endpoint):
        self.subscription_key = subscription_key
        self.endpoint = endpoint

        self.computervision_client = ComputerVisionClient(self.endpoint,
                                                          CognitiveServicesCredentials(self.subscription_key))
        self.information_dict = {}

    def read_text(self, image):
        try:
            text_recognition_url = self.endpoint + "/vision/v3.1/read/analyze"
            image.seek(0)
            headers = {'Ocp-Apim-Subscription-Key': self.subscription_key,
                       'Content-Type': 'application/octet-stream'}
            params = {'language': 'unk',
                      'detectOrientation': 'true'}
            response = requests.post(
                text_recognition_url,
                headers=headers,
                data=image)
            response.raise_for_status()
            # operation_url = response.headers["Operation-Location"]

            analysis = {}
            poll = True
            while poll:
                response_final = requests.get(
                    response.headers["Operation-Location"], headers=headers)
                analysis = response_final.json()
                time.sleep(1)
                if "analyzeResult" in analysis:
                    poll = False
                if ("status" in analysis) and (analysis['status'] == 'failed'):
                    poll = False

            polygons = []
            if "analyzeResult" in analysis:
                # Extract the recognized text, with bounding boxes.
                polygons = [(line["boundingBox"], line["text"])
                            for line in analysis["analyzeResult"]["readResults"][0]["lines"]]

            print('>>', polygons)

        except Exception as e:
            print(e)

    def object_detection(self, image):
        try:
            image.seek(0)
            object_information = []
            detect_objects_results = self.computervision_client.detect_objects_in_stream(image)
            if len(detect_objects_results.objects) == 0:
                self.information_dict.update({"object": False,
                                              "object_information": object_information})
            if len(detect_objects_results.objects):
                # stored as [x1, y1, x2, y2]
                for obj in detect_objects_results.objects:
                    # print('>>', obj.object_property)
                    coords = [
                        obj.rectangle.x,
                        obj.rectangle.y,
                        obj.rectangle.x + obj.rectangle.w,
                        obj.rectangle.y + obj.rectangle.h
                    ]
                    object_information.append({"object": obj.object_property,
                                               "coords": coords})

                self.information_dict.update({"object": True,
                                              "object_information": object_information})

        except Exception as e:
            print(e)

    def domain_specific_celebrity(self, image):
        try:
            image.seek(0)
            celebs_results = self.computervision_client.analyze_image_by_domain_in_stream("celebrities",
                                                                                          image)
            celebrities = []
            if len(celebs_results.result["celebrities"]) == 0:
                self.information_dict.update({"celebrity": False,
                                              "celeb_names": celebrities})
            else:
                for celeb in celebs_results.result["celebrities"]:
                    celebrities.append(celeb)
                self.information_dict.update({"celebrity": True,
                                              "celeb_names": celebrities})
        except Exception as e:
            print(e)

    def domain_specific_landmark(self, image):
        try:
            image.seek(0)
            landmark_results = self.computervision_client.analyze_image_by_domain_in_stream("landmarks",
                                                                                            image)
            landmarks = []
            if len(landmark_results.result["landmarks"]) == 0:
                self.information_dict.update({"landmark": False,
                                              "lmark_names": landmarks})
            else:
                for landmark in landmark_results.result["landmarks"]:
                    landmarks.append(landmark)
                self.information_dict.update({"landmark": True,
                                              "lmark_names": landmarks})

        except Exception as e:
            print(e)

    def analyze_image(self, image):
        try:
            image.seek(0)
            features = ["brands", "categories", "faces"]

            analysis_results = self.computervision_client.analyze_image_in_stream(image, features)
            brand_information = []
            if len(analysis_results.brands) == 0:
                self.information_dict.update({"brands": False,
                                              "brand_information": brand_information})
            if len(analysis_results.brands):

                for brand in analysis_results.brands:
                    individual_information = {"brand_name": brand.name,
                                              "confidence": brand.confidence * 100,
                                              "coords": [
                                                  brand.rectangle.x,
                                                  brand.rectangle.y,
                                                  brand.rectangle.x + brand.rectangle.w,
                                                  brand.rectangle.y + brand.rectangle.h
                                              ]
                                              }

                    brand_information.append(individual_information)
                self.information_dict.update({"brands": True,
                                              "brand_information": brand_information})

            category_information = []
            if len(analysis_results.categories) == 0:
                self.information_dict.update({"category": False,
                                              "category_information": category_information})

            if len(analysis_results.categories):
                for category in analysis_results.categories:
                    category_information.append({"name": category.name,
                                                 "score": category.score * 100})

                self.information_dict.update({"category": True,
                                              "category_information": category_information})

            face_information = []
            if len(analysis_results.faces) == 0:
                self.information_dict.update({"face": False,
                                              "face_information": face_information})
            if len(analysis_results.faces):
                for face in analysis_results.faces:
                    face_information.append(
                        {
                            "gender": face.gender,
                            "age": face.age,
                            "coords": [
                                face.face_rectangle.left,
                                face.face_rectangle.top,
                                face.face_rectangle.left + face.face_rectangle.width,
                                face.face_rectangle.top + face.face_rectangle.height
                            ]
                        }
                    )
                self.information_dict.update({"face": True,
                                              "face_information": face_information})
        except Exception as e:
            print(e)

    def get_features(self, image):
        """ returns a json for every frame """
        # self.object_detection(image=image)
        # self.analyze_image(image=image)
        # self.domain_specific_celebrity(image=image)
        # self.domain_specific_landmark(image=image)

        self.read_text(image=image)
        ## ocr, text detection
        return self.information_dict
