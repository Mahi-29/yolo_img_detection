
import cv2
import torch
import traceback

class Detection:

    def __init__(self, base_path) -> None:
        self.image_base_path = base_path
        self.image_width = 1024
        self.image_height = 1024
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
        self.model.conf = 0.5
        self.model.iou = 0.45

    def preprocessing_image(self, image):
        print("inside_preprocessing")
        resized_image = cv2.resize(image, (self.image_width, self.image_height))
        return resized_image

    def get_bounding_box(self, image, bbox_data):
        try:
            for i in range(len(bbox_data)):
                image = cv2.rectangle(image, (int(bbox_data["xmin"][i]), int(bbox_data["ymin"][i])), (int(bbox_data["xmax"][i]), int(bbox_data["ymax"][i])), (36,255,12), 1)
                image = cv2.putText(image, f"{bbox_data['name'][i]} {round(bbox_data['confidence'][i], 2)}", (int(bbox_data["xmin"][i]), int(bbox_data["ymin"][i])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
        except Exception as e:
            print("test",e)
            traceback.print_exc()
        return image

    def inference(self, image_path):
        try:
            image = cv2.imread(image_path)
            print("inside_inference")
            preprocessed_image = self.preprocessing_image(image)
        except:
            traceback.print_exc()
            return False
        
        try:
            results = self.model(preprocessed_image)
            bbdata = results.pandas().xyxy[0]
            preprocessed_image = self.get_bounding_box(preprocessed_image, bbdata)
            cv2.imwrite(self.image_base_path,preprocessed_image)
            return True
        except:
            traceback.print_exc
            return False
