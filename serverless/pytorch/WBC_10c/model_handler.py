import numpy as np
from ultralytics import YOLO
from PIL import Image

class ModelHandler:
    def __init__(self, model_path, labels, logger):
        self.labels = labels
        self.logger = logger
        self.model_path = model_path
        self.input_size = 1920
        self.model = YOLO(self.model_path)
        self.model.to('cuda')
        self.logger.info("Model loaded on GPU")


    def infer(self, image_pil: Image, threshold: float):
        image_results = self.model(image_pil, conf=threshold, agnostic_nms=True ,imgsz=self.input_size, verbose=False)
        self.logger.info(f"Model returned {len(image_results[0].boxes)} detections.")
        results = []
        for bbox in image_results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = bbox
            predicted_class_id = int(class_id)
            if predicted_class_id in self.labels:
                results.append({
                    "confidence": f"{confidence:.2f}",
                    "label": self.labels[predicted_class_id],
                    "points": [int(x1), int(y1), int(x2), int(y2)],
                    "type": "rectangle",
                })

        self.logger.info(f"Returning {len(results)} final annotations to CVAT.")
        self.unload()
        return results
    
    def unload(self):
        import torch
        if torch.cuda.is_available():
            self.logger.info("Unloading model from GPU memory.")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            self.logger.info("GPU memory cleared.")