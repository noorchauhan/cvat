import json
import base64
import io
import traceback
from PIL import Image
from model_handler import ModelHandler

CLASS_NAMES = {
    0: "agg_platelet",
    1: "giant_platelet"
}

def init_context(context):
    context.logger.info("Initializing Ultralytics YOLO context...")
    model_path = "/opt/nuclio/PLT_2400-1280_agg_giant_aug_b4_n_70-30_fold_4_best.pt" # add your model path here
    model = ModelHandler(model_path, CLASS_NAMES, context.logger)
    context.user_data.model = model


def handler(context, event):
    try:
        data = event.body
        buf = io.BytesIO(base64.b64decode(data["image"]))
        threshold = float(data.get("threshold", 0.40))
        image = Image.open(buf).convert("RGB")
        try:
            results = context.user_data.model.infer(image, threshold)
            return context.Response(body=json.dumps(results), headers={},
                content_type='application/json', status_code=200)
        except Exception as e:
            context.user_data.model.unload() 
            raise       
        results = context.user_data.model.infer(image, threshold)

        return context.Response(body=json.dumps(results), headers={},
            content_type='application/json', status_code=200)

    except Exception:
        context.logger.error(f"Error during inference: {traceback.format_exc()}")
        return context.Response(body=traceback.format_exc(), status_code=500)