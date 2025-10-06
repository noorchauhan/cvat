import json
import base64
import io
import traceback
from PIL import Image
from model_handler import ModelHandler

CLASS_NAMES = {
    0: "basophils",
    1: "eosinophils",
    2: "lymphocytes",
    3: "monocytes",
    4: "neutrophils",
    5: "unclassified"
}

def init_context(context):
    model_path = "/opt/nuclio/best.pt"
    model = ModelHandler(model_path, CLASS_NAMES, context.logger)
    context.user_data.model = model


def handler(context, event):
    try:
        data = event.body
        buf = io.BytesIO(base64.b64decode(data["image"]))
        threshold = float(data.get("threshold", 0.35))
        context.logger.info(f"Using confidence threshold: {threshold}")
        image = Image.open(buf).convert("RGB")
        try:
            results = context.user_data.model.infer(image, threshold)
            return context.Response(body=json.dumps(results), headers={},
                content_type='application/json', status_code=200)
        except Exception as e:
            context.user_data.model.unload()
            raise

    except Exception:
        context.logger.error(f"Error during inference: {traceback.format_exc()}")
        return context.Response(body=traceback.format_exc(), status_code=500)