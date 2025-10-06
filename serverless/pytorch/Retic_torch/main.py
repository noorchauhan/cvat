import json
import base64
import io
import traceback
from PIL import Image
from model_handler import ModelHandler

CLASS_NAMES = {
    0: "RBC",
    1: "reticulocyte"
}

def init_context(context):
    context.logger.info("Initializing RBC/Retic PyTorch context...")
    model_path = "/opt/nuclio/best.pt"      # Enter your model name .pt file, keep it in the same directory
    model = ModelHandler(model_path, CLASS_NAMES, context.logger)
    context.user_data.model = model
    context.logger.info("RBC/Retic PyTorch context initialization complete.")

def handler(context, event):
    try:
        context.logger.info("Handling new request...")
        data = event.body
        buf = io.BytesIO(base64.b64decode(data["image"]))
        threshold = float(data.get("threshold", 0.35))
        image = Image.open(buf).convert("RGB") # You can adjust the preprocessing here according to your suite
        results = context.user_data.model.infer(image, threshold)
        return context.Response(body=json.dumps(results), headers={},
            content_type='application/json', status_code=200)

    except Exception:
        context.logger.error(f"Error during inference: {traceback.format_exc()}")
        return context.Response(body=traceback.format_exc(), status_code=500)
