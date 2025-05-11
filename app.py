import sys
from flask import Flask, request, jsonify

# Configuration and custom modules
from config import logger, MODEL_BACKEND, DEFAULT_OCR_PROMPT # Import logger and config
from device_manager import DeviceManager
from model_provider import ModelProvider
from inference_engine import InferenceEngine

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Global Service Objects ---
# These will be initialized at startup
device_manager_instance = None
model_provider_instance = None
inference_engine_instance = None

def initialize_services():
  """Initializes and loads all necessary services (device, model, processor, engine)."""
  global device_manager_instance, model_provider_instance, inference_engine_instance

  logger.info(f"Starting service initialization with backend: {MODEL_BACKEND.upper()}")
  try:
    device_manager_instance = DeviceManager()

    model_provider_instance = ModelProvider(device_manager_instance)
    model_provider_instance.load_resources() # This loads model and processor

    inference_engine_instance = InferenceEngine(
      model=model_provider_instance.model,
      processor=model_provider_instance.processor,
      device_manager=device_manager_instance
    )
    logger.info("All services initialized successfully.")
  except Exception as e:
    logger.critical(f"Fatal error during service initialization: {e}", exc_info=True)
    # Exit if essential services fail to load, as the app cannot function.
    sys.exit(f"Service initialization failed: {e}")


@app.route('/infer', methods=['POST'])
def infer_route():
  global inference_engine_instance
  if not inference_engine_instance:
    logger.error("Inference engine not available. Services might not have initialized correctly.")
    return jsonify({"code": 503, "error": "Service Unavailable: Inference engine not initialized"}), 503

  try:
    request_data = request.get_json()
    if not request_data:
      logger.warning("No JSON data received in request.")
      return jsonify({"code": 400, "error": "Bad Request: No JSON data received"}), 400

    image_urls = request_data.get('images')
    custom_prompt = request_data.get('prompt', DEFAULT_OCR_PROMPT)

    if not image_urls or not isinstance(image_urls, list) or not all(isinstance(url, str) for url in image_urls):
      logger.warning(f"Invalid 'images' field: {image_urls}")
      return jsonify({"code": 400, "error": "Bad Request: Invalid or missing 'images' field. It should be a list of URLs."}), 400
    if not image_urls: # Check after type validation
      logger.warning("'images' list is empty.")
      return jsonify({"code": 400, "error": "Bad Request: 'images' list cannot be empty."}), 400

    response_payload, http_status_code = inference_engine_instance.run_inference(image_urls, custom_prompt)
    return jsonify(response_payload), http_status_code

  except Exception as e:
    logger.error(f"Unhandled error during /infer request: {str(e)}", exc_info=True)
    # Construct a detailed error message for the client if appropriate,
    # but avoid leaking too much internal detail in production.
    return jsonify({"code": 500, "error": "Internal Server Error", "details": "An unexpected error occurred during inference."}), 500


if __name__ == '__main__':
  # Ensure 'accelerate' is installed for device_map="auto" to work effectively
  # And other dependencies like 'transformers', 'torch', 'modelscope' (if used), 'flask'

  # Initialize services before starting the Flask app
  initialize_services()

  # Determine port based on platform (as in original script)
  port = 8080 if sys.platform == "darwin" else 5000
  logger.info(f"Starting Flask application on host 0.0.0.0, port {port}")
  app.run(host='0.0.0.0', port=port, debug=False) # debug=False for production
