import time
import torch
#from transformers import AutoProcessor, AutoModelForImageTextToText
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from flask import Flask, request, jsonify
import logging
import json
import re
import sys

# --- Global Constants ---
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# --- Environment Setup ---
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables for Model and Processor ---
model = None
processor = None
# primary_device will be set to "cuda:0", "mps", or "cpu"
# It indicates the main device for initial tensor placement and for single-device scenarios.
primary_device = None
# model_dtype will be set based on device capabilities
model_dtype = None


def load_model_and_processor():
  global model, processor, primary_device, model_dtype, MODEL_NAME

  logger.info(f"Initiating model and processor loading sequence for model: {MODEL_NAME}...")
  overall_start_time = time.perf_counter()

  # --- Advanced Device Detection ---
  rocm_detected = False
  if torch.cuda.is_available():
    # Check if this CUDA is actually ROCm/HIP
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
      rocm_detected = True
      primary_device = "cuda:0" # ROCm GPUs are addressed as cuda devices
      model_dtype = torch.bfloat16 # Or torch.float16 if bfloat16 is problematic on your AMD GPU
      logger.info(f"AMD ROCm detected. {torch.cuda.device_count()} GPU(s) found. Using device_map='auto'. Primary device: {primary_device}, dtype: {model_dtype}")
    elif torch.cuda.device_count() > 0: # NVIDIA CUDA
      primary_device = "cuda:0"
      model_dtype = torch.bfloat16
      logger.info(f"NVIDIA CUDA detected. {torch.cuda.device_count()} GPU(s) found. Using device_map='auto'. Primary device: {primary_device}, dtype: {model_dtype}")
    else: # CUDA available but no devices (edge case)
      logger.warning("torch.cuda.is_available() is True, but no CUDA devices found. Checking MPS.")
      primary_device = None # Fall through to MPS or CPU

  if primary_device is None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    primary_device = "mps"
    model_dtype = torch.float16 # bfloat16 not well supported on MPS, float16 or float32
    logger.info(f"Apple MPS detected. Using device: {primary_device}, dtype: {model_dtype}. device_map='auto' will target MPS.")

  if primary_device is None: # Fallback to CPU
    primary_device = "cpu"
    model_dtype = torch.float32
    logger.info(f"No CUDA, ROCm, or MPS detected. Using CPU. Device: {primary_device}, dtype: {model_dtype}")

  logger.info(f"Selected primary_device for operations: {primary_device}, model_dtype: {model_dtype}")

  # --- 模型加载 ---
  logger.info(f"开始加载模型 {MODEL_NAME} (using device_map='auto')...")
  start_time_model_load = time.perf_counter()

  try:
    # device_map="auto" will try to use available GPUs (CUDA/ROCm) or fall back to primary_device (MPS/CPU)
    # For MPS, "auto" should correctly map to the single "mps" device.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      MODEL_NAME,
      torch_dtype=model_dtype,
      device_map="auto",  # Let accelerate handle device mapping
      trust_remote_code=True
    )
    logger.info(f"Model {MODEL_NAME} loaded.")
    if hasattr(model, 'hf_device_map'):
      logger.info(f"Model device map: {model.hf_device_map}")
      # The device_map might put parts of the model on CPU if GPU memory is insufficient.
      # The 'model.device' attribute might point to the device of the first/last parameter or be 'meta'.
      # We can check where the first parameter actually landed as an indication.
      first_param_device = next(model.parameters()).device
      logger.info(f"First model parameter is on device: {first_param_device}")
      # If device_map led to CPU usage even if a GPU was primary_device, primary_device for inputs should reflect that.
      # However, inputs.to(primary_device) is generally fine, as accelerate handles internal transfers.
    else:
      logger.info(f"Model loaded to single device: {model.device if hasattr(model, 'device') else 'Unknown'}. (No hf_device_map attribute implies not sharded by accelerate, or on CPU/MPS).")
      # If not sharded, model.device should be reliable.
      if hasattr(model, 'device') and primary_device != str(model.device):
        logger.warning(f"Primary device was {primary_device} but model ended up on {model.device}. This might happen if device_map='auto' chose differently.")
        # primary_device = str(model.device) # Could update primary_device here but let's see

  except Exception as e:
    logger.error(f"Error loading model {MODEL_NAME}: {e}", exc_info=True)
    raise

  # Synchronization for timing
  if primary_device.startswith("cuda"): # Covers NVIDIA CUDA and AMD ROCm
    torch.cuda.synchronize(device=primary_device.split(':')[0] + (f":{torch.cuda.current_device()}" if ':' not in primary_device else primary_device) if torch.cuda.device_count() > 0 else None) # Sync current device on that backend
  elif primary_device == "mps":
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'): # Check if mps module and sync exist
      torch.mps.synchronize()

  end_time_model_load = time.perf_counter()
  logger.info(f"模型加载耗时: {end_time_model_load - start_time_model_load:.4f} 秒")

  # --- 处理器加载 ---
  logger.info(f"开始加载处理器 for {MODEL_NAME}...")
  start_time_processor_load = time.perf_counter()
  processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
  end_time_processor_load = time.perf_counter()
  logger.info(f"处理器加载耗时: {end_time_processor_load - start_time_processor_load:.4f} 秒")

  overall_end_time = time.perf_counter()
  logger.info(f"模型和处理器总加载耗时: {overall_end_time - overall_start_time:.4f} 秒")


@app.route('/infer', methods=['POST'])
def infer_images():
  global model, processor, primary_device, model_dtype # model_dtype is mostly for loading, but good to have
  if not model or not processor:
    logger.error("Model or processor not loaded before inference call.")
    return jsonify({"code": 503, "error": "Service Unavailable: Model or processor not loaded"}), 503

  try:
    request_data = request.get_json()
    # ... (input validation as before)
    if not request_data:
      logger.warning("No JSON data received in request.")
      return jsonify({"code": 400, "error": "Bad Request: No JSON data received"}), 400

    image_urls = request_data.get('images')
    custom_prompt = request_data.get('prompt', "Your task is to act as an OCR tool. Please extract all text from the provided image(s) and return it structured as a JSON object. If multiple distinct pieces of information are present (like on an ID card), ensure each piece is a key-value pair in the JSON. For example, for an ID card, keys might include 'name', 'number', 'address', etc.")


    if not image_urls or not isinstance(image_urls, list) or not all(isinstance(url, str) for url in image_urls):
      logger.warning(f"Invalid 'images' field: {image_urls}")
      return jsonify({"code": 400, "error": "Bad Request: Invalid or missing 'images' field. It should be a list of URLs."}), 400
    if not image_urls:
      logger.warning("'images' list is empty.")
      return jsonify({"code": 400, "error": "Bad Request: 'images' list cannot be empty."}), 400

    logger.info(f"Received {len(image_urls)} image(s) for inference. Using prompt: '{custom_prompt}'")
    inference_total_start_time = time.perf_counter()

    user_content = []
    for img_url in image_urls:
      user_content.append({"type": "image", "image": img_url})
    user_content.append({"type": "text", "text": custom_prompt})

    messages = [
      {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant. You must respond in JSON format as requested by the user."}]},
      {"role": "user", "content": user_content}
    ]

    logger.info("开始准备输入数据...")
    start_time_input_prep = time.perf_counter()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    # Inputs should be moved to the primary_device for consistency before model.generate
    # If device_map="auto" sharded the model, model.generate will handle internal transfers.
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(primary_device)

    # Synchronize after data transfer to the primary device
    if primary_device.startswith("cuda"): # Covers NVIDIA and ROCm
      torch.cuda.synchronize(device=primary_device)
    elif primary_device == "mps":
      if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
        torch.mps.synchronize()

    end_time_input_prep = time.perf_counter()
    logger.info(f"输入数据准备耗时 (包括 .to('{primary_device}')): {end_time_input_prep - start_time_input_prep:.4f} 秒")

    logger.info("开始推理 (模型生成)...")
    start_time_inference = time.perf_counter()
    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    # Synchronize on the device where the output tensor landed
    output_device = generated_ids.device
    logger.info(f"Generated IDs tensor is on device: {output_device}")
    if output_device.type == 'cuda': # Covers NVIDIA and ROCm
      torch.cuda.synchronize(device=output_device)
    elif output_device.type == 'mps':
      if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
        torch.mps.synchronize()

    end_time_inference = time.perf_counter()
    logger.info(f"推理 (model.generate) 耗时: {end_time_inference - start_time_inference:.4f} 秒")

    logger.info("开始后处理 (解码)...")
    start_time_post_processing = time.perf_counter()
    # Move generated_ids to CPU for decoding
    generated_ids_cpu = generated_ids.to('cpu')
    # Ensure input_ids are also on CPU for length calculation if they were on GPU
    input_ids_cpu = inputs.input_ids.to('cpu') if inputs.input_ids.device.type != 'cpu' else inputs.input_ids


    generated_ids_trimmed = [
      out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids_cpu, generated_ids_cpu)
    ]
    output_text_list = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output_text = output_text_list[0] if output_text_list else ""
    end_time_post_processing = time.perf_counter()
    logger.info(f"后处理 (解码) 耗时: {end_time_post_processing - start_time_post_processing:.4f} 秒")
    logger.info(f"模型原始输出: {output_text}")

    # --- Response Formatting ---
    # (Your existing response formatting logic - unchanged)
    parsed_data = None
    response_payload = {}
    http_status_code = 200

    try:
      parsed_data = json.loads(output_text)
      logger.info("Successfully parsed model output directly as JSON.")
    except json.JSONDecodeError:
      logger.info("Direct JSON parsing failed. Attempting to extract from markdown code block.")
      match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", output_text, re.IGNORECASE)
      if match:
        json_str = match.group(1).strip()
        logger.info(f"Extracted potential JSON string from markdown: '{json_str}'")
        if not json_str:
          error_msg = "JSON code block was found but was empty."
          logger.warning(error_msg)
          response_payload = {"code": 422, "error": "Unprocessable Entity: Model output contained an empty JSON code block.", "raw_model_output": output_text}
          http_status_code = 422
        else:
          try:
            parsed_data = json.loads(json_str)
            logger.info("Successfully parsed JSON extracted from markdown.")
          except json.JSONDecodeError as e_inner:
            error_msg = f"Failed to parse JSON extracted from markdown: {e_inner}"
            logger.error(error_msg)
            logger.error(f"Problematic JSON string after extraction: '{json_str}'")
            response_payload = {"code": 422, "error": "Unprocessable Entity: Model output contained a JSON-like block, but its content was malformed.", "details": str(e_inner), "raw_model_output": output_text}
            http_status_code = 422
      else:
        error_msg = "Model output was not in a parsable JSON format and no JSON code block was found."
        logger.warning(error_msg)
        response_payload = {"code": 422, "error": "Unprocessable Entity: Model output was not in the expected JSON format.", "raw_model_output": output_text}
        http_status_code = 422

    if not response_payload:
      if parsed_data is not None:
        response_payload = {"code": 200, "data": parsed_data}
      else:
        logger.error("Internal logic error: parsed_data is None but no error response was constructed.")
        response_payload = {"code": 500, "error": "Internal Server Error: Failed to process model output due to an unexpected state.", "raw_model_output": output_text}
        http_status_code = 500

    inference_total_end_time = time.perf_counter()
    logger.info(f"总推理请求处理耗时 (从接收到图片到生成文本): {inference_total_end_time - inference_total_start_time:.4f} 秒")

    return jsonify(response_payload), http_status_code

  except Exception as e:
    logger.error(f"Unhandled error during inference: {str(e)}", exc_info=True)
    return jsonify({"code": 500, "error": "Internal Server Error", "details": str(e)}), 500


if __name__ == '__main__':
  # Ensure 'accelerate' is installed for device_map="auto" to work effectively
  # pip install accelerate
  load_model_and_processor()
  app.run(host='0.0.0.0', port=8080 if sys.platform == "darwin" else 5000, debug=False)
