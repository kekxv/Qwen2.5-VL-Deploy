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

# --- Global Constants ---
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct" # Model name as a global constant

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
primary_device = None

def load_model_and_processor():
    global model, processor, primary_device, MODEL_NAME # Include MODEL_NAME if it were to be modified, but here it's just used

    logger.info(f"Initiating model and processor loading sequence for model: {MODEL_NAME}...")
    overall_start_time = time.perf_counter()

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        primary_device = "cuda:0"
        model_dtype = torch.bfloat16
        logger.info(f"CUDA is available. {torch.cuda.device_count()} GPU(s) found. Will use device_map='auto'. Primary device for inputs: {primary_device}")
    else:
        primary_device = "cpu"
        model_dtype = torch.float32
        logger.info("CUDA not available or no GPUs found. Using CPU. device_map='auto' will likely result in CPU usage.")
    logger.info(f"Selected primary_device for initial tensor placement: {primary_device}")

    logger.info(f"开始加载模型 {MODEL_NAME} (using device_map='auto')...")
    start_time_model_load = time.perf_counter()

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME, # Use the global constant
            torch_dtype=model_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info(f"Model {MODEL_NAME} loaded with device_map='auto'.")
        if hasattr(model, 'hf_device_map'):
            logger.info(f"Model device map: {model.hf_device_map}")
        else:
            logger.info("Model does not have hf_device_map attribute.")
        logger.info(f"Model's main device attribute: {model.device}")

    except Exception as e:
        logger.error(f"Error loading model {MODEL_NAME} with device_map='auto': {e}", exc_info=True)
        raise

    end_time_model_load = time.perf_counter()
    logger.info(f"模型加载耗时: {end_time_model_load - start_time_model_load:.4f} 秒")

    logger.info(f"开始加载处理器 for {MODEL_NAME}...")
    start_time_processor_load = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True) # Use the global constant
    end_time_processor_load = time.perf_counter()
    logger.info(f"处理器加载耗时: {end_time_processor_load - start_time_processor_load:.4f} 秒")

    overall_end_time = time.perf_counter()
    logger.info(f"模型和处理器总加载耗时: {overall_end_time - overall_start_time:.4f} 秒")


@app.route('/infer', methods=['POST'])
def infer_images():
    global model, processor, primary_device # MODEL_NAME is accessed directly as a global constant
    if not model or not processor:
        logger.error("Model or processor not loaded before inference call.")
        return jsonify({"code": 503, "error": "Service Unavailable: Model or processor not loaded"}), 503

    try:
        request_data = request.get_json()
        if not request_data:
            logger.warning("No JSON data received in request.")
            return jsonify({"code": 400, "error": "Bad Request: No JSON data received"}), 400

        image_urls = request_data.get('images')
        # Updated default prompt to be more generic for OCR
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
        
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(primary_device)

        if "cuda" in primary_device:
            torch.cuda.synchronize(device=primary_device)
        end_time_input_prep = time.perf_counter()
        logger.info(f"输入数据准备耗时 (包括 .to('{primary_device}')): {end_time_input_prep - start_time_input_prep:.4f} 秒")

        logger.info("开始推理 (模型生成)...")
        start_time_inference = time.perf_counter()
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        
        output_device = generated_ids.device
        if output_device.type == 'cuda':
             torch.cuda.synchronize(device=output_device)

        end_time_inference = time.perf_counter()
        logger.info(f"推理 (model.generate) 耗时 (output on {output_device}): {end_time_inference - start_time_inference:.4f} 秒")

        logger.info("开始后处理 (解码)...")
        start_time_post_processing = time.perf_counter()
        generated_ids_cpu = generated_ids.to('cpu')
        input_ids_cpu = inputs.input_ids.to('cpu')

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids_cpu, generated_ids_cpu)
        ]
        output_text_list = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        output_text = output_text_list[0] if output_text_list else ""
        end_time_post_processing = time.perf_counter()
        logger.info(f"后处理 (解码) 耗时: {end_time_post_processing - start_time_post_processing:.4f} 秒")
        logger.info(f"模型原始输出: {output_text}")

        # --- Response Formatting ---
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
    load_model_and_processor()
    app.run(host='0.0.0.0', port=5000, debug=False)
