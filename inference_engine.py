import time
import torch
import json
import re
from config import logger
from qwen_vl_utils import process_vision_info
class InferenceEngine:
  def __init__(self, model, processor, device_manager):
    self.model = model
    self.processor = processor
    self.device_manager = device_manager # Instance of DeviceManager

    if not self.model or not self.processor:
      raise ValueError("Model and processor must be provided to InferenceEngine.")

  def run_inference(self, image_urls, custom_prompt):
    logger.info(f"Received {len(image_urls)} image(s) for inference. Using prompt: '{custom_prompt}'")
    inference_total_start_time = time.perf_counter()

    primary_device_str = self.device_manager.primary_device

    user_content = []
    for img_url in image_urls:
      user_content.append({"type": "image", "image": img_url})
    user_content.append({"type": "text", "text": custom_prompt})

    messages = [
      {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant. You must respond in JSON format as requested by the user."}]},
      {"role": "user", "content": user_content}
    ]

    logger.info("Preparing input data...")
    start_time_input_prep = time.perf_counter()

    # process_vision_info is expected to return image paths/URLs and video paths/URLs
    # The Qwen processor can handle image URLs directly.
    image_inputs_for_processor, video_inputs_for_processor = process_vision_info(messages)

    text_for_template = self.processor.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
    )

    inputs = self.processor(
      text=[text_for_template],
      images=image_inputs_for_processor,
      videos=video_inputs_for_processor, # Pass if you handle videos
      padding=True,
      return_tensors="pt"
    )
    inputs = inputs.to(primary_device_str) # Move inputs to the primary device

    self.device_manager.synchronize_device(torch.device(primary_device_str)) # Sync after data transfer
    end_time_input_prep = time.perf_counter()
    logger.info(f"Input data preparation time (incl. .to('{primary_device_str}')): {end_time_input_prep - start_time_input_prep:.4f} seconds")

    logger.info("Starting inference (model.generate)...")
    start_time_inference = time.perf_counter()
    with torch.no_grad(): # Important for inference
      generated_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    output_device = generated_ids.device
    logger.info(f"Generated IDs tensor is on device: {output_device}")
    self.device_manager.synchronize_device(output_device) # Sync on the device where output tensor landed
    end_time_inference = time.perf_counter()
    logger.info(f"Inference (model.generate) time: {end_time_inference - start_time_inference:.4f} seconds")

    logger.info("Starting post-processing (decoding)...")
    start_time_post_processing = time.perf_counter()

    generated_ids_cpu = generated_ids.to('cpu')
    input_ids_cpu = inputs.input_ids.to('cpu') if inputs.input_ids.device.type != 'cpu' else inputs.input_ids

    generated_ids_trimmed = [
      out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids_cpu, generated_ids_cpu)
    ]
    output_text_list = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output_text = output_text_list[0] if output_text_list else ""
    end_time_post_processing = time.perf_counter()
    logger.info(f"Post-processing (decoding) time: {end_time_post_processing - start_time_post_processing:.4f} seconds")
    logger.info(f"Raw model output: {output_text}")

    # --- Response Formatting ---
    parsed_data = None
    response_payload = {}
    http_status_code = 200

    try:
      # First, try to remove any potential `</s>` or other EOS tokens if not handled by skip_special_tokens
      # Qwen models sometimes add "Assistant:" or similar prefixes if not perfectly templated.
      # The `clean_up_tokenization_spaces=False` and `skip_special_tokens=True` should handle most, but let's be safe.
      # Try to find the start of JSON (either { or [)
      json_start_curly = output_text.find('{')
      json_start_square = output_text.find('[')

      json_start_index = -1
      if json_start_curly != -1 and json_start_square != -1:
        json_start_index = min(json_start_curly, json_start_square)
      elif json_start_curly != -1:
        json_start_index = json_start_curly
      elif json_start_square != -1:
        json_start_index = json_start_square

      cleaned_output_text = output_text
      if json_start_index != -1:
        cleaned_output_text = output_text[json_start_index:]
        # Attempt to find matching end brace/bracket (very basic, might not be robust for nested structures if there's trailing text)
        # This part is tricky. A full parser is better. For now, simple loads.
      else: # No JSON structure detected at all
        logger.warning("No JSON object/array start token ({ or [) found in model output.")


      parsed_data = json.loads(cleaned_output_text)
      logger.info("Successfully parsed model output directly as JSON.")
    except json.JSONDecodeError:
      logger.info("Direct JSON parsing failed. Attempting to extract from markdown code block.")
      # Regex to find JSON within markdown code blocks (```json ... ``` or ``` ... ```)
      match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", output_text, re.IGNORECASE)
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

    if not response_payload: # If no error payload was set above
      if parsed_data is not None:
        response_payload = {"code": 200, "data": parsed_data}
      else:
        # This case should ideally be covered by the try-except for JSON parsing
        logger.error("Internal logic error: parsed_data is None but no error response was constructed during JSON parsing.")
        response_payload = {"code": 500, "error": "Internal Server Error: Failed to process model output into JSON.", "raw_model_output": output_text}
        http_status_code = 500

    inference_total_end_time = time.perf_counter()
    logger.info(f"Total inference request processing time: {inference_total_end_time - inference_total_start_time:.4f} seconds")

    return response_payload, http_status_code
