import time
import torch
from config import logger, MODEL_NAME, MODEL_BACKEND

# Conditional imports based on backend
if MODEL_BACKEND == "huggingface":
  from transformers import AutoModelForImageTextToText, AutoProcessor
  ModelClass = AutoModelForImageTextToText
  ProcessorClass = AutoProcessor
  logger.info("Using Hugging Face Transformers for model and processor.")
elif MODEL_BACKEND == "modelscope":
  from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor as ModelScopeAutoProcessor
  # Note: modelscope.AutoProcessor might be different or a wrapper around transformers.AutoProcessor
  # For Qwen-VL, transformers.AutoProcessor is generally compatible.
  # If ModelScope's specific processor is strictly needed, use ModelScopeAutoProcessor.
  # Let's stick to the original imports for fidelity.
  ModelClass = Qwen2_5_VLForConditionalGeneration
  ProcessorClass = ModelScopeAutoProcessor # Using ModelScope's AutoProcessor as per original
  logger.info("Using ModelScope for model and processor.")
else:
  # This case should ideally be caught in qwen_config.py
  raise ImportError(f"Unsupported model backend: {MODEL_BACKEND}. Check qwen_config.py.")


class ModelProvider:
  def __init__(self, device_manager):
    self.model = None
    self.processor = None
    self.device_manager = device_manager # Instance of DeviceManager

  def load_resources(self):
    logger.info(f"Initiating model and processor loading sequence for backend: {MODEL_BACKEND.upper()}...")
    overall_start_time = time.perf_counter()

    primary_device_str = self.device_manager.primary_device
    model_dtype = self.device_manager.model_dtype

    # --- Model Loading ---
    logger.info(f"Starting to load model {MODEL_NAME} (using device_map='auto')...")
    start_time_model_load = time.perf_counter()
    try:
      self.model = ModelClass.from_pretrained(
        MODEL_NAME,
        torch_dtype=model_dtype,
        device_map="auto",
        trust_remote_code=True
      )
      logger.info(f"Model {MODEL_NAME} loaded.")
      if hasattr(self.model, 'hf_device_map'):
        logger.info(f"Model device map: {self.model.hf_device_map}")
        first_param_device = next(self.model.parameters()).device
        logger.info(f"First model parameter is on device: {first_param_device}")
      else:
        model_device_attr = getattr(self.model, 'device', 'Unknown')
        logger.info(f"Model loaded to single device: {model_device_attr}. (No hf_device_map).")
        if primary_device_str != str(model_device_attr) and model_device_attr != 'meta':
          logger.warning(f"Primary device was {primary_device_str} but model ended up on {model_device_attr}.")

    except Exception as e:
      logger.error(f"Error loading model {MODEL_NAME} with {MODEL_BACKEND} backend: {e}", exc_info=True)
      raise

    self.device_manager.synchronize_device() # Sync after model load to primary device context
    end_time_model_load = time.perf_counter()
    logger.info(f"Model loading time: {end_time_model_load - start_time_model_load:.4f} seconds")

    # --- Processor Loading ---
    logger.info(f"Starting to load processor for {MODEL_NAME} using {MODEL_BACKEND} backend...")
    start_time_processor_load = time.perf_counter()
    try:
      self.processor = ProcessorClass.from_pretrained(MODEL_NAME, trust_remote_code=True)
      logger.info(f"Processor for {MODEL_NAME} loaded.")
    except Exception as e:
      logger.error(f"Error loading processor for {MODEL_NAME} with {MODEL_BACKEND} backend: {e}", exc_info=True)
      raise
    end_time_processor_load = time.perf_counter()
    logger.info(f"Processor loading time: {end_time_processor_load - start_time_processor_load:.4f} seconds")

    overall_end_time = time.perf_counter()
    logger.info(f"Total model and processor loading time: {overall_end_time - overall_start_time:.4f} seconds")

    if not self.model or not self.processor:
      raise RuntimeError("Model or processor failed to initialize properly.")
