import os
import logging
import sys

# --- Global Constants ---
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_OCR_PROMPT = (
  "Your task is to act as an OCR tool. Please extract all text from the provided image(s) "
  "and return it structured as a JSON object. If multiple distinct pieces of information are present "
  "(like on an ID card), ensure each piece is a key-value pair in the JSON. For example, "
  "for an ID card, keys might include 'name', 'number', 'address', etc."
)

# --- Backend Selection ---
# Set MODEL_BACKEND environment variable to "modelscope" or "huggingface"
# Defaults to "huggingface" if not set or invalid
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "huggingface").lower()
VALID_BACKENDS = ["huggingface", "modelscope"]
if MODEL_BACKEND not in VALID_BACKENDS:
  print(f"Warning: Invalid MODEL_BACKEND '{MODEL_BACKEND}'. Defaulting to 'huggingface'.", file=sys.stderr)
  MODEL_BACKEND = "huggingface"

# --- Environment Setup ---
os.environ['CURL_CA_BUNDLE'] = '' # Should be used with caution
os.environ['HF_ENDPOINT'] = os.getenv('HF_ENDPOINT', 'https://huggingface.co') # Default to official HF if not set
# For hf-mirror.com specifically:
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# --- Configure Logging ---
def setup_logging(name="VLApp"):
  logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  )
  return logging.getLogger(name)

logger = setup_logging()

logger.info(f"Using model backend: {MODEL_BACKEND.upper()}")
logger.info(f"Model name: {MODEL_NAME}")
logger.info(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
if not os.environ.get('CURL_CA_BUNDLE'):
  logger.warning("CURL_CA_BUNDLE is not set. This might be insecure if downloading from untrusted sources without SSL verification.")
