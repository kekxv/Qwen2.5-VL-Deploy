import torch
import time
from config import logger

class DeviceManager:
  def __init__(self):
    self.primary_device = None
    self.model_dtype = None
    self._detect_device_and_dtype()

  def _detect_device_and_dtype(self):
    logger.info("Initiating advanced device detection...")
    rocm_detected = False
    if torch.cuda.is_available():
      if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        rocm_detected = True
        self.primary_device = "cuda:0"
        self.model_dtype = torch.bfloat16
        logger.info(f"AMD ROCm detected. {torch.cuda.device_count()} GPU(s) found. Using device_map='auto'. Primary device: {self.primary_device}, dtype: {self.model_dtype}")
      elif torch.cuda.device_count() > 0:
        self.primary_device = "cuda:0"
        self.model_dtype = torch.bfloat16
        logger.info(f"NVIDIA CUDA detected. {torch.cuda.device_count()} GPU(s) found. Using device_map='auto'. Primary device: {self.primary_device}, dtype: {self.model_dtype}")
      else:
        logger.warning("torch.cuda.is_available() is True, but no CUDA devices found. Checking MPS.")
        self.primary_device = None

    if self.primary_device is None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
      self.primary_device = "mps"
      self.model_dtype = torch.float16
      logger.info(f"Apple MPS detected. Using device: {self.primary_device}, dtype: {self.model_dtype}. device_map='auto' will target MPS.")

    if self.primary_device is None:
      self.primary_device = "cpu"
      self.model_dtype = torch.float32
      logger.info(f"No CUDA, ROCm, or MPS detected. Using CPU. Device: {self.primary_device}, dtype: {self.model_dtype}")

    logger.info(f"Selected primary_device for operations: {self.primary_device}, model_dtype: {self.model_dtype}")

  def synchronize_device(self, device_to_sync=None):
    """Synchronizes the specified device, or the primary_device if None."""
    sync_target = device_to_sync if device_to_sync else self.primary_device

    if not sync_target:
      logger.debug("No device specified or detected for synchronization.")
      return

    device_type = sync_target.type if isinstance(sync_target, torch.device) else str(sync_target).split(':')[0]

    if device_type == "cuda":
      # For ROCm, primary_device is "cuda:0". For NVIDIA, it's also "cuda:X".
      # torch.cuda.synchronize() syncs the current CUDA device by default.
      # If a specific device (e.g., "cuda:0") is given, it syncs that one.
      target_cuda_device = sync_target if isinstance(sync_target, torch.device) else torch.device(sync_target)
      if torch.cuda.is_available() and target_cuda_device.index is not None and target_cuda_device.index < torch.cuda.device_count():
        logger.debug(f"Synchronizing CUDA device: {target_cuda_device}")
        torch.cuda.synchronize(device=target_cuda_device)
      elif torch.cuda.is_available(): # E.g. if sync_target was just "cuda"
        logger.debug("Synchronizing current CUDA device.")
        torch.cuda.synchronize()
      else:
        logger.warning(f"Attempted to synchronize CUDA device {sync_target}, but CUDA not available or device index invalid.")
    elif device_type == "mps":
      if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
        logger.debug("Synchronizing MPS device.")
        torch.mps.synchronize()
      else:
        logger.warning("Attempted to synchronize MPS, but torch.mps.synchronize not available.")
    elif device_type == "cpu":
      logger.debug("CPU synchronization is a no-op.")
    else:
      logger.warning(f"Synchronization not implemented for device type: {device_type}")

