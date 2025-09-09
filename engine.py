# File: engine.py
# Core TTS model loading and speech generation logic.

import logging
import random
import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path

from chatterbox.tts import ChatterboxTTS  # Main TTS engine class
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # Multilingual TTS engine class
from chatterbox.models.s3gen.const import (
    S3GEN_SR,
)  # Default sample rate from the engine

# Import the singleton config_manager
from config import config_manager

logger = logging.getLogger(__name__)

# --- Global Module Variables ---
chatterbox_model: Optional[ChatterboxTTS] = None
chatterbox_multilingual_model: Optional[ChatterboxMultilingualTTS] = None
MODEL_LOADED: bool = False
MULTILINGUAL_MODEL_LOADED: bool = False
model_device: Optional[str] = (
    None  # Stores the resolved device string ('cuda' or 'cpu')
)


def set_seed(seed_value: int):
    """
    Sets the seed for torch, random, and numpy for reproducibility.
    This is called if a non-zero seed is provided for generation.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    logger.info(f"Global seed set to: {seed_value}")


def _test_cuda_functionality() -> bool:
    """
    Tests if CUDA is actually functional, not just available.

    Returns:
        bool: True if CUDA works, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.cuda()
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"CUDA functionality test failed: {e}")
        return False


def _test_mps_functionality() -> bool:
    """
    Tests if MPS is actually functional, not just available.

    Returns:
        bool: True if MPS works, False otherwise.
    """
    if not torch.backends.mps.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.to("mps")
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"MPS functionality test failed: {e}")
        return False


def load_model() -> bool:
    """
    Loads both English and Multilingual TTS models.
    This version directly attempts to load from the Hugging Face repository (or its cache)
    using `from_pretrained`, bypassing the local `paths.model_cache` directory.
    Updates global variables `chatterbox_model`, `chatterbox_multilingual_model`, `MODEL_LOADED`, 
    `MULTILINGUAL_MODEL_LOADED`, and `model_device`.

    Returns:
        bool: True if at least one model was loaded successfully, False otherwise.
    """
    global chatterbox_model, chatterbox_multilingual_model, MODEL_LOADED, MULTILINGUAL_MODEL_LOADED, model_device

    if MODEL_LOADED and MULTILINGUAL_MODEL_LOADED:
        logger.info("Both TTS models are already loaded.")
        return True

    try:
        # NVIDIA GPU ONLY - No CPU/MPS fallback
        if not _test_cuda_functionality():
            logger.error("❌ CUDA NOT AVAILABLE! This container requires NVIDIA GPU.")
            logger.error("Make sure you're running with --gpus all flag")
            raise RuntimeError("NVIDIA GPU required but not available")
        
        resolved_device_str = "cuda"
        model_device = resolved_device_str
        logger.info(f"🚀 GPU CONFIRMED: Using CUDA device")

        # Get configured model_repo_id for logging and context,
        # though from_pretrained might use its own internal default if not overridden.
        model_repo_id_config = config_manager.get_string(
            "model.repo_id", "ResembleAI/chatterbox"
        )

        logger.info(
            f"Attempting to load both English and Multilingual models using from_pretrained (expected from Hugging Face repository: {model_repo_id_config} or library default)."
        )
        
        # Load English model
        if not MODEL_LOADED:
            try:
                logger.info("Loading English ChatterboxTTS model...")
                chatterbox_model = ChatterboxTTS.from_pretrained(device=model_device)
                MODEL_LOADED = True
                logger.info(
                    f"Successfully loaded English TTS model on {model_device}. Engine sample rate: {chatterbox_model.sr} Hz."
                )
            except Exception as e_en:
                logger.error(
                    f"Failed to load English model: {e_en}",
                    exc_info=True,
                )
                chatterbox_model = None
                MODEL_LOADED = False

        # Load Multilingual model
        if not MULTILINGUAL_MODEL_LOADED:
            try:
                logger.info("Loading Multilingual ChatterboxMultilingualTTS model...")
                chatterbox_multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=model_device)
                MULTILINGUAL_MODEL_LOADED = True
                logger.info(
                    f"Successfully loaded Multilingual TTS model on {model_device}. Engine sample rate: {chatterbox_multilingual_model.sr} Hz."
                )
            except Exception as e_ml:
                logger.error(
                    f"Failed to load Multilingual model: {e_ml}",
                    exc_info=True,
                )
                chatterbox_multilingual_model = None
                MULTILINGUAL_MODEL_LOADED = False

        # Check if at least one model loaded successfully
        if not MODEL_LOADED and not MULTILINGUAL_MODEL_LOADED:
            logger.error("Failed to load both English and Multilingual models!")
            return False
        
        if MODEL_LOADED and not MULTILINGUAL_MODEL_LOADED:
            logger.warning("Only English model loaded. Multilingual features will not be available.")
        elif not MODEL_LOADED and MULTILINGUAL_MODEL_LOADED:
            logger.warning("Only Multilingual model loaded. English-specific optimizations may not be available.")
        else:
            logger.info("Both English and Multilingual models loaded successfully!")

        return True

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during model loading: {e}", exc_info=True
        )
        chatterbox_model = None
        MODEL_LOADED = False
        return False


def synthesize(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    language: str = "en",
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Synthesizes audio from text using the appropriate TTS model (English or Multilingual).

    Args:
        text: The text to synthesize.
        audio_prompt_path: Path to an audio file for voice cloning or predefined voice.
        temperature: Controls randomness in generation.
        exaggeration: Controls expressiveness.
        cfg_weight: Classifier-Free Guidance weight.
        seed: Random seed for generation. If 0, default randomness is used.
              If non-zero, a global seed is set for reproducibility.
        language: Language code (e.g., 'en', 'es', 'fr'). Uses English model for 'en', 
                 Multilingual model for other languages.

    Returns:
        A tuple containing the audio waveform (torch.Tensor) and the sample rate (int),
        or (None, None) if synthesis fails.
    """
    global chatterbox_model, chatterbox_multilingual_model

    # Determine which model to use based on language
    use_multilingual = language != "en"
    
    if use_multilingual:
        if not MULTILINGUAL_MODEL_LOADED or chatterbox_multilingual_model is None:
            logger.error(f"Multilingual TTS model is not loaded. Cannot synthesize audio for language '{language}'.")
            return None, None
        selected_model = chatterbox_multilingual_model
        model_name = "Multilingual"
    else:
        if not MODEL_LOADED or chatterbox_model is None:
            logger.error("English TTS model is not loaded. Cannot synthesize audio.")
            return None, None
        selected_model = chatterbox_model
        model_name = "English"

    try:
        # Set seed globally if a specific seed value is provided and is non-zero.
        if seed != 0:
            logger.info(f"Applying user-provided seed for generation: {seed}")
            set_seed(seed)
        else:
            logger.info(
                "Using default (potentially random) generation behavior as seed is 0."
            )

        logger.debug(
            f"Synthesizing with {model_name} model for language '{language}': audio_prompt='{audio_prompt_path}', temp={temperature}, "
            f"exag={exaggeration}, cfg_weight={cfg_weight}, seed_applied_globally_if_nonzero={seed}"
        )

        # Call the appropriate model's generate method
        if use_multilingual:
            # Multilingual model has different parameters (language_id instead of language)
            wav_tensor = selected_model.generate(
                text=text,
                language_id=language,  # Multilingual model uses language_id
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
        else:
            # English model (original parameters)
            wav_tensor = selected_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

        # Both models should return a CPU tensor
        return wav_tensor, selected_model.sr

    except Exception as e:
        logger.error(f"Error during TTS synthesis: {e}", exc_info=True)
        return None, None


# --- End File: engine.py ---
