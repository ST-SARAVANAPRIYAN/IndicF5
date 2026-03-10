"""Inference wrapper for IndicF5 model with proper error handling"""

import os
import tempfile
import time
import contextlib
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IndicF5InferenceEngine:
    """Wrapper for IndicF5 model inference with proper error handling"""
    
    def __init__(self):
        self.config = get_config()
        self.device = torch.device(self.config.get_device())
        self.model = None
        self._warmed_up = False
        self._configure_runtime()
        logger.info(f"IndicF5 inference engine initialized on device: {self.device}")

    def _configure_runtime(self) -> None:
        """Apply runtime performance settings."""
        if torch.cuda.is_available():
            if self.config.model.enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = self.config.model.cudnn_benchmark
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    def _autocast_context(self):
        """Return autocast context when enabled and supported."""
        if not self.config.model.use_mixed_precision:
            return contextlib.nullcontext()
        if self.device.type != "cuda":
            return contextlib.nullcontext()

        dtype = torch.float16
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _gpu_mem_gb(self) -> Tuple[float, float]:
        if not torch.cuda.is_available():
            return 0.0, 0.0
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return allocated, reserved

    def _warmup(self) -> None:
        """Optional one-time warmup to reduce first-request latency."""
        if self._warmed_up or not self.config.model.warmup_on_load or self.model is None:
            return
        try:
            dummy_audio = None
            prompts_dir = self.config.root / "prompts"
            if prompts_dir.exists():
                for ext in ("*.wav", "*.mp3", "*.flac"):
                    found = list(prompts_dir.glob(ext))
                    if found:
                        dummy_audio = str(found[0])
                        break
            if not dummy_audio:
                self._warmed_up = True
                return
            logger.info("Running one-time warmup inference...")
            with torch.no_grad():
                _ = self.model(
                    text="warmup",
                    ref_audio_path=dummy_audio,
                    ref_text="warmup",
                )
            self._warmed_up = True
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup skipped: {str(e)}")
            self._warmed_up = True
    
    def load_model(self) -> bool:
        """
        Load the IndicF5 model
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading model from {self.config.model.model_repo}...")
            
            # Import here to avoid issues if transformers not available
            from transformers import AutoModel
            
            self.model = AutoModel.from_pretrained(
                self.config.model.model_repo, 
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)

            if self.config.model.torch_compile and hasattr(torch, "compile"):
                try:
                    self.model = torch.compile(self.model, mode=self.config.model.compile_mode)
                    logger.info(f"Applied torch.compile(mode={self.config.model.compile_mode})")
                except Exception as e:
                    logger.warning(f"torch.compile skipped: {str(e)}")

            self.model.eval()
            self._warmup()
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            return False
    
    def load_all_models(self) -> bool:
        """Load model (remote model already bundles inference stack)"""
        return self.load_model()
    
    def move_to_device(self, device: str) -> None:
        """
        Move models to specified device
        
        Args:
            device: Device string ('cuda' or 'cpu')
        """
        try:
            self.device = torch.device(device)
            self._configure_runtime()
            if self.model is not None:
                self.model = self.model.to(self.device)
            logger.info(f"Models moved to device: {device}")
        except Exception as e:
            logger.error(f"Failed to move models to device: {str(e)}", exc_info=True)
            raise
    
    def clear_cache(self) -> None:
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    def _preprocess_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load and preprocess audio
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            audio, sr = torchaudio.load(audio_path)
            
            # Normalize loudness
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            rms = torch.sqrt(torch.mean(torch.square(audio)))
            if rms < self.config.audio.target_rms:
                audio = audio * self.config.audio.target_rms / rms
            
            # Resample if needed
            if sr != self.config.audio.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.config.audio.sample_rate)
                audio = resampler(audio)
            
            return audio.to(self.device), self.config.audio.sample_rate
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}", exc_info=True)
            raise
    
    def _prepare_text(self, ref_text: str, gen_text: str) -> str:
        """
        Prepare and validate text
        
        Args:
            ref_text: Reference text
            gen_text: Text to generate
            
        Returns:
            Normalized generation text
        """
        if not ref_text or not gen_text:
            raise ValueError("Both reference text and generation text are required")
        
        return gen_text.strip()
    
    @torch.inference_mode()
    def synthesize(
        self,
        ref_audio_path: str,
        ref_text: str,
        gen_text: str,
        speed: float = 1.0,
        nfe_steps: int | None = None,
        cfg_strength: float | None = None,
        remove_silence: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text
        
        Args:
            ref_audio_path: Path to reference audio
            ref_text: Reference text 
            gen_text: Text to generate
            speed: Speed multiplier
            remove_silence: Whether to remove silence
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            start_time = time.time()
            input_chars = len(gen_text.strip())
            gpu_alloc_before, gpu_reserved_before = self._gpu_mem_gb()
            
            # Check model is loaded
            if self.model is None:
                logger.warning("Models not loaded, loading now...")
                if not self.load_all_models():
                    raise RuntimeError("Failed to load models")
            
            logger.info("Starting synthesis...")
            
            # Prepare text
            logger.debug("Preparing text...")
            normalized_gen_text = self._prepare_text(ref_text, gen_text)
            effective_nfe_steps = int(nfe_steps if nfe_steps is not None else self.config.inference.nfe_steps)
            effective_cfg_strength = float(
                cfg_strength if cfg_strength is not None else self.config.inference.cfg_strength
            )
            effective_nfe_steps = max(8, min(64, effective_nfe_steps))
            effective_cfg_strength = max(0.5, min(5.0, effective_cfg_strength))

            if hasattr(self.model, "config"):
                try:
                    self.model.config.speed = float(speed)
                    if hasattr(self.model.config, "remove_sil"):
                        self.model.config.remove_sil = bool(remove_silence)
                except Exception as e:
                    logger.warning(f"Could not apply runtime model config: {str(e)}")

            infer_start = time.time()
            generated_wave = None

            if hasattr(self.model, "ema_model") and hasattr(self.model, "vocoder"):
                try:
                    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

                    logger.info(
                        "[GEN] fast-path=ema_model nfe_steps=%d cfg_strength=%.2f speed=%.2f",
                        effective_nfe_steps,
                        effective_cfg_strength,
                        float(speed),
                    )

                    ref_audio, processed_ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)

                    with torch.no_grad(), self._autocast_context():
                        generated_wave, _, _ = infer_process(
                            ref_audio,
                            processed_ref_text,
                            normalized_gen_text,
                            self.model.ema_model,
                            self.model.vocoder,
                            nfe_step=effective_nfe_steps,
                            cfg_strength=effective_cfg_strength,
                            sway_sampling_coef=self.config.inference.sway_sampling_coef,
                            speed=max(0.5, float(speed)),
                            device=self.device,
                        )
                except Exception as e:
                    logger.warning(f"Fast-path failed, falling back to HF wrapper: {str(e)}")

            if generated_wave is None:
                logger.info("[GEN] fast-path unavailable; using HF wrapper")
                with torch.no_grad(), self._autocast_context():
                    generated_wave = self.model(
                        text=normalized_gen_text,
                        ref_audio_path=ref_audio_path,
                        ref_text=ref_text,
                    )

            infer_elapsed = time.time() - infer_start

            if isinstance(generated_wave, torch.Tensor):
                generated_wave = generated_wave.detach().cpu().numpy()
            generated_wave = np.asarray(generated_wave).squeeze()

            if generated_wave.dtype == np.int16:
                generated_wave = generated_wave.astype(np.float32) / 32768.0
            else:
                generated_wave = generated_wave.astype(np.float32)

            sr = self.config.audio.sample_rate
            
            # Remove silence if requested
            if remove_silence:
                logger.debug("Removing silence...")
                generated_wave = self._remove_silence(generated_wave)
            
            latency = time.time() - start_time
            audio_sec = max(len(generated_wave) / sr, 1e-6)
            rtf = latency / audio_sec
            gpu_alloc_after, gpu_reserved_after = self._gpu_mem_gb()

            logger.info(
                "[GEN] chars=%d speed=%.2f device=%s time=%.2fs infer=%.2fs audio=%.2fs rtf=%.2f "
                "gpu_alloc=%.2fGB->%.2fGB gpu_reserved=%.2fGB->%.2fGB",
                input_chars,
                speed,
                self.device,
                latency,
                infer_elapsed,
                audio_sec,
                rtf,
                gpu_alloc_before,
                gpu_alloc_after,
                gpu_reserved_before,
                gpu_reserved_after,
            )
            
            return generated_wave, sr
            
        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}", exc_info=True)
            raise
    
    def _remove_silence(self, audio: np.ndarray, min_silence_len: int = 1000, 
                       silence_thresh: int = -50, keep_silence: int = 500) -> np.ndarray:
        """
        Remove silence from audio
        
        Args:
            audio: Audio array
            min_silence_len: Minimum silence length in ms
            silence_thresh: Silence threshold in dB
            keep_silence: Keep silence length in ms
            
        Returns:
            Audio array with silence removed
        """
        try:
            from pydub import AudioSegment, silence
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, self.config.audio.sample_rate)
                temp_path = f.name
            
            # Process
            aseg = AudioSegment.from_file(temp_path)
            non_silent_segs = silence.split_on_silence(
                aseg,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=keep_silence,
                seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for seg in non_silent_segs:
                non_silent_wave += seg
            
            non_silent_wave.export(temp_path, format="wav")
            processed_audio, _ = torchaudio.load(temp_path)
            
            # Cleanup
            os.remove(temp_path)
            
            return processed_audio.squeeze().cpu().numpy()
            
        except ImportError:
            logger.warning("pydub not available, skipping silence removal")
            return audio
        except Exception as e:
            logger.warning(f"Error removing silence: {str(e)}, continuing without silence removal")
            return audio


# Singleton
_inference_engine = None


def get_inference_engine() -> IndicF5InferenceEngine:
    """Get or create the inference engine"""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = IndicF5InferenceEngine()
    return _inference_engine
