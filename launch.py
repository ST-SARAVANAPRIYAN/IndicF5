#!/usr/bin/env python3
"""
IndicF5 Neo: Fast and Scalable Text-to-Speech

Main entry point for the application.
Run: python launch.py
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import soundfile as sf
import torch
import gradio as gr

# Import our modules
from src.config import get_config
from src.utils.logger import LoggerSetup, get_logger
from src.utils.device_manager import DeviceManager
from src.inference.engine import get_inference_engine
from src.data_management.profile_manager import VoiceProfileManager

# Setup logging
config = get_config()
LoggerSetup.setup(log_dir=config.paths.logs_dir)
logger = get_logger(__name__)

# Initialize managers
device_mgr = DeviceManager()
profile_mgr = VoiceProfileManager(profiles_dir=str(config.paths.profiles_dir))
inference_engine = get_inference_engine()

# Output directory
output_dir = config.paths.outputs_dir / "history"
output_dir.mkdir(parents=True, exist_ok=True)

logger.info("=" * 50)
logger.info("IndicF5 Neo - Text-to-Speech Application")
logger.info("=" * 50)
logger.info(
    "Perf mode: device=%s mixed_precision=%s tf32=%s cudnn_benchmark=%s torch_compile=%s",
    config.get_device(),
    config.model.use_mixed_precision,
    config.model.enable_tf32,
    config.model.cudnn_benchmark,
    config.model.torch_compile,
)


class UIState:
    """Manage UI state"""
    def __init__(self):
        self.model_loaded = False


ui_state = UIState()


def get_history_files():
    """Get list of generated audio files"""
    try:
        files = sorted(output_dir.glob("*.wav"), key=os.path.getmtime, reverse=True)
        return [[str(f), f.name] for f in files[:50]]  # Limit to 50 most recent
    except Exception as e:
        logger.error(f"Error getting history files: {str(e)}")
        return []


def get_profile_table_rows():
    """Get profile table rows"""
    rows = []
    for name in profile_mgr.list_profiles():
        profile = profile_mgr.get_profile(name) or {}
        rows.append([name, profile.get("ref_text", "")])
    return rows


def synthesize(
    profile_name,
    ref_audio,
    ref_text,
    gen_text,
    remove_silence,
    speed,
    nfe_steps,
    cfg_strength,
    device_type,
):
    """Synthesize speech from text"""
    try:
        req_start = time.time()
        logger.info(f"Synthesis request: profile={profile_name}, device={device_type}")
        
        # Change device if needed
        current_device = str(device_mgr.get_current_device()).replace(":0", "")
        if device_type == "cuda" and not torch.cuda.is_available():
            device_type = "cpu"
        if device_type != current_device:
            logger.info(f"Switching to device: {device_type}")
            device_mgr.set_device(device_type)
            inference_engine.move_to_device(device_type)
        
        # Load models if needed
        if not ui_state.model_loaded:
            logger.info("Loading models...")
            if not inference_engine.load_all_models():
                raise gr.Error("Failed to load models. Check logs for details.")
            ui_state.model_loaded = True
        
        # Get reference audio and text from profile if selected
        if profile_name and profile_name != "None":
            profile = profile_mgr.get_profile(profile_name)
            if profile:
                ref_audio = profile["audio_path"]
                ref_text = profile["ref_text"]
                logger.info(f"Using profile: {profile_name}")
        
        # Validate inputs
        if not ref_audio:
            raise gr.Error("Reference audio is required")
        if not ref_text:
            raise gr.Error("Reference text is required")
        if not gen_text:
            raise gr.Error("Text to generate is required")
        
        # Synthesize
        logger.info("Starting synthesis...")
        audio, sr = inference_engine.synthesize(
            ref_audio_path=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            speed=speed,
            nfe_steps=nfe_steps,
            cfg_strength=cfg_strength,
            remove_silence=remove_silence,
        )
        
        # Save to history
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"gen_{timestamp}.wav"
        sf.write(str(output_path), audio, sr)
        logger.info(f"Audio saved to {output_path}")

        elapsed = time.time() - req_start
        return (sr, audio), f"✓ Generated in {elapsed:.2f}s", get_history_files()
        
    except gr.Error:
        raise
    except Exception as e:
        logger.error(f"Synthesis failed: {str(e)}", exc_info=True)
        raise gr.Error(f"Synthesis failed: {str(e)}")


def save_profile(name, audio, ref_text, save_processed):
    """Save a voice profile"""
    try:
        if not name:
            return gr.Dropdown(), "Profile name is required"
        if not audio:
            return gr.Dropdown(), "Audio file is required"
        if not ref_text:
            return gr.Dropdown(), "Reference text is required"
        
        # Save the profile
        profile_mgr.save_profile(name, audio, ref_text)
        
        # Update dropdown
        profiles = ["None"] + profile_mgr.list_profiles()
        
        logger.info(f"Profile saved: {name}")
        return (
            gr.Dropdown(choices=profiles, value=name),
            f"Profile '{name}' saved successfully!",
        )
    except Exception as e:
        logger.error(f"Failed to save profile: {str(e)}")
        return gr.Dropdown(), f"Failed to save profile: {str(e)}"


def delete_profile(name):
    """Delete a profile"""
    try:
        if not name:
            profiles = ["None"] + profile_mgr.list_profiles()
            return gr.Dropdown(choices=profiles, value="None"), "Select a profile to delete", get_profile_table_rows()
        
        if profile_mgr.delete_profile(name):
            profiles = ["None"] + profile_mgr.list_profiles()
            logger.info(f"Profile deleted: {name}")
            return (
                gr.Dropdown(choices=profiles, value="None"),
                f"Profile '{name}' deleted",
                get_profile_table_rows(),
            )
        else:
            profiles = ["None"] + profile_mgr.list_profiles()
            return gr.Dropdown(choices=profiles, value="None"), "Failed to delete profile", get_profile_table_rows()
    except Exception as e:
        logger.error(f"Error deleting profile: {str(e)}")
        profiles = ["None"] + profile_mgr.list_profiles()
        return gr.Dropdown(choices=profiles, value="None"), f"Error: {str(e)}", get_profile_table_rows()


def offload_model():
    """Offload model to CPU"""
    try:
        inference_engine.move_to_device('cpu')
        inference_engine.clear_cache()
        logger.info("Model offloaded to CPU")
        return "✓ Model offloaded to CPU"
    except Exception as e:
        logger.error(f"Error offloading model: {str(e)}")
        return f"Error: {str(e)}"


def refresh_profiles():
    """Refresh profile list"""
    profiles = ["None"] + profile_mgr.list_profiles()
    return gr.Dropdown(choices=profiles)


def load_audio_from_history(table_data, evt: gr.SelectData):
    """Load audio from history"""
    try:
        if not evt or not table_data:
            return None
        if hasattr(evt, "index") and isinstance(evt.index, (list, tuple)) and len(evt.index) > 0:
            row_idx = evt.index[0]
            if isinstance(row_idx, int) and 0 <= row_idx < len(table_data):
                selected = table_data[row_idx][0]
                if isinstance(selected, str) and os.path.exists(selected):
                    return selected
    except Exception as e:
        logger.warning(f"Error loading from history: {str(e)}")
    return None


# Build Gradio Interface
with gr.Blocks(
    title="IndicF5 Neo"
) as app:
    
    gr.HTML("""
    <div class="container">
        <div class="header">
            <h1>🚀 IndicF5 Neo</h1>
            <p>Fast & Scalable Text-to-Speech for Indic Languages</p>
        </div>
    </div>
    """)
    
    with gr.Tabs():
        
        # ============ GENERATE TAB ============
        with gr.Tab("⚡ Generate Speech"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Voice Selection")
                    profile_dropdown = gr.Dropdown(
                        choices=["None"] + profile_mgr.list_profiles(),
                        value="None",
                        label="Voice Profile"
                    )
                    refresh_btn = gr.Button("🔄 Refresh", scale=1)
                    
                    gr.Markdown("### Reference Audio")
                    with gr.Accordion("Upload Reference", open=True):
                        ref_audio_in = gr.Audio(
                            label="Audio File",
                            type="filepath"
                        )
                        ref_text_in = gr.Textbox(
                            label="Reference Text",
                            lines=2,
                            placeholder="Transcription of the reference audio..."
                        )
                
                with gr.Column(scale=2):
                    gr.Markdown("### Text Generation")
                    gen_text_in = gr.Textbox(
                        label="Text to Generate",
                        lines=6,
                        placeholder="Enter the text you want to synthesize..."
                    )
                    
                    with gr.Row():
                        speed_slider = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.2,
                            step=0.1,
                            label="Speed"
                        )
                        device_radio = gr.Radio(
                            choices=["cuda", "cpu"],
                            value="cuda" if torch.cuda.is_available() else "cpu",
                            label="Device",
                            interactive=True
                        )
                    
                    with gr.Row():
                        remove_silence_chk = gr.Checkbox(
                            label="Remove Silence",
                            value=False
                        )
                        offload_btn = gr.Button("♻️ Offload", scale=1)

                    with gr.Accordion("Advanced Performance", open=False):
                        with gr.Row():
                            nfe_steps_slider = gr.Slider(
                                minimum=8,
                                maximum=32,
                                value=16,
                                step=1,
                                label="NFE Steps (lower = faster)"
                            )
                            cfg_strength_slider = gr.Slider(
                                minimum=0.5,
                                maximum=3.0,
                                value=1.5,
                                step=0.1,
                                label="CFG Strength"
                            )

                    with gr.Row():
                        generate_btn = gr.Button("🎤 Synthesize", variant="primary", scale=2)
                        stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)
            
            with gr.Row():
                audio_out = gr.Audio(label="Generated Audio", type="numpy")
                status_out = gr.Textbox(label="Status", interactive=False)
        
        
        # ============ PROFILES TAB ============
        with gr.Tab("👥 Manage Profiles"):
            gr.Markdown("### Save New Profile")
            with gr.Row():
                with gr.Column():
                    profile_name_in = gr.Textbox(
                        label="Profile Name",
                        placeholder="e.g., My Voice"
                    )
                    save_profile_btn = gr.Button("💾 Save Profile", variant="primary")
                
                with gr.Column():
                    gr.Markdown("")  # Spacing
                    save_status = gr.Textbox(interactive=False)
            
            gr.Markdown("### Existing Profiles")
            with gr.Row():
                profile_table = gr.Dataframe(
                    headers=["Name", "Reference Text"],
                    value=get_profile_table_rows(),
                    interactive=False,
                    label="Your Profiles"
                )
            
            gr.Markdown("### Delete Profile")
            with gr.Row():
                del_profile_name = gr.Dropdown(
                    choices=profile_mgr.list_profiles(),
                    label="Profile to Delete"
                )
                del_btn = gr.Button("🗑️ Delete", variant="stop")
        
        
        # ============ HISTORY TAB ============
        with gr.Tab("📜 History"):
            gr.Markdown("### Recent Generations")
            history_table = gr.Dataframe(
                headers=["Path", "Filename"],
                value=get_history_files(),
                interactive=False,
                label="Generated Audio Files"
            )
            
            history_audio = gr.Audio(label="Playback", type="filepath")
            refresh_hist_btn = gr.Button("🔄 Refresh History")
        
        
        # ============ SETTINGS TAB ============
        with gr.Tab("⚙️ Settings"):
            gr.Markdown("### Device Information")
            
            device_info = f"""
            - **Current Device:** {device_mgr.get_device_string()}
            - **CUDA Available:** {torch.cuda.is_available()}
            - **PyTorch Version:** {torch.__version__}
            - **Python Version:** {sys.version.split()[0]}
            """
            gr.Markdown(device_info)
            
            if torch.cuda.is_available():
                mem_info = device_mgr.get_gpu_memory_info()
                mem_text = f"""
                ### GPU Memory
                - **Total:** {mem_info['total_memory'] / 1e9:.2f} GB
                - **Allocated:** {mem_info['allocated_memory'] / 1e9:.2f} GB
                - **Reserved:** {mem_info['reserved_memory'] / 1e9:.2f} GB
                - **Free:** {mem_info['free_memory'] / 1e9:.2f} GB
                """
                gr.Markdown(mem_text)
    
    
    # ============ EVENT HANDLERS ============
    
    # Generate button
    generate_event = generate_btn.click(
        fn=synthesize,
        inputs=[
            profile_dropdown,
            ref_audio_in,
            ref_text_in,
            gen_text_in,
            remove_silence_chk,
            speed_slider,
            nfe_steps_slider,
            cfg_strength_slider,
            device_radio,
        ],
        outputs=[audio_out, status_out, history_table],
        queue=True,
    )

    stop_btn.click(
        fn=lambda: "⏹ Generation interrupted",
        outputs=[status_out],
        cancels=[generate_event],
        queue=False,
    )
    
    # Profile management
    save_profile_btn.click(
        fn=save_profile,
        inputs=[profile_name_in, ref_audio_in, ref_text_in, gr.Checkbox(visible=False, value=False)],
        outputs=[profile_dropdown, save_status]
    )
    
    del_btn.click(
        fn=delete_profile,
        inputs=[del_profile_name],
        outputs=[profile_dropdown, save_status, profile_table]
    )
    
    # Refresh buttons
    refresh_btn.click(
        fn=refresh_profiles,
        outputs=[profile_dropdown]
    )
    
    refresh_hist_btn.click(
        fn=get_history_files,
        outputs=[history_table]
    )
    
    # Offload button
    offload_btn.click(
        fn=offload_model,
        outputs=[status_out]
    )
    
    # History audio playback
    history_table.select(
        fn=load_audio_from_history,
        inputs=[history_table],
        outputs=[history_audio]
    )


def main():
    """Main entry point"""
    try:
        logger.info(f"Starting IndicF5 Neo on {config.ui.host}:{config.ui.port}")
        
        # Pre-load models
        logger.info("Pre-loading models...")
        if inference_engine.load_all_models():
            ui_state.model_loaded = True
            logger.info("Models loaded successfully")
        else:
            logger.warning("Failed to pre-load models, will load on first use")

        try:
            app.queue(default_concurrency_limit=1, max_size=config.ui.max_queue_size)
            logger.info(f"Queue enabled with max_size={config.ui.max_queue_size}")
        except TypeError:
            app.queue(concurrency_count=1, max_size=config.ui.max_queue_size)
            logger.info(f"Queue enabled (legacy args) with max_size={config.ui.max_queue_size}")
        
        # Launch app
        app.launch(
            server_name=config.ui.host,
            server_port=config.ui.port,
            share=config.ui.share,
            debug=config.ui.debug,
            theme=gr.themes.Soft(),
            css="""
            .container { max-width: 1200px; margin: auto; }
            .header { text-align: center; margin-bottom: 20px; }
            .status-box { padding: 10px; border-radius: 5px; margin: 10px 0; }
            """,
        )
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
