"""SolarRenderer custom nodes for ComfyUI's native MiniMax-H3.

- SolarRendererConditioning: structure video + optional first frame + six-part
  prompt -> ref2va conditioning faithful to SolarRenderer's training encoding.
- SolarRendererSampler: DMD-style re-noise Euler sampler for the 4-step LoRA.

The text encoder, VAE and DiT are ComfyUI's own; SolarRenderer LoRAs load
through the stock LoraLoaderModelOnly after a one-time key conversion with
convert_solarrenderer_lora.py. No ComfyUI source file is modified.
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .solarrenderer_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as exc:  # keep the rest of ComfyUI loading if this node fails
    print("[ComfyUI-SolarRenderer] import failed:", exc)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
