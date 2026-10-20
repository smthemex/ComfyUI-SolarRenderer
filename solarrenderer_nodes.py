"""SolarRenderer conditioning for ComfyUI native MiniMax-H3.

Reproduces SolarRenderer's training-time encoding with ComfyUI's own stack:

- the structure video and the optional first frame are cropped/resized onto the
  target canvas and encoded with ComfyUI's native H3 video VAE;
- both latents are attached as ``minimax_keyframes`` at frame 0. ComfyUI's
  PackedLayout places keyframe cond rows on the target's own rotary timeline and
  spatial grid, pinned at t=0.999 - exactly SolarRenderer's "aligned
  observations" (structure laid over every target frame, first frame at latent
  0), as trained in SolarWM's stage0.5 layout;
- the prompt is presented to Qwen3-VL with ``<Video 1>`` (structure, 2 fps with
  timestamps) and ``<Video 2>`` (first frame), matching the six-part prompt
  form of skills/solarrenderer-caption.

The text encoder, VAE and DiT are ComfyUI's own; no ComfyUI source is modified.
"""

from __future__ import annotations

import torch

import nodes
import comfy.utils
import node_helpers


def snap_length(length: int) -> int:
    """Snap a pixel frame count up onto H3's 17k+5 grid."""
    length = max(5, int(length))
    while length % 17 != 5:
        length += 1
    return length


def latent_t_for(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def _resize_to_canvas(frames: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """[T, H, W, C] -> [T, height, width, 3]: centre-crop to the canvas aspect,
    LANCZOS-resize to the canvas - what SolarRenderer's ``aligned_image`` does."""
    frames = frames[..., :3].movedim(-1, 1)
    frames = comfy.utils.common_upscale(frames, width, height, "lanczos", "center")
    return frames.movedim(1, -1)


def _empty_av_latent(width: int, height: int, length: int,fps=24):
    from comfy import model_management

    frame_count = snap_length(length)
    latent_t = latent_t_for(frame_count)
    audio_t = round(frame_count / fps * 40)
    video = torch.zeros([1, 24, latent_t, height // 16, width // 16],
                        device=model_management.intermediate_device())
    audio = torch.zeros([1, 32, 2, audio_t],
                        device=model_management.intermediate_device())
    import comfy.nested_tensor

    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _qwen_video_item(frames: torch.Tensor, timestamps) -> dict:
    return {"type": "video", "data": frames, "timestamps": list(timestamps)}


class SolarRendererConditioning:
    """Structure video (+ optional first frame) + six-part prompt -> SolarRenderer conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE (e.g. minimax_h3_video_vae_fp16.safetensors)"}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Six-part SolarRenderer prompt (skills/solarrenderer-caption). "
                                                 "References <Video 1> (structure) / <Video 2> (first frame)."}),
                "structure_video": ("IMAGE", {"tooltip": "Structure clip frames at 24 fps, at least `length` frames; "
                                                         "frame-aligned with the video being rendered."}),
                "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {"default": 158, "min": 5, "max": 3600, "step": 17,
                                   "tooltip": "Frame count at 24 fps, snapped up to the 17k+5 grid "
                                              "(158 = 6.6 s, SolarRenderer's training length)."}),
                "fps": ("INT", {"default": 24, "min": 5, "max": 120, "step": 1,
                                 "tooltip": "Frames per second for the video being rendered."}),
                                           },
            "optional": {
                "first_frame": ("IMAGE", { "tooltip": "Optional first frame in the target look "
                                                                       "(<Video 2> of the prompt)."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "encode"
    CATEGORY = "conditioning/video_models"
    DESCRIPTION = ("SolarRenderer ref2va conditioning: the structure video is encoded on the target canvas and "
                   "packed on the target's own timeline/grid (aligned observation), the first frame as a keyframe "
                   "at frame 0; the prompt reaches Qwen3-VL as <Video 1>/<Video 2>.")

    def encode(self, clip, vae, prompt, structure_video, width, height, length,fps, first_frame=None):
        frame_count = snap_length(length)
        expected_latent_t = latent_t_for(frame_count)

        latent, frame_count = _empty_av_latent(width, height, frame_count,fps=fps)

        if structure_video is None or structure_video.shape[0] < frame_count:
            got = 0 if structure_video is None else int(structure_video.shape[0])
            raise ValueError(
                "the structure video needs at least {} frames at {} fps (got {}); trim the loader or lower length".format(
                    frame_count, fps, got))
        structure = _resize_to_canvas(structure_video[:frame_count].float(), width, height)

        keyframes = []
        if first_frame is not None:
            first = _resize_to_canvas(first_frame[:1].float(), width, height)
            first_latent = vae.encode(first)
            if first_latent.shape[2] != 1:
                raise ValueError("first frame VAE encode produced {} latent frames, expected 1".format(
                    first_latent.shape[2]))
            # row order mirrors SolarRenderer: first frame rows, then the structure rows
            keyframes.append({"resolved_frame_index": 0, "latent": first_latent})

        structure_latent = vae.encode(structure)
        if structure_latent.shape[2] != expected_latent_t:
            raise ValueError("structure VAE encode produced {} latent frames, expected {}".format(
                structure_latent.shape[2], expected_latent_t))
        keyframes.append({"resolved_frame_index": 0, "latent": structure_latent})

        # Qwen3-VL presentation, in the trained order: <Video 1> structure at 2 fps
        # with timestamps, then <Video 2> the first frame (held ~0.5 s, as the
        # 13-frame still of SolarRenderer's front end decodes to two 2 fps samples).

        sample_idx = list(range(0, frame_count, int(fps/2)))
        ref_items = [_qwen_video_item(structure[sample_idx], [i / 2.0 for i in sample_idx])]
        if first_frame is not None:
            ref_items.append(_qwen_video_item(torch.cat([first[:1], first[:1]], dim=0), [0.0, 0.5]))

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes})
        return (cond, latent)


class SolarRendererSampler:
    """DMD-style re-noise Euler sampler, as SolarRenderer's 4-step student was trained."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = ("Fresh-noise Euler sampling: each step predicts x0, then re-noises it to the next sigma. "
                   "Use a simple scheduler with 4 steps and MiniMaxH3SigmaShift at video=12, audio=3.")

    def get_sampler(self):
        import comfy.samplers
        from .solarrenderer_sampling import renoise_euler_sampler

        return (comfy.samplers.KSAMPLER(renoise_euler_sampler),)


NODE_CLASS_MAPPINGS = {
    "SolarRendererConditioning": SolarRendererConditioning,
    "SolarRendererSampler": SolarRendererSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SolarRendererConditioning": "SolarRenderer Conditioning (Structure to Video)",
    "SolarRendererSampler": "SolarRenderer Renoise Sampler (4-step)",
}
