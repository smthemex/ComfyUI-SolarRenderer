"""DMD-style re-noise Euler sampling for the SolarRenderer 4-step student.

Matches SolarWM ``H3Stage0p5Core.generate(..., renoise=True)``: each evaluation
predicts x0, then the sample is rebuilt as
``x_next = (1 - sigma_next) * x0 + sigma_next * fresh_noise``
instead of carrying the old noise along - the loop the four-step LoRA was
post-trained for. ComfyUI's model wrapper already handles the audio stream's
own shifted schedule, so one loop covers both streams of the packed AV latent.
"""

from __future__ import annotations

import torch
from tqdm.auto import trange


@torch.no_grad()
def renoise_euler_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
    import comfy.k_diffusion.sampling

    extra_args = {} if extra_args is None else extra_args
    noise_sampler = comfy.k_diffusion.sampling.default_noise_sampler(x, seed=extra_args.get("seed"))
    s_in = x.new_ones((x.shape[0],))
    for index in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[index]
        sigma_next = sigmas[index + 1]
        denoised = model(x, sigma * s_in, **extra_args)
        if float(sigma_next) == 0.0:
            x = denoised
        else:
            noise = noise_sampler(sigma, sigma_next)
            x = (1.0 - sigma_next) * denoised + sigma_next * noise
        if callback is not None:
            callback({
                "i": index,
                "denoised": denoised,
                "x": x,
                "sigma": sigma,
                "sigma_hat": sigma,
            })
    return x
