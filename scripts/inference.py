# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
import cv2  # Import necessary library for video processing
from gfpgan import GFPGANer
from codeformer import CodeFormer
def calculate_resolution_ratio(input_video_path, output_video_path):
    input_video = cv2.VideoCapture(input_video_path)
    output_video = cv2.VideoCapture(output_video_path)

    input_frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_frame_width = int(output_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_frame_height = int(output_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    input_resolution = input_frame_width * input_frame_height
    output_resolution = output_frame_width * output_frame_height

    return output_resolution / input_resolution if input_resolution > 0 else 1

def apply_gfpgan_superres(video_out_path):
    gfpgan_model = GFPGANer()
    gfpgan_model.enhance(video_out_path)

def apply_codeformer_superres(video_out_path):
    codeformer_model = CodeFormer()
    codeformer_model.enhance(video_out_path)
# def main(config, args):
#     # Check if the GPU supports float16
#     is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
#     dtype = torch.float16 if is_fp16_supported else torch.float32

#     print(f"Input video path: {args.video_path}")
#     print(f"Input audio path: {args.audio_path}")
#     print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

#     scheduler = DDIMScheduler.from_pretrained("configs")

#     if config.model.cross_attention_dim == 768:
#         whisper_model_path = "checkpoints/whisper/small.pt"
#     elif config.model.cross_attention_dim == 384:
#         whisper_model_path = "checkpoints/whisper/tiny.pt"
#     else:
#         raise NotImplementedError("cross_attention_dim must be 768 or 384")

#     audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

#     vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
#     vae.config.scaling_factor = 0.18215
#     vae.config.shift_factor = 0

#     unet, _ = UNet3DConditionModel.from_pretrained(
#         OmegaConf.to_container(config.model),
#         args.inference_ckpt_path,  # load checkpoint
#         device="cpu",
#     )

#     unet = unet.to(dtype=dtype)

#     # set xformers
#     if is_xformers_available():
#         unet.enable_xformers_memory_efficient_attention()

#     pipeline = LipsyncPipeline(
#         vae=vae,
#         audio_encoder=audio_encoder,
#         unet=unet,
#         scheduler=scheduler,
#     ).to("cuda")

#     if args.seed != -1:
#         set_seed(args.seed)
#     else:
#         torch.seed()

#     print(f"Initial seed: {torch.initial_seed()}")

#     pipeline(
#         video_path=args.video_path,
#         audio_path=args.audio_path,
#         video_out_path=args.video_out_path,
#         video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),
#         num_frames=config.data.num_frames,
#         num_inference_steps=args.inference_steps,
#         guidance_scale=args.guidance_scale,
#         weight_dtype=dtype,
#         width=config.data.resolution,
#         height=config.data.resolution,
#     )
def main(config, args):
    # Check if the GPU supports float16
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,  # load checkpoint
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    # set xformers
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

<<<<<<< HEAD
    # Handling the superres argument (for GFPGAN, CodeFormer, or both)
    superres = args.superres

    # Validate superres argument
    if superres not in ["GFPGAN", "CodeFormer", "None", "GFPGAN,CodeFormer"]:
        raise ValueError("Invalid superres model choice. Use 'GFPGAN', 'CodeFormer', 'GFPGAN,CodeFormer' or 'None'.")

    # If both models are provided (GFPGAN and CodeFormer), split the models into a list
    if superres == "GFPGAN,CodeFormer":
        models = ["GFPGAN", "CodeFormer"]
    else:
        # If only one model is provided, split by comma if necessary (e.g., "GFPGAN" or "CodeFormer")
        models = superres.split(",")

    print(f"Applying super-resolution models: {models}")

    # Add logic to calculate resolution ratio between input and output
    video_resolution_ratio = calculate_resolution_ratio(args.video_path, args.video_out_path)

    # Apply super-resolution if the output resolution is poorer than the input
    if video_resolution_ratio > 1:
        if "GFPGAN" in models:
            print("Applying GFPGAN super-resolution...")
            apply_gfpgan_superres(args.video_out_path)  # Apply GFPGAN super-resolution only to generated part
        if "CodeFormer" in models:
            print("Applying CodeFormer super-resolution...")
            apply_codeformer_superres(args.video_out_path)  # Apply CodeFormer super-resolution only to generated part

    # Run the inference pipeline
    print("Running the pipeline...")
=======
>>>>>>> 5289c629cd23b4b3dffaebf805e0e012ea90ed23
    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)

    main(config, args)
