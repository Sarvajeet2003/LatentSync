# LatentSync Enhancements and Super-Resolution Integration

This repository implements a system for generating lipsynced videos by enhancing video quality using super-resolution models such as **GFPGAN** and **CodeFormer**. The following changes have been made to integrate super-resolution functionality using both models.

## Changes Overview

### 1. **Super-Resolution Integration**
   I have integrated two super-resolution models, **GFPGAN** and **CodeFormer**, to improve the quality of the generated part of the video. You can now specify which super-resolution model(s) to apply using the `--superres` argument when running the inference script.

   - **GFPGAN**: This model is used to enhance the generated part of the video.
   - **CodeFormer**: This model is also used for improving the video quality.
   - **Both Models**: Both **GFPGAN** and **CodeFormer** can be applied sequentially to the video by passing `"GFPGAN,CodeFormer"` in the `--superres` argument.
   - **No Super-Resolution**: If you do not want to apply any super-resolution, use `"None"`.

### 2. **Modifications in the `inference.sh` Script**
   I modified the `inference.sh` script to support the `--superres` argument. This allows me to pass either `GFPGAN`, `CodeFormer`, `GFPGAN,CodeFormer`, or `None` to control which super-resolution models should be applied to the generated video.

   **Example Usage:**
   - To apply **GFPGAN**:  
     ```bash
     ./inference.sh GFPGAN
     ```
   - To apply **CodeFormer**:  
     ```bash
     ./inference.sh CodeFormer
     ```
   - To apply **both GFPGAN and CodeFormer**:  
     ```bash
     ./inference.sh GFPGAN,CodeFormer
     ```
   - To **not apply any super-resolution**:  
     ```bash
     ./inference.sh None
     ```

### 3. **Modifications in the `inference.py` Script**
   The main inference logic was updated to check the `--superres` argument. If the argument specifies a super-resolution model, it calculates the resolution ratio between the input video and the output video. If the generated video has a lower resolution than the input video, the chosen super-resolution model(s) are applied to enhance the generated video.

   **Changes in the `main()` function:**
   - The function accepts the `--superres` argument, which can be one of the following values:
     - `"GFPGAN"`: Apply only GFPGAN.
     - `"CodeFormer"`: Apply only CodeFormer.
     - `"GFPGAN,CodeFormer"`: Apply both models sequentially.
     - `"None"`: Apply no super-resolution.
   - The resolution ratio between input and output videos is calculated using the `calculate_resolution_ratio()` function. If the output resolution is poorer, the specified super-resolution model(s) are applied.

   **Super-Resolution Model Application:**
   - If **GFPGAN** is specified, it is applied to the generated part of the video using `apply_gfpgan_superres()`.
   - If **CodeFormer** is specified, it is applied to the generated part of the video using `apply_codeformer_superres()`.

   **Helper Functions Added:**
   - `calculate_resolution_ratio(input_video_path, output_video_path)`: This function calculates the resolution ratio between the input and output videos.
   - `apply_gfpgan_superres(video_out_path)`: This function applies GFPGAN super-resolution to the output video.
   - `apply_codeformer_superres(video_out_path)`: This function applies CodeFormer super-resolution to the output video.

### 4. **Updated `requirements.txt`**
   To ensure the new super-resolution models work, I updated the `requirements.txt` to include the necessary dependencies:
   - **GFPGAN**: The package was added to the `requirements.txt` to ensure the model can be used in the project.
   - **CodeFormer**: I added the dependency for **CodeFormer** to the `requirements.txt` to make sure the model is available for enhancing video quality.

   Example entries in `requirements.txt`:
   ```txt
   torch>=1.9.0
   gfpgan
   codeformer
   xformers==0.0.25  # For memory-efficient attention if needed
