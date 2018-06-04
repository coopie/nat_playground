# vae-nat
using Noise-As-Targets to train VAEs


# Making a video out of the stored frames:
    ffmpeg -framerate 25 -pattern_type glob -i '<images_dir>/*.png' -c:v libx264 -pix_fmt yuv420p <output_name>
