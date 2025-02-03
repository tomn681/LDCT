from config import config
from diffusers import UNet2DConditionModel

model = UNet2DConditionModel(
    sample_size=config.image_size, # the target image resolution
    in_channels=config.channels,  # the number of input channels, 3 for RGB images
    out_channels=config.channels,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    only_cross_attention=False,
    layers_per_block=2,
    block_out_channels=(320, 640, 1280, 1280),
    down_block_types=(
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    mid_block_type="UNetMidBlock2DCrossAttn",
    up_block_types=(
        "UpBlock2D", 
        "CrossAttnUpBlock2D", 
        "CrossAttnUpBlock2D", 
        "CrossAttnUpBlock2D"),
    encoder_hid_dim_type="image_proj" # "image_proj" if image conditioning, "text_proj" if text conditioning, "text_image_proj" if dual.
    )
