#!/bin/sh

cd SynthText
python gen.py \
    --begin_index=0 \
    --end_index=30 \
    --max_time=100 \
    --gen_data_path= \
    --bg_data_path= \
    --instance_per_image=5 \
    --output_path= \
    --jobs=5

python to_image.py \
    --input_folder= \
    --output_folder=
