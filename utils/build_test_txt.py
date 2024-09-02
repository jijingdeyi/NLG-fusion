import os
import random
from pathlib import Path

# set the path of test images
image_dir = Path('/data/ykx/RoadScene/test/ir')

# set the save path of pred.txt
pred_file = Path('/data/ykx/RoadScene/test/meta/pred.txt')

# ensure that the directory of the output file exists
pred_file.parent.mkdir(parents=True, exist_ok=True)

# supported image file formats
supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}

# get all supported image files in different formats
image_files = [f.name for f in image_dir.glob('*') if f.suffix.lower() in supported_formats]
image_files.sort()

# write the training set file name to pred.txt
with pred_file.open('w') as f:
    for image_file in image_files:
        f.write(image_file + '\n')

print(f'file pred.txt completed, include {len(image_files)} images.')
