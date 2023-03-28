# Copyright 2023 (C) antillia.com. All Rights Reserved.
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
# ==============================================================================

# create_master.py
# 2023/03/28 to-arai
#

import glob
import os
from PIL import Image
import traceback
import shutil

def split(input_dir, category, ouput_dir):
  [src, dest] = category
  pattern = input_dir + "/" + src + "*.tif" 
  files = glob.glob(pattern)
  print("Pattern {}".format(pattern))

  print("Files {}".format(files))
  #input("HIT")
  for file in files:
     basename = os.path.basename(file)
     output_sub_dir = output_dir + "/" + dest + "/"
     if not os.path.exists(output_sub_dir):
       os.makedirs(output_sub_dir)
     filename = basename.split(".")[0]
     
     # 1 Create a save file name as jpg
     output_file = os.path.join(output_sub_dir, filename + ".jpg")

     # 2 Resize the original image to 1/5
     img = Image.open(file)
     w, h = img.size
     rw = int(w/5)
     rh = int(h/5)
     img_resized = img.resize((rw, rh), Image.LANCZOS)

     # 3 Save the resized_imaged as a JPEG file
     img_resized.save(output_file, "JPEG", quality=100)

if __name__ == "__main__":
  try:
    input_dir = "./Pictures"
    categories = [["","MYC_IHC"]]
    #pos_or_neg = ["pos", "neg"]
    output_dir = "./Lymphoma_images_master"
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    for category in categories:
        split(input_dir, category, output_dir)

    input_dir = "./Pictures/dir_out"
    categories = [["dab", "DAB"], 
                  ["dist", "DistanceMap"], 
                  ["hem",  "Hematoxylin"]
                 ]
    pos_or_neg = ["pos", "neg"]
    output_dir = "./Lymphoma_images_master"
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    for category in categories:
        split(input_dir, category, output_dir)

  except:
    traceback.print_exc()

