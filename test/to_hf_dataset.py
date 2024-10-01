import sys, os
sys.path.append("../pix2pix_zero/src/utils")
from datasets import load_dataset, Dataset
from transformers import CLIPModel, CLIPProcessor, CLIPImageProcessor
from lavis.models import load_model_and_preprocess

from PIL import Image
from pathlib import Path
import torch
from tqdm import tqdm
import json
import random


seed = 42
torch.manual_seed(seed)
random.seed(seed)   


device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = 'CompVis/stable-diffusion-v1-4'
model_path = 'runwayml/stable-diffusion-v1-5'

torch_dtype = torch.float16
num_ddim_steps = 100
null_prompt = False

# load DDIM
if null_prompt:

    from null_inversion import NullInversion
    null_inversion = NullInversion(model_path, device=device)

else:
    from ddim_inv import DDIMInversion
    from scheduler import DDIMInverseScheduler
    pipe_invert = DDIMInversion.from_pretrained(model_path, torch_dtype=torch_dtype).to(device)
    pipe_invert.scheduler = DDIMInverseScheduler.from_config(pipe_invert.scheduler.config)


model_blip, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=torch.device(device))


# load datasets
dataset_list = []
with open("edit_sessions.json", "r") as f, open("global_descriptions.json", "r") as f2:
    requests = json.load(f)
    keys = list(requests.keys())
 
    descriptions = json.load(f2)
    print("Generating captions...")
    for folder in tqdm(keys):
                     
        #print("Folder: ",folder, ", num. requests: ",len(requests[folder]))
        for i, request in enumerate(requests[folder]):
            from_original = i==0
            #print("    Request ",i,": ")
            input_caption = descriptions[folder][request["input"]]
            #print("        Input: ",input_caption)
            edit_caption = request["instruction"]
            #print("        Request: ", edit_caption)
            output_caption = descriptions[folder][request["output"]]
            #print("        Output: ",output_caption)
            input_image = "images/"+folder+"/"+request["input"]
            #print("        Path input: ",input_image)
            output_image = "images/"+folder+"/"+request["output"]
            #print("        Path output: ",output_image)

            img = Image.open(input_image).resize((512,512), Image.Resampling.LANCZOS).convert("RGB")
            _image = vis_processors["eval"](img).unsqueeze(0).to(device)
            generated_input_caption = model_blip.generate({"image": _image})[0]

            dataset_list += [{
                "input_caption": input_caption,
                "generated_input_caption": generated_input_caption,
                "edit_caption": edit_caption,
                "output_caption": output_caption,
                "input_image": input_image,
                "output_image": output_image,
                "from_original": from_original
            }]

random.shuffle(dataset_list)
final_daset_list = []
for dict in tqdm(dataset_list):
    input_caption = dict["input_caption"]
    input_image = dict["input_image"]

    # invert image
    if null_prompt:
        _, x_inv, _ = null_inversion.invert(input_image, input_caption, offsets=(0,0,200,0), verbose=True)
    else:
        img = Image.open(input_image).resize((512,512), Image.Resampling.LANCZOS).convert("RGB")
        x_inv, _, _ = pipe_invert(
            input_caption, 
            guidance_scale=1,
            num_inversion_steps=num_ddim_steps,
            img=img,
            torch_dtype=torch_dtype
        )

    if null_prompt:
        _, xg_inv, _ = null_inversion.invert(input_image, generated_input_caption, offsets=(0,0,200,0), verbose=True)
    else:
        img = Image.open(input_image).resize((512,512), Image.Resampling.LANCZOS).convert("RGB")
        xg_inv, _, _ = pipe_invert(
            generated_input_caption, 
            guidance_scale=1,
            num_inversion_steps=num_ddim_steps,
            img=img,
            torch_dtype=torch_dtype
        )

    dict["x_inv"] = x_inv 
    dict["xg_inv"] = xg_inv    
    final_daset_list.append(dict)

# list to dataset
dataset = Dataset.from_list(final_daset_list)
print(dataset)
#print(dataset[0])

# save dataset
dataset.save_to_disk("magicbrush_testset-100-sd1.5-100steps")
   
