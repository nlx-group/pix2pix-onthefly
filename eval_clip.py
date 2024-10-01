import sys, os
sys.path.append("pix2pix_zero/src/utils")
from datasets import load_dataset, load_from_disk
from transformers import CLIPModel, CLIPProcessor, CLIPImageProcessor
from PIL import Image
from pathlib import Path
import torch
from tqdm import tqdm
from pix2pix_decoder import Pix2Pix_decoder
import transformers
import accelerate

import matplotlib.pyplot as plt

import logging

logging.disable(logging.WARNING)
# load dataset
dataset = load_from_disk("test/magicbrush_testset-100-sd1.5-100steps")
#shuffle
dataset = dataset.shuffle(seed=42)
print(dataset)

# load clip
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#load embedder clip
embedder_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cpu")
embedder_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# load editor
editor = Pix2Pix_decoder()

save = False

img_tgt_list = []
txt_tgt_list = []

img_tgt_cosine_avg = 0
img_src_cosine_avg = 0
text_tgt_cosine_avg = 0
text_src_cosine_avg = 0

original_img_avg = 0
reconstructed_img_avg = 0
original_text_avg = 0
reconstructed_text_avg = 0

n_count = 0
with tqdm(dataset, unit="folders") as pbar:
    for folder in pbar:
        n_count += 1
        torch.cuda.empty_cache()

        image_src = "test/"+folder["input_image"]
        image_tgt = "test/"+folder["output_image"] 
        prompt_tgt = folder["output_caption"]
        prompt_src = folder["input_caption"]      

        ###################TESTE###################
        # image_src = Image.open(image_src).resize((512,512), Image.Resampling.LANCZOS)
        # image_tgt = Image.open(image_tgt).resize((512,512), Image.Resampling.LANCZOS)
        # image_tgt_encoded = embedder_processor(images=image_tgt, return_tensors="pt").to("cpu")
        # image_src_encoded = embedder_processor(images=image_src, return_tensors="pt").to("cpu")

        # image_tgt_features = embedder_clip.get_image_features(**image_tgt_encoded)
        # torch.cuda.empty_cache()
        # image_src_features = embedder_clip.get_image_features(**image_src_encoded)
        # torch.cuda.empty_cache()
  

        # img_dif = (image_tgt_features-image_src_features)                                                                       
        ###################TESTE###################
    
        image_edit, image_recon  = editor.edit_real(folder["xg_inv"][0], folder["edit_caption"], folder["generated_input_caption"], folder["output_caption"])
        torch.cuda.empty_cache()

        ##################TESTE###################
        # image_recon_encoded = embedder_processor(images=image_recon, return_tensors="pt").to("cpu")
        # image_recon_features = embedder_clip.get_image_features(**image_recon_encoded)
        # torch.cuda.empty_cache()
        # img_diff = (image_tgt_features-image_recon_features)
        # image_edit, image_recon  = editor.edit_real(folder["xg_inv"][0], folder["edit_caption"], folder["generated_input_caption"], folder["output_caption"], img_dif)

        ###################TESTE###################
        image_src = Image.open(image_src).resize((512,512), Image.Resampling.LANCZOS) # Uncomment this line to use the original code
        image_tgt = Image.open(image_tgt).resize((512,512), Image.Resampling.LANCZOS) # Uncomment this line to use the original code

        if save:
            width = image_src.width + image_recon[0].width + image_edit[0].width + image_tgt.width
            height = max(image_src.height, image_recon[0].height, image_edit[0].height, image_tgt.height)
            dst = Image.new('RGB', (width, height))
            dst.paste(image_src, (0, 0))
            dst.paste(image_recon[0], (image_src.width, 0))
            dst.paste(image_edit[0], (image_src.width+image_recon[0].width, 0))
            dst.paste(image_tgt, (image_src.width+image_recon[0].width+image_edit[0].width, 0))
            dst.save("output/"+"mb_edit"+folder["edit_caption"]+str(n_count)+".png")
            
            
        # encode images
        image_edit_encoded = processor(images=image_edit, return_tensors="pt").to("cpu")
        image_tgt_encoded = processor(images=image_tgt, return_tensors="pt").to("cpu")
        image_src_encoded = processor(images=image_src, return_tensors="pt").to("cpu")
        image_recon_encoded = processor(images=image_recon, return_tensors="pt").to("cpu")

        # encode text
        tgt_encoded = processor(text=prompt_tgt, return_tensors="pt").to("cpu")
        src_encoded = processor(text=prompt_src, return_tensors="pt").to("cpu")

        # get image features
        image_edit_features = clip.get_image_features(**image_edit_encoded)
        torch.cuda.empty_cache()
        image_tgt_features = clip.get_image_features(**image_tgt_encoded)
        torch.cuda.empty_cache()
        image_src_features = clip.get_image_features(**image_src_encoded)
        torch.cuda.empty_cache()
        image_recon_features = clip.get_image_features(**image_recon_encoded)

        # get text features
        tgt_features = clip.get_text_features(**tgt_encoded)
        src_features = clip.get_text_features(**src_encoded)

        # get cosine similarity
        cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        img_tgt_cosine_avg += cosine_sim(image_edit_features, image_tgt_features).item()
        img_tgt_list.append(cosine_sim(image_edit_features, image_tgt_features).item())
        img_src_cosine_avg += cosine_sim(image_edit_features, image_src_features).item()
        text_tgt_cosine_avg += cosine_sim(image_edit_features, tgt_features).item()
        img_tgt_list.append(cosine_sim(image_edit_features, tgt_features).item())
        text_src_cosine_avg += cosine_sim(image_edit_features, src_features).item()

        original_img_avg += cosine_sim(image_src_features, image_tgt_features).item()
        reconstructed_img_avg += cosine_sim(image_recon_features, image_tgt_features).item()
        original_text_avg += cosine_sim(image_src_features, tgt_features).item()
        reconstructed_text_avg += cosine_sim(image_recon_features, tgt_features).item()
        

        pbar.set_postfix(tgt_img = img_tgt_cosine_avg/n_count, src_img = img_src_cosine_avg/n_count, tgt_txt = text_tgt_cosine_avg/n_count, src_txt = text_src_cosine_avg/n_count)
        
        # if save and n_count >5:
        #     break

print("Editing cosine similarity to:")
print("    Target image: ", img_tgt_cosine_avg/n_count)
print("    Source image: ", img_src_cosine_avg/n_count)
print("    Target text: ", text_tgt_cosine_avg/n_count)
print("    Source text: ", text_src_cosine_avg/n_count)
print("Baseline* cosine similarity  to target image:")
print("    *Original image: ", original_img_avg/n_count)
print("    *Reconstructed image: ", reconstructed_img_avg/n_count)
print("Baseline* cosine similarity  to target text:")
print("    *Original image: ", original_text_avg/n_count)
print("    *Reconstructed image: ", reconstructed_text_avg/n_count)

value_05 = 0
for value in img_tgt_list:
    if value >= 0.5:
        value_05 += 1
print("Number of img values above 0.5: ", value_05)

value_05 = 0
for value in txt_tgt_list:
    if value >= 0.5:
        value_05 += 1
print("Number of txt values above 0.5: ", value_05)

plt.hist(img_tgt_list)
plt.xlabel("Cosine similarity")
plt.ylabel("Count")
plt.savefig("output/hist_clip-i.png")

plt.hist(txt_tgt_list)
plt.xlabel("Cosine similarity")
plt.ylabel("Count")
plt.savefig("output/hist_clip-t.png")
