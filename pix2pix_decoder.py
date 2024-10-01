import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffusers import DDIMScheduler
from pix2pix_zero.src.utils.edit_directions import construct_direction
from sd_utils import ReeditingPipeline
from decoder_utils import CreateDirectionCaptions
from lavis.models import load_model_and_preprocess

from PIL import Image
from pix2pix_zero.src.utils.edit_pipeline import EditingPipeline


def load_sentence_embeddings(l_sentences, tokenizer, text_encoder, device):
    with torch.no_grad():
        l_embeddings = []
        for sent in l_sentences:
            text_inputs = tokenizer(
                    sent,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=None)[0]
            l_embeddings.append(prompt_embeds)
    return torch.concatenate(l_embeddings, dim=0).mean(dim=0).unsqueeze(0)

class Pix2Pix_decoder():

    def __init__(self):
        
        torch.manual_seed(42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_path = 'CompVis/stable-diffusion-v1-4'
        self.model_path = 'runwayml/stable-diffusion-v1-5'
        self.torch_dtype = torch.float16
        self.num_ddim_steps = 100
        self.xa_guidance = 0.15
        self.negative_guidance_scale = 5.0
        self.x = torch.randn((1,4,64,64), device=self.device)
        self.results_folder = 'output/'
        

        torch.set_default_device(self.device) 
        

        self.model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

        # load the BLIP model
        # self.model_blip, self.vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=torch.device(self.device))

        # # load inversion model
        # self.pipe_invert = DDIMInversion.from_pretrained(self.model_path, torch_dtype=self.torch_dtype).to(self.device)
        # self.pipe_invert.scheduler = DDIMInverseScheduler.from_config(self.pipe_invert.scheduler.config)

        # Make the editing pipeline
        self.pipe = EditingPipeline.from_pretrained(self.model_path, torch_dtype=self.torch_dtype).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # load caption creation model
        self.captions_creator = CreateDirectionCaptions("microsoft/phi-2", "prompts/0shot_1.txt", "magicbrush_dev_25", n_captions=1, n_shots=0)


   
       
        

    def edit_real(self, x_inv, edit_string, prompt_str=None, target_str=None, img_dif=None):
       
        x_inv = x_inv

        # Make the edit direction      
        #text_negative, text_positive = self.captions_creator.generate(edit_string, prompt_str)
        text_negative = [prompt_str]
        text_positive = [target_str]
        #print(text_negative)
        #print(text_positive)
        
        negative_emb = load_sentence_embeddings(text_negative, self.pipe.tokenizer, self.pipe.text_encoder, device=self.device)
        positive_emb = load_sentence_embeddings(text_positive, self.pipe.tokenizer, self.pipe.text_encoder, device=self.device)
        edit_direction = (positive_emb.mean(0)-negative_emb.mean(0)).unsqueeze(0)
        if img_dif is not None:
            edit_direction = img_dif.to(self.device)
     

        torch.cuda.empty_cache()
        # Make the edit
        rec_pil, edit_pil = self.pipe(prompt_str,
            num_inference_steps=self.num_ddim_steps,
            x_in=torch.tensor(x_inv).unsqueeze(0),
            edit_dir=edit_direction,
            guidance_amount=self.xa_guidance,
            guidance_scale=self.negative_guidance_scale,
            negative_prompt=prompt_str # use the unedited prompt for the negative prompt
            )
        torch.cuda.empty_cache()

        # return rec_pil, edit_pil, x_inv_image, x_dec_img, prompt_str
        return edit_pil, rec_pil
    