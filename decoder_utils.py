import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from datasets import Dataset
import inflect
import random


phy = True

class CreateDirectionCaptions():
    def __init__(self, model_name_or_path, prompt_path, example_path=None, n_captions=1, n_shots=1, device="cuda"):
        torch.set_default_device(device)
        self.n_captions = n_captions
        self.n_shots = n_shots

        if phy:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto", trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        else:
            model_id = "mistralai/Mistral-7B-Instruct-v0.2"
            model_id = "meta-llama/Llama-2-7b-hf"
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

        if example_path is not None:
            self.examples_dataset = Dataset.load_from_disk(example_path)
        else:
            self.examples_dataset = None

        with open(prompt_path, "r") as f:
            self.prompt = f.read()

    def _create_few_shot_examples(self, number):
        few_shot_examples = ""
        prompt = self.prompt.replace("[NUMBER]", number)
        # random n_shot number of examples
        index_list = random.sample(range(len(self.examples_dataset)), self.n_shots)
        #print(index_list)
        for i in index_list:
            example = self.examples_dataset[i]
            instruction = example["instruction"]
            shot_example = prompt.replace("[TRANSFORMATION]", instruction)
            for i, negative in enumerate(example["negative"][:self.n_captions]):
                if i == 0:
                    shot_example += " " + negative + "\n"
                else:
                    num= i+1
                    shot_example += "Caption " +str(num)+": " + negative + "\n"
            
            shot_example += "\nAfter transformation:\n\n"

            for i, positive in enumerate(example["positive"][:self.n_captions]):
                num = i+1
                shot_example += "Caption " +str(num)+": " + positive + "\n"
            
            shot_example += "\n"

            few_shot_examples += shot_example
        
        return few_shot_examples

            

    def generate(self, input_text, prompt_str=None):
        text_negative, text_positive = None, None
        p = inflect.engine()
        number = p.number_to_words(self.n_captions) + " ("+str(self.n_captions)+")"
        
        n_tries = 0
        text_negative_list = []
        text_positive_list = []
        while text_negative is None or text_positive is None:
            
            ## Allow for a few tries before giving up
            if n_tries > 0 and n_tries < 10:
                print("Error number",n_tries,"in text, trying again...")
            if n_tries >= 10:
                if len(text_negative_list) > 0:
                    if prompt_str is not None:
                        text_negative_list = list(set(text_negative_list)) # remove duplicates
                        # guarantee that the prompt is in the negative captions
                        text_negative_list.remove(prompt_str) # remove the prompt from the list
                        text_negative = text_negative_list[:self.n_captions-1] # retrieve n_captions-1 captions
                        text_negative += [prompt_str] # add the prompt back
                    else:
                        text_negative_list = list(set(text_negative_list))
                        text_negative = text_negative_list[:self.n_captions]
                    print("Using previous text with length",len(text_negative),"for negative captions")
                else:
                    text_negative = [" "]
                    print("Using empty text for negative captions")

                # same for positive captions
                if len(text_positive_list) > 0:
                    text_positive_list = list(set(text_positive_list))
                    text_positive = text_positive_list[:self.n_captions]
                    print("Using previous text with length",len(text_positive),"for positive captions")
                else:
                    text_positive = [" "]
                    print("Using empty text for positive captions")
        
                return text_negative, text_positive
            
            ## Create the input text
            in_text = self.prompt.replace("[NUMBER]", number)
            in_text = in_text.replace("[TRANSFORMATION]", input_text)
            if prompt_str is not None:
                in_text = in_text + " " + prompt_str+"\n"

            # Add few shot examples
            if self.examples_dataset is not None and self.n_shots > 0:
                few_shot_examples = self._create_few_shot_examples(number)
                in_text = few_shot_examples + in_text
            #print(in_text)
            inputs = self.tokenizer(in_text, return_tensors="pt", return_attention_mask=False)

            ## Generate the text
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=3400, do_sample=True)
            text = self.tokenizer.batch_decode(outputs)[0]

            ## Strip the text into negative and positive captions
            text = text.split("<|endoftext|>")[0]
            try:
                text_negative, text_positive = self._strip(text)
            except:
                text_negative, text_positive = None, None
            n_tries += 1

            ## Check if number of captions is correct and if not, try again. Except if number of captions is > needed number of captions
            if text_negative != None:
                if len(text_negative) == 0:
                    text_negative = None
                elif len(text_negative) < self.n_captions:
                    text_negative_list += text_negative
                    text_negative = None
                elif len(text_negative) > self.n_captions and text_positive!=None:
                    if len(text_positive) > self.n_captions:
                        text_negative = text_negative[:self.n_captions]
                    else:
                        text_negative_list += text_negative
                        text_negative = None
                elif len(text_negative) > self.n_captions:
                    text_negative_list += text_negative
                    text_negative = None
            
            if text_positive != None:
                if len(text_positive) == 0:
                    text_positive = None
                elif len(text_positive) < self.n_captions:
                    text_positive_list += text_positive
                    text_positive = None
                elif len(text_positive) > self.n_captions and text_negative!=None:
                    if len(text_negative) > self.n_captions:
                        text_positive = text_positive[:self.n_captions]
                    else:
                        text_positive_list += text_positive
                        text_positive = None
                elif len(text_positive) > self.n_captions:
                    text_positive_list += text_positive
                    text_positive = None
           
            #print(text_negative, text_positive)
        return text_negative, text_positive


    def _strip(self, text):
        #text = text.split("Output: ")[-1]
        text = text.split("Before transformation")[-1]
        text = text.split("After transformation")
        
        text_negative = text[0].split("\n")
        text_positive = text[1].split("\n")    

        text_negative = [t.split(": ")[-1] for t in text_negative if t.startswith("Caption")]
        text_positive = [t.split(": ")[-1] for t in text_positive if t.startswith("Caption")]
       

        return text_negative, text_positive




