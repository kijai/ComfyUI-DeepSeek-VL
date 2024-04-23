import os
import torch
from torchvision.transforms import ToPILImage

from transformers import AutoModelForCausalLM
from .deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM

import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))

class deepseek_vl_model_loader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                [   
                    "deepseek-vl-1.3b-chat",
                    "deepseek-vl-1.3b-base",
                    "deepseek-vl-7b-chat",
                    "deepseek-vl-7b-base",
                ],
                {
                "default": "deepseek-vl-7b-chat"
                }),
            },
        }

    RETURN_TYPES = ("DEEPSEEKVLMODEL",)
    RETURN_NAMES = ("deepseek_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "deepseek-vl"

    def loadmodel(self, model):
        mm.soft_empty_cache()
        dtype = mm.vae_dtype()
        device = mm.get_torch_device()
        custom_config = {
            "model": model,
        }
        if not hasattr(self, "model") or custom_config != self.current_config:
            self.current_config = custom_config
            model_dir = (os.path.join(folder_paths.models_dir, "LLM", "deepseek-vl"))
            checkpoint_path = os.path.join(model_dir, model)

            if not os.path.exists(checkpoint_path):
                print(f"Downloading {model}")
                from huggingface_hub import snapshot_download
                
                snapshot_download(repo_id=f"deepseek-ai/{model}",  
                                  local_dir=checkpoint_path, 
                                  local_dir_use_symlinks=False
                                  )
                model_path = checkpoint_path
            else:
                model_path = os.path.join(folder_paths.models_dir, "LLM", "deepseek-vl", model)
            print(f"Loading model from {model_path}")

            vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
            tokenizer = vl_chat_processor.tokenizer

            vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            vl_gpt = vl_gpt.to(dtype).to(device).eval()

            deepseek_vl_model = {
                "chat_processor": vl_chat_processor,
                "model": vl_gpt,
                "tokenizer": tokenizer
            }
   
        return (deepseek_vl_model,)
            
class deepseek_vl_inference:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            "deepseek_vl_model": ("DEEPSEEKVLMODEL",),
            #"parameters": ("LLAMACPPARAMS", ),
            "prompt": ("STRING", {"multiline": True, "default": "Describe the image in detail.",}),

            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "Llama-cpp"

    def process(self, images, deepseek_vl_model, prompt):
        
        mm.soft_empty_cache()
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        images = images.permute(0, 3, 1, 2)
        to_pil = ToPILImage()
       

        vl_chat_processor = deepseek_vl_model["chat_processor"]
        model = deepseek_vl_model["model"]
        tokenizer = deepseek_vl_model["tokenizer"]
        ## single image conversation example

        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{prompt}",
            },
            {"role": "Assistant", "content": ""},
        ]

        ## multiple images (or in-context learning) conversation example
        # conversation = [
        #     {
        #         "role": "User",
        #         "content": "<image_placeholder>A dog wearing nothing in the foreground, "
        #                    "<image_placeholder>a dog wearing a santa hat, "
        #                    "<image_placeholder>a dog wearing a wizard outfit, and "
        #                    "<image_placeholder>what"s the dog wearing?",
        #         "images": [
        #             "images/dog_a.png",
        #             "images/dog_b.png",
        #             "images/dog_c.png",
        #             "images/dog_d.png",
        #         ],
        #     },
        #     {"role": "Assistant", "content": ""}
        # ]
        pbar = ProgressBar(len(images))
        answer_list = []
        model.to(device)
        for img in images:
            pil_image = to_pil(img)
            prepare_inputs = vl_chat_processor(
                conversations=conversation,
                images=[pil_image],
                force_batchify=True
            ).to(device)

            # run image encoder to get the image embeddings
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

            # run the model to get the response
            outputs = model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )
            answer = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
            answer = answer.lstrip(" [User]\n\n")
            answer_list.append(answer)
            pbar.update(1)

        model.to(offload_device)
        #print(f"{prepare_inputs['sft_format'][0]}", answer)
        if (len(images)) > 1:
            return (answer_list,)
        else:
            return (answer_list[0],)

class parameters:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "max_tokens": ("INT", {"default": 32, "min": 0, "max": 4096, "step": 1}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 1000, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tfs_z": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mirostat_mode": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                }        
        }

    RETURN_TYPES = ("LLAMACPPARAMS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "process"
    CATEGORY = "Llama-cpp"

    def process(self, max_tokens, top_k, top_p, min_p, typical_p, temperature, repeat_penalty, 
                frequency_penalty, presence_penalty, tfs_z, mirostat_mode, mirostat_eta, mirostat_tau, 
                ):
        
        parameters_dict = {
        "max_tokens": max_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "typical_p": typical_p,
        "temperature": temperature,
        "repeat_penalty": repeat_penalty,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "tfs_z": tfs_z,
        "mirostat_mode": mirostat_mode,
        "mirostat_eta": mirostat_eta,
        "mirostat_tau": mirostat_tau,
        } 
        return (parameters_dict,)
    
NODE_CLASS_MAPPINGS = {
    "deepseek_vl_model_loader": deepseek_vl_model_loader,
    "deepseek_vl_inference": deepseek_vl_inference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "deepseek_vl_model_loader": "DeepSeek-VL Model Loader",
    "deepseek_vl_inference": "DeepSeek-VL Inference",
}

