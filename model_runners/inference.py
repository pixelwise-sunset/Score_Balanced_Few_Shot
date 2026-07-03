from transformers import AutoModelForImageTextToText, AutoProcessor
import draccus
from dataclasses import dataclass
import json
import torch
from tqdm import tqdm

@dataclass
class infer_args:
    model_path: str = ""
    data_path: str = ""
    file_name: str = ""
    run_id: str = ""
    device: str = "auto"
@draccus.wrap()
def start_inference(inf_args:infer_args):
    #load the data
    DATA_PATH = inf_args.data_path
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    if inf_args.device != "auto":
        device = {"": inf_args.device}
    else:
        device = "auto"
        
    model = AutoModelForImageTextToText.from_pretrained(
        inf_args.model_path,
        dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map=device
    )

    processor = AutoProcessor.from_pretrained(inf_args.model_path)

    results = []
    for m in tqdm(range(len(data))):
        message = [data[m]]

        inputs = processor.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        #append the result
        result = {
            "input": message,          
            "response": output_text        
        }
        results.append(result)
    file_path = f"{inf_args.file_name}"
    with open(file_path, 'x') as f:
        json.dump(results, f, indent = 2)

if __name__ == "__main__":
    start_inference()

