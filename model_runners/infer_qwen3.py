from dataclasses import dataclass
import json
from pathlib import Path
import sys

import draccus
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from exp.few_shot.scripts.make_shot import flatten_content


@dataclass
class infer_args:
    model_path: str = ""
    data_path: str = ""
    file_name: str = ""
    run_id: str = ""
    device: str = "auto"
    max_new_tokens: int = 256


def load_chat_processor(model_path: str):
    try:
        return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


@draccus.wrap()
def start_inference(inf_args: infer_args):
    with open(inf_args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = AutoModelForCausalLM.from_pretrained(
        inf_args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=inf_args.device,
        trust_remote_code=True,
    )
    processor = load_chat_processor(inf_args.model_path)

    pad_token_id = getattr(processor, "pad_token_id", None)
    eos_token_id = getattr(processor, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    results = []
    for item in tqdm(data):
        user_text = flatten_content(item["content"])
        key = item["key"]
        message = [{"role": "user", "content": user_text, "key": key}]

        chat_template_kwargs = dict(
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        try:
            inputs = processor.apply_chat_template(
                message,
                enable_thinking=False,
                **chat_template_kwargs,
            )
        except TypeError:
            inputs = processor.apply_chat_template(message, **chat_template_kwargs)

        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=inf_args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        results.append(
            {
                "input": message,
                "response": output_text,
            }
        )

    with open(inf_args.file_name, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    start_inference()
