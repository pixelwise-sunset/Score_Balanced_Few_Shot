import pandas as pd
import json
from utils.dataset_helper.get_data import *
import itertools
from exp.few_shot.scripts.split_shot_df import *
from exp.few_shot.scripts.prompts import *

def prepare_shots(infer_df:pd.DataFrame,shot_df: pd.DataFrame, metrics, exclude_image:bool = False, include_explain:bool = False, continuous_output:bool=False):
    if shot_df is None:
        shot_df = pd.DataFrame(columns=infer_df.columns)

    num_shot_img = int(len(shot_df) / len(metrics))

    if len(shot_df) == 0:
        base_prompt = get_prompt_template(shots = 0, zero_shot=True, continuous_output=continuous_output, metrics=metrics)
    else:
        base_prompt = get_prompt_template(shots=num_shot_img, metrics=metrics, continuous_output=continuous_output)

    if exclude_image:
        base_prompt = noimg_template(metrics=metrics)
    # df = pd.concat([shot_df, infer_df]).reset_index(drop=True)
    # print(infer_df)
    grouped_shot_df = grouping(shot_df, group_type=['metric'])
    grouped_infer_df = grouping(infer_df, group_type = ['metric'])

    shot_key_cols = list(grouped_shot_df.keys)
    infer_key_cols = list(grouped_infer_df.keys)

    # 2. Find the exact index of the path column
    path_col_name = 'image_path' 
    shot_path_idx = shot_key_cols.index(path_col_name)
    infer_path_idx = infer_key_cols.index(path_col_name)

    #for storing
    group_shot_list = []
    keys_shot_img_path = []

    group_infer_list = []
    keys_infer_img_path = []

    for key_vals, group in grouped_shot_df:
        group_shot_list.append(group)
        keys_shot_img_path.append(key_vals[shot_path_idx])

    for key_vals, group in grouped_infer_df:
        group_infer_list.append(group)
        keys_infer_img_path.append(key_vals[infer_path_idx])

    shot_groups = group_shot_list
    infer_groups = group_infer_list

    if infer_groups is None:
        print("found")

    content = [{"type": "text", "text": base_prompt}]

    img_index = 0
    for s in shot_groups:
        shot_query = s['query_text'].iloc[0]
        shot_response = s['candidate'].iloc[0]
        
        # build few-shot example
        rating_string = ""
        for m in metrics:
            metric_value = s['label'][s['metric'] == m].iloc[0]
            rating_string += f"{m}: {metric_value}\n"

        example_string = f"Image {img_index + 1}:\nQuery:\n\"{shot_query}\"\n\nLLM response:\n\"{shot_response}\"\nRatings:\n{rating_string}"
        # if not exclude_image:
        #     example_string = f"Image {img_index + 1}:\nQuery:\n\"{shot_query}\"\n\nLLM response:\n\"{shot_response}\"\nRatings:\n{rating_string}"
        # else:
        #     example_string=f"Query:\n\"{shot_query}\"\n\nLLM response:\n\"{shot_response}\"\nRatings:\n{rating_string}"

        content.append({"type": "text", "text": example_string})

        if not exclude_image:
            content.append({"type": "image", "image": keys_shot_img_path[img_index]})
        img_index += 1
    #add explanation

    infer_index = 0

    if include_explain:
            content.append({"type": "text", "text": prompt_detailedEXP14()})

    if infer_groups:
        content.append({"type": "text", "text": 
                        f"""
                            Now rate the response below. Remember to output only JSON. 
                            Do **not** add any Markdown formatting, backticks, or explanations. 
                            The output must be a valid JSON object or array only.
                            Your response shoud strictly follow: \n{prompt_outputTemp(metrics=metrics)}
                        """})
    # else:
    #     raise ValueError('empty infer group')

    for i in infer_groups:
        infer_query = i['query_text'].iloc[0] if isinstance(i, pd.DataFrame) else i['query_text']
        infer_response = i['candidate'].iloc[0] if isinstance(i, pd.DataFrame) else i['candidate']

        infer_index += 1
        infer_section = f"Image {num_shot_img + infer_index}:\nQuery:\n\"{infer_query}\"\n\nLLM response:\n\"{infer_response}\""
        content.append({"type": "text", "text": infer_section})
        if not exclude_image:
            content.append({"type": "image", "image": keys_infer_img_path[infer_index - 1]})
        # print(infer_query)


    return {
            "role": "user",
            "content": content,
            "key": None
        }

def prepare_gold_texts_shot(infer_df:pd.DataFrame,shot_df: pd.DataFrame, metrics):
    base_prompt = gold_text_template(metrics=metrics)

    grouped_shot_df = grouping_with_gold(shot_df, group_type=['metric'])
    grouped_infer_df = grouping_with_gold(infer_df, group_type = ['metric'])

    shot_key_cols = list(grouped_shot_df.keys)
    infer_key_cols = list(grouped_infer_df.keys)

    #for storing
    group_shot_list = []
    keys_shot_img_path = []

    group_infer_list = []
    keys_infer_img_path = []

    path_col_name = 'image_path' 
    shot_path_idx = shot_key_cols.index(path_col_name)
    infer_path_idx = infer_key_cols.index(path_col_name)
    
    for key_vals, group in grouped_shot_df:
        group_shot_list.append(group)
        keys_shot_img_path.append(key_vals[shot_path_idx])

    for key_vals, group in grouped_infer_df:
        group_infer_list.append(group)
        keys_infer_img_path.append(key_vals[infer_path_idx])

    shot_groups = group_shot_list
    infer_groups = group_infer_list

    sample_idx = 0
    for s in shot_groups:
        shot_query = s['query_text'].iloc[0]
        shot_response = s['candidate'].iloc[0]
        shot_gold_response = str(s['gold_texts'].iloc[0])

        # build few-shot example
        rating_string = ""
        for m in metrics:
            metric_value = s['label'][s['metric'] == m].iloc[0]
            rating_string += f"{m}: {metric_value}\n"

        example_string = f"Sample {sample_idx + 1}:\nQuery:\n\"{shot_query}\"\n\nLLM response:\n\"{shot_response}\"\ngold responses:{shot_gold_response}\nRatings:\n{rating_string}"
        sample_idx += 1

        base_prompt = base_prompt + "\n\n" + example_string

    base_prompt = base_prompt + f"""
        Now rate the response below. Remember to output only JSON. 
        Do **not** add any Markdown formatting, backticks, or explanations. 
        The output must be a valid JSON object or array only.
        Your response shoud strictly follow: \n{prompt_outputTemp(metrics=metrics)}
                        """ + "\n\n"    
    for i in infer_groups:
        infer_query = i['query_text'].iloc[0]
        infer_response = i['candidate'].iloc[0]
        infer_gold_response = str(i['gold_texts'].iloc[0])

        infer_string = f"The sample you need to rate:\nQuery:\n\"{infer_query}\"\n\nLLM response:\n\"{infer_response}\"\ngold responses:{infer_gold_response}\n"
        base_prompt = base_prompt + infer_string

    
    return {
            "role": "user",
            "content": base_prompt,
            "key": None
        }  

def make_identity(df:pd.DataFrame):
    template = [{
        'dataset':df['dataset'].iloc[0],
        'encounter_id':df['encounter_id'].iloc[0],
        'lang':df['lang'].iloc[0],
        'candidate_author_id':df['candidate_author_id'].iloc[0]
    }]

    return template

def flatten_content(content):
    if isinstance(content, list):
        return "\n".join(
            c["text"] for c in content if c.get("type") == "text"
        )
    return content
        


if  __name__ == "__main__":
    train_path = "exp/few_shot/datasets/train.csv"


    metrics = METRICS
    # shot_df = pd.read_csv(f"exp/few_shot/datasets/shots/shot{i}.csv")
    shot_df = split_shot(split_from=train_path, shot_num = 1, metrics = metrics, save_path="exp/few_shot/runs/exp-all-metrics")
    infer_df = pd.read_csv("exp/few_shot/datasets/val.csv")
    # _, infer_df = retrieve_data(df = aligned_df,lang = ['en'], sample_n=0, system = ['1','2','3']) 
    llm_input = []
    #group the inference dataframe so that every iteration is a sample 
    infer_group = grouping(df = infer_df, group_type=['metric'])
    
    
    group_keys = []
    for key_val,infer in infer_group:
        shot = prepare_gold_texts_shot(infer_df = infer, shot_df=shot_df,metrics=metrics)
        key = make_identity(infer)

        shot["key"] = key
        llm_input.append(shot)
        group_keys.append(key_val)
        

    # print(len(infer_df))

    with open(f'my_test/shot_gold.json', "w", encoding="utf-8") as f:
        json.dump(llm_input, f, indent=2, ensure_ascii=False)

    print(len(llm_input))
    print(llm_input[0]["content"])

    # print(f"length of the json: {len(llm_input)}")

