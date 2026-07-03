import json

def prompt_disFlagEXP():
    EXP = """
    **1. disagree_flag (1 if disagree, 0 otherwise)**

*   **Purpose:** This flag indicates whether the LLM's response fundamentally contradicts established medical knowledge or the specific details provided in the query, to the point where it's considered incorrect or potentially harmful.
*   **How to Rate:**
    *   **1 (Disagree):** The LLM provides a diagnosis or information that is clearly wrong, contradicts the image findings, ignores crucial details in the patient history, or suggests inappropriate/harmful actions. Examples:
        *   Suggesting a treatment known to be contraindicated for the suspected condition.
        *   Misidentifying a clear visual sign (e.g., calling a clearly purulent wound "clean").
        *   Ignoring a key symptom mentioned (e.g., the patient states it's intensely itchy, and the LLM suggests a non-itchy condition without acknowledging the itch).
        *   Giving a diagnosis that is highly improbable given the patient's age, history, and presentation.
        *   In Image 1, the LLM suggests Granuloma Annulare, which is a possible differential but the description (red rashes turning leukoplakia, reddish on rubbing) doesn't strongly fit, and the history of eczema/vitiligo treatment makes it less likely without further investigation. The response is very non-committal ("Unclear diagnosis...") which might be acceptable, but the suggestion of GA specifically could be debated depending on the expert's view of the image and description fit. However, the *overall* rating is low because it's not helpful.
        *   In Image 6, the LLM suggests melasma/PIH for a dark spot on the lip, which is unlikely. This is a clear disagreement with likely possibilities (e.g., melanoma, pigmented lesion).
        *   In Image 7, the LLM suggests monitoring for parapsoriasis and Behcet's. While monitoring is part of management, the response doesn't acknowledge the specific features described (dry scales, unclear borders, symmetry) which are more suggestive of mycosis fungoides (a type of cutaneous T-cell lymphoma) than typical
"""

    return EXP

def prompt_detailedEXP14():
    EXP = """
    Here's a breakdown of how to rate each criterion, incorporating the nuances observed in the examples:

**1. disagree_flag (1 if disagree, 0 otherwise)**

*   **Definition:** This flag indicates whether the expert fundamentally disagrees with the VLM's assessment or conclusion. It's a strong indicator of a problematic response.
*   **How to Rate:**
    *   **1:** Assign this if the VLM's diagnosis, advice, or interpretation of the image is clearly incorrect, misleading, or potentially harmful based on established medical knowledge. Examples:
        *   Image 1: The VLM suggests granuloma annulare, but the description (red rashes turning leukoplakic, worsening, new macules) and the image (erythematous, slightly raised lesion) don't strongly fit, and the patient history is complex. A simple "unclear diagnosis without biopsy" might be considered a weak response, but the *disagreement* flag is set because the VLM offers a specific, potentially incorrect differential diagnosis without sufficient justification or caution.
        *   Image 6: The VLM suggests melasma or post-inflammatory hyperpigmentation for a dark spot on the lip. This is questionable, especially given the patient's age and lack of history. The expert disagrees with this specific diagnosis.
        *   Image 7: The VLM suggests monitoring for persistent rashes and ulcers in a patient with a history suggestive of Behcet's disease. The expert disagrees with this minimal response, likely because it doesn't address the underlying condition or provide adequate guidance.
        *   Image 9: The VLM states the wound requires professional assessment but then gives potentially incorrect advice about gauze and medication. The expert disagrees with the overall handling of the situation.
        *   Image 10: The VLM states the wound requires reassessment but gives an inaccurate assessment of the wound's condition (factual accuracy is 0.0). The expert disagrees with the assessment.
        *   Image 12: The VLM states "It is fine for you" for a potentially infected wound, which is clearly incorrect and potentially harmful.
        *   Image 13: The VLM states "Yes, this is a large bruise" for what appears to be a more complex skin lesion, potentially missing a more serious diagnosis.
    *   **0:** Assign this if the VLM's response, while perhaps incomplete or not perfectly worded, doesn't contain a fundamentally incorrect statement or dangerous advice that the expert strongly objects to. Examples:
        *   Image 2: The VLM suggests chronic eczema or psoriasis and recommends seeing a dermatologist. This is a reasonable, albeit general, response to the image and description.
        *   Image 3: Similar to Image 2, the VLM suggests chronic eczema and recommends a dermatologist.
        *   Image 4: The VLM simply says "Eczema." While brief, it's not necessarily *wrong*, just incomplete and potentially lacking nuance.
        *   Image 5: The VLM suggests psoriasis and provides basic treatment information. This is a plausible diagnosis and reasonable advice.
        *   Image 8: The VLM correctly identifies the scar and gives appropriate advice.
        *   Image 11: The VLM correctly advises on the need for stitch removal.
        *   Image 14: The VLM correctly advises on the rabies protection and wound care, even if the writing style is basic.

**2. Completeness (1=Perfect, 0.5=Partial, 0=Not Complete)**

*   **Definition:** Does the response address all aspects of the user's query? Does it provide sufficient information given the context?
*   **How to Rate:**
    *   **1:** The response fully addresses the question, providing all necessary information or guidance requested. Example: Image 5 provides a likely diagnosis (psoriasis) and basic treatment/management information. Image 8 provides a diagnosis (hypertrophic scar) and management advice. Image 14 addresses the rabies concern and wound care.
    *   **0.5:** The response addresses some parts of the question but misses key elements or provides only a partial answer. Examples:
        *   Image 2: Suggests possible diagnoses and recommends seeing a dermatologist, but doesn't elaborate on specific features or differentials.
        *   Image 3: Similar to Image 2.
        *   Image 4: Simply states "Eczema" without further explanation or advice.
        *   Image 6: Suggests possible diagnoses but doesn't fully explore the implications or next steps.
        *   Image 7: Offers very minimal advice ("Monitor closely...consult dermatologist") without addressing the specific concerns raised (rashes, ulcers, Behcet's history).
        *   Image 9: Advises on gauze and medication but contradicts itself by saying the wound needs professional assessment.
        *   Image 11: Advises on stitch removal but the writing style is poor.
    *   **0:** The response fails to address significant parts of the question or provides information that is irrelevant or nonsensical in the context. Examples:
        *   Image 1: The VLM offers a vague response ("Unclear diagnosis...") and a single differential ("granuloma annulare") without explaining why or addressing the patient's complex history and worsening symptoms. It fails to acknowledge the uncertainty or the need for further investigation.
        *   Image 12: The VLM gives a simplistic and potentially incorrect reassurance ("It is fine for you") without assessing the wound properly.
        *   Image 13: The VLM provides a basic description ("large bruise") that may not be accurate and doesn't address the user's concern about what's happening.

**3. Factual Accuracy (1=Perfect, 0.5=Partial, 0=Not Accurate)**

*   **Definition:** Is the information provided by the VLM medically correct and consistent with current dermatological/wound care knowledge?
*   **How to Rate:**
    *   **1:** All statements made are factually correct. Examples:
        *   Image 2: Suggesting eczema or psoriasis as possibilities for itchy, recurrent, excoriated lesions is factually accurate.
        *   Image 5: Suggesting psoriasis and mentioning topical steroids, phototherapy, and that it's controllable but not curable is factually accurate.
        *   Image 8: Identifying it as a hypertrophic scar and suggesting treatments like silicone gel, steroid injections, or laser therapy is factually accurate.
        *   Image 11: Stating that stitches need removal by a professional is factually accurate.
        *   Image 14: Stating that the rabies vaccine protects and advising wound cleaning is factually accurate.
    *   **0.5:** Some information is accurate, but there might be minor inaccuracies, oversimplifications, or missing crucial caveats. This rating is less common in the provided examples.
    *   **0:** The response contains clear factual errors or misinformation. Examples:
        *   Image 1: Suggesting granuloma annulare without strong justification might be considered inaccurate in this context, especially given the patient's history.
        *   Image 6: Suggesting melasma or post-inflammatory hyperpigmentation for a lip lesion without further investigation might be inaccurate.
        *   Image 7: The response is so minimal it's hard to assess factual accuracy, but it doesn't address the likely underlying condition (Behcet's) or provide useful factual information.
        *   Image 9: Stating that professional assessment is needed but then giving potentially incorrect advice about gauze and medication introduces factual inaccuracy.
        *   Image 10: Stating the wound requires reassessment but implying it's *not* infected (based on the 0.0 factual accuracy rating) might be inaccurate given the delayed healing.
        *   Image 12: Stating "It is fine for you" for a potentially infected wound is factually inaccurate and dangerous.
        *   Image 13: Stating "Yes, this is a large bruise" when the image shows something more complex is factually inaccurate.

**4. Overall (1=Perfect Medical Accuracy, 0.5=Partial, 0=Not Accurate)**

*   **Definition:** This is a holistic assessment of the medical accuracy of the *entire* response, regardless of completeness or relevance. It focuses purely on whether the medical information presented is correct.
*   **How to Rate:** This rating often aligns closely with "Factual Accuracy," but it considers the overall medical soundness of the response.
    *   **1:** The response is entirely medically sound and accurate. Examples: Image 2, Image 5, Image 8, Image 11, Image 14.
    *   **0.5:** There are some minor inaccuracies or oversimplifications, but the overall medical message isn't fundamentally wrong. (Less common in these examples).
    *   **0:** The response contains significant medical inaccuracies or potentially harmful advice. Examples: Image 1, Image 6, Image 7, Image 9, Image 10, Image 12, Image 13.

**5. Relevance (1=Relevant, 0.5=Partially Relevant, 0=Irrelevant)**

*   **Definition:** Does the response directly address the user's question and the information provided in the image and text?
*   **How to Rate:**
    *   **1:** The response is directly relevant to the query and the image. Examples: Image 2, Image 3, Image 4, Image 5, Image 8, Image 9, Image 10, Image 11, Image 14.
    *   **0.5:** The response touches upon relevant aspects but might include tangential information or miss the main point slightly. (Less common in these examples).
    *   **0:** The response is unrelated to the query or image. (Not present in these examples).

**6. Writing Style (1=Appropriate, 0.5=Partial, 0=Inappropriate)**

*   **Definition:** Is the language clear, concise, professional, and appropriate for communicating medical information to a patient? Is it easy to understand?
*   **How to Rate:**
    *   **1:** The writing is clear, professional, easy to understand, and uses appropriate medical terminology without being overly technical or condescending. Examples: Image 2, Image 5, Image 8, Image 9, Image 10, Image 11, Image 14.
    *   **0.5:** The writing is understandable but might be slightly awkward, repetitive, lack clarity, or use less-than-ideal phrasing. Examples:
        *   Image 3: Similar to Image 2.
        *   Image 4: Very brief ("Eczema."), lacks explanation.
        *   Image 7: Extremely brief and unhelpful.
        *   Image 12: Reassuring but potentially dangerously simplistic ("It is fine for you").
        *   Image 13: Basic description, lacks depth.
    *   **0:** The writing is confusing, uses inappropriate language, is overly technical, or is otherwise unsuitable for patient communication. (Not strongly present in these examples, but Image 1's vague response could arguably fall here due to lack of clarity).
    """

    return EXP
    

def prompt_guidance(metrics:list[str]):
    disagree_flag = "disagree_flag:put 1 if disagree with an answer, 0 otherwise"

    completeness = "completeness:put 1 for perfect completeness, 0.5 partial, 0 not complete"

    factual_accuracy = "factual_accuracy:Has all necessary information required to give been given to the patient based on the question present? \n" \
                        "1 is for perfect factual accuracy, 0.5 partial, 0 for not accurate."

    factual_consistency_wgold = "factual_consistency_wgold:compare the candidate response against the provided gold doctor responses. " \
                                "Put 1 if it is fully consistent, 0.5 if partially consistent or missing key details, 0 if inconsistent or medically wrong."
    
    overall =   "overall: Is everything in the response medically accurate (regardless on if it is complete or relevant)\n" \
                "1 is for perfect medical accuracy, 0.5 partial, 0 for not accurate."
    
    relevance = "relevance: 1 for relevant answer to the question, 0.5 partially relevant, 0.0 irrelevant information"

    writing_style = "writing style: 1 for appropriate writing style, 0.5 partial, 0 otherwise"

    metric_dict = {"disagree_flag":disagree_flag,
                   "completeness":completeness,
                   "factual-accuracy":factual_accuracy,
                   "factual-consistency-wgold": factual_consistency_wgold,
                   "overall": overall,
                   "relevance": relevance,
                   "writing-style": writing_style
                   }
    
    guidance = "Here are the rating rules:\n\n"
    for m in metrics:
        guidance = guidance + metric_dict[m] + "\n\n"

    general = """
In general for a guide on number ratings:

0: Terrible response (incorrect or/and harmful). May lead to grievous misinformation or dangerous outcomes. Should never be shown to the patient.

0.5: Response relevant and will not necessary cause harm but missing critical items or containing incorrect information. Should not be shown to the patient before correction.

1: Response gives complete and accurate answers. Can be safely presented to a patient
"""
    return guidance + general

def prompt_outputTemp(metrics:list[str], continuous_output:bool = False):
    template = [{}]

    if not continuous_output:
        for m in metrics:
            if m == "disagree_flag":
                template[0][m] = "{0.0,1.0}"
            else:
                template[0][m] = "{0.0,0.5,1.0}"

    else:
        for m in metrics:
            template[0][m] = "any number between 0 and 1, which is your confidence level of whether it should receive a perfect score 1"


    template = json.dumps(template)
    
    return str(template)

def get_prompt_template(shots:int, zero_shot:bool = False, metrics = list[str], continuous_output:bool = False):

    PROMPT_TEMPLATE = f"""You are an expert in dermatology and woundcare. Think silently if needed.
Your job is to give accurate ratings of the response generated by an LLM. The response is an answer to the question about an image which contains either a skin disease or a wound.
You should rate the response according to the following rules:

{prompt_guidance(metrics=metrics)}

I will give you a few examples. After that, you should provide your answers to a sample containing an image, a question and a response from an LLM.


Your output format must be strict JSON, with the following structure:

{prompt_outputTemp(metrics=metrics, continuous_output=continuous_output)}

**Important:**
- Output only JSON. Do NOT include explanations or extra text.
- Each object corresponds to one image to rate.
- Do not add extra commas or trailing text.
"""
    ZERO_SHOT_TEMPLATE =  f"""You are an expert in dermatology and wound care. Think silently if needed
Your job is to give accurate ratings of the response generated by an LLM. The response is an answer to the query about an image which contains either a skin disease or a wound.
The output format must be strict JSON, with the following structure:

{prompt_guidance(metrics=metrics)}

Your output format must be strict JSON, with the following structure:

{prompt_outputTemp(metrics=metrics)}

**Important:**
- Output only JSON. Do NOT include explanations or extra text.
- Each object corresponds to one image to rate.
- Do not add extra commas or trailing text.
"""
    if zero_shot:
        return ZERO_SHOT_TEMPLATE
    else:
        return PROMPT_TEMPLATE
    
def noimg_template(metrics):
    NOIMG_TEMPLATE = f"""You are an expert in dermatology and wound care. Think silently if needed
    Your job is to give accurate ratings of the response generated by an LLM. The response is an answer to the question about an image which contains either a skin disease or a wound.
    You should rate the response according to the following rules:

    I will give you a few examples. After that, you should provide your answers to a sample containing an image, a question and a response from an LLM.. However, no image will be provided.
    Your output format must be strict JSON, with the following structure:

    {prompt_guidance(metrics=metrics)}

    Your output format must be strict JSON, with the following structure:

    {prompt_outputTemp(metrics=metrics)}

    **Important:**
    - Output only JSON. Do NOT include explanations or extra text.
    - Each object corresponds to one image to rate.
    - Do not add extra commas or trailing text.

    """
    
    return NOIMG_TEMPLATE

def gold_text_template(metrics):
    TEMPLATE = f"""
    You are an expert in dermatology and wound care. Think silently if needed
    Your job is to give accurate ratings of the response generated by an LLM, which includes a diagnosis and a treatment.
    The response is an answer to a question from a patient asking about a disease that he has. 
    You will be supplied with gold responses given by professional doctors. 
    You should rate the LLM's response by comparing it very carefully with the gold responses and treat them as ground truth.

    
    You should rate the response according to the following rules:

    {prompt_guidance(metrics=metrics)}

    Your output format must be strict JSON, with the following structure:

    {prompt_outputTemp(metrics=metrics)}

    Your output format must be strict JSON, with the following structure:

    **Important:**
    - Output only JSON. Do NOT include explanations or extra text.
    - Each object corresponds to one image to rate.
    - Do not add extra commas or trailing text.

    I will give you a few examples with ratings. After that, you should provide your ratings to a sample, which contains only a query, a response and gold responses.
""" 
    
    return TEMPLATE
    
