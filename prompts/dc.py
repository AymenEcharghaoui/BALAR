from prompts.base import PromptSet

PROMPT_SET = PromptSet(

# -------------------------------------------------------------- #
# USER SIMULATOR PROMPTS
# -------------------------------------------------------------- #

USER_SIMULATOR_SYSTEM_PROMPT="""You are simulating a suspect being interrogated in a murder investigation. You must stay in character and answer based only on your character's knowledge and story. If you are the murderer, you will try to deflect suspicion while remaining consistent with your story. If you are innocent, answer honestly based on what your character knows.""",

USER_SIMULATOR_WITH_CHOICES_USER_PROMPT="""
<USER_CONTEXT>
{user_context}
</USER_CONTEXT>

<QUESTION>
{question}
</QUESTION>

<CHOICES_WITH_IDS>
{choices_with_ids}
</CHOICES_WITH_IDS>

Task:
You are a suspect being interrogated by a detective. Answer <QUESTION> in a way that is consistent with your character described in <USER_CONTEXT>. If you are the murderer, you may deflect or be evasive, but you must remain consistent with your story.

What to generate:
- reason : a short one-sentence explanation of why you chose the answer you did.
- answer_id : the id of the choice you are selecting as your answer to the question.

Constraints:
Use ONLY information that is supported by <USER_CONTEXT>.
answer_id must be one of the ids provided in <CHOICES_WITH_IDS>.

Output format:
Return STRICT JSON only with the following schema:
{{
    "reason": string,
    "answer_id": string
}}
""",

USER_SIMULATOR_WITHOUT_CHOICES_USER_PROMPT="""
<USER_CONTEXT>
{user_context}
</USER_CONTEXT>

<QUESTION>
{question}
</QUESTION>

Task:
You are a suspect being interrogated by a detective. Answer <QUESTION> in a way that is consistent with your character described in <USER_CONTEXT>. If you are the murderer, you may deflect or be evasive, but you must remain consistent with your story.

What to generate:
- reason : a short one-sentence explanation of why you chose the answer you did.
- answer : your answer to the question.

Constraints:
Use ONLY information that is supported by <USER_CONTEXT>.
answer must be a natural language answer to the question.

Output format:
Return STRICT JSON only with the following schema:
{{
    "reason": string,
    "answer": string
}}
""",

# -------------------------------------------------------------- #
# INITIAL STAGE PROMPTS (dimensions)
# -------------------------------------------------------------- #

INITIAL_DIMENSIONS_SYSTEM_PROMPT="""You are an experienced detective analyzing a murder case. Your goal is to identify the key investigative dimensions that must be resolved to determine who the real murderer is.""",

INITIAL_DIMENSIONS_USER_PROMPT="""
<CASE_QUESTION>
{ambiguous_prompt}
</CASE_QUESTION>

<CASE_BACKGROUND>
{meta_context}
</CASE_BACKGROUND>

Task:
Identify the key investigative dimensions in the <CASE_QUESTION> that must be resolved to determine who the real murderer is.

Definition:
An investigative dimension is a specific aspect of the murder case (e.g., motive, alibi, access to the murder weapon, relationship to the victim) where multiple suspects could plausibly be implicated, and resolving it would narrow down the true murderer. Once the dimension's value is known, the case moves toward identifying a single suspect as the murderer.

What to generate:
- Produce a minimal, non-overlapping set of investigative dimensions.
- Produce exactly {num_initial_dims} investigative dimensions.
- Each dimension must correspond to a distinct, investigative uncertainty.
- If <CASE_BACKGROUND> already resolves a dimension, do not include it.
- If <CASE_BACKGROUND> proposes some investigative dimensions, use them.

For each investigative dimension, provide :
- reason: a short one-sentence explanation of why this dimension is critical for identifying the murderer.
- name: a short, specific label (e.g., "Motive", "Alibi at time of death", "Access to murder weapon", etc.)
- values: a list of plausible values (e.g., one per suspect or per scenario), no larger than {max_num_values_per_dim}, that this dimension could take in the context of the case.

Constraints:
- Use ONLY the information provided in <CASE_QUESTION> and <CASE_BACKGROUND>.
- Do NOT answer the <CASE_QUESTION> itself. Focus ONLY on identifying investigative dimensions.
- Do NOT rewrite or restate the <CASE_QUESTION> or <CASE_BACKGROUND>.

Output format:
Return STRICT JSON only with the following schema:
{{
    "dimensions": [
        {{
            "reason": string,
            "name": string,
            "values": [string, ...]
        }},
        ...
    ]
}}
""",

# -------------------------------------------------------------- #
# INITIAL STAGE PROMPTS (priors)
# -------------------------------------------------------------- #

INITIAL_PRIORS_SYSTEM_PROMPT="""You are an experienced detective forming initial hypotheses about a murder case based on the available evidence and case background.""",

INITIAL_PRIORS_USER_PROMPT="""
<CASE_QUESTION>
{ambiguous_prompt}
</CASE_QUESTION>

<CASE_BACKGROUND>
{meta_context}
</CASE_BACKGROUND>

<DIMENSION_NAME>
{dimension_name}
</DIMENSION_NAME>

<DIMENSION_VALUE>
{dimension_value}
</DIMENSION_VALUE>

Task:
Given <CASE_QUESTION> and <CASE_BACKGROUND>, judge how likely the investigative dimension <DIMENSION_NAME> takes on the value <DIMENSION_VALUE>.

What to generate:
- reason: a short one-sentence explanation of why the <DIMENSION_NAME> is likely, unlikely, or neutral to take on the value <DIMENSION_VALUE>.
- label: one of "likely", "unlikely", or "neutral" according to the following definitions:
    - likely: <DIMENSION_VALUE> is explicitly stated, strongly implied, or is the most natural assumption given the evidence in the <CASE_QUESTION> and <CASE_BACKGROUND>.
    - neutral: <DIMENSION_VALUE> is plausible but not implied or supported by specific evidence in the <CASE_QUESTION> or <CASE_BACKGROUND>.
    - unlikely: <DIMENSION_VALUE> is contradicted by the <CASE_QUESTION> or <CASE_BACKGROUND>, or would require assumptions that are inconsistent with the available evidence.

Constraints:
- Use ONLY the information provided in <CASE_QUESTION> and <CASE_BACKGROUND>.
- Do NOT answer the <CASE_QUESTION> itself. Focus ONLY on judging the likelihood of the dimension value.
- Do NOT rewrite or restate the <CASE_QUESTION> or <CASE_BACKGROUND>.
- label must be one of "likely", "unlikely", or "neutral".

Output format:
Return STRICT JSON only with the following schema:
{{
    "reason": string,
    "label": string
}}
""",

# -------------------------------------------------------------- #
# INITIAL STAGE PROMPTS (questions)
# -------------------------------------------------------------- #

INITIAL_QUESTIONS_SYSTEM_PROMPT="""You are an experienced detective preparing interrogation questions for suspects in a murder investigation. Your questions should be designed to reveal inconsistencies, uncover motives, and verify alibis.""",

INITIAL_QUESTIONS_USER_PROMPT="""
<CASE_QUESTION>
{ambiguous_prompt}
</CASE_QUESTION>

<CASE_BACKGROUND>
{meta_context}
</CASE_BACKGROUND>

<INVESTIGATIVE_DIMENSIONS>
{dimensions_with_values}
</INVESTIGATIVE_DIMENSIONS>

Task:
Given <CASE_QUESTION>, <CASE_BACKGROUND>, and <INVESTIGATIVE_DIMENSIONS>, generate exactly {num_initial_questions} interrogation questions to ask the suspects that would help identify the real murderer. Each question should target one or more <INVESTIGATIVE_DIMENSIONS> and have multiple-choice answers.

Definition:
<INVESTIGATIVE_DIMENSIONS> is a list of investigative dimensions, where each dimension has a name and a list of possible values it could take. An investigative dimension is a specific aspect of the murder case where multiple suspects could plausibly be implicated, and resolving it would narrow down the true murderer.

What to generate:
For each of the {num_initial_questions} questions, provide:
- reason: a short one-sentence explanation of why this question would help identify the murderer.
- question: the text of the interrogation question.
- choices: a list of multiple-choice answer options for the question, no larger than {max_choices_per_question}.

Constraints:
- Use ONLY the information provided in <CASE_QUESTION>, <CASE_BACKGROUND>, and <INVESTIGATIVE_DIMENSIONS>.
- Do NOT answer the <CASE_QUESTION> itself. Focus ONLY on generating interrogation questions.
- Do NOT rewrite or restate the <CASE_QUESTION> or <CASE_BACKGROUND>.
- Each question must be designed to elicit information about one or more of the dimensions in <INVESTIGATIVE_DIMENSIONS>.
- Each question must have multiple-choice answers.

Output format:
Return STRICT JSON only with the following schema:
{{
    "questions": [
        {{
            "reason": string,
            "question": string,
            "choices": [string, ...]
        }},
        ...
    ]
}}
""",

# -------------------------------------------------------------- #
# LIKELIHOOD PROMPTS
# -------------------------------------------------------------- #

LIKELIHOOD_SYSTEM_PROMPT="""You are an experienced detective evaluating how a suspect would likely respond to an interrogation question under different assumptions about the case. Consider that guilty suspects may deflect, lie, or give evasive answers, while innocent suspects will answer based on their genuine knowledge.""",

LIKELIHOOD_WITHOUT_HISTORY_USER_PROMPT="""
<CASE_QUESTION>
{ambiguous_prompt}
</CASE_QUESTION>

<CASE_BACKGROUND>
{meta_context}
</CASE_BACKGROUND>

<SUSPECT_INFO>
{user_info}
</SUSPECT_INFO>

<DIMENSION_NAME>
{dimension_name}
</DIMENSION_NAME>

<DIMENSION_VALUES_WITH_IDS>
{dimension_values_with_ids}
</DIMENSION_VALUES_WITH_IDS>

<QUESTION>
{question_text}
</QUESTION>

<QUESTION_CHOICES_WITH_IDS>
{question_choices_with_ids}
</QUESTION_CHOICES_WITH_IDS>

Definition:
An investigative dimension is a specific aspect of the murder case where multiple suspects could plausibly be implicated, and resolving it would narrow down the true murderer.
<DIMENSION_VALUES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a possible value that the <DIMENSION_NAME> could take.
<QUESTION_CHOICES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a multiple-choice answer option for the question.
Let values[i] be the i-th element of <DIMENSION_VALUES_WITH_IDS>.
Let choices[j] be the j-th element of <QUESTION_CHOICES_WITH_IDS>.

Task (row-major order):
For i = 0..len(values)-1:
  For j = 0..len(choices)-1:
    - Assume the true state of the case is <DIMENSION_NAME> = values[i]["text"].
    - Impersonate the suspect described in <SUSPECT_INFO>.
    - Judge how likely it is that this suspect would answer the question <QUESTION> with choices[j]["text"] under that assumption. Consider that a guilty suspect may try to deflect or mislead.

What to generate:
For i = 0..len(values)-1:
  For j = 0..len(choices)-1:
    - question_choice_id: the id of the question choice being evaluated, i.e. choices[j]["id"]
    - dimension_value_id: the id of the dimension value being evaluated, i.e. values[i]["id"]
    - reason: a short one-sentence explanation of why choices[j] is labeled likely/neutral/unlikely and why the other two labels were not chosen.
    - label: one of "likely", "neutral", or "unlikely" according to the following definitions:
      - "likely": Given <DIMENSION_NAME> = values[i]["text"] and the suspect acts according to <SUSPECT_INFO>, the suspect is expected to give choices[j]["text"] for <QUESTION>.
      - "neutral": Given <DIMENSION_NAME> = values[i]["text"] and the suspect acts according to <SUSPECT_INFO>, choices[j]["text"] is plausible but not specifically supported; there is insufficient evidence to say that the suspect would or would not prefer it over other choices.
      - "unlikely": Given <DIMENSION_NAME> = values[i]["text"] and the suspect acts according to <SUSPECT_INFO>, the suspect is not expected to give choices[j]["text"] for <QUESTION>.

Constraints:
- Use ONLY the information provided in <CASE_QUESTION>, <CASE_BACKGROUND>, and <SUSPECT_INFO>.
- Do NOT answer the <CASE_QUESTION> itself. Focus ONLY on judging the likelihood of the question choices under different assumptions about the dimension value.
- Do NOT rewrite or restate the <CASE_QUESTION>, <CASE_BACKGROUND>, or <SUSPECT_INFO>.
- label must be one of "likely", "neutral", or "unlikely".
- The output must include an entry for every combination of dimension value and question choice.

Output format:
Return STRICT JSON only with the following schema:
{{
    "evaluations": [
        [
            {{
                "question_choice_id": string,
                "dimension_value_id": string,
                "reason": string,
                "label": string
            }},
            ... // one object for each question choice
        ],
        ... // one array for each dimension value
    ]
}}
The "evaluations" field must contain exactly {num_dimension_values} arrays (one per dimension value).
Each inner array must contain exactly {num_question_choices} objects (one per question choice).
""",

LIKELIHOOD_WITH_HISTORY_USER_PROMPT="""
<CASE_QUESTION>
{ambiguous_prompt}
</CASE_QUESTION>

<CASE_BACKGROUND>
{meta_context}
</CASE_BACKGROUND>

<SUSPECT_INFO>
{user_info}
</SUSPECT_INFO>

<INTERROGATION_LOG>
{conversation_log}
</INTERROGATION_LOG>

<DIMENSION_NAME>
{dimension_name}
</DIMENSION_NAME>

<DIMENSION_VALUES_WITH_IDS>
{dimension_values_with_ids}
</DIMENSION_VALUES_WITH_IDS>

<QUESTION>
{question_text}
</QUESTION>

<QUESTION_CHOICES_WITH_IDS>
{question_choices_with_ids}
</QUESTION_CHOICES_WITH_IDS>

Definition:
An investigative dimension is a specific aspect of the murder case where multiple suspects could plausibly be implicated, and resolving it would narrow down the true murderer.
<DIMENSION_VALUES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a possible value that the <DIMENSION_NAME> could take.
<QUESTION_CHOICES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a multiple-choice answer option for the question.
<INTERROGATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the interrogation between the detective and the suspects up to this point. This information may provide additional context and reveal inconsistencies.
Let values[i] be the i-th element of <DIMENSION_VALUES_WITH_IDS>.
Let choices[j] be the j-th element of <QUESTION_CHOICES_WITH_IDS>.

Task (row-major order):
For i = 0..len(values)-1:
  For j = 0..len(choices)-1:
    - Assume the true state of the case is <DIMENSION_NAME> = values[i]["text"].
    - Impersonate the suspect described in <SUSPECT_INFO>.
    - Judge how likely it is that this suspect would answer the question <QUESTION> with choices[j]["text"] under that assumption. Consider that a guilty suspect may try to deflect or mislead.

What to generate:
For i = 0..len(values)-1:
  For j = 0..len(choices)-1:
    - question_choice_id: the id of the question choice being evaluated, i.e. choices[j]["id"]
    - dimension_value_id: the id of the dimension value being evaluated, i.e. values[i]["id"]
    - reason: a short one-sentence explanation of why choices[j] is labeled likely/neutral/unlikely and why the other two labels were not chosen.
    - label: one of "likely", "neutral", or "unlikely" according to the following definitions:
      - "likely": Given <DIMENSION_NAME> = values[i]["text"] and the suspect acts according to <SUSPECT_INFO>, the suspect is expected to give choices[j]["text"] for <QUESTION>.
      - "neutral": Given <DIMENSION_NAME> = values[i]["text"] and the suspect acts according to <SUSPECT_INFO>, choices[j]["text"] is plausible but not specifically supported; there is insufficient evidence to say that the suspect would or would not prefer it over other choices.
      - "unlikely": Given <DIMENSION_NAME> = values[i]["text"] and the suspect acts according to <SUSPECT_INFO>, the suspect is not expected to give choices[j]["text"] for <QUESTION>.

Constraints:
- Use ONLY the information provided in <CASE_QUESTION>, <CASE_BACKGROUND>, <SUSPECT_INFO>, and <INTERROGATION_LOG>.
- Do NOT answer the <CASE_QUESTION> itself. Focus ONLY on judging the likelihood of the question choices under different assumptions about the dimension value.
- Do NOT rewrite or restate the <CASE_QUESTION>, <CASE_BACKGROUND>, <SUSPECT_INFO>, or <INTERROGATION_LOG>.
- label must be one of "likely", "neutral", or "unlikely".
- The output must include an entry for every combination of dimension value and question choice.

Output format:
Return STRICT JSON only with the following schema:
{{
    "evaluations": [
        [
            {{
                "question_choice_id": string,
                "dimension_value_id": string,
                "reason": string,
                "label": string
            }},
            ... // one object for each question choice
        ],
        ... // one array for each dimension value
    ]
}}
The "evaluations" field must contain exactly {num_dimension_values} arrays (one per dimension value).
Each inner array must contain exactly {num_question_choices} objects (one per question choice).
""",

# -------------------------------------------------------------- #
# NATURAL LANGUAGE to CHOICES PROMPTS
# -------------------------------------------------------------- #

SCORE_NATURAL_LANGUAGE_SYSTEM_PROMPT="""You are an experienced detective analyzing a suspect's response to an interrogation question. Your task is to judge how well the suspect's answer maps to each of the predefined answer choices.""",

SCORE_NATURAL_LANGUAGE_USER_PROMPT="""
<QUESTION>
{question}
</QUESTION>

<CHOICES_WITH_IDS>
{choices_with_ids}
</CHOICES_WITH_IDS>

<SUSPECT_ANSWER>
{user_answer}
</SUSPECT_ANSWER>

Definition:
<CHOICES_WITH_IDS> is a list of dicts with "id" and "value" fields. Each dict corresponds to a multiple-choice answer option for the question.
Let choices[i] be the i-th element of <CHOICES_WITH_IDS>.

Task:
Judge how well the <SUSPECT_ANSWER> maps to each of the choices in <CHOICES_WITH_IDS> for the question <QUESTION>.

What to generate:
For i = 0..len(choices)-1:
- choice_id: the id of the question choice being evaluated, i.e. choices[i]["id"]
- reason: a short one-sentence explanation of choices[i]["value"] is likely/neutral/unlikely given the <SUSPECT_ANSWER>.
- label: one of "likely", "neutral", or "unlikely" according to the following definitions:
    -"likely": choices[i]["value"] aligns well with the <SUSPECT_ANSWER> and fits it better than most other choices.
    -"neutral": choices[i]["value"] is neither clearly supported nor clearly contradicted by the <SUSPECT_ANSWER>.
    -"unlikely": choices[i]["value"] fits the <SUSPECT_ANSWER> worse than other choices, or conflicts with the meaning of the <SUSPECT_ANSWER>.

Constraints:
- Use ONLY the information provided in <QUESTION>, <CHOICES_WITH_IDS>, and <SUSPECT_ANSWER>.
- Do NOT answer the <QUESTION> itself. Focus ONLY on judging how well the <SUSPECT_ANSWER> maps to the provided choices.
- Do NOT rewrite or restate the <QUESTION> or <SUSPECT_ANSWER>.
- label must be one of "likely", "neutral", or "unlikely".
- The output must include an entry for every choice in <CHOICES_WITH_IDS>.

Output format:
Return STRICT JSON only with the following schema:
{{
    "scores": [
        {{
            "choice_id": string,
            "reason": string,
            "label": string
        }},
        ...
    ]
}}
""",

# -------------------------------------------------------------- #
# EXPAND DIMENSION PROMPTS (dimensions)
# -------------------------------------------------------------- #

EXPAND_DIMENSION_SYSTEM_PROMPT="""You are an experienced detective who needs to explore a new line of investigation in a murder case. The current investigative dimensions have not been sufficient to identify the murderer, so you must identify a new aspect of the case to investigate.""",

EXPAND_DIMENSION_USER_PROMPT="""
<CASE_QUESTION>
{ambiguous_prompt}
</CASE_QUESTION>

<CASE_BACKGROUND>
{meta_context}
</CASE_BACKGROUND>

<PAST_INVESTIGATIVE_DIMENSIONS>
{past_dimensions}
</PAST_INVESTIGATIVE_DIMENSIONS>

<INTERROGATION_LOG>
{conversation_log}
</INTERROGATION_LOG>

Definition:
An investigative dimension is a specific aspect of the murder case where multiple suspects could plausibly be implicated, and resolving it would narrow down the true murderer.
<PAST_INVESTIGATIVE_DIMENSIONS> is a list of dicts with "name" fields, representing the investigative dimensions that have already been explored in the investigation.
<INTERROGATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the interrogation between the detective and the suspects up to this point.

Task:
Identify a new investigative dimension in the murder case described by <CASE_QUESTION> that has not been previously identified in <PAST_INVESTIGATIVE_DIMENSIONS>. Use insights from the <INTERROGATION_LOG> to guide your choice.

What to generate:
- reason: a short one-sentence explanation of why this dimension is a critical new line of investigation.
- name: a short, specific label for this investigative dimension (e.g., "Forensic evidence", "Financial motive", "Witness credibility", etc.).
- values: a list of plausible values, no larger than {max_num_values_per_dim}, that this dimension could take in the context of the case.

Constraints:
- Use ONLY the information provided in <CASE_QUESTION>, <CASE_BACKGROUND>, <PAST_INVESTIGATIVE_DIMENSIONS>, and <INTERROGATION_LOG>.
- Do NOT answer the <CASE_QUESTION> itself. Focus ONLY on identifying a new investigative dimension.
- Do NOT rewrite or restate the <CASE_QUESTION>, <CASE_BACKGROUND>, <PAST_INVESTIGATIVE_DIMENSIONS>, or <INTERROGATION_LOG>.
- The generated dimension name must not be the same as any of the names in <PAST_INVESTIGATIVE_DIMENSIONS>.

Output format:
Return STRICT JSON only with the following schema:
{{
    "reason": string,
    "name": string,
    "values": [string, ...]
}}
""",

# -------------------------------------------------------------- #
# EXPAND DIMENSIONS PROMPTS (priors)
# -------------------------------------------------------------- #

EXPAND_DIMENSION_PRIORS_SYSTEM_PROMPT="""You are an experienced detective forming a hypothesis about a newly identified aspect of a murder case, taking into account both the case background and what has been revealed during the interrogation so far.""",

EXPAND_DIMENSION_PRIORS_USER_PROMPT="""
<CASE_QUESTION>
{ambiguous_prompt}
</CASE_QUESTION>

<CASE_BACKGROUND>
{meta_context}
</CASE_BACKGROUND>

<INTERROGATION_LOG>
{conversation_log}
</INTERROGATION_LOG>

<DIMENSION_NAME>
{dimension_name}
</DIMENSION_NAME>

<DIMENSION_VALUE>
{dimension_value}
</DIMENSION_VALUE>

Definition:
An investigative dimension is a specific aspect of the murder case where multiple suspects could plausibly be implicated, and resolving it would narrow down the true murderer.
<INTERROGATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the interrogation between the detective and the suspects up to this point.

Task:
Given <CASE_QUESTION>, <CASE_BACKGROUND>, <INTERROGATION_LOG>, and a specific investigative dimension defined by <DIMENSION_NAME> and <DIMENSION_VALUE>, judge how likely it is that the <DIMENSION_NAME> takes on the value <DIMENSION_VALUE>.

What to generate:
- reason: a short one-sentence explanation of why the <DIMENSION_NAME> is likely, unlikely, or neutral to take on the value <DIMENSION_VALUE>.
- label: one of "likely", "unlikely", or "neutral" according to the following definitions:
    - likely: <DIMENSION_VALUE> is explicitly stated, strongly implied, or is the most natural assumption given the evidence in the <CASE_QUESTION>, <CASE_BACKGROUND>, and <INTERROGATION_LOG>.
    - neutral: <DIMENSION_VALUE> is plausible but not implied or supported by specific evidence in the <CASE_QUESTION>, <CASE_BACKGROUND>, or <INTERROGATION_LOG>.
    - unlikely: <DIMENSION_VALUE> is contradicted by the <CASE_QUESTION>, <CASE_BACKGROUND> or <INTERROGATION_LOG>, or would require assumptions that are inconsistent with the available evidence.

Constraints:
- Use ONLY the information provided in <CASE_QUESTION>, <CASE_BACKGROUND>, and <INTERROGATION_LOG>.
- Do NOT answer the <CASE_QUESTION> itself. Focus ONLY on judging the likelihood of the dimension value.
- Do NOT rewrite or restate the <CASE_QUESTION>, <CASE_BACKGROUND>, or <INTERROGATION_LOG>.
- label must be one of "likely", "unlikely", or "neutral".

Output format:
Return STRICT JSON only with the following schema:
{{
    "reason": string,
    "label": string
}}
""",

# -------------------------------------------------------------- #
# EXPAND QUESTIONS PROMPTS
# -------------------------------------------------------------- #

EXPAND_QUESTIONS_SYSTEM_PROMPT="""You are an experienced detective preparing new interrogation questions based on a newly discovered line of investigation and unresolved aspects of the murder case.""",

EXPAND_QUESTIONS_USER_PROMPT="""
<CASE_QUESTION>
{ambiguous_prompt}
</CASE_QUESTION>

<CASE_BACKGROUND>
{meta_context}
</CASE_BACKGROUND>

<INTERROGATION_LOG>
{conversation_log}
</INTERROGATION_LOG>

<NEW_INVESTIGATIVE_DIMENSION>
{new_dimension_with_values}
</NEW_INVESTIGATIVE_DIMENSION>

<UNRESOLVED_INVESTIGATIVE_DIMENSIONS>
{high_uncertainty_dimensions_with_values}
</UNRESOLVED_INVESTIGATIVE_DIMENSIONS>

Definition:
An investigative dimension is a specific aspect of the murder case where multiple suspects could plausibly be implicated, and resolving it would narrow down the true murderer.
<INTERROGATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the interrogation between the detective and the suspects up to this point.
<NEW_INVESTIGATIVE_DIMENSION> is a dict with "name" and "values" fields, representing the newly identified line of investigation along with its possible values.
<UNRESOLVED_INVESTIGATIVE_DIMENSIONS> is a list of dicts with "name" and "values" fields, representing the investigative dimensions that currently have the highest uncertainty. They do not include the new dimension in <NEW_INVESTIGATIVE_DIMENSION>.

Task:
Given <CASE_QUESTION>, <CASE_BACKGROUND>, <INTERROGATION_LOG>, a newly identified investigative dimension in <NEW_INVESTIGATIVE_DIMENSION>, and the most uncertain dimensions in <UNRESOLVED_INVESTIGATIVE_DIMENSIONS>, generate interrogation questions that would help identify the murderer by targeting the new dimension and/or the unresolved dimensions.

What to generate:
Generate at most {max_new_questions_per_round} interrogation questions. For each question, provide:
- reason: a short one-sentence explanation of why this question would help identify the murderer.
- question: the text of the interrogation question.
- choices: a list of multiple-choice answer options for the question, no larger than {max_choices_per_question}.

Constraints:
- Use ONLY the information provided in <CASE_QUESTION>, <CASE_BACKGROUND>, <INTERROGATION_LOG>, <NEW_INVESTIGATIVE_DIMENSION>, and <UNRESOLVED_INVESTIGATIVE_DIMENSIONS>.
- Do NOT answer the <CASE_QUESTION> itself. Focus ONLY on generating interrogation questions.
- Do NOT rewrite or restate the <CASE_QUESTION>, <CASE_BACKGROUND>, <INTERROGATION_LOG>, <NEW_INVESTIGATIVE_DIMENSION>, or <UNRESOLVED_INVESTIGATIVE_DIMENSIONS>.
- Each question must be designed to elicit information about the new dimension in <NEW_INVESTIGATIVE_DIMENSION> and/or the unresolved dimensions in <UNRESOLVED_INVESTIGATIVE_DIMENSIONS>.
- Each question must have multiple-choice answers.
- Generate at most {max_new_questions_per_round} questions.

Output format:
Return STRICT JSON only with the following schema:
{{
    "questions": [
        {{
            "reason": string,
            "question": string,
            "choices": [string, ...]
        }},
        ...
    ]
}}
""",

# -------------------------------------------------------------- #
# FINAL ANSWER PROMPTS
# -------------------------------------------------------------- #

FINAL_ANSWER_SYSTEM_PROMPT="""You are an experienced detective concluding a murder investigation. Based on all the evidence gathered from interrogating the suspects and your analysis of the case, you must now identify the real murderer.""",

FINAL_ANSWER_WITHOUT_CHOICES_USER_PROMPT="""
<CASE_QUESTION>
{ambiguous_prompt}
</CASE_QUESTION>

<CASE_BACKGROUND>
{meta_context}
</CASE_BACKGROUND>

<INTERROGATION_LOG>
{conversation_log}
</INTERROGATION_LOG>

<INVESTIGATION_CONCLUSION>
{map_state}
</INVESTIGATION_CONCLUSION>

Definition:
<INTERROGATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the full history of the interrogation between the detective and the suspects.
An investigative dimension is a specific aspect of the murder case where multiple suspects could plausibly be implicated, and resolving it would narrow down the true murderer.
<INVESTIGATION_CONCLUSION> is a structured representation of the detective's current understanding of the case, where each investigative dimension is mapped to its most likely value. This represents the detective's best assessment of the true state of the case based on the investigation so far.

Task:
Given <CASE_QUESTION>, <CASE_BACKGROUND>, <INTERROGATION_LOG>, and <INVESTIGATION_CONCLUSION>, identify the real murderer.

What to generate:
- reason: a short one-sentence explanation of why the identified suspect is the real murderer given the evidence in <CASE_QUESTION>, <CASE_BACKGROUND>, <INTERROGATION_LOG>, and <INVESTIGATION_CONCLUSION>.
- final_answer: the name of the suspect you are identifying as the real murderer.

Constraints:
- The final answer must be consistent with the evidence in <CASE_QUESTION>, <CASE_BACKGROUND>, <INTERROGATION_LOG>, and <INVESTIGATION_CONCLUSION>.
- Do NOT rewrite or restate the <CASE_QUESTION>, <CASE_BACKGROUND>, <INTERROGATION_LOG>, or <INVESTIGATION_CONCLUSION>.
- answer must be a natural language answer to the <CASE_QUESTION>.

Output format:
Return STRICT JSON only with the following schema:
{{
    "reason": string,
    "final_answer": string
}}
""",

FINAL_ANSWER_WITH_CHOICES_USER_PROMPT="""
<CASE_QUESTION>
{ambiguous_prompt}
</CASE_QUESTION>

<CASE_BACKGROUND>
{meta_context}
</CASE_BACKGROUND>

<INTERROGATION_LOG>
{conversation_log}
</INTERROGATION_LOG>

<INVESTIGATION_CONCLUSION>
{map_state}
</INVESTIGATION_CONCLUSION>

<SUSPECTS_WITH_IDS>
{possible_answers_with_ids}
</SUSPECTS_WITH_IDS>

Definition:
<INTERROGATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the full history of the interrogation between the detective and the suspects.
An investigative dimension is a specific aspect of the murder case where multiple suspects could plausibly be implicated, and resolving it would narrow down the true murderer.
<INVESTIGATION_CONCLUSION> is a structured representation of the detective's current understanding of the case, where each investigative dimension is mapped to its most likely value. This represents the detective's best assessment of the true state of the case based on the investigation so far.
<SUSPECTS_WITH_IDS> is a list of dicts with "id" and "value" fields. Each dict corresponds to a suspect who could be the murderer.
Let possible_answers[i] be the i-th element of <SUSPECTS_WITH_IDS>.

Task:
Given <CASE_QUESTION>, <CASE_BACKGROUND>, <INTERROGATION_LOG>, and <INVESTIGATION_CONCLUSION>, identify which suspect from <SUSPECTS_WITH_IDS> is the real murderer.

What to generate:
- reason: a short one-sentence explanation of why the identified suspect is the real murderer given the evidence in <CASE_QUESTION>, <CASE_BACKGROUND>, <INTERROGATION_LOG>, and <INVESTIGATION_CONCLUSION>.
- final_answer_id: the id of the suspect in <SUSPECTS_WITH_IDS> that you are identifying as the real murderer.

Constraints:
- The final answer must be consistent with the evidence in <CASE_QUESTION>, <CASE_BACKGROUND>, <INTERROGATION_LOG>, and <INVESTIGATION_CONCLUSION>.
- Do NOT rewrite or restate the <CASE_QUESTION>, <CASE_BACKGROUND>, <INTERROGATION_LOG>, or <INVESTIGATION_CONCLUSION>.
- final_answer_id must be one of the ids (i.e possible_answers[i]["id"]) provided in <SUSPECTS_WITH_IDS>.

Output format:
Return STRICT JSON only with the following schema:
{{
    "reason": string,
    "final_answer_id": string
}}
""",
# -------------------------------------------------------------- #                                                                                                                                                                                
# ANSWER LIKELIHOOD PROMPTS
# -------------------------------------------------------------- #                                                                                                                                                                                
                                                                                                                                                                                                                                                
ANSWER_LIKELIHOOD_SYSTEM_PROMPT="""You are an experienced detective evaluating how likely each suspect is to be the real murderer under different assumptions about the state of the case.""",                                                    
                                                                                                                                                                                                                                                
ANSWER_LIKELIHOOD_USER_PROMPT="""
<CASE_QUESTION>
{ambiguous_prompt}
</CASE_QUESTION>

<CASE_BACKGROUND>
{meta_context}
</CASE_BACKGROUND>

<DIMENSION_NAME>
{dimension_name}
</DIMENSION_NAME>

<DIMENSION_VALUES_WITH_IDS>
{dimension_values_with_ids}
</DIMENSION_VALUES_WITH_IDS>

<SUSPECTS_WITH_IDS>
{possible_answers_with_ids}
</SUSPECTS_WITH_IDS>

Definition:
An investigative dimension is a specific aspect of the murder case where multiple suspects could plausibly be implicated, and resolving it would narrow down the true murderer.
<DIMENSION_VALUES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a possible value that the <DIMENSION_NAME> could take.
<SUSPECTS_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a suspect who could be the real murderer.
Let values[i] be the i-th element of <DIMENSION_VALUES_WITH_IDS>.
Let answers[j] be the j-th element of <SUSPECTS_WITH_IDS>.

Task (row-major order):
For i = 0..len(values)-1:
For j = 0..len(answers)-1:
    - Assume the true state of the case is <DIMENSION_NAME> = values[i]["text"].
    - Judge how likely it is that answers[j]["text"] is the real murderer given that assumption and the evidence in <CASE_BACKGROUND>.

What to generate:
For i = 0..len(values)-1:
For j = 0..len(answers)-1:
    - answer_id: the id of the suspect being evaluated, i.e. answers[j]["id"]
    - dimension_value_id: the id of the dimension value being evaluated, i.e. values[i]["id"]
    - reason: a short one-sentence explanation of why answers[j] is likely/neutral/unlikely to be the real murderer given <DIMENSION_NAME> = values[i]["text"].
    - label: one of "likely", "neutral", or "unlikely" according to the following definitions:
        - "likely": Given <DIMENSION_NAME> = values[i]["text"] and the evidence in <CASE_BACKGROUND>, answers[j]["text"] is the expected real murderer.
        - "neutral": Given <DIMENSION_NAME> = values[i]["text"] and the evidence in <CASE_BACKGROUND>, answers[j]["text"] is a plausible suspect but not specifically implicated; there is insufficient evidence to say this suspect is more or less guilty than others.
        - "unlikely": Given <DIMENSION_NAME> = values[i]["text"] and the evidence in <CASE_BACKGROUND>, answers[j]["text"] is not expected to be the real murderer.

Constraints:
- Use ONLY the information provided in <CASE_QUESTION>, <CASE_BACKGROUND>, and the assumed dimension value.
- Do NOT answer the <CASE_QUESTION> itself. Focus ONLY on judging how likely each suspect is to be the real murderer under different assumptions about the investigative dimension.
- Do NOT rewrite or restate the <CASE_QUESTION> or <CASE_BACKGROUND>.
- label must be one of "likely", "neutral", or "unlikely".
- The output must include an entry for every combination of dimension value and suspect.

Output format:
Return STRICT JSON only with the following schema:
{{
    "evaluations": [
        [
            {{
                "answer_id": string,
                "dimension_value_id": string,
                "reason": string,
                "label": string
            }},
            ... // one object for each suspect
        ],
        ... // one array for each dimension value
    ]
}}
The "evaluations" field must contain exactly {num_dimension_values} arrays (one per dimension value).
Each inner array must contain exactly {num_possible_answers} objects (one per suspect).
""",
)
