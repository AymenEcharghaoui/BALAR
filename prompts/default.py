
# -------------------------------------------------------------- #
# USER SIMULATOR PROMPTS
# -------------------------------------------------------------- #

USER_SIMULATOR_SYSTEM_PROMPT = """You are simulating a user."""
USER_SIMULATOR_WITH_CHOICES_USER_PROMPT = """
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
Answer <QUESTION> in a way that is consistent with <USER_CONTEXT>. 

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
"""
USER_SIMULATOR_WITHOUT_CHOICES_USER_PROMPT = """
<USER_CONTEXT>
{user_context}
</USER_CONTEXT>

<QUESTION>
{question}
</QUESTION>

Task:
Answer <QUESTION> in a way that is consistent with <USER_CONTEXT>. 

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
"""

# -------------------------------------------------------------- #
# INITIAL STAGE PROMPTS (dimensions)
# -------------------------------------------------------------- #

INITIAL_DIMENSIONS_SYSTEM_PROMPT = """"""
INITIAL_DIMENSIONS_USER_PROMPT = """
<AMBIGUOUS_PROMPT>
{ambiguous_prompt}
</AMBIGUOUS_PROMPT>

<META_CONTEXT>
{meta_context}
</META_CONTEXT>

Task:
Identify the dimensions of ambiguity in the <AMBIGUOUS_PROMPT>.

Definition:
A dimension of ambiguity is a specific aspect of the <AMBIGUOUS_PROMPT> that can be interpreted in multiple valid ways, leading to different possible meanings or outcomes. Once the dimension's value is known, the prompt moves toward a single dominant interpretation with a well-defined ground-truth answer.

What to generate:
- Produce a minimal, non-overlapping set of dimensions of ambiguity.
- Produce exactly {num_initial_dims} dimensions of ambiguity.
- Each dimension must correspond to a distinct, intent-level uncertainty.
- If <META_CONTEXT> already resolves a dimension, do not include it.
- If <META_CONTEXT> context proposes some dimensions of ambiguity, use them.

For each dimension of ambiguity, provide :
- reason: a short one-sentence explanation of why this dimension is a source of ambiguity in the prompt.
- name: a short, specific label
- values: a list of plausible values, no larger than {max_num_values_per_dim}, that this dimension could take in the context of the prompt.

Constraints:
- Use ONLY the information provided in <AMBIGUOUS_PROMPT> and <META_CONTEXT>.
- Do NOT answer the <AMBIGUOUS_PROMPT> itself. Focus ONLY on identifying dimensions of ambiguity.
- Do NOT rewrite or restate the <AMBIGUOUS_PROMPT> or <META_CONTEXT>.

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
"""

# -------------------------------------------------------------- #
# INITIAL STAGE PROMPTS (priors)
# -------------------------------------------------------------- #

INITIAL_PRIORS_SYSTEM_PROMPT = """"""
INITIAL_PRIORS_USER_PROMPT = """
<AMBIGUOUS_PROMPT>
{ambiguous_prompt}
</AMBIGUOUS_PROMPT>

<META_CONTEXT>
{meta_context}
</META_CONTEXT>

<DIMENSION_NAME>
{dimension_name}
</DIMENSION_NAME>

<DIMENSION_VALUE>
{dimension_value}
</DIMENSION_VALUE>

Task:
Given <AMBIGUOUS_PROMPT> and <META_CONTEXT>, judge how likely the <DIMENSION_NAME> takes on the value <DIMENSION_VALUE>.

What to generate:
- reason: a short one-sentence explanation of why the <DIMENSION_NAME> is likely, unlikely, or neutral to take on the value <DIMENSION_VALUE>.
- label: one of "likely", "unlikely", or "neutral" according to the following definitions:
    - likely: <DIMENSION_VALUE> is explicitly stated, strongly implied, or is the most natural assumption given the information in the <AMBIGUOUS_PROMPT> and <META_CONTEXT>.
    - neutral: <DIMENSION_VALUE> is plausible but not implied or supported by specific evidence in the <AMBIGUOUS_PROMPT> or <META_CONTEXT>.
    - unlikely: <DIMENSION_VALUE> is contradicted by the <AMBIGUOUS_PROMPT> or <META_CONTEXT>, or would require assumptions that are inconsistent with the provided information.

Constraints:
- Use ONLY the information provided in <AMBIGUOUS_PROMPT> and <META_CONTEXT>.
- Do NOT answer the <AMBIGUOUS_PROMPT> itself. Focus ONLY on judging the likelihood of the dimension value.
- Do NOT rewrite or restate the <AMBIGUOUS_PROMPT> or <META_CONTEXT>.
- label must be one of "likely", "unlikely", or "neutral".

Output format:
Return STRICT JSON only with the following schema:
{{
    "reason": string,
    "label": string
}}
"""

# -------------------------------------------------------------- #
# INITIAL STAGE PROMPTS (questions)
# -------------------------------------------------------------- #

INITIAL_QUESTIONS_SYSTEM_PROMPT = """"""
INITIAL_QUESTIONS_USER_PROMPT = """
<AMBIGUOUS_PROMPT>
{ambiguous_prompt}
</AMBIGUOUS_PROMPT>

<META_CONTEXT>
{meta_context}
</META_CONTEXT>

<DIMENSIONS>
{dimensions_with_values}
</DIMENSIONS>

Task:
Given <AMBIGUOUS_PROMPT>, <META_CONTEXT>, and <DIMENSIONS>, generate exactly {num_initial_questions} clarifying questions that would help disambiguate <AMBIGUOUS_PROMPT>. Each question should target one or more <DIMENSIONS> and have multiple-choice answers.

Definition:
<DIMENSIONS> is a list of dimensions of ambiguity, where each dimension has a name and a list of possible values it could take. A dimension of ambiguity is a specific aspect of the <AMBIGUOUS_PROMPT> that can be interpreted in multiple valid ways, leading to different possible meanings or outcomes.

What to generate:
For each of the {num_initial_questions} questions, provide:
- reason: a short one-sentence explanation of why this question would help disambiguate the prompt.
- question: the text of the clarifying question.
- choices: a list of multiple-choice answer options for the question, no larger than {max_choices_per_question}.

Constraints:
- Use ONLY the information provided in <AMBIGUOUS_PROMPT>, <META_CONTEXT>, and <DIMENSIONS>.
- Do NOT answer the <AMBIGUOUS_PROMPT> itself. Focus ONLY on generating clarifying questions.
- Do NOT rewrite or restate the <AMBIGUOUS_PROMPT> or <META_CONTEXT>.
- Each question must be designed to elicit information about one or more of the dimensions in <DIMENSIONS>.
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
"""

# -------------------------------------------------------------- #
# LIKELIHOOD PROMPTS
# -------------------------------------------------------------- #

LIKELIHOOD_SYSTEM_PROMPT = """"""
LIKELIHOOD_WITHOUT_HISTORY_USER_PROMPT = """
<AMBIGUOUS_PROMPT>
{ambiguous_prompt}
</AMBIGUOUS_PROMPT>

<META_CONTEXT>
{meta_context}
</META_CONTEXT>

<USER_INFO>
{user_info}
</USER_INFO>

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
A dimension of ambiguity is a specific aspect of the <AMBIGUOUS_PROMPT> that can be interpreted in multiple valid ways, leading to different possible meanings or outcomes.
<DIMENSION_VALUES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a possible value that the <DIMENSION_NAME> could take.
<QUESTION_CHOICES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a multiple-choice answer option for the question.
Let values[i] be the i-th element of <DIMENSION_VALUES_WITH_IDS>.
Let choices[j] be the j-th element of <QUESTION_CHOICES_WITH_IDS>.

Task (row-major order):
For i = 0..len(values)-1:
  For j = 0..len(choices)-1:
    - Assume the true state of the world is <DIMENSION_NAME> = values[i]["text"].
    - Impersonate the user described in <USER_INFO>.
    - Judge how likely it is that this user would answer the question <QUESTION> with choices[j]["text"] under that assumption.

What to generate:
For i = 0..len(values)-1:
  For j = 0..len(choices)-1:
    - question_choice_id: the id of the question choice being evaluated, i.e. choices[j]["id"]
    - dimension_value_id: the id of the dimension value being evaluated, i.e. values[i]["id"]
    - reason: a short one-sentence explanation of why choices[j] is labeled likely/neutral/unlikely and why the other two labels were not chosen.
    - label: one of "likely", "neutral", or "unlikely" according to the following definitions:
      - "likely": Given <DIMENSION_NAME> = values[i]["text"] and the user acts according to <USER_INFO>, the user is expected to give choices[j]["text"] for <QUESTION>.
      - "neutral": Given <DIMENSION_NAME> = values[i]["text"] and the user acts according to <USER_INFO>, choices[j]["text"] is plausible but not specifically supported; there is insufficient evidence to say that the user would or would not prefer it over other choices.
      - "unlikely": Given <DIMENSION_NAME> = values[i]["text"] and the user acts according to <USER_INFO>, the user is not expected to give choices[j]["text"] for <QUESTION>.

Constraints:
- Use ONLY the information provided in <AMBIGUOUS_PROMPT>, <META_CONTEXT>, and <USER_INFO>.
- Do NOT answer the <AMBIGUOUS_PROMPT> itself. Focus ONLY on judging the likelihood of the question choices under different assumptions about the dimension value.
- Do NOT rewrite or restate the <AMBIGUOUS_PROMPT>, <META_CONTEXT>, or <USER_INFO>.
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
"""
LIKELIHOOD_WITH_HISTORY_USER_PROMPT = """
<AMBIGUOUS_PROMPT>
{ambiguous_prompt}
</AMBIGUOUS_PROMPT>

<META_CONTEXT>
{meta_context}
</META_CONTEXT>

<USER_INFO>
{user_info}
</USER_INFO>

<CONVERSATION_LOG>
{conversation_log}
</CONVERSATION_LOG>

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
A dimension of ambiguity is a specific aspect of the <AMBIGUOUS_PROMPT> that can be interpreted in multiple valid ways, leading to different possible meanings or outcomes.
<DIMENSION_VALUES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a possible value that the <DIMENSION_NAME> could take.
<QUESTION_CHOICES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a multiple-choice answer option for the question.
<CONVERSATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the conversation between the user and the agent up to this point. This information may provide additional context.
Let values[i] be the i-th element of <DIMENSION_VALUES_WITH_IDS>.
Let choices[j] be the j-th element of <QUESTION_CHOICES_WITH_IDS>.

Task (row-major order):
For i = 0..len(values)-1:
  For j = 0..len(choices)-1:
    - Assume the true state of the world is <DIMENSION_NAME> = values[i]["text"].
    - Impersonate the user described in <USER_INFO>.
    - Judge how likely it is that this user would answer the question <QUESTION> with choices[j]["text"] under that assumption.

What to generate:
For i = 0..len(values)-1:
  For j = 0..len(choices)-1:
    - question_choice_id: the id of the question choice being evaluated, i.e. choices[j]["id"]
    - dimension_value_id: the id of the dimension value being evaluated, i.e. values[i]["id"]
    - reason: a short one-sentence explanation of why choices[j] is labeled likely/neutral/unlikely and why the other two labels were not chosen.
    - label: one of "likely", "neutral", or "unlikely" according to the following definitions:
      - "likely": Given <DIMENSION_NAME> = values[i]["text"] and the user acts according to <USER_INFO>, the user is expected to give choices[j]["text"] for <QUESTION>.
      - "neutral": Given <DIMENSION_NAME> = values[i]["text"] and the user acts according to <USER_INFO>, choices[j]["text"] is plausible but not specifically supported; there is insufficient evidence to say that the user would or would not prefer it over other choices.
      - "unlikely": Given <DIMENSION_NAME> = values[i]["text"] and the user acts according to <USER_INFO>, the user is not expected to give choices[j]["text"] for <QUESTION>.

Constraints:
- Use ONLY the information provided in <AMBIGUOUS_PROMPT>, <META_CONTEXT>, <USER_INFO>, and <CONVERSATION_LOG>.
- Do NOT answer the <AMBIGUOUS_PROMPT> itself. Focus ONLY on judging the likelihood of the question choices under different assumptions about the dimension value.
- Do NOT rewrite or restate the <AMBIGUOUS_PROMPT>, <META_CONTEXT>, <USER_INFO>, or <CONVERSATION_LOG>.
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
"""

# -------------------------------------------------------------- #
# NATURAL LANGUAGE to CHOICES PROMPTS
# -------------------------------------------------------------- #

SCORE_NATURAL_LANGUAGE_SYSTEM_PROMPT = """"""
SCORE_NATURAL_LANGUAGE_USER_PROMPT = """
<QUESTION>
{question}
</QUESTION>

<CHOICES_WITH_IDS>
{choices_with_ids}
</CHOICES_WITH_IDS>

<USER_ANSWER>
{user_answer}
</USER_ANSWER>

Definition:
<CHOICES_WITH_IDS> is a list of dicts with "id" and "value" fields. Each dict corresponds to a multiple-choice answer option for the question.
Let choices[i] be the i-th element of <CHOICES_WITH_IDS>.

Task:
Judge how well the <USER_ANSWER> maps to each of the choices in <CHOICES_WITH_IDS> for the question <QUESTION>.

What to generate:
For i = 0..len(choices)-1:
- choice_id: the id of the question choice being evaluated, i.e. choices[i]["id"]
- reason: a short one-sentence explanation of choices[i]["value"] is likely/neutral/unlikely given the <USER_ANSWER>.
- label: one of "likely", "neutral", or "unlikely" according to the following definitions:
    -"likely": choices[i]["value"] aligns well with the <USER_ANSWER> and fits it better than most other choices.
    -"neutral": choices[i]["value"] is neither clearly supported nor clearly contradicted by the <USER_ANSWER>.
    -"unlikely": choices[i]["value"] fits the <USER_ANSWER> worse than other choices, or conflicts with the meaning of the <USER_ANSWER>.

Constraints:
- Use ONLY the information provided in <QUESTION>, <CHOICES_WITH_IDS>, and <USER_ANSWER>.
- Do NOT answer the <QUESTION> itself. Focus ONLY on judging how well the <USER_ANSWER> maps to the provided choices.
- Do NOT rewrite or restate the <QUESTION> or <USER_ANSWER>.
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
"""

# -------------------------------------------------------------- #
# EXPAND DIMENSION PROMPTS (dimensions)
# -------------------------------------------------------------- #

EXPAND_DIMENSION_SYSTEM_PROMPT = """"""
EXPAND_DIMENSION_USER_PROMPT = """
<AMBIGUOUS_PROMPT>
{ambiguous_prompt}
</AMBIGUOUS_PROMPT>

<META_CONTEXT>
{meta_context}
</META_CONTEXT>

<PAST_DIMENSIONS>
{past_dimensions}
</PAST_DIMENSIONS>

<CONVERSATION_LOG>
{conversation_log}
</CONVERSATION_LOG>

Definition:
A dimension of ambiguity is a specific aspect of the <AMBIGUOUS_PROMPT> that can be interpreted in multiple valid ways, leading to different possible meanings or outcomes.
<PAST_DIMENSIONS> is a list of dicts with "name" fields, representing the dimensions that have already been used in previous iterations of the conversation.
<CONVERSATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the conversation between the user and the agent up to this point.

Task:
Identify a new dimension of ambiguity in the <AMBIGUOUS_PROMPT> that has not been previously identified in <PAST_DIMENSIONS>.

What to generate:
- reason: a short one-sentence explanation of why this dimension is a source of ambiguity in the prompt.
- name: a short, specific label for this dimension of ambiguity.
- values: a list of plausible values, no larger than {max_num_values_per_dim}, that this dimension could take in the context of the prompt.

Constraints:
- Use ONLY the information provided in <AMBIGUOUS_PROMPT>, <META_CONTEXT>, <PAST_DIMENSIONS>, and <CONVERSATION_LOG>.
- Do NOT answer the <AMBIGUOUS_PROMPT> itself. Focus ONLY on identifying a new dimension of ambiguity.
- Do NOT rewrite or restate the <AMBIGUOUS_PROMPT>, <META_CONTEXT>, <PAST_DIMENSIONS>, or <CONVERSATION_LOG>.
- The generated dimension name must not be the same as any of the names in <PAST_DIMENSIONS>.

Output format:
Return STRICT JSON only with the following schema:
{{
    "reason": string,
    "name": string,
    "values": [string, ...]
}}
"""

# -------------------------------------------------------------- #
# EXPAND DIMENSIONS PROMPTS (priors)
# -------------------------------------------------------------- #

EXPAND_DIMENSION_PRIORS_SYSTEM_PROMPT = """"""
EXPAND_DIMENSION_PRIORS_USER_PROMPT = """
<AMBIGUOUS_PROMPT>
{ambiguous_prompt}
</AMBIGUOUS_PROMPT>

<META_CONTEXT>
{meta_context}
</META_CONTEXT>

<CONVERSATION_LOG>
{conversation_log}
</CONVERSATION_LOG>

<DIMENSION_NAME>
{dimension_name}
</DIMENSION_NAME>

<DIMENSION_VALUE>
{dimension_value}
</DIMENSION_VALUE>

Definition:
A dimension of ambiguity is a specific aspect of the <AMBIGUOUS_PROMPT> that can be interpreted in multiple valid ways, leading to different possible meanings or outcomes.
<CONVERSATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the conversation between the user and the agent up to this point.

Task:
Given <AMBIGUOUS_PROMPT>, <META_CONTEXT>, <CONVERSATION_LOG>, and a specific dimension of ambiguity defined by <DIMENSION_NAME> and <DIMENSION_VALUE>, judge how likely it is that the <DIMENSION_NAME> takes on the value <DIMENSION_VALUE>.

What to generate:
- reason: a short one-sentence explanation of why the <DIMENSION_NAME> is likely, unlikely, or neutral to take on the value <DIMENSION_VALUE>.
- label: one of "likely", "unlikely", or "neutral" according to the following definitions:
    - likely: <DIMENSION_VALUE> is explicitly stated, strongly implied, or is the most natural assumption given the information in the <AMBIGUOUS_PROMPT>, <META_CONTEXT>, and <CONVERSATION_LOG>.
    - neutral: <DIMENSION_VALUE> is plausible but not implied or supported by specific evidence in the <AMBIGUOUS_PROMPT>, <META_CONTEXT>, or <CONVERSATION_LOG>.
    - unlikely: <DIMENSION_VALUE> is contradicted by the <AMBIGUOUS_PROMPT>, <META_CONTEXT> or <CONVERSATION_LOG>, or would require assumptions that are inconsistent with the provided information.

Constraints:
- Use ONLY the information provided in <AMBIGUOUS_PROMPT>, <META_CONTEXT>, and <CONVERSATION_LOG>.
- Do NOT answer the <AMBIGUOUS_PROMPT> itself. Focus ONLY on judging the likelihood of the dimension value.
- Do NOT rewrite or restate the <AMBIGUOUS_PROMPT>, <META_CONTEXT>, or <CONVERSATION_LOG>.
- label must be one of "likely", "unlikely", or "neutral".

Output format:
Return STRICT JSON only with the following schema:
{{
    "reason": string,
    "label": string
}}
"""

# -------------------------------------------------------------- #
# EXPAND QUESTIONS PROMPTS
# -------------------------------------------------------------- #

EXPAND_QUESTIONS_SYSTEM_PROMPT = """"""
EXPAND_QUESTIONS_USER_PROMPT = """
<AMBIGUOUS_PROMPT>
{ambiguous_prompt}
</AMBIGUOUS_PROMPT>

<META_CONTEXT>
{meta_context}
</META_CONTEXT>

<CONVERSATION_LOG>
{conversation_log}
</CONVERSATION_LOG>

<NEW_DIMENSION_WITH_VALUES>
{new_dimension_with_values}
</NEW_DIMENSION_WITH_VALUES>

<HIGH_UNCERTAINTY_DIMENSIONS_WITH_VALUES>
{high_uncertainty_dimensions_with_values}
</HIGH_UNCERTAINTY_DIMENSIONS_WITH_VALUES>

Definition:
A dimension of ambiguity is a specific aspect of the <AMBIGUOUS_PROMPT> that can be interpreted in multiple valid ways, leading to different possible meanings or outcomes.
<CONVERSATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the conversation between the user and the agent up to this point
<NEW_DIMENSION_WITH_VALUES> is a dict with "name" and "values" fields, representing the new dimension of ambiguity that was just identified along with its possible values.
<HIGH_UNCERTAINTY_DIMENSIONS_WITH_VALUES> is a list of dicts with "name" and "values" fields, representing the dimensions of ambiguity that currently have the highest uncertainty in terms of their true value. They do not include the new dimension in <NEW_DIMENSION_WITH_VALUES>.

Task:
Given <AMBIGUOUS_PROMPT>, <META_CONTEXT>, <CONVERSATION_LOG>, a newly identified dimension of ambiguity in <NEW_DIMENSION_WITH_VALUES>, and the dimensions with the highest uncertainty in <HIGH_UNCERTAINTY_DIMENSIONS_WITH_VALUES>, generate a set of clarifying questions that would help disambiguate the prompt by targeting the new dimension and/or the high-uncertainty dimensions.

What to generate:
Generate at most {max_new_questions_per_round} clarifying questions. For each question, provide:
- reason: a short one-sentence explanation of why this question would help disambiguate the prompt.
- question: the text of the clarifying question.
- choices: a list of multiple-choice answer options for the question, no larger than {max_choices_per_question}.

Constraints:
- Use ONLY the information provided in <AMBIGUOUS_PROMPT>, <META_CONTEXT>, <CONVERSATION_LOG>, <NEW_DIMENSION_WITH_VALUES>, and <HIGH_UNCERTAINTY_DIMENSIONS_WITH_VALUES>.
- Do NOT answer the <AMBIGUOUS_PROMPT> itself. Focus ONLY on generating clarifying questions.
- Do NOT rewrite or restate the <AMBIGUOUS_PROMPT>, <META_CONTEXT>, <CONVERSATION_LOG>, <NEW_DIMENSION_WITH_VALUES>, or <HIGH_UNCERTAINTY_DIMENSIONS_WITH_VALUES>.
- Each question must be designed to elicit information about the new dimension in <NEW_DIMENSION_WITH_VALUES> and/or the high-uncertainty dimensions in <HIGH_UNCERTAINTY_DIMENSIONS_WITH_VALUES>.
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
"""

# -------------------------------------------------------------- #
# FINAL ANSWER PROMPTS
# -------------------------------------------------------------- #

FINAL_ANSWER_SYSTEM_PROMPT = """"""
FINAL_ANSWER_WITHOUT_CHOICES_USER_PROMPT = """
<AMBIGUOUS_PROMPT>
{ambiguous_prompt}
</AMBIGUOUS_PROMPT>

<META_CONTEXT>
{meta_context}
</META_CONTEXT>

<CONVERSATION_LOG>
{conversation_log}
</CONVERSATION_LOG>

<MAP_STATE>
{map_state}
</MAP_STATE>

Definition:
<CONVERSATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the conversation between the user and the agent up to this point.
A dimension of ambiguity is a specific aspect of the <AMBIGUOUS_PROMPT> that can be interpreted in multiple valid ways, leading to different possible meanings or outcomes.
<MAP_STATE> is a structured representation of the current state of the agent's understanding of the ambiguous prompt, where each dimension of ambiguity is mapped to a specific value. This represents the agent's best guess of the true state of the world based on the conversation so far.

Task:
Given <AMBIGUOUS_PROMPT>, <META_CONTEXT>, <CONVERSATION_LOG>, and <MAP_STATE>, provide a final answer to the <AMBIGUOUS_PROMPT>.

What to generate:
- reason: a short one-sentence explanation of why the final answer is correct given the information in <AMBIGUOUS_PROMPT>, <META_CONTEXT>, <CONVERSATION_LOG>, and <MAP_STATE>.
- final_answer: your final answer to the <AMBIGUOUS_PROMPT>.

Constraints:
- The final answer must be consistent with the information in <AMBIGUOUS_PROMPT>, <META_CONTEXT>, <CONVERSATION_LOG>, and <MAP_STATE>.
- Do NOT rewrite or restate the <AMBIGUOUS_PROMPT>, <META_CONTEXT>, <CONVERSATION_LOG>, or <MAP_STATE>.
- answer must be a natural language answer to the <AMBIGUOUS_PROMPT>.

Output format:
Return STRICT JSON only with the following schema:
{{
    "reason": string,
    "final_answer": string
}}
"""
FINAL_ANSWER_WITH_CHOICES_USER_PROMPT = """
<AMBIGUOUS_PROMPT>
{ambiguous_prompt}
</AMBIGUOUS_PROMPT>

<META_CONTEXT>
{meta_context}
</META_CONTEXT>

<CONVERSATION_LOG>
{conversation_log}
</CONVERSATION_LOG>

<MAP_STATE>
{map_state}
</MAP_STATE>

<POSSIBLE_ANSWERS_WITH_IDS>
{possible_answers_with_ids}
</POSSIBLE_ANSWERS_WITH_IDS>

Definition:
<CONVERSATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the conversation between the user and the agent up to this point.
A dimension of ambiguity is a specific aspect of the <AMBIGUOUS_PROMPT> that can be interpreted in multiple valid ways, leading to different possible meanings or outcomes.
<MAP_STATE> is a structured representation of the current state of the agent's understanding of the ambiguous prompt, where each dimension of ambiguity is mapped to a specific value. This represents the agent's best guess of the true state of the world based on the conversation so far.
<POSSIBLE_ANSWERS_WITH_IDS> is a list of dicts with "id" and "value" fields. Each dict corresponds to a possible answer choice for <AMBIGUOUS_PROMPT>.
Let possible_answers[i] be the i-th element of <POSSIBLE_ANSWERS_WITH_IDS>.

Task:
Given <AMBIGUOUS_PROMPT>, <META_CONTEXT>, <CONVERSATION_LOG>, and <MAP_STATE>, provide a final answer to the <AMBIGUOUS_PROMPT>. 

What to generate:
- reason: a short one-sentence explanation of why the final answer is correct given the information in <AMBIGUOUS_PROMPT>, <META_CONTEXT>, <CONVERSATION_LOG>, and <MAP_STATE>.
- final_answer_id: the id of the choice in <POSSIBLE_ANSWERS_WITH_IDS> that you are selecting as your final answer to the <AMBIGUOUS_PROMPT>.

Constraints:
- The final answer must be consistent with the information in <AMBIGUOUS_PROMPT>, <META_CONTEXT>, <CONVERSATION_LOG>, and <MAP_STATE>.
- Do NOT rewrite or restate the <AMBIGUOUS_PROMPT>, <META_CONTEXT>, <CONVERSATION_LOG>, or <MAP_STATE>.
- final_answer_id must be one of the ids (i.e possible_answers[i]["id"]) provided in <POSSIBLE_ANSWERS_WITH_IDS>.

Output format:
Return STRICT JSON only with the following schema:
{{
    "reason": string,
    "final_answer_id": string
}}
"""


# -------------------------------------------------------------- #                                                                                                                                 
# Verifier prompts                                                                                                                                                                 
# -------------------------------------------------------------- #                                                                                                                                 
VERIFIER_SYSTEM_PROMPT = """You are a verification agent. Your task is to review a response generated by another LLM and check it for logical inconsistencies, hallucinations, or errors."""       
                                                                                                                                                                                                   
VERIFIER_USER_PROMPT = """
<ORIGINAL_SYSTEM_PROMPT>                                                                                                                                                 
{system_prompt}                                                                                                                               
</ORIGINAL_SYSTEM_PROMPT>                                                                                                                                                                          
                                                                                                                                                                                                   
<ORIGINAL_USER_PROMPT>                                                                                                                                                                             
{user_prompt}                                                                                                                                                                                      
</ORIGINAL_USER_PROMPT>                                                                                                                                                                            
                                                                                                                                                                                                   
<PROPOSED_RESPONSE>                                                                                                                                                                                
{response}                                                                                                                                                                                         
</PROPOSED_RESPONSE>                                                                                                                                                                               
                                                                                                                                                                                                   
Task:                                                                                                                                                                                              
Review the <PROPOSED_RESPONSE> generated for the task described in <ORIGINAL_SYSTEM_PROMPT> and <ORIGINAL_USER_PROMPT>.                                                                            
                                                                                                                                                                                                   
Check for:                                                                                                                                                                                         
1. Logical inconsistencies between the reasoning provided and the assigned labels.                                                                                                                 
2. Labels that contradict the information provided in the original prompts.                                                                                                                        
3. Any hallucinated or fabricated information not supported by the original prompts.                                                                                                                
4. Whether each reason actually supports its assigned label (e.g., a reason describing something as expected should not have an "unlikely" label).                                                 
                                                                                                                                                                                                   
What to generate:                                                                                                                                                                                  
- is_valid: true if the response is logically consistent and well-reasoned, false if issues were found.                                                                                            
- feedback: if is_valid is false, provide specific feedback about which entries are wrong and why. If is_valid is true, this should be an empty string.                                            
                                                                                                                                                                                                   
Output format:                                                                                                                                                                                     
Return STRICT JSON only with the following schema:                                                                                                                                                 
{{                                                                                                                                                                                                 
    "is_valid": boolean,                                                                                                                                                                           
    "feedback": string
}}                                                                                                                                                                                                 
"""                                                                                                                                                                                                
                                                                                                                                                                                                  
VERIFIER_CORRECTION_ADD = """                                                                                                                                                                 
                                                                                                                                                                                                   
<VERIFIER_FEEDBACK>                                                                                                                                                                                
A verification step found issues with a previous attempt at this task. Here is the feedback:                                                                                                       
{feedback}                                                                                                                                                                                         
                                                                                                                                                                                                   
The previous response was:                                                                                                                                                                         
{previous_response}                                                                                                                                                                                
</VERIFIER_FEEDBACK>                                                                                                                                                                               
                                                                                                                                                                                                   
Please redo the task, taking into account the <VERIFIER_FEEDBACK>. Make sure to correct the issues identified by the verifier.                                                                     
"""    

# -------------------------------------------------------------- #                                                                                                                                 
# Answer likelihood prompts                                                                                                                                                                 
# -------------------------------------------------------------- #  
ANSWER_LIKELIHOOD_SYSTEM_PROMPT = """"""
ANSWER_LIKELIHOOD_USER_PROMPT = """
<AMBIGUOUS_PROMPT>
{ambiguous_prompt}
</AMBIGUOUS_PROMPT>

<META_CONTEXT>
{meta_context}
</META_CONTEXT>

<DIMENSION_NAME>
{dimension_name}
</DIMENSION_NAME>

<DIMENSION_VALUES_WITH_IDS>
{dimension_values_with_ids}
</DIMENSION_VALUES_WITH_IDS>

<POSSIBLE_ANSWERS_WITH_IDS>
{possible_answers_with_ids}
</POSSIBLE_ANSWERS_WITH_IDS>

Definition:
A dimension of ambiguity is a specific aspect of the <AMBIGUOUS_PROMPT> that can be interpreted in multiple valid ways, leading to different possible meanings or outcomes.
<DIMENSION_VALUES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a possible value that the <DIMENSION_NAME> could take.
<POSSIBLE_ANSWERS_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a candidate final answer to the <AMBIGUOUS_PROMPT>.
Let values[i] be the i-th element of <DIMENSION_VALUES_WITH_IDS>.
Let answers[j] be the j-th element of <POSSIBLE_ANSWERS_WITH_IDS>.

Task (row-major order):
For i = 0..len(values)-1:
For j = 0..len(answers)-1:
    - Assume the true state of the world is <DIMENSION_NAME> = values[i]["text"].
    - Judge how likely it is that answers[j]["text"] is the correct final answer to <AMBIGUOUS_PROMPT> under that assumption.

What to generate:
For i = 0..len(values)-1:
For j = 0..len(answers)-1:
    - answer_id: the id of the candidate answer being evaluated, i.e. answers[j]["id"]
    - dimension_value_id: the id of the dimension value being evaluated, i.e. values[i]["id"]
    - reason: a short one-sentence explanation of why answers[j] is likely/neutral/unlikely to be the correct final answer given <DIMENSION_NAME> = values[i]["text"].
    - label: one of "likely", "neutral", or "unlikely" according to the following definitions:
        - "likely": Given <DIMENSION_NAME> = values[i]["text"], answers[j]["text"] is the expected correct answer to <AMBIGUOUS_PROMPT>.
        - "neutral": Given <DIMENSION_NAME> = values[i]["text"], answers[j]["text"] is a plausible answer but not specifically supported; there is insufficient evidence to say it is more or less correct than other answers.
        - "unlikely": Given <DIMENSION_NAME> = values[i]["text"], answers[j]["text"] is not expected to be the correct answer to <AMBIGUOUS_PROMPT>.

Constraints:
- Use ONLY the information provided in <AMBIGUOUS_PROMPT>, <META_CONTEXT>, and the assumed dimension value.
- Do NOT answer the <AMBIGUOUS_PROMPT> itself. Focus ONLY on judging how likely each candidate answer is to be correct under different assumptions about the dimension value.
- Do NOT rewrite or restate the <AMBIGUOUS_PROMPT> or <META_CONTEXT>.
- label must be one of "likely", "neutral", or "unlikely".
- The output must include an entry for every combination of dimension value and candidate answer.

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
            ... // one object for each candidate answer
        ],
        ... // one array for each dimension value
    ]
}}
The "evaluations" field must contain exactly {num_dimension_values} arrays (one per dimension value).
Each inner array must contain exactly {num_possible_answers} objects (one per candidate answer).
"""

