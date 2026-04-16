from prompts.base import PromptSet

PROMPT_SET = PromptSet(

# -------------------------------------------------------------- #
# USER SIMULATOR PROMPTS
# -------------------------------------------------------------- #

USER_SIMULATOR_SYSTEM_PROMPT="""You are simulating a patient being examined by a physician. Answer questions based ONLY on your medical record and symptoms. Use facts from your medical record. Do not give information that was not specifically asked about.""",

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
You are a patient being examined by a physician. Answer <QUESTION> in a way that is consistent with your medical record and symptoms described in <USER_CONTEXT>. Use facts from your medical record. Do not give information that was not specifically asked about.

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
You are a patient being examined by a physician. Answer <QUESTION> in a way that is consistent with your medical record and symptoms described in <USER_CONTEXT>. Use facts from your medical record. Do not give information that was not specifically asked about.

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

INITIAL_DIMENSIONS_SYSTEM_PROMPT="""You are an experienced physician performing a diagnosis. Your goal is to identify the key clinical dimensions that must be clarified to arrive at the correct diagnosis or clinical decision.""",

INITIAL_DIMENSIONS_USER_PROMPT="""
<CLINICAL_QUESTION>
{ambiguous_prompt}
</CLINICAL_QUESTION>

<PATIENT_INFORMATION>
{meta_context}
</PATIENT_INFORMATION>

Task:
Identify the key clinical dimensions in the <CLINICAL_QUESTION> that must be clarified to arrive at the correct diagnosis or clinical decision.

Definition:
A clinical dimension is a specific clinical factor (e.g., symptom characterization, lab finding, risk factor, past medical history) where different values would point toward different diagnoses or clinical decisions. Once the dimension's value is known, the diagnosis narrows toward a single correct answer.

What to generate:
- Produce a minimal, non-overlapping set of clinical dimensions.
- Produce exactly {num_initial_dims} clinical dimensions.
- Each dimension must correspond to a distinct clinical uncertainty.
- If <PATIENT_INFORMATION> already resolves a dimension, do not include it.
- If <PATIENT_INFORMATION> proposes some clinical dimensions, use them.

For each clinical dimension, provide :
- reason: a short one-sentence explanation of why this clinical factor is discriminating between diagnoses.
- name: a short, specific clinical label
- values: a list of clinically plausible values, no larger than {max_num_values_per_dim}, that this dimension could take.

Constraints:
- Use ONLY the information provided in <CLINICAL_QUESTION> and <PATIENT_INFORMATION>.
- Do NOT answer the <CLINICAL_QUESTION> itself. Focus ONLY on identifying clinical dimensions.
- Do NOT rewrite or restate the <CLINICAL_QUESTION> or <PATIENT_INFORMATION>.

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

INITIAL_PRIORS_SYSTEM_PROMPT="""You are an experienced physician forming initial clinical hypotheses based on the available patient information. Use your clinical knowledge to assess the likelihood of different clinical findings.""",

INITIAL_PRIORS_USER_PROMPT="""
<CLINICAL_QUESTION>
{ambiguous_prompt}
</CLINICAL_QUESTION>

<PATIENT_INFORMATION>
{meta_context}
</PATIENT_INFORMATION>

<DIMENSION_NAME>
{dimension_name}
</DIMENSION_NAME>

<DIMENSION_VALUE>
{dimension_value}
</DIMENSION_VALUE>

Task:
Given <CLINICAL_QUESTION> and <PATIENT_INFORMATION>, judge how likely the clinical dimension <DIMENSION_NAME> takes on the value <DIMENSION_VALUE>.

What to generate:
- reason: a short one-sentence explanation of why the <DIMENSION_NAME> is likely, unlikely, or neutral to take on the value <DIMENSION_VALUE>.
- label: one of "likely", "unlikely", or "neutral" according to the following definitions:
    - likely: <DIMENSION_VALUE> is explicitly stated, strongly implied, or is the most natural clinical assumption given the patient information in <CLINICAL_QUESTION> and <PATIENT_INFORMATION>.
    - neutral: <DIMENSION_VALUE> is clinically plausible but not implied or supported by specific evidence in the <CLINICAL_QUESTION> or <PATIENT_INFORMATION>.
    - unlikely: <DIMENSION_VALUE> is contradicted by the <CLINICAL_QUESTION> or <PATIENT_INFORMATION>, or would require assumptions that are inconsistent with the patient's presentation.

Constraints:
- Use ONLY the information provided in <CLINICAL_QUESTION> and <PATIENT_INFORMATION>.
- Do NOT answer the <CLINICAL_QUESTION> itself. Focus ONLY on judging the likelihood of the dimension value.
- Do NOT rewrite or restate the <CLINICAL_QUESTION> or <PATIENT_INFORMATION>.
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

INITIAL_QUESTIONS_SYSTEM_PROMPT="""You are an experienced physician conducting a patient interview to gather clinical information for a diagnosis. Your questions should be targeted, clinically relevant, and designed to discriminate between competing diagnoses.""",

INITIAL_QUESTIONS_USER_PROMPT="""
<CLINICAL_QUESTION>
{ambiguous_prompt}
</CLINICAL_QUESTION>

<PATIENT_INFORMATION>
{meta_context}
</PATIENT_INFORMATION>

<CLINICAL_DIMENSIONS>
{dimensions_with_values}
</CLINICAL_DIMENSIONS>

Task:
Given <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, and <CLINICAL_DIMENSIONS>, generate exactly {num_initial_questions} clinical questions to ask the patient that would help arrive at the correct diagnosis. Each question should target one or more <CLINICAL_DIMENSIONS> and have multiple-choice answers.

Definition:
<CLINICAL_DIMENSIONS> is a list of clinical dimensions, where each dimension has a name and a list of possible values it could take. A clinical dimension is a specific clinical factor where different values would point toward different diagnoses.

What to generate:
For each of the {num_initial_questions} questions, provide:
- reason: a short one-sentence explanation of why this question would help narrow the diagnosis.
- question: the text of the clinical question to ask the patient.
- choices: a list of multiple-choice answer options for the question, no larger than {max_choices_per_question}.

Constraints:
- Use ONLY the information provided in <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, and <CLINICAL_DIMENSIONS>.
- Do NOT answer the <CLINICAL_QUESTION> itself. Focus ONLY on generating clinical questions.
- Do NOT rewrite or restate the <CLINICAL_QUESTION> or <PATIENT_INFORMATION>.
- Each question must be designed to elicit information about one or more of the dimensions in <CLINICAL_DIMENSIONS>.
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

LIKELIHOOD_SYSTEM_PROMPT="""You are an experienced physician evaluating how a patient would likely respond to a clinical question under different assumptions about their underlying condition.""",

LIKELIHOOD_WITHOUT_HISTORY_USER_PROMPT="""
<CLINICAL_QUESTION>
{ambiguous_prompt}
</CLINICAL_QUESTION>

<PATIENT_INFORMATION>
{meta_context}
</PATIENT_INFORMATION>

<PATIENT_PROFILE>
{user_info}
</PATIENT_PROFILE>

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
A clinical dimension is a specific clinical factor where different values would point toward different diagnoses or clinical decisions.
<DIMENSION_VALUES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a possible value that the <DIMENSION_NAME> could take.
<QUESTION_CHOICES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a multiple-choice answer option for the question.
Let values[i] be the i-th element of <DIMENSION_VALUES_WITH_IDS>.
Let choices[j] be the j-th element of <QUESTION_CHOICES_WITH_IDS>.

Task (row-major order):
For i = 0..len(values)-1:
  For j = 0..len(choices)-1:
    - Assume the patient's true clinical state is <DIMENSION_NAME> = values[i]["text"].
    - Impersonate the patient described in <PATIENT_PROFILE>.
    - Judge how likely it is that this patient would answer the question <QUESTION> with choices[j]["text"] under that assumption.

What to generate:
For i = 0..len(values)-1:
  For j = 0..len(choices)-1:
    - question_choice_id: the id of the question choice being evaluated, i.e. choices[j]["id"]
    - dimension_value_id: the id of the dimension value being evaluated, i.e. values[i]["id"]
    - reason: a short one-sentence explanation of why choices[j] is labeled likely/neutral/unlikely and why the other two labels were not chosen.
    - label: one of "likely", "neutral", or "unlikely" according to the following definitions:
      - "likely": Given <DIMENSION_NAME> = values[i]["text"] and the patient acts according to <PATIENT_PROFILE>, the patient is expected to give choices[j]["text"] for <QUESTION>.
      - "neutral": Given <DIMENSION_NAME> = values[i]["text"] and the patient acts according to <PATIENT_PROFILE>, choices[j]["text"] is plausible but not specifically supported; there is insufficient evidence to say that the patient would or would not prefer it over other choices.
      - "unlikely": Given <DIMENSION_NAME> = values[i]["text"] and the patient acts according to <PATIENT_PROFILE>, the patient is not expected to give choices[j]["text"] for <QUESTION>.

Constraints:
- Use ONLY the information provided in <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, and <PATIENT_PROFILE>.
- Do NOT answer the <CLINICAL_QUESTION> itself. Focus ONLY on judging the likelihood of the question choices under different assumptions about the dimension value.
- Do NOT rewrite or restate the <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, or <PATIENT_PROFILE>.
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
<CLINICAL_QUESTION>
{ambiguous_prompt}
</CLINICAL_QUESTION>

<PATIENT_INFORMATION>
{meta_context}
</PATIENT_INFORMATION>

<PATIENT_PROFILE>
{user_info}
</PATIENT_PROFILE>

<CLINICAL_INTERVIEW_LOG>
{conversation_log}
</CLINICAL_INTERVIEW_LOG>

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
A clinical dimension is a specific clinical factor where different values would point toward different diagnoses or clinical decisions.
<DIMENSION_VALUES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a possible value that the <DIMENSION_NAME> could take.
<QUESTION_CHOICES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a multiple-choice answer option for the question.
<CLINICAL_INTERVIEW_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the clinical interview between the physician and the patient up to this point. This information may reveal additional symptoms or clinical details.
Let values[i] be the i-th element of <DIMENSION_VALUES_WITH_IDS>.
Let choices[j] be the j-th element of <QUESTION_CHOICES_WITH_IDS>.

Task (row-major order):
For i = 0..len(values)-1:
  For j = 0..len(choices)-1:
    - Assume the patient's true clinical state is <DIMENSION_NAME> = values[i]["text"].
    - Impersonate the patient described in <PATIENT_PROFILE>.
    - Judge how likely it is that this patient would answer the question <QUESTION> with choices[j]["text"] under that assumption.

What to generate:
For i = 0..len(values)-1:
  For j = 0..len(choices)-1:
    - question_choice_id: the id of the question choice being evaluated, i.e. choices[j]["id"]
    - dimension_value_id: the id of the dimension value being evaluated, i.e. values[i]["id"]
    - reason: a short one-sentence explanation of why choices[j] is labeled likely/neutral/unlikely and why the other two labels were not chosen.
    - label: one of "likely", "neutral", or "unlikely" according to the following definitions:
      - "likely": Given <DIMENSION_NAME> = values[i]["text"] and the patient acts according to <PATIENT_PROFILE>, the patient is expected to give choices[j]["text"] for <QUESTION>.
      - "neutral": Given <DIMENSION_NAME> = values[i]["text"] and the patient acts according to <PATIENT_PROFILE>, choices[j]["text"] is plausible but not specifically supported; there is insufficient evidence to say that the patient would or would not prefer it over other choices.
      - "unlikely": Given <DIMENSION_NAME> = values[i]["text"] and the patient acts according to <PATIENT_PROFILE>, the patient is not expected to give choices[j]["text"] for <QUESTION>.

Constraints:
- Use ONLY the information provided in <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, <PATIENT_PROFILE>, and <CLINICAL_INTERVIEW_LOG>.
- Do NOT answer the <CLINICAL_QUESTION> itself. Focus ONLY on judging the likelihood of the question choices under different assumptions about the dimension value.
- Do NOT rewrite or restate the <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, <PATIENT_PROFILE>, or <CLINICAL_INTERVIEW_LOG>.
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

SCORE_NATURAL_LANGUAGE_SYSTEM_PROMPT="""You are an experienced physician interpreting a patient's response to a clinical question. Your task is to judge how well the patient's answer maps to each of the predefined answer choices.""",

SCORE_NATURAL_LANGUAGE_USER_PROMPT="""
<QUESTION>
{question}
</QUESTION>

<CHOICES_WITH_IDS>
{choices_with_ids}
</CHOICES_WITH_IDS>

<PATIENT_ANSWER>
{user_answer}
</PATIENT_ANSWER>

Definition:
<CHOICES_WITH_IDS> is a list of dicts with "id" and "value" fields. Each dict corresponds to a multiple-choice answer option for the question.
Let choices[i] be the i-th element of <CHOICES_WITH_IDS>.

Task:
Judge how well the <PATIENT_ANSWER> maps to each of the choices in <CHOICES_WITH_IDS> for the question <QUESTION>.

What to generate:
For i = 0..len(choices)-1:
- choice_id: the id of the question choice being evaluated, i.e. choices[i]["id"]
- reason: a short one-sentence explanation of choices[i]["value"] is likely/neutral/unlikely given the <PATIENT_ANSWER>.
- label: one of "likely", "neutral", or "unlikely" according to the following definitions:
    -"likely": choices[i]["value"] aligns well with the <PATIENT_ANSWER> and fits it better than most other choices.
    -"neutral": choices[i]["value"] is neither clearly supported nor clearly contradicted by the <PATIENT_ANSWER>.
    -"unlikely": choices[i]["value"] fits the <PATIENT_ANSWER> worse than other choices, or conflicts with the meaning of the <PATIENT_ANSWER>.

Constraints:
- Use ONLY the information provided in <QUESTION>, <CHOICES_WITH_IDS>, and <PATIENT_ANSWER>.
- Do NOT answer the <QUESTION> itself. Focus ONLY on judging how well the <PATIENT_ANSWER> maps to the provided choices.
- Do NOT rewrite or restate the <QUESTION> or <PATIENT_ANSWER>.
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

EXPAND_DIMENSION_SYSTEM_PROMPT="""You are an experienced physician who needs to explore a new line of clinical inquiry. The clinical dimensions investigated so far have not been sufficient to arrive at a definitive diagnosis, so you must identify a new clinical factor to assess.""",

EXPAND_DIMENSION_USER_PROMPT="""
<CLINICAL_QUESTION>
{ambiguous_prompt}
</CLINICAL_QUESTION>

<PATIENT_INFORMATION>
{meta_context}
</PATIENT_INFORMATION>

<PAST_CLINICAL_DIMENSIONS>
{past_dimensions}
</PAST_CLINICAL_DIMENSIONS>

<CLINICAL_INTERVIEW_LOG>
{conversation_log}
</CLINICAL_INTERVIEW_LOG>

Definition:
A clinical dimension is a specific clinical factor where different values would point toward different diagnoses or clinical decisions.
<PAST_CLINICAL_DIMENSIONS> is a list of dicts with "name" fields, representing the clinical dimensions that have already been assessed in the patient interview.
<CLINICAL_INTERVIEW_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the clinical interview between the physician and the patient up to this point.

Task:
Identify a new clinical dimension relevant to the <CLINICAL_QUESTION> that has not been previously identified in <PAST_CLINICAL_DIMENSIONS>. Use insights from the <CLINICAL_INTERVIEW_LOG> to guide your choice — the patient's answers may reveal the need to investigate additional clinical factors.

What to generate:
- reason: a short one-sentence explanation of why this clinical factor is important for narrowing the diagnosis.
- name: a short, specific clinical label for this dimension.
- values: a list of clinically plausible values, no larger than {max_num_values_per_dim}, that this dimension could take.

Constraints:
- Use ONLY the information provided in <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, <PAST_CLINICAL_DIMENSIONS>, and <CLINICAL_INTERVIEW_LOG>.
- Do NOT answer the <CLINICAL_QUESTION> itself. Focus ONLY on identifying a new clinical dimension.
- Do NOT rewrite or restate the <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, <PAST_CLINICAL_DIMENSIONS>, or <CLINICAL_INTERVIEW_LOG>.
- The generated dimension name must not be the same as any of the names in <PAST_CLINICAL_DIMENSIONS>.

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

EXPAND_DIMENSION_PRIORS_SYSTEM_PROMPT="""You are an experienced physician forming a clinical hypothesis about a newly identified clinical factor, taking into account the patient's baseline information and what has been revealed during the clinical interview so far.""",

EXPAND_DIMENSION_PRIORS_USER_PROMPT="""
<CLINICAL_QUESTION>
{ambiguous_prompt}
</CLINICAL_QUESTION>

<PATIENT_INFORMATION>
{meta_context}
</PATIENT_INFORMATION>

<CLINICAL_INTERVIEW_LOG>
{conversation_log}
</CLINICAL_INTERVIEW_LOG>

<DIMENSION_NAME>
{dimension_name}
</DIMENSION_NAME>

<DIMENSION_VALUE>
{dimension_value}
</DIMENSION_VALUE>

Definition:
A clinical dimension is a specific clinical factor where different values would point toward different diagnoses or clinical decisions.
<CLINICAL_INTERVIEW_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the clinical interview between the physician and the patient up to this point.

Task:
Given <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, <CLINICAL_INTERVIEW_LOG>, and a specific clinical dimension defined by <DIMENSION_NAME> and <DIMENSION_VALUE>, judge how likely it is that the <DIMENSION_NAME> takes on the value <DIMENSION_VALUE>.

What to generate:
- reason: a short one-sentence explanation of why the <DIMENSION_NAME> is likely, unlikely, or neutral to take on the value <DIMENSION_VALUE>.
- label: one of "likely", "unlikely", or "neutral" according to the following definitions:
    - likely: <DIMENSION_VALUE> is explicitly stated, strongly implied, or is the most natural clinical assumption given the patient information in <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, and <CLINICAL_INTERVIEW_LOG>.
    - neutral: <DIMENSION_VALUE> is clinically plausible but not implied or supported by specific evidence in the <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, or <CLINICAL_INTERVIEW_LOG>.
    - unlikely: <DIMENSION_VALUE> is contradicted by the <CLINICAL_QUESTION>, <PATIENT_INFORMATION> or <CLINICAL_INTERVIEW_LOG>, or would require assumptions that are inconsistent with the patient's presentation.

Constraints:
- Use ONLY the information provided in <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, and <CLINICAL_INTERVIEW_LOG>.
- Do NOT answer the <CLINICAL_QUESTION> itself. Focus ONLY on judging the likelihood of the dimension value.
- Do NOT rewrite or restate the <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, or <CLINICAL_INTERVIEW_LOG>.
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

EXPAND_QUESTIONS_SYSTEM_PROMPT="""You are an experienced physician preparing additional clinical questions for a patient interview based on a newly identified clinical factor and unresolved aspects of the diagnosis.""",

EXPAND_QUESTIONS_USER_PROMPT="""
<CLINICAL_QUESTION>
{ambiguous_prompt}
</CLINICAL_QUESTION>

<PATIENT_INFORMATION>
{meta_context}
</PATIENT_INFORMATION>

<CLINICAL_INTERVIEW_LOG>
{conversation_log}
</CLINICAL_INTERVIEW_LOG>

<NEW_CLINICAL_DIMENSION>
{new_dimension_with_values}
</NEW_CLINICAL_DIMENSION>

<UNRESOLVED_CLINICAL_DIMENSIONS>
{high_uncertainty_dimensions_with_values}
</UNRESOLVED_CLINICAL_DIMENSIONS>

Definition:
A clinical dimension is a specific clinical factor where different values would point toward different diagnoses or clinical decisions.
<CLINICAL_INTERVIEW_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the clinical interview between the physician and the patient up to this point.
<NEW_CLINICAL_DIMENSION> is a dict with "name" and "values" fields, representing the newly identified clinical factor along with its possible values.
<UNRESOLVED_CLINICAL_DIMENSIONS> is a list of dicts with "name" and "values" fields, representing the clinical dimensions that currently have the highest diagnostic uncertainty. They do not include the new dimension in <NEW_CLINICAL_DIMENSION>.

Task:
Given <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, <CLINICAL_INTERVIEW_LOG>, a newly identified clinical dimension in <NEW_CLINICAL_DIMENSION>, and the most uncertain dimensions in <UNRESOLVED_CLINICAL_DIMENSIONS>, generate clinical questions to ask the patient that would help narrow the diagnosis by targeting the new dimension and/or the unresolved dimensions.

What to generate:
Generate at most {max_new_questions_per_round} clinical questions. For each question, provide:
- reason: a short one-sentence explanation of why this question would help narrow the diagnosis.
- question: the text of the clinical question to ask the patient.
- choices: a list of multiple-choice answer options for the question, no larger than {max_choices_per_question}.

Constraints:
- Use ONLY the information provided in <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, <CLINICAL_INTERVIEW_LOG>, <NEW_CLINICAL_DIMENSION>, and <UNRESOLVED_CLINICAL_DIMENSIONS>.
- Do NOT answer the <CLINICAL_QUESTION> itself. Focus ONLY on generating clinical questions.
- Do NOT rewrite or restate the <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, <CLINICAL_INTERVIEW_LOG>, <NEW_CLINICAL_DIMENSION>, or <UNRESOLVED_CLINICAL_DIMENSIONS>.
- Each question must be designed to elicit information about the new dimension in <NEW_CLINICAL_DIMENSION> and/or the unresolved dimensions in <UNRESOLVED_CLINICAL_DIMENSIONS>.
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

FINAL_ANSWER_SYSTEM_PROMPT="""You are an experienced physician concluding a clinical assessment. Based on all the clinical information gathered from the patient interview and your analysis of the clinical dimensions, you must now provide your diagnosis or clinical decision.""",

FINAL_ANSWER_WITHOUT_CHOICES_USER_PROMPT="""
<CLINICAL_QUESTION>
{ambiguous_prompt}
</CLINICAL_QUESTION>

<PATIENT_INFORMATION>
{meta_context}
</PATIENT_INFORMATION>

<CLINICAL_INTERVIEW_LOG>
{conversation_log}
</CLINICAL_INTERVIEW_LOG>

<CLINICAL_ASSESSMENT>
{map_state}
</CLINICAL_ASSESSMENT>

Definition:
<CLINICAL_INTERVIEW_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the full history of the clinical interview between the physician and the patient.
A clinical dimension is a specific clinical factor where different values would point toward different diagnoses or clinical decisions.
<CLINICAL_ASSESSMENT> is a structured representation of the physician's current understanding of the patient's condition, where each clinical dimension is mapped to its most likely value. This represents the physician's best assessment of the true clinical state based on the interview so far.

Task:
Given <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, <CLINICAL_INTERVIEW_LOG>, and <CLINICAL_ASSESSMENT>, provide your diagnosis or clinical decision.

What to generate:
- reason: a short one-sentence explanation of why this is the correct diagnosis or clinical decision given the clinical evidence in <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, <CLINICAL_INTERVIEW_LOG>, and <CLINICAL_ASSESSMENT>.
- final_answer: your diagnosis or clinical decision.

Constraints:
- The final answer must be consistent with the clinical evidence in <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, <CLINICAL_INTERVIEW_LOG>, and <CLINICAL_ASSESSMENT>.
- Do NOT rewrite or restate the <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, <CLINICAL_INTERVIEW_LOG>, or <CLINICAL_ASSESSMENT>.
- answer must be a natural language answer to the <CLINICAL_QUESTION>.

Output format:
Return STRICT JSON only with the following schema:
{{
    "reason": string,
    "final_answer": string
}}
""",

FINAL_ANSWER_WITH_CHOICES_USER_PROMPT="""
<CLINICAL_QUESTION>
{ambiguous_prompt}
</CLINICAL_QUESTION>

<PATIENT_INFORMATION>
{meta_context}
</PATIENT_INFORMATION>

<CLINICAL_INTERVIEW_LOG>
{conversation_log}
</CLINICAL_INTERVIEW_LOG>

<CLINICAL_ASSESSMENT>
{map_state}
</CLINICAL_ASSESSMENT>

<DIAGNOSTIC_OPTIONS_WITH_IDS>
{possible_answers_with_ids}
</DIAGNOSTIC_OPTIONS_WITH_IDS>

Definition:
<CLINICAL_INTERVIEW_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the full history of the clinical interview between the physician and the patient.
A clinical dimension is a specific clinical factor where different values would point toward different diagnoses or clinical decisions.
<CLINICAL_ASSESSMENT> is a structured representation of the physician's current understanding of the patient's condition, where each clinical dimension is mapped to its most likely value. This represents the physician's best assessment of the true clinical state based on the interview so far.
<DIAGNOSTIC_OPTIONS_WITH_IDS> is a list of dicts with "id" and "value" fields. Each dict corresponds to a possible diagnosis or clinical decision.
Let possible_answers[i] be the i-th element of <DIAGNOSTIC_OPTIONS_WITH_IDS>.

Task:
Given <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, <CLINICAL_INTERVIEW_LOG>, and <CLINICAL_ASSESSMENT>, select the correct diagnosis or clinical decision from <DIAGNOSTIC_OPTIONS_WITH_IDS>.

What to generate:
- reason: a short one-sentence explanation of why this is the correct diagnosis or clinical decision given the clinical evidence in <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, <CLINICAL_INTERVIEW_LOG>, and <CLINICAL_ASSESSMENT>.
- final_answer_id: the id of the choice in <DIAGNOSTIC_OPTIONS_WITH_IDS> that you are selecting as your diagnosis or clinical decision.

Constraints:
- The final answer must be consistent with the clinical evidence in <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, <CLINICAL_INTERVIEW_LOG>, and <CLINICAL_ASSESSMENT>.
- Do NOT rewrite or restate the <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, <CLINICAL_INTERVIEW_LOG>, or <CLINICAL_ASSESSMENT>.
- final_answer_id must be one of the ids (i.e possible_answers[i]["id"]) provided in <DIAGNOSTIC_OPTIONS_WITH_IDS>.

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
                                                                                                                                                                                                                                                
ANSWER_LIKELIHOOD_SYSTEM_PROMPT="""You are an experienced physician evaluating how likely each candidate diagnosis or clinical decision is to be correct under different assumptions about the patient's clinical state.""",                      

ANSWER_LIKELIHOOD_USER_PROMPT="""
<CLINICAL_QUESTION>
{ambiguous_prompt}
</CLINICAL_QUESTION>

<PATIENT_INFORMATION>
{meta_context}
</PATIENT_INFORMATION>

<DIMENSION_NAME>
{dimension_name}
</DIMENSION_NAME>

<DIMENSION_VALUES_WITH_IDS>
{dimension_values_with_ids}
</DIMENSION_VALUES_WITH_IDS>

<DIAGNOSTIC_OPTIONS_WITH_IDS>
{possible_answers_with_ids}
</DIAGNOSTIC_OPTIONS_WITH_IDS>

Definition:
A clinical dimension is a specific clinical factor where different values would point toward different diagnoses or clinical decisions.
<DIMENSION_VALUES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a possible value that the <DIMENSION_NAME> could take.
<DIAGNOSTIC_OPTIONS_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a candidate diagnosis or clinical decision for the <CLINICAL_QUESTION>.
Let values[i] be the i-th element of <DIMENSION_VALUES_WITH_IDS>.
Let answers[j] be the j-th element of <DIAGNOSTIC_OPTIONS_WITH_IDS>.

Task (row-major order):
For i = 0..len(values)-1:
For j = 0..len(answers)-1:
    - Assume the patient's true clinical state is <DIMENSION_NAME> = values[i]["text"].
    - Judge how likely it is that answers[j]["text"] is the correct diagnosis or clinical decision for <CLINICAL_QUESTION> under that assumption.

What to generate:
For i = 0..len(values)-1:
For j = 0..len(answers)-1:
    - answer_id: the id of the candidate diagnosis being evaluated, i.e. answers[j]["id"]
    - dimension_value_id: the id of the dimension value being evaluated, i.e. values[i]["id"]
    - reason: a short one-sentence explanation of why answers[j] is likely/neutral/unlikely to be the correct diagnosis given <DIMENSION_NAME> = values[i]["text"].
    - label: one of "likely", "neutral", or "unlikely" according to the following definitions:
        - "likely": Given <DIMENSION_NAME> = values[i]["text"] and the patient information in <PATIENT_INFORMATION>, answers[j]["text"] is the expected correct diagnosis or clinical decision for <CLINICAL_QUESTION>.
        - "neutral": Given <DIMENSION_NAME> = values[i]["text"] and the patient information in <PATIENT_INFORMATION>, answers[j]["text"] is a plausible diagnosis but not specifically supported; there is insufficient clinical evidence to say it is more or less correct than other options.
        - "unlikely": Given <DIMENSION_NAME> = values[i]["text"] and the patient information in <PATIENT_INFORMATION>, answers[j]["text"] is not expected to be the correct diagnosis or clinical decision for <CLINICAL_QUESTION>.

Constraints:
- Use ONLY the information provided in <CLINICAL_QUESTION>, <PATIENT_INFORMATION>, and the assumed dimension value.
- Do NOT answer the <CLINICAL_QUESTION> itself. Focus ONLY on judging how likely each candidate diagnosis is to be correct under different assumptions about the clinical dimension.
- Do NOT rewrite or restate the <CLINICAL_QUESTION> or <PATIENT_INFORMATION>.
- label must be one of "likely", "neutral", or "unlikely".
- The output must include an entry for every combination of dimension value and candidate diagnosis.

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
            ... // one object for each candidate diagnosis
        ],
        ... // one array for each dimension value
    ]
}}
The "evaluations" field must contain exactly {num_dimension_values} arrays (one per dimension value).
Each inner array must contain exactly {num_possible_answers} objects (one per candidate diagnosis).
""",
)