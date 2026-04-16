from prompts.base import PromptSet

PROMPT_SET = PromptSet(

# -------------------------------------------------------------- #
# USER SIMULATOR PROMPTS
# -------------------------------------------------------------- #

USER_SIMULATOR_SYSTEM_PROMPT="""You are the host of a thinking puzzle (situation puzzle). You know the hidden explanation behind the puzzle. Answer the solver's questions truthfully based on the solution. Never reveal the full solution directly; only confirm or deny specific aspects when asked.""",

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
You are the host of a thinking puzzle. A solver is asking you <QUESTION> to try to figure out the hidden explanation. Answer based on your knowledge of the puzzle solution described in <USER_CONTEXT>. Give a truthful answer — do not mislead, but do not give extra information.

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
You are the host of a thinking puzzle. A solver is asking you <QUESTION> to try to figure out the hidden explanation. Answer based on your knowledge of the puzzle solution described in <USER_CONTEXT>. Give a truthful answer — do not mislead, but do not volunteer extra information.

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

INITIAL_DIMENSIONS_SYSTEM_PROMPT="""You are an expert thinking puzzle solver. Your goal is to identify the hidden dimensions of the puzzle — the unstated aspects of the scenario whose true values would explain the strange situation presented.""",

INITIAL_DIMENSIONS_USER_PROMPT="""
<PUZZLE>
{ambiguous_prompt}
</PUZZLE>

<PUZZLE_CONTEXT>
{meta_context}
</PUZZLE_CONTEXT>

Task:
Identify the hidden dimensions of the thinking puzzle in <PUZZLE> that must be uncovered to explain the strange scenario.

Definition:
A puzzle dimension is a hidden aspect of the scenario — such as a non-obvious word meaning, an unstated context, a surprising identity, or an unusual causal mechanism — where knowing its true value would explain the puzzle. Once the dimension's value is known, the puzzle moves toward a single coherent explanation.

What to generate:
- Produce a minimal, non-overlapping set of puzzle dimensions.
- Produce exactly {num_initial_dims} puzzle dimensions.
- Each dimension must correspond to a distinct hidden aspect of the puzzle.
- If <PUZZLE_CONTEXT> already resolves a dimension, do not include it.
- If <PUZZLE_CONTEXT> proposes some puzzle dimensions, use them.

For each puzzle dimension, provide :
- reason: a short one-sentence explanation of why this dimension is a key unknown in the puzzle.
- name: a short, specific label
- values: a list of plausible interpretations, no larger than {max_num_values_per_dim}, that this dimension could take.

Constraints:
- Use ONLY the information provided in <PUZZLE> and <PUZZLE_CONTEXT>.
- Do NOT solve the <PUZZLE> itself. Focus ONLY on identifying hidden dimensions.
- Do NOT rewrite or restate the <PUZZLE> or <PUZZLE_CONTEXT>.

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

INITIAL_PRIORS_SYSTEM_PROMPT="""You are an expert thinking puzzle solver forming initial hypotheses about the hidden aspects of a puzzle based on the scenario description.""",

INITIAL_PRIORS_USER_PROMPT="""
<PUZZLE>
{ambiguous_prompt}
</PUZZLE>

<PUZZLE_CONTEXT>
{meta_context}
</PUZZLE_CONTEXT>

<DIMENSION_NAME>
{dimension_name}
</DIMENSION_NAME>

<DIMENSION_VALUE>
{dimension_value}
</DIMENSION_VALUE>

Task:
Given <PUZZLE> and <PUZZLE_CONTEXT>, judge how likely the puzzle dimension <DIMENSION_NAME> takes on the value <DIMENSION_VALUE>.

What to generate:
- reason: a short one-sentence explanation of why the <DIMENSION_NAME> is likely, unlikely, or neutral to take on the value <DIMENSION_VALUE>.
- label: one of "likely", "unlikely", or "neutral" according to the following definitions:
    - likely: <DIMENSION_VALUE> is explicitly suggested by, strongly implied by, or is the most natural interpretation given the clues in the <PUZZLE> and <PUZZLE_CONTEXT>.
    - neutral: <DIMENSION_VALUE> is a plausible interpretation but not implied or supported by specific clues in the <PUZZLE> or <PUZZLE_CONTEXT>.
    - unlikely: <DIMENSION_VALUE> is contradicted by the <PUZZLE> or <PUZZLE_CONTEXT>, or would require assumptions that are inconsistent with the scenario.

Constraints:
- Use ONLY the information provided in <PUZZLE> and <PUZZLE_CONTEXT>.
- Do NOT solve the <PUZZLE> itself. Focus ONLY on judging the likelihood of the dimension value.
- Do NOT rewrite or restate the <PUZZLE> or <PUZZLE_CONTEXT>.
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

INITIAL_QUESTIONS_SYSTEM_PROMPT="""You are an expert thinking puzzle solver. Generate clarifying questions to ask the puzzle host that will help you uncover the hidden explanation. Good puzzle questions test specific hypotheses about what is really going on in the scenario.""",

INITIAL_QUESTIONS_USER_PROMPT="""
<PUZZLE>
{ambiguous_prompt}
</PUZZLE>

<PUZZLE_CONTEXT>
{meta_context}
</PUZZLE_CONTEXT>

<PUZZLE_DIMENSIONS>
{dimensions_with_values}
</PUZZLE_DIMENSIONS>

Task:
Given <PUZZLE>, <PUZZLE_CONTEXT>, and <PUZZLE_DIMENSIONS>, generate exactly {num_initial_questions} clarifying questions to ask the puzzle host that would help uncover the hidden explanation. Each question should target one or more <PUZZLE_DIMENSIONS> and have "yes"/"no" answers.

Definition:
<PUZZLE_DIMENSIONS> is a list of puzzle dimensions, where each dimension has a name and a list of possible values it could take. A puzzle dimension is a hidden aspect of the scenario where knowing its true value would explain the puzzle.

What to generate:
For each of the {num_initial_questions} questions, provide:
- reason: a short one-sentence explanation of why this question would help solve the puzzle.
- question: the text of the clarifying question to ask the puzzle host.
- choices: ["yes", "no"] as the {max_choices_per_question} multiple-choice answer options for the question.

Constraints:
- Use ONLY the information provided in <PUZZLE>, <PUZZLE_CONTEXT>, and <PUZZLE_DIMENSIONS>.
- Do NOT solve the <PUZZLE> itself. Focus ONLY on generating clarifying questions.
- Do NOT rewrite or restate the <PUZZLE> or <PUZZLE_CONTEXT>.
- Each question must be designed to elicit information about one or more of the dimensions in <PUZZLE_DIMENSIONS>.
- Each question must have "yes"/"no" answers.
- Keep each question short: at most 20 words.
- Keep each reason short: at most 15 words.

Output format:
Return STRICT JSON only with the following schema:
{{
    "questions": [
        {{
            "reason": string,
            "question": string,
            "choices": ["yes", "no"]
        }},
        ...
    ]
}}
""",

# -------------------------------------------------------------- #
# LIKELIHOOD PROMPTS
# -------------------------------------------------------------- #

LIKELIHOOD_SYSTEM_PROMPT="""You are an expert thinking puzzle analyst. Evaluate how a puzzle host who knows the hidden explanation would likely respond to a solver's question under different assumptions about the puzzle's hidden aspects. The host answers truthfully, without giving extra information.""",

LIKELIHOOD_WITHOUT_HISTORY_USER_PROMPT="""
<PUZZLE>
{ambiguous_prompt}
</PUZZLE>

<PUZZLE_CONTEXT>
{meta_context}
</PUZZLE_CONTEXT>

<HOST_INFO>
{user_info}
</HOST_INFO>

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
A puzzle dimension is a hidden aspect of the scenario where knowing its true value would explain the puzzle.
<DIMENSION_VALUES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a possible value that the <DIMENSION_NAME> could take.
<QUESTION_CHOICES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a multiple-choice answer option for the question.
Let values[i] be the i-th element of <DIMENSION_VALUES_WITH_IDS>.
Let choices[j] be the j-th element of <QUESTION_CHOICES_WITH_IDS>.

Task (row-major order):
For i = 0..len(values)-1:
  For j = 0..len(choices)-1:
    - Assume the hidden explanation is such that <DIMENSION_NAME> = values[i]["text"].
    - Impersonate the puzzle host described in <HOST_INFO>, who knows the hidden explanation.
    - Judge how likely it is that this host would answer the question <QUESTION> with choices[j]["text"] under that assumption. The host answers truthfully.

What to generate:
For i = 0..len(values)-1:
  For j = 0..len(choices)-1:
    - question_choice_id: the id of the question choice being evaluated, i.e. choices[j]["id"]
    - dimension_value_id: the id of the dimension value being evaluated, i.e. values[i]["id"]
    - reason: a short one-sentence explanation of why choices[j] is labeled likely/neutral/unlikely and why the other two labels were not chosen.
    - label: one of "likely", "neutral", or "unlikely" according to the following definitions:
      - "likely": Given <DIMENSION_NAME> = values[i]["text"] and the host acts according to <HOST_INFO>, the host is expected to give choices[j]["text"] for <QUESTION>.
      - "neutral": Given <DIMENSION_NAME> = values[i]["text"] and the host acts according to <HOST_INFO>, choices[j]["text"] is plausible but not specifically supported; there is insufficient evidence to say that the host would or would not prefer it over other choices.
      - "unlikely": Given <DIMENSION_NAME> = values[i]["text"] and the host acts according to <HOST_INFO>, the host is not expected to give choices[j]["text"] for <QUESTION>.

Constraints:
- Use ONLY the information provided in <PUZZLE>, <PUZZLE_CONTEXT>, and <HOST_INFO>.
- Do NOT solve the <PUZZLE> itself. Focus ONLY on judging the likelihood of the question choices under different assumptions about the dimension value.
- Do NOT rewrite or restate the <PUZZLE>, <PUZZLE_CONTEXT>, or <HOST_INFO>.
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
<PUZZLE>
{ambiguous_prompt}
</PUZZLE>

<PUZZLE_CONTEXT>
{meta_context}
</PUZZLE_CONTEXT>

<HOST_INFO>
{user_info}
</HOST_INFO>

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
A puzzle dimension is a hidden aspect of the scenario where knowing its true value would explain the puzzle.
<DIMENSION_VALUES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a possible value that the <DIMENSION_NAME> could take.
<QUESTION_CHOICES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a multiple-choice answer option for the question.
<CONVERSATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the conversation between the solver and the host up to this point. This information may provide additional clues.
Let values[i] be the i-th element of <DIMENSION_VALUES_WITH_IDS>.
Let choices[j] be the j-th element of <QUESTION_CHOICES_WITH_IDS>.

Task (row-major order):
For i = 0..len(values)-1:
  For j = 0..len(choices)-1:
    - Assume the hidden explanation is such that <DIMENSION_NAME> = values[i]["text"].
    - Impersonate the puzzle host described in <HOST_INFO>, who knows the hidden explanation.
    - Judge how likely it is that this host would answer the question <QUESTION> with choices[j]["text"] under that assumption. The host answers truthfully.

What to generate:
For i = 0..len(values)-1:
  For j = 0..len(choices)-1:
    - question_choice_id: the id of the question choice being evaluated, i.e. choices[j]["id"]
    - dimension_value_id: the id of the dimension value being evaluated, i.e. values[i]["id"]
    - reason: a short one-sentence explanation of why choices[j] is labeled likely/neutral/unlikely and why the other two labels were not chosen.
    - label: one of "likely", "neutral", or "unlikely" according to the following definitions:
      - "likely": Given <DIMENSION_NAME> = values[i]["text"] and the host acts according to <HOST_INFO>, the host is expected to give choices[j]["text"] for <QUESTION>.
      - "neutral": Given <DIMENSION_NAME> = values[i]["text"] and the host acts according to <HOST_INFO>, choices[j]["text"] is plausible but not specifically supported; there is insufficient evidence to say that the host would or would not prefer it over other choices.
      - "unlikely": Given <DIMENSION_NAME> = values[i]["text"] and the host acts according to <HOST_INFO>, the host is not expected to give choices[j]["text"] for <QUESTION>.

Constraints:
- Use ONLY the information provided in <PUZZLE>, <PUZZLE_CONTEXT>, <HOST_INFO>, and <CONVERSATION_LOG>.
- Do NOT solve the <PUZZLE> itself. Focus ONLY on judging the likelihood of the question choices under different assumptions about the dimension value.
- Do NOT rewrite or restate the <PUZZLE>, <PUZZLE_CONTEXT>, <HOST_INFO>, or <CONVERSATION_LOG>.
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

SCORE_NATURAL_LANGUAGE_SYSTEM_PROMPT="""You are an expert thinking puzzle analyst. Your task is to judge how well a puzzle host's response maps to each of the predefined answer choices. The host gives truthful answers ("yes" or "no").""",

SCORE_NATURAL_LANGUAGE_USER_PROMPT="""
<QUESTION>
{question}
</QUESTION>

<CHOICES_WITH_IDS>
{choices_with_ids}
</CHOICES_WITH_IDS>

<HOST_ANSWER>
{user_answer}
</HOST_ANSWER>

Definition:
<CHOICES_WITH_IDS> is a list of dicts with "id" and "value" fields. Each dict corresponds to a multiple-choice answer option for the question.
Let choices[i] be the i-th element of <CHOICES_WITH_IDS>.

Task:
Judge how well the <HOST_ANSWER> maps to each of the choices in <CHOICES_WITH_IDS> for the question <QUESTION>.

What to generate:
For i = 0..len(choices)-1:
- choice_id: the id of the question choice being evaluated, i.e. choices[i]["id"]
- reason: a short one-sentence explanation of choices[i]["value"] is likely/neutral/unlikely given the <HOST_ANSWER>.
- label: one of "likely", "neutral", or "unlikely" according to the following definitions:
    -"likely": choices[i]["value"] aligns well with the <HOST_ANSWER> and fits it better than most other choices.
    -"neutral": choices[i]["value"] is neither clearly supported nor clearly contradicted by the <HOST_ANSWER>.
    -"unlikely": choices[i]["value"] fits the <HOST_ANSWER> worse than other choices, or conflicts with the meaning of the <HOST_ANSWER>.

Constraints:
- Use ONLY the information provided in <QUESTION>, <CHOICES_WITH_IDS>, and <HOST_ANSWER>.
- Do NOT answer the <QUESTION> itself. Focus ONLY on judging how well the <HOST_ANSWER> maps to the provided choices.
- Do NOT rewrite or restate the <QUESTION> or <HOST_ANSWER>.
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

EXPAND_DIMENSION_SYSTEM_PROMPT="""You are an expert thinking puzzle solver. The puzzle dimensions explored so far have not been sufficient to explain the puzzle, so you must identify a new hidden aspect of the scenario to investigate.""",

EXPAND_DIMENSION_USER_PROMPT="""
<PUZZLE>
{ambiguous_prompt}
</PUZZLE>

<PUZZLE_CONTEXT>
{meta_context}
</PUZZLE_CONTEXT>

<PAST_PUZZLE_DIMENSIONS>
{past_dimensions}
</PAST_PUZZLE_DIMENSIONS>

<CONVERSATION_LOG>
{conversation_log}
</CONVERSATION_LOG>

Definition:
A puzzle dimension is a hidden aspect of the scenario where knowing its true value would explain the puzzle.
<PAST_PUZZLE_DIMENSIONS> is a list of dicts with "name" fields, representing the puzzle dimensions that have already been explored.
<CONVERSATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the conversation between the solver and the host up to this point.

Task:
Identify a new puzzle dimension in <PUZZLE> that has not been previously identified in <PAST_PUZZLE_DIMENSIONS>. Use insights from the <CONVERSATION_LOG> to guide your choice.

What to generate:
- reason: a short one-sentence explanation of why this dimension is a key unknown in the puzzle.
- name: a short, specific label for this puzzle dimension.
- values: a list of plausible interpretations, no larger than {max_num_values_per_dim}, that this dimension could take.

Constraints:
- Use ONLY the information provided in <PUZZLE>, <PUZZLE_CONTEXT>, <PAST_PUZZLE_DIMENSIONS>, and <CONVERSATION_LOG>.
- Do NOT solve the <PUZZLE> itself. Focus ONLY on identifying a new puzzle dimension.
- Do NOT rewrite or restate the <PUZZLE>, <PUZZLE_CONTEXT>, <PAST_PUZZLE_DIMENSIONS>, or <CONVERSATION_LOG>.
- The generated dimension name must not be the same as any of the names in <PAST_PUZZLE_DIMENSIONS>.

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

EXPAND_DIMENSION_PRIORS_SYSTEM_PROMPT="""You are an expert thinking puzzle solver forming a hypothesis about a newly identified hidden aspect of the puzzle, taking into account both the puzzle scenario and what the host has revealed so far.""",

EXPAND_DIMENSION_PRIORS_USER_PROMPT="""
<PUZZLE>
{ambiguous_prompt}
</PUZZLE>

<PUZZLE_CONTEXT>
{meta_context}
</PUZZLE_CONTEXT>

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
A puzzle dimension is a hidden aspect of the scenario where knowing its true value would explain the puzzle.
<CONVERSATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the conversation between the solver and the host up to this point.

Task:
Given <PUZZLE>, <PUZZLE_CONTEXT>, <CONVERSATION_LOG>, and a specific puzzle dimension defined by <DIMENSION_NAME> and <DIMENSION_VALUE>, judge how likely it is that the <DIMENSION_NAME> takes on the value <DIMENSION_VALUE>.

What to generate:
- reason: a short one-sentence explanation of why the <DIMENSION_NAME> is likely, unlikely, or neutral to take on the value <DIMENSION_VALUE>.
- label: one of "likely", "unlikely", or "neutral" according to the following definitions:
    - likely: <DIMENSION_VALUE> is explicitly suggested by, strongly implied by, or is the most natural interpretation given the clues in the <PUZZLE>, <PUZZLE_CONTEXT>, and <CONVERSATION_LOG>.
    - neutral: <DIMENSION_VALUE> is plausible but not implied or supported by specific clues in the <PUZZLE>, <PUZZLE_CONTEXT>, or <CONVERSATION_LOG>.
    - unlikely: <DIMENSION_VALUE> is contradicted by the <PUZZLE>, <PUZZLE_CONTEXT> or <CONVERSATION_LOG>, or would require assumptions that are inconsistent with the scenario.

Constraints:
- Use ONLY the information provided in <PUZZLE>, <PUZZLE_CONTEXT>, and <CONVERSATION_LOG>.
- Do NOT solve the <PUZZLE> itself. Focus ONLY on judging the likelihood of the dimension value.
- Do NOT rewrite or restate the <PUZZLE>, <PUZZLE_CONTEXT>, or <CONVERSATION_LOG>.
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

EXPAND_QUESTIONS_SYSTEM_PROMPT="""You are an expert thinking puzzle solver generating new questions to ask the puzzle host. Focus on testing specific hypotheses about the hidden explanation, especially targeting newly discovered or still-uncertain aspects of the puzzle.""",

EXPAND_QUESTIONS_USER_PROMPT="""
<PUZZLE>
{ambiguous_prompt}
</PUZZLE>

<PUZZLE_CONTEXT>
{meta_context}
</PUZZLE_CONTEXT>

<CONVERSATION_LOG>
{conversation_log}
</CONVERSATION_LOG>

<NEW_PUZZLE_DIMENSION>
{new_dimension_with_values}
</NEW_PUZZLE_DIMENSION>

<UNCERTAIN_PUZZLE_DIMENSIONS>
{high_uncertainty_dimensions_with_values}
</UNCERTAIN_PUZZLE_DIMENSIONS>

Definition:
A puzzle dimension is a hidden aspect of the scenario where knowing its true value would explain the puzzle.
<CONVERSATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the history of the conversation between the solver and the host up to this point.
<NEW_PUZZLE_DIMENSION> is a dict with "name" and "values" fields, representing the newly identified hidden aspect of the puzzle along with its possible interpretations.
<UNCERTAIN_PUZZLE_DIMENSIONS> is a list of dicts with "name" and "values" fields, representing the puzzle dimensions that currently have the highest uncertainty. They do not include the new dimension in <NEW_PUZZLE_DIMENSION>.

Task:
Given <PUZZLE>, <PUZZLE_CONTEXT>, <CONVERSATION_LOG>, a newly identified puzzle dimension in <NEW_PUZZLE_DIMENSION>, and the most uncertain dimensions in <UNCERTAIN_PUZZLE_DIMENSIONS>, generate questions to ask the puzzle host that would help uncover the hidden explanation by targeting the new dimension and/or the uncertain dimensions.

What to generate:
Generate at most {max_new_questions_per_round} clarifying questions. For each question, provide:
- reason: a short one-sentence explanation of why this question would help solve the puzzle.
- question: the text of the clarifying question to ask the puzzle host.
- choices: ["yes", "no"] as the {max_choices_per_question} multiple-choice answer options for the question.

Constraints:
- Use ONLY the information provided in <PUZZLE>, <PUZZLE_CONTEXT>, <CONVERSATION_LOG>, <NEW_PUZZLE_DIMENSION>, and <UNCERTAIN_PUZZLE_DIMENSIONS>.
- Do NOT solve the <PUZZLE> itself. Focus ONLY on generating clarifying questions.
- Do NOT rewrite or restate the <PUZZLE>, <PUZZLE_CONTEXT>, <CONVERSATION_LOG>, <NEW_PUZZLE_DIMENSION>, or <UNCERTAIN_PUZZLE_DIMENSIONS>.
- Each question must be designed to elicit information about the new dimension in <NEW_PUZZLE_DIMENSION> and/or the uncertain dimensions in <UNCERTAIN_PUZZLE_DIMENSIONS>.
- Each question must have multiple-choice answers.
- Generate at most {max_new_questions_per_round} questions.
- Keep each question short: at most 20 words.
- Keep each reason short: at most 15 words.

Output format:
Return STRICT JSON only with the following schema:
{{
    "questions": [
        {{
            "reason": string,
            "question": string,
            "choices": ["yes", "no"]
        }},
        ...
    ]
}}
""",

# -------------------------------------------------------------- #
# FINAL ANSWER PROMPTS
# -------------------------------------------------------------- #

FINAL_ANSWER_SYSTEM_PROMPT="""You are an expert thinking puzzle solver. Based on all the clues gathered from the puzzle host's responses and your analysis of the puzzle dimensions, you must now provide the hidden explanation of the puzzle.""",

FINAL_ANSWER_WITHOUT_CHOICES_USER_PROMPT="""
<PUZZLE>
{ambiguous_prompt}
</PUZZLE>

<PUZZLE_CONTEXT>
{meta_context}
</PUZZLE_CONTEXT>

<CONVERSATION_LOG>
{conversation_log}
</CONVERSATION_LOG>

<PUZZLE_SOLUTION_STATE>
{map_state}
</PUZZLE_SOLUTION_STATE>

Definition:
<CONVERSATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the full history of the conversation between the solver and the puzzle host.
A puzzle dimension is a hidden aspect of the scenario where knowing its true value would explain the puzzle.
<PUZZLE_SOLUTION_STATE> is a structured representation of the solver's current understanding of the puzzle, where each puzzle dimension is mapped to its most likely value. This represents the solver's best guess of the hidden explanation based on the clues gathered so far.

Task:
Given <PUZZLE>, <PUZZLE_CONTEXT>, <CONVERSATION_LOG>, and <PUZZLE_SOLUTION_STATE>, provide the hidden explanation of the puzzle.

What to generate:
- reason: a short one-sentence explanation of why this is the correct explanation given the clues in <PUZZLE>, <PUZZLE_CONTEXT>, <CONVERSATION_LOG>, and <PUZZLE_SOLUTION_STATE>.
- final_answer: the hidden explanation of the puzzle.

Constraints:
- The final answer must be consistent with the clues in <PUZZLE>, <PUZZLE_CONTEXT>, <CONVERSATION_LOG>, and <PUZZLE_SOLUTION_STATE>.
- Do NOT rewrite or restate the <PUZZLE>, <PUZZLE_CONTEXT>, <CONVERSATION_LOG>, or <PUZZLE_SOLUTION_STATE>.
- answer must be a natural language explanation of the puzzle.

Output format:
Return STRICT JSON only with the following schema:
{{
    "reason": string,
    "final_answer": string
}}
""",

FINAL_ANSWER_WITH_CHOICES_USER_PROMPT="""
<PUZZLE>
{ambiguous_prompt}
</PUZZLE>

<PUZZLE_CONTEXT>
{meta_context}
</PUZZLE_CONTEXT>

<CONVERSATION_LOG>
{conversation_log}
</CONVERSATION_LOG>

<PUZZLE_SOLUTION_STATE>
{map_state}
</PUZZLE_SOLUTION_STATE>

<POSSIBLE_EXPLANATIONS_WITH_IDS>
{possible_answers_with_ids}
</POSSIBLE_EXPLANATIONS_WITH_IDS>

Definition:
<CONVERSATION_LOG> is a list of dicts with "question_text", "user_name", and "user_answer" fields, representing the full history of the conversation between the solver and the puzzle host.
A puzzle dimension is a hidden aspect of the scenario where knowing its true value would explain the puzzle.
<PUZZLE_SOLUTION_STATE> is a structured representation of the solver's current understanding of the puzzle, where each puzzle dimension is mapped to its most likely value. This represents the solver's best guess of the hidden explanation based on the clues gathered so far.
<POSSIBLE_EXPLANATIONS_WITH_IDS> is a list of dicts with "id" and "value" fields. Each dict corresponds to a possible explanation of the puzzle.
Let possible_answers[i] be the i-th element of <POSSIBLE_EXPLANATIONS_WITH_IDS>.

Task:
Given <PUZZLE>, <PUZZLE_CONTEXT>, <CONVERSATION_LOG>, and <PUZZLE_SOLUTION_STATE>, select the correct explanation from <POSSIBLE_EXPLANATIONS_WITH_IDS>.

What to generate:
- reason: a short one-sentence explanation of why this is the correct explanation given the clues in <PUZZLE>, <PUZZLE_CONTEXT>, <CONVERSATION_LOG>, and <PUZZLE_SOLUTION_STATE>.
- final_answer_id: the id of the explanation in <POSSIBLE_EXPLANATIONS_WITH_IDS> that you are selecting as the hidden explanation of the puzzle.

Constraints:
- The final answer must be consistent with the clues in <PUZZLE>, <PUZZLE_CONTEXT>, <CONVERSATION_LOG>, and <PUZZLE_SOLUTION_STATE>.
- Do NOT rewrite or restate the <PUZZLE>, <PUZZLE_CONTEXT>, <CONVERSATION_LOG>, or <PUZZLE_SOLUTION_STATE>.
- final_answer_id must be one of the ids (i.e possible_answers[i]["id"]) provided in <POSSIBLE_EXPLANATIONS_WITH_IDS>.

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
                                                                                                                                                                                                                                                
ANSWER_LIKELIHOOD_SYSTEM_PROMPT="""You are an expert thinking puzzle analyst. Evaluate how likely each candidate explanation is to be the correct hidden explanation of the puzzle under different assumptions about the puzzle's hidden dimensions.""",                                                                                                                                                                                                                                   
                
ANSWER_LIKELIHOOD_USER_PROMPT="""
<PUZZLE>
{ambiguous_prompt}
</PUZZLE>

<PUZZLE_CONTEXT>
{meta_context}
</PUZZLE_CONTEXT>

<DIMENSION_NAME>
{dimension_name}
</DIMENSION_NAME>

<DIMENSION_VALUES_WITH_IDS>
{dimension_values_with_ids}
</DIMENSION_VALUES_WITH_IDS>

<POSSIBLE_EXPLANATIONS_WITH_IDS>
{possible_answers_with_ids}
</POSSIBLE_EXPLANATIONS_WITH_IDS>

Definition:
A puzzle dimension is a hidden aspect of the scenario where knowing its true value would explain the puzzle.
<DIMENSION_VALUES_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a possible value that the <DIMENSION_NAME> could take.
<POSSIBLE_EXPLANATIONS_WITH_IDS> is a list of dicts with "id" and "text" fields. Each dict corresponds to a candidate hidden explanation of the <PUZZLE>.
Let values[i] be the i-th element of <DIMENSION_VALUES_WITH_IDS>.
Let answers[j] be the j-th element of <POSSIBLE_EXPLANATIONS_WITH_IDS>.

Task (row-major order):
For i = 0..len(values)-1:
For j = 0..len(answers)-1:
    - Assume the hidden explanation is such that <DIMENSION_NAME> = values[i]["text"].
    - Judge how likely it is that answers[j]["text"] is the correct hidden explanation of the <PUZZLE> under that assumption.

What to generate:
For i = 0..len(values)-1:
For j = 0..len(answers)-1:
    - answer_id: the id of the candidate explanation being evaluated, i.e. answers[j]["id"]
    - dimension_value_id: the id of the dimension value being evaluated, i.e. values[i]["id"]
    - reason: a short one-sentence explanation of why answers[j] is likely/neutral/unlikely to be the correct hidden explanation given <DIMENSION_NAME> = values[i]["text"].
    - label: one of "likely", "neutral", or "unlikely" according to the following definitions:
        - "likely": Given <DIMENSION_NAME> = values[i]["text"] and the clues in <PUZZLE_CONTEXT>, answers[j]["text"] is the expected correct hidden explanation of the <PUZZLE>.
        - "neutral": Given <DIMENSION_NAME> = values[i]["text"] and the clues in <PUZZLE_CONTEXT>, answers[j]["text"] is a plausible explanation but not specifically supported; there is insufficient evidence to say it is more or less correct than other explanations.
        - "unlikely": Given <DIMENSION_NAME> = values[i]["text"] and the clues in <PUZZLE_CONTEXT>, answers[j]["text"] is not expected to be the correct hidden explanation of the <PUZZLE>.

Constraints:
- Use ONLY the information provided in <PUZZLE>, <PUZZLE_CONTEXT>, and the assumed dimension value.
- Do NOT solve the <PUZZLE> itself. Focus ONLY on judging how likely each candidate explanation is to be correct under different assumptions about the puzzle dimension.
- Do NOT rewrite or restate the <PUZZLE> or <PUZZLE_CONTEXT>.
- label must be one of "likely", "neutral", or "unlikely".
- The output must include an entry for every combination of dimension value and candidate explanation.

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
            ... // one object for each candidate explanation
        ],
        ... // one array for each dimension value
    ]
}}
The "evaluations" field must contain exactly {num_dimension_values} arrays (one per dimension value).
Each inner array must contain exactly {num_possible_answers} objects (one per candidate explanation).
""",
)
