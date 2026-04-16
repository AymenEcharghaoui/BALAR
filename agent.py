import shutil
import asyncio
from openai import AsyncOpenAI
import numpy as np
import torch
import json
import os
import time
from typing import List, Dict, Any, Tuple, Optional, Literal, Type, Union, Annotated
from dataclasses import dataclass, field
from pydantic import BaseModel, conlist, Field
from pathlib import Path
from prompts.base import PromptSet
from prompts.default import VERIFIER_SYSTEM_PROMPT, VERIFIER_USER_PROMPT, VERIFIER_CORRECTION_ADD
from dotenv import load_dotenv
load_dotenv()
# no variable name like AGENT_* must be used unless inside run_loop function

_OPENAI_AGENT_API_KEY = os.getenv("OPENAI_AGENT_API_KEY", "_")
_OPENAI_USER_API_KEY = os.getenv("OPENAI_USER_API_KEY", "_")
OPENAI_AGENT_CLIENT = AsyncOpenAI(api_key=_OPENAI_AGENT_API_KEY)
OPENAI_USER_CLIENT = AsyncOpenAI(api_key=_OPENAI_USER_API_KEY)

_VLLM_AGENT_BASE_URL = os.getenv("VLLM_AGENT_BASE_URL", "_")
_VLLM_USER_BASE_URL = os.getenv("VLLM_USER_BASE_URL", "_")
VLLM_AGENT_CLIENT = AsyncOpenAI(api_key="_", base_url=_VLLM_AGENT_BASE_URL)
VLLM_USER_CLIENT = AsyncOpenAI(api_key="_", base_url=_VLLM_USER_BASE_URL)

DEFAULT_MAX_RETRIES = 10
SEM = asyncio.Semaphore(200)

NON_EMPTY_STRING = Annotated[str, Field(min_length=1)]
@dataclass
class LLMConfig:
    model_name: str
    model_class: Literal["openai", "qwen", "llama"]
    reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh"]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 100_000
    top_p: Optional[float] = None

@dataclass
class AgentConfig:
    agent_model_config: LLMConfig
    user_model_config: LLMConfig
    max_rounds: int
    max_ask_rounds: int
    alpha: float
    num_initial_dims: int
    max_num_values_per_dim: int
    num_initial_questions: int
    max_choices_per_question: int
    max_new_questions_per_round: int
    num_top_uncertainty_dims: int
    expand_multiplier: float = 1. # if int, then allow for that many times more rounds before expanding
    max_total_states: int = 50000
    use_verifier: bool = False
    beta: float = 1.
    # each instance gets its own dict
    labels2probs: Dict[str, float] = field(default_factory=lambda: {
        "likely": 0.8,
        "neutral": 0.5,
        "unlikely": 0.2
    })

class VerifierScheme(BaseModel):                                                                                                                                                      
    is_valid: bool                                                                                                                                                                     
    feedback: str 

async def raw_inference(model_config: LLMConfig, system_prompt: str, user_prompt: str, scheme: Type[BaseModel], use_user: bool=False) -> Any:
    if model_config.model_class == "openai":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        client = OPENAI_USER_CLIENT if use_user else OPENAI_AGENT_CLIENT
        async with SEM:
            response = await client.responses.parse(
                model=model_config.model_name,
                input=messages,
                text_format=scheme,
                reasoning={
                    "effort": model_config.reasoning_effort
                },
                max_output_tokens=model_config.max_tokens,
            )
            return response.output_parsed
    elif model_config.model_class in ["qwen", "llama"]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        client = VLLM_USER_CLIENT if use_user else VLLM_AGENT_CLIENT
        async with SEM:
            response = await client.chat.completions.create(
                model=model_config.model_name,
                messages=messages,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
                max_tokens=model_config.max_tokens,
                # extra_body={
                #     "structured_outputs": {"json": scheme.model_json_schema()}
                # }
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": scheme.__name__,
                        "schema": scheme.model_json_schema()
                    },
                }
            )
            return scheme.model_validate_json(response.choices[0].message.content)

    else:
        raise NotImplementedError(f"Model class {model_config.model_class} not implemented yet.")

async def inference(model_config: LLMConfig, system_prompt: str, user_prompt: str, scheme: Type[BaseModel], use_verifier: bool = False, use_user: bool = False, verifier_log: Optional[List] = None) -> Any:
    response = await raw_inference(model_config, system_prompt, user_prompt, scheme, use_user=use_user)
    if not use_verifier:
        return response

    response_json = response.model_dump_json(indent=2)
    verifier_response = await raw_inference(
        model_config=model_config,
        system_prompt=VERIFIER_SYSTEM_PROMPT,
        user_prompt=VERIFIER_USER_PROMPT.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response_json
        ),
        scheme=VerifierScheme,
        use_user=use_user
    )

    if verifier_response.is_valid:
        if verifier_log is not None:
            verifier_log.append({"corrected": False, "feedback": verifier_response.feedback})
        return response

    if verifier_log is not None:
        verifier_log.append({"corrected": True, "feedback": verifier_response.feedback})
    corrected_user_prompt = user_prompt + VERIFIER_CORRECTION_ADD.format(
        feedback=verifier_response.feedback,
        previous_response=response_json
    )
    return await raw_inference(model_config, system_prompt, corrected_user_prompt, scheme, use_user=use_user)


async def get_user_answer(model_config: LLMConfig, user_info: List[str], question: str, prompts: PromptSet,choices: Optional[List[str]] = None) -> Dict[str, str]:
    if choices is None:
        class RESPONSE_WITHOUT_CHOICES(BaseModel):
            reason: NON_EMPTY_STRING
            answer: NON_EMPTY_STRING
        response = await inference(
            model_config=model_config,
            system_prompt=prompts.USER_SIMULATOR_SYSTEM_PROMPT,
            user_prompt=prompts.USER_SIMULATOR_WITHOUT_CHOICES_USER_PROMPT.format(
                user_context=json.dumps(user_info, ensure_ascii=False),
                question=question,
            ),
            scheme=RESPONSE_WITHOUT_CHOICES,
            use_user=True
        )
        return {"reason": response.reason, "answer": response.answer}
    else:
        ids2choices = {f"choice_{i}": choice for i, choice in enumerate(choices)}
        class RESPONSE_WITH_CHOICES(BaseModel):
            reason: NON_EMPTY_STRING
            answer_id: Literal[tuple(ids2choices.keys())]
        response = await inference(
            model_config=model_config,
            system_prompt=prompts.USER_SIMULATOR_SYSTEM_PROMPT,
            user_prompt=prompts.USER_SIMULATOR_WITH_CHOICES_USER_PROMPT.format(
                user_context=json.dumps(user_info, ensure_ascii=False),
                question=question,
                choices_with_ids=json.dumps([{"id": cid, "choice_value": choice} for cid, choice in ids2choices.items()], ensure_ascii=False)
            ),
            scheme=RESPONSE_WITH_CHOICES,
            use_user=True
        )
        return {"reason": response.reason, "answer": ids2choices[response.answer_id]}

class RunLogger:                                                                                                                                                                                                                
    def __init__(self, logs_path: Optional[str]=None):
        self.logs_path = logs_path
        self.lines = []
        if self.logs_path is not None:
            Path(self.logs_path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, *args):
        line = " ".join([str(a) for a in args])
        self.lines.append(line)
        if self.logs_path is not None:
            with open(self.logs_path, "a") as f:
                f.write(line + "\n")

    def log_json(self, tag: str, obj: Any):
        def default(o):
            if isinstance(o, torch.Tensor):
                return o.tolist()
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.generic):
                return o.item()
            raise TypeError(f"Not serializable: {type(o)}")
        self.log(f"[{tag}]", json.dumps(obj, default=default, ensure_ascii=False))

    def save(self, logs_path):
        Path(logs_path).parent.mkdir(parents=True, exist_ok=True)
        with open(logs_path, "w") as f:
            f.write("\n".join(self.lines))

def compute_entropy(pi: torch.Tensor, dim: int=0) -> torch.Tensor:
    return -torch.sum(torch.special.xlogy(pi, pi), dim=dim)

def compute_MI(pi: torch.Tensor, K: torch.Tensor) -> float:
    # pi(s) = Prob(\theta=s), shape (S,)
    # K(s,y) = Prob(Y=y|\theta=s), shape (S, C)
    numer = K * pi.unsqueeze(-1)
    py = torch.sum(numer, dim=0)
    post_y = numer / py.unsqueeze(0).clamp(min=1e-12)
    H_prior = compute_entropy(pi)
    H_post = compute_entropy(post_y, dim=0)
    mi = H_prior - torch.sum(py * H_post)
    return mi.item()

def batch_compute_MI(log_prior_flat: torch.Tensor, likelihoods: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    # log_prior_flat (S,) (S=num_states)
    # likelihoods (N, S, C) (N=num_pairs, C=max_num_question_choices) (max over all questions to batch) 
    # masks (N, C) (1 if valid choice, 0 if padding) (likelihoods padded to max number of choices)
    # output (N,) MI for each pair
    prior_probs = torch.exp(log_prior_flat) # (S,)
    pi = prior_probs.unsqueeze(0).unsqueeze(-1) # (1, S, 1)
    numer = likelihoods * pi # (N, S, C)
    py = torch.sum(numer, dim=1) # (N, C)
    py = py * masks # (N, C)
    post_y = numer / py.unsqueeze(1).clamp(min=1e-12) # (N, S, C)
    # H(prior), scalar
    H_prior = -torch.sum(torch.special.xlogy(prior_probs, prior_probs))
    # H(posterior|y), (N, C)
    H_post = -torch.sum(torch.special.xlogy(post_y, post_y), dim=1)
    H_post = H_post * masks # (N, C)
    # (N,)
    return H_prior - torch.sum(py * H_post, dim=1)

async def score_natural_language_answer(model_config: LLMConfig, question: str, choices: List[str], user_answer: str, labels2probs: Dict[str, float], prompts: PromptSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
    ids2choices = {f"choice_{i}": choice for i, choice in enumerate(choices)}
    class LABEL_SCHEME(BaseModel):
        choice_id: Literal[tuple(ids2choices.keys())]
        reason: NON_EMPTY_STRING
        label: Literal["likely", "neutral", "unlikely"]
    class RESPONSE_SCHEME(BaseModel):
        scores: conlist(LABEL_SCHEME, min_length=len(choices), max_length=len(choices))
    response = await inference(
        model_config=model_config,
        system_prompt=prompts.SCORE_NATURAL_LANGUAGE_SYSTEM_PROMPT,
        user_prompt=prompts.SCORE_NATURAL_LANGUAGE_USER_PROMPT.format(
            question=question,
            choices_with_ids=json.dumps([{"id": cid, "value": choice} for cid, choice in ids2choices.items()], ensure_ascii=False),
            user_answer=user_answer
        ),
        scheme=RESPONSE_SCHEME
    )
    scores = []
    reasons = []
    score_map = {score.choice_id: score for score in response.scores}
    for choice_id in ids2choices:
        if choice_id not in score_map:
            raise ValueError(f"No score found for choice_id {choice_id}")
        score = score_map[choice_id]
        scores.append(labels2probs[score.label])
        reasons.append(score.reason)
    scores = torch.tensor(scores)
    return {"scores": scores / torch.sum(scores), "reasons": reasons}


@dataclass
class DIMENSION:
    name: NON_EMPTY_STRING
    reason: NON_EMPTY_STRING
    values: List[NON_EMPTY_STRING]
    round: int
    id: int
    distribution: Dict[int, Dict[str, Union[torch.Tensor, List[str]]]]  # round -> {"probabilities": Tensor, "reasons": List[str], "labels": List[str]}

    @property
    def latest_distribution(self) -> torch.Tensor:
        return self.distribution[max(self.distribution.keys())]["probabilities"]

class DIMENSIONS:
    def __init__(self):
        self.dimensions: Dict[int, DIMENSION] = {}

    def add_dimension(self, name: str, reason: str, values: List[str], round: int) -> DIMENSION:
        new_id = len(self.dimensions)
        self.dimensions[new_id] = DIMENSION(name, reason, values, round, new_id, distribution={})
        return self.dimensions[new_id]

    def update_distribution(self, dim_id: int, distribution: torch.Tensor, round: int, reasons: Optional[List[str]] = None, labels: Optional[List[str]] = None) -> None:
        if dim_id in self.dimensions:
            self.dimensions[dim_id].distribution[round] = {
                "probabilities": distribution,
            }
            if reasons is not None:
                self.dimensions[dim_id].distribution[round]["reasons"] = reasons
            if labels is not None:
                self.dimensions[dim_id].distribution[round]["labels"] = labels
        else:
            raise ValueError(f"Dimension ID {dim_id} not found.")

    def __iter__(self):
        return iter(self.dimensions.items())

    def __getitem__(self, dim_id: int) -> DIMENSION:
        if dim_id in self.dimensions:
            return self.dimensions[dim_id]
        else:
            raise ValueError(f"Dimension ID {dim_id} not found.")

@dataclass
class QUESTION:
    text: str
    reason: str
    choices: List[str]
    id: int
    round: int

class QUESTIONS:
    def __init__(self):
        self.questions: Dict[int, QUESTION] = {}
        self.rounds2questionIds: Dict[int, List[int]] = {}

    def add_question(self, text: str, reason: str, choices: List[str], round: int) -> int:
        new_id = len(self.questions)
        self.questions[new_id] = QUESTION(text, reason, choices, new_id, round)
        if round not in self.rounds2questionIds:
            self.rounds2questionIds[round] = []
        self.rounds2questionIds[round].append(new_id)
        return new_id

    def __iter__(self):
        return iter(self.questions.items())

    def __getitem__(self, question_id: int) -> QUESTION:
        if question_id in self.questions:
            return self.questions[question_id]
        else:
            raise ValueError(f"Question ID {question_id} not found.")

@dataclass
class USER:
    id: int
    private_info: List[str]
    public_info: List[str]
    name: Optional[str] = None

class USERS:
    def __init__(self):
        self.users: Dict[int, USER] = {}

    def add_user(self, private_info: List[str], public_info: List[str], name: Optional[str] = None) -> int:
        new_id = len(self.users)
        self.users[new_id] = USER(new_id, private_info, public_info, name)
        return new_id

    def get_user_private_info(self, user_id: int) -> List[str]:
        if user_id in self.users:
            return self.users[user_id].private_info
        else:
            raise ValueError(f"User ID {user_id} not found.")

    def get_user_public_info(self, user_id: int) -> List[str]:
        if user_id in self.users:
            return self.users[user_id].public_info
        else:
            raise ValueError(f"User ID {user_id} not found.")

    def __iter__(self):
        return iter(self.users.items())

    def __getitem__(self, user_id: int) -> USER:
        if user_id in self.users:
            return self.users[user_id]
        else:
            raise ValueError(f"User ID {user_id} not found.")

class LIKELIHOOD_TABLES:
    def __init__(self):
        # key is (question_id, user_id, dimension_id)
        # value is a dict with keys: probabilities, reasons, labels
        # probabilities: torch.Tensor of shape (num_dimension_values, num_question_choices)
        self.likelihood_tables: Dict[Tuple[int, int, int], Dict[str, Union[torch.Tensor, List[List[str]]]]] = {}

    def update_table(self, question_id: int, user_id: int, dimension_id: int, probs: torch.Tensor, reasons: List[List[str]], labels: List[List[str]]) -> None:
        self.likelihood_tables[(question_id, user_id, dimension_id)] = {
            "probabilities": probs,
            "reasons": reasons,
            "labels": labels
        }

    def __iter__(self):
        return iter(self.likelihood_tables.items())
    
    def __getitem__(self, key: Tuple[int, int, int]) -> torch.Tensor:
        if key in self.likelihood_tables:
            return self.likelihood_tables[key]["probabilities"]
        else:
            raise ValueError(f"Likelihood table for key {key} not found.")


class BeliefState:
    # joint over all dimensions, represented in log-space 

    def __init__(self, dimension_sizes: List[int], log_joint: torch.Tensor):
        self.dimension_sizes = dimension_sizes
        self.log_joint = log_joint  # shape (n_1, n_2, ..., n_D), n_d is number of values for dimension d

    @staticmethod
    def from_marginals(marginals: List[torch.Tensor]) -> 'BeliefState':
        # joint from independent marginals in log-space
        dimension_sizes = [marginal.shape[0] for marginal in marginals]
        log_joint = torch.log(marginals[0])
        for marginal in marginals[1:]:
            # (5,1) + (4,) -> (5,4)
            # (5,4,1) + (3,) -> (5,4,3)
            log_joint = log_joint.unsqueeze(-1) +  torch.log(marginal)
        return BeliefState(dimension_sizes, log_joint)

    def bayesian_update(self, log_update: torch.Tensor) -> None:
        # update in log-space
        self.log_joint = self.log_joint + log_update
        self.log_joint = self.log_joint - torch.logsumexp(self.log_joint.flatten(), dim=0)

    def expand_dimension(self, marginal: torch.Tensor) -> None:
        # add new dimension (given independence)
        num_new_dimension_values = marginal.shape[0]
        self.dimension_sizes.append(num_new_dimension_values)
        shape = [1] * len(self.dimension_sizes)
        shape[-1] = num_new_dimension_values
        log_marginal = torch.log(marginal).reshape(shape) # (1,1,1,7)
        # (2,4,3,1) + (1,1,1,7) -> (2,4,3,7)
        self.log_joint = self.log_joint.unsqueeze(-1) + log_marginal
        # normalize
        self.log_joint = self.log_joint - torch.logsumexp(self.log_joint.flatten(), dim=0)

    def marginal(self, dimension_index: int) -> torch.Tensor:
        # marginalize out all other dimensions except dimension_index, return as probs 
        if len(self.dimension_sizes) == 1:
            return torch.exp(self.log_joint)
        dimensions_to_sum = [i for i in range(len(self.dimension_sizes)) if i != dimension_index]
        log_marginal = torch.logsumexp(self.log_joint, dim=dimensions_to_sum) # (n_d,)
        return torch.exp(log_marginal) # (n_d,)

    def map_state_indices(self) -> Tuple[int, ...]:
        # indices of map state (one per dimension)
        flat_idx = torch.argmax(self.log_joint.flatten())
        indices = torch.unravel_index(flat_idx, self.log_joint.shape)
        return tuple(int(idx.item()) for idx in indices)

    def max_prob(self) -> float:
        return torch.exp(self.log_joint.flatten().max()).item()

    def entropy(self) -> float:
        probs = torch.exp(self.log_joint.flatten())
        return -torch.sum(torch.special.xlogy(probs, probs)).item()

    def flatten_log_probs(self) -> torch.Tensor:
        return self.log_joint.flatten()

    def num_states(self) -> int:
        return self.log_joint.numel()


def compute_state_log_likelihood(dimensions_log_likelihoods: List[torch.Tensor], dimension_sizes: List[int]) -> torch.Tensor:
    # dimensions_log_likelihoods is list of D tensors, each of shape (n_d, num_question_choices)
    # dimension_sizes is list of D ints
    # output shape is (n_1*n_2*...*n_D, num_question_choices)
    num_dimensions = len(dimensions_log_likelihoods)
    num_question_choices = dimensions_log_likelihoods[0].shape[1]
    # (1,1,1,10)
    result = torch.zeros([1] * num_dimensions + [num_question_choices])
    for d in range(num_dimensions):
        shape = [1] * (num_dimensions + 1)  # [1,1,1,1]
        shape[d] = dimension_sizes[d] #[1,4,1,1]
        shape[-1] = num_question_choices #[1,4,1,10]
        # (1,1,1,10) + (3,1,1,10) -> (3,1,1,10)
        # (3,1,1,10) + (1,4,1,10) -> (3,4,1,10)
        result = result + dimensions_log_likelihoods[d].reshape(shape)
    # (3,4,6,10) -> (72,10)
    result = result.reshape(-1, num_question_choices)
    # normalize
    return result - torch.logsumexp(result, dim=1, keepdim=True)

async def one_likelihood_table(user_id: int, user_info: List[str], question_id: int, question_text: str, question_choices: List[str], dimension_id: int, dimension_name: str, dimension_values: List[str], prompts: PromptSet, ambiguous_prompt: str, meta_context: str, model_config: LLMConfig, conversation_log: Optional[List[Dict[str, str]]], use_verifier: bool = False, verifier_log: Optional[List] = None) -> Dict[str, Any]:
    ids2question_choices = {f"choice_{i}": choice for i, choice in enumerate(question_choices)}
    ids2dimension_values = {f"value_{i}": value for i, value in enumerate(dimension_values)}
    class CELL_SCHEME(BaseModel):
        question_choice_id : Literal[tuple(ids2question_choices.keys())]
        dimension_value_id : Literal[tuple(ids2dimension_values.keys())]
        reason: NON_EMPTY_STRING
        label: Literal["likely", "unlikely", "neutral"]
    class RESPONSE_SCHEME(BaseModel):
        evaluations: conlist(
            conlist(CELL_SCHEME, min_length=len(question_choices), max_length=len(question_choices)),
            min_length=len(dimension_values),
            max_length=len(dimension_values)
        )
    if conversation_log is None:
        user_prompt = prompts.LIKELIHOOD_WITHOUT_HISTORY_USER_PROMPT.format(
            ambiguous_prompt=ambiguous_prompt,
            meta_context=meta_context,
            user_info=json.dumps(user_info, ensure_ascii=False),
            dimension_name=dimension_name,
            num_dimension_values=len(dimension_values),
            dimension_values_with_ids=json.dumps([{"id": vid, "text": value} for vid, value in ids2dimension_values.items()], ensure_ascii=False),
            question_text=question_text,
            num_question_choices=len(question_choices),
            question_choices_with_ids=json.dumps([{"id": qid, "text": choice} for qid, choice in ids2question_choices.items()], ensure_ascii=False)
        )
    else:
        user_prompt = prompts.LIKELIHOOD_WITH_HISTORY_USER_PROMPT.format(
            ambiguous_prompt=ambiguous_prompt,
            meta_context=meta_context,
            user_info=json.dumps(user_info, ensure_ascii=False),
            dimension_name=dimension_name,
            num_dimension_values=len(dimension_values),
            dimension_values_with_ids=json.dumps([{"id": vid, "text": value} for vid, value in ids2dimension_values.items()], ensure_ascii=False),
            question_text=question_text,
            num_question_choices=len(question_choices),
            question_choices_with_ids=json.dumps([{"id": qid, "text": choice} for qid, choice in ids2question_choices.items()], ensure_ascii=False),
            conversation_log=json.dumps(conversation_log, ensure_ascii=False)
        )
    for attempt in range(DEFAULT_MAX_RETRIES):
        try:
            response = await inference(
                model_config=model_config,
                system_prompt=prompts.LIKELIHOOD_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                scheme=RESPONSE_SCHEME,
                use_verifier=use_verifier,
                verifier_log=verifier_log
            )

            # no duplicates
            seen_combinations = set()
            for row in response.evaluations:
                for eval in row:
                    combo = (eval.dimension_value_id, eval.question_choice_id)
                    if combo in seen_combinations:
                        raise ValueError(f"Duplicate combination found: {combo}")
                    seen_combinations.add(combo)
            
            # no missing
            expected_combinations = {
                (f"value_{i}", f"choice_{j}")
                for i in range(len(dimension_values))
                for j in range(len(question_choices))
            }
            
            missing = expected_combinations - seen_combinations
            if missing:
                raise ValueError(f"Missing combinations: {missing}")
            
            return {
                "user_id": user_id,
                "dimension_id": dimension_id,
                "question_id": question_id,
                "evaluations": [{
                    "question_choice": ids2question_choices[eval.question_choice_id],
                    "dimension_value": ids2dimension_values[eval.dimension_value_id],
                    "label": eval.label,
                    "reason": eval.reason
                } for row in response.evaluations for eval in row]
            }
        except Exception as e:
            if attempt == DEFAULT_MAX_RETRIES - 1:
                raise e
            print(f"Attempt {attempt + 1} failed for one_likelihood_table() : {e}. Retrying...")

def parse_and_store_likelihood_tables(result: Dict[str, Any], dimensions: DIMENSIONS, current_questions: QUESTIONS, likelihood_tables: LIKELIHOOD_TABLES, config: AgentConfig) -> None:
    dimension_values = dimensions[result["dimension_id"]].values
    question_choices  = current_questions[result["question_id"]].choices
    dim_ques2lab_rea = {(eval["dimension_value"], eval["question_choice"]): (eval["label"], eval["reason"]) for eval in result["evaluations"]}
    # (num_dimension_values, num_question_choices)
    probs = torch.tensor(
        [
            [config.labels2probs[dim_ques2lab_rea[(value, choice)][0]] for choice in question_choices]
            for value in dimension_values
        ],
    )
    reasons = [
        [dim_ques2lab_rea[(value, choice)][1] for choice in question_choices]
        for value in dimension_values
    ]
    labels = [
        [dim_ques2lab_rea[(value, choice)][0] for choice in question_choices]
        for value in dimension_values
    ]
    likelihood_tables.update_table(
        question_id=result["question_id"],
        user_id=result["user_id"],
        dimension_id=result["dimension_id"],
        probs=probs / torch.sum(probs, dim=1, keepdim=True),
        reasons=reasons,
        labels=labels
    )
    return

async def one_answer_likelihood_table(dimension_name: str, dimension_values: List[str], possible_answers: List[str], ambiguous_prompt: str, meta_context: str, model_config: LLMConfig, labels2probs: Dict[str, float], prompts: PromptSet) -> torch.Tensor:
    ids2dimension_values = {f"value_{i}": value for i, value in enumerate(dimension_values)}
    ids2answers = {f"answer_{i}": answer for i, answer in enumerate(possible_answers)}
    class CELL_SCHEME(BaseModel):
        answer_id: Literal[tuple(ids2answers.keys())]
        dimension_value_id: Literal[tuple(ids2dimension_values.keys())]
        reason: NON_EMPTY_STRING
        label: Literal["likely", "unlikely", "neutral"]
    class RESPONSE_SCHEME(BaseModel):
        evaluations: conlist(
            conlist(CELL_SCHEME, min_length=len(possible_answers), max_length=len(possible_answers)),
            min_length=len(dimension_values),
            max_length=len(dimension_values)
        )
    user_prompt = prompts.ANSWER_LIKELIHOOD_USER_PROMPT.format(
        ambiguous_prompt=ambiguous_prompt,
        meta_context=meta_context,
        dimension_name=dimension_name,
        dimension_values_with_ids=json.dumps([{"id": vid, "text": value} for vid, value in ids2dimension_values.items()], ensure_ascii=False),
        possible_answers_with_ids=json.dumps([{"id": aid, "text": answer} for aid, answer in ids2answers.items()], ensure_ascii=False),
        num_dimension_values=len(dimension_values),
        num_possible_answers=len(possible_answers)
    )
    for attempt in range(DEFAULT_MAX_RETRIES):
        try:
            response = await inference(
                model_config=model_config,
                system_prompt=prompts.ANSWER_LIKELIHOOD_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                scheme=RESPONSE_SCHEME
            )
            seen = set()
            for row in response.evaluations:
                for cell in row:
                    combo = (cell.dimension_value_id, cell.answer_id)
                    if combo in seen:
                        raise ValueError(f"Duplicate combination: {combo}")
                    seen.add(combo)
            expected = {(f"value_{i}", f"answer_{j}") for i in range(len(dimension_values)) for j in range(len(possible_answers))}
            missing = expected - seen
            if missing:
                raise ValueError(f"Missing combinations: {missing}")
            label_map = {(cell.dimension_value_id, cell.answer_id): cell.label for row in response.evaluations for cell in row}
            probs = torch.tensor([
                [labels2probs[label_map[(f"value_{i}", f"answer_{j}")]] for j in range(len(possible_answers))] for i in range(len(dimension_values))
            ])
            return probs / probs.sum(dim=1, keepdim=True)
        except Exception as e:
            if attempt == DEFAULT_MAX_RETRIES - 1:
                raise e
            print(f"Attempt {attempt + 1} failed for one_answer_likelihood_table(): {e}. Retrying...")

def compute_answer_probs(log_joint: torch.Tensor, answer_likelihoods: List[torch.Tensor]) -> torch.Tensor:
    # log_joint: shape (n_1, ..., n_D)
    # answer_likelihoods[d]: shape (n_d, num_answers)
    num_dims = log_joint.dim()
    num_answers = answer_likelihoods[0].shape[1]
    log_p_answers = []
    for a in range(num_answers):
        log_lik_a = torch.zeros_like(log_joint)
        for d, L_d in enumerate(answer_likelihoods):
            shape = [1] * num_dims
            shape[d] = -1
            log_lik_a = log_lik_a + torch.log(L_d[:, a].clamp(min=1e-9)).reshape(shape)
        log_p_answers.append(torch.logsumexp((log_joint + log_lik_a).flatten(), dim=0))
    log_p = torch.stack(log_p_answers)
    log_p = log_p - torch.logsumexp(log_p, dim=0)
    return log_p.exp() # shape (num_answers,)

def compute_target_entropy(alpha: float, size: int) -> float:
    return -(1-alpha) * np.log(1-alpha) - alpha * np.log(alpha / (size - 1)) if size > 1 else 0.0

class TrajectoryRecorder:    
    # assumes less than 1000 rounds
    """
    trajectory_dir/
        meta.json                      # config, users, prompt
        dimensions.jsonl               # one line per dimension (append-only)
        questions.jsonl                # one line per question (append-only)
        dim_likelihoods/               # per-dim likelihood tables (written once) q{qid}_u{uid}_d{did}.pt
        rounds/                        # per-round belief snapshots round_{NNN}.pt
        src/                           # copy of source code (written once)
    """

    def __init__(self, trajectory_dir: str):
        self.dir = Path(trajectory_dir)
        self.rounds_dir = self.dir / "rounds"
        self.dim_likelihoods_dir = self.dir / "dim_likelihoods"
        self.answer_likelihoods_dir = self.dir / "answer_likelihoods"
        for d in [self.rounds_dir, self.dim_likelihoods_dir, self.answer_likelihoods_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self._dims_file = self.dir / "dimensions.jsonl"
        self._qs_file = self.dir / "questions.jsonl"
        # Clear these on init so retries don't accumulate stale entries from a prior failed attempt
        self._dims_file.write_text("")
        self._qs_file.write_text("")
        
    def save_meta(self, ambiguous_prompt: str, meta_context: str, users_info: List[Dict[str, Any]], config: AgentConfig):
        with open(self.dir / "meta.json", "w") as f:
            json.dump({
                "ambiguous_prompt": ambiguous_prompt,
                "meta_context": meta_context,
                "users": users_info,
                "config": {
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "max_rounds": config.max_rounds,
                    "num_initial_dims": config.num_initial_dims,
                    "max_num_values_per_dim": config.max_num_values_per_dim,
                    "num_initial_questions": config.num_initial_questions,
                    "max_choices_per_question": config.max_choices_per_question,
                    "max_new_questions_per_round": config.max_new_questions_per_round,
                    "expand_multiplier": config.expand_multiplier,
                    "labels2probs": config.labels2probs,
                    "model_name": config.agent_model_config.model_name,
                },
            }, f, ensure_ascii=False, indent=2)

    def copy_source(self, src_dir: str):
        dst = self.dir / "src"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_dir, dst, ignore=shutil.ignore_patterns("__pycache__"))

    def append_dimension(self, dim_id: int, dim: DIMENSION):
        with open(self._dims_file, "a") as f:
            f.write(json.dumps({"id": dim_id, "name": dim.name, "reason": dim.reason, "values": dim.values, "round_created": dim.round}, ensure_ascii=False) + "\n")

    def append_question(self, q_id: int, q: QUESTION):
        with open(self._qs_file, "a") as f:
            f.write(json.dumps({"id": q_id, "text": q.text, "reason": q.reason, "choices": q.choices, "round_created": q.round}, ensure_ascii=False) + "\n")

    def save_answer_likelihood(self, dim_id: int, table: torch.Tensor):
        torch.save(table, self.answer_likelihoods_dir / f"d{dim_id}.pt")

    def save_dim_likelihood(self, q_id: int, u_id: int, d_id: int, table: Dict):
        torch.save(table, self.dim_likelihoods_dir / f"q{q_id}_u{u_id}_d{d_id}.pt")

    def save_round(self, round_num: int, action: str, belief: BeliefState, dims_ordered: List[int], extra: Optional[Dict] = None):
        snapshot = {
            "round": round_num,
            "action": action,
            "dimensions_ids_ordered": list(dims_ordered),
            "dimension_sizes": list(belief.dimension_sizes),
            "log_joint": belief.log_joint.clone(),
        }
        if extra:
            snapshot.update(extra)
        torch.save(snapshot, self.rounds_dir / f"round_{round_num:03d}.pt")

async def run_agent(
        ambiguous_prompt: str, 
        meta_context: str, 
        users_info: List[Dict[str, Any]], 
        config: AgentConfig, 
        prompts: PromptSet,
        possible_answers: Optional[List[str]]=None,
        logger: Optional[RunLogger]=None,
        trajectory_dir: Optional[str]=None
    ) -> str:

    recorder = TrajectoryRecorder(trajectory_dir) if trajectory_dir else None
    if recorder:
        recorder.copy_source(str(Path(__file__).parent))
        recorder.save_meta(ambiguous_prompt, meta_context, users_info, config)

    AGENT_VERIFIER_LOG: List[Dict[str, Any]] = []  # accumulates verifier events across the run

    def _verifier_stats(events: List[Dict[str, Any]]) -> Dict[str, Any]:
        corrected = [e for e in events if e["corrected"]]
        return {
            "total_verifier_calls": len(events),
            "corrections": len(corrected),
            "correction_feedbacks": [e["feedback"] for e in corrected],
        }

    # -------------------------------------------------------------- #
    # Step 1: Generate initial dimensions (names+values)
    # AGENT_DIMENSIONS
    # -------------------------------------------------------------- #
    AGENT_DIMENSIONS = DIMENSIONS()
    class DIMENSION_SCHEME(BaseModel):
        reason: NON_EMPTY_STRING
        name: NON_EMPTY_STRING
        values: conlist(NON_EMPTY_STRING, min_length=2, max_length=config.max_num_values_per_dim)
    class RESPONSE_SCHEME(BaseModel):
        dimensions: conlist(DIMENSION_SCHEME, min_length=1, max_length=config.num_initial_dims)
    response = await inference(
        model_config=config.agent_model_config,
        system_prompt=prompts.INITIAL_DIMENSIONS_SYSTEM_PROMPT,
        user_prompt=prompts.INITIAL_DIMENSIONS_USER_PROMPT.format(
            ambiguous_prompt=ambiguous_prompt,
            meta_context=meta_context,
            num_initial_dims=config.num_initial_dims,
            max_num_values_per_dim=config.max_num_values_per_dim
        ),
        scheme=RESPONSE_SCHEME
    )
    for dimension in response.dimensions:
        AGENT_DIMENSIONS.add_dimension(name=dimension.name, reason=dimension.reason, values=dimension.values, round=0)

    if logger:
        logger.log_json("INIT_DIMENSIONS", [{"id": did, "name": d.name, "reason": d.reason, "values": d.values} for did, d in AGENT_DIMENSIONS])
    if recorder:
          for did, d in AGENT_DIMENSIONS:
              recorder.append_dimension(did, d)

    # -------------------------------------------------------------- #
    # Step 2: Generate priors over dimensions
    # AGENT_DIMENSIONS
    # -------------------------------------------------------------- #
    async def one_prior(dimension_id: int, dimension_name: str, dimension_value: str) -> Tuple[int, str, str, str, str]:
        class RESPONSE_SCHEME(BaseModel):
            reason: NON_EMPTY_STRING
            label: Literal["likely", "unlikely", "neutral"]
        user_prompt = prompts.INITIAL_PRIORS_USER_PROMPT.format(
            ambiguous_prompt=ambiguous_prompt,
            meta_context=meta_context,
            dimension_name=dimension_name,
            dimension_value=dimension_value
        )
        response = await inference(
            model_config=config.agent_model_config,
            system_prompt=prompts.INITIAL_PRIORS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            scheme=RESPONSE_SCHEME
        )
        return (dimension_id, dimension_name, dimension_value, response.label, response.reason)
    tasks = []
    for dimension_id, dimension in AGENT_DIMENSIONS:
        dimension_name = dimension.name
        for dimension_value in dimension.values:
            tasks.append(one_prior(dimension_id, dimension_name, dimension_value))
    results = await asyncio.gather(*tasks)
    results_by_dim = {}
    for dimension_id, dimension_name, dimension_value, label, reason in results:
        results_by_dim.setdefault(dimension_id, {})[dimension_value] = (label, reason)
    for dimension_id, dimension in AGENT_DIMENSIONS:
        dimension_results = results_by_dim[dimension_id]
        distribution = []
        reasons = []
        labels = []
        total_prob = 0.0
        for dimension_value in dimension.values:
            label, reason = dimension_results[dimension_value]
            prob = config.labels2probs[label]
            total_prob += prob
            distribution.append(prob)
            reasons.append(reason)
            labels.append(label)
        distribution = torch.tensor(distribution) / total_prob
        AGENT_DIMENSIONS.update_distribution(dim_id=dimension_id, distribution=distribution, round=0, reasons=reasons, labels=labels)

    if logger:
        logger.log_json("INIT_PRIORS", [
            {
                "dimension_id": did,
                "name": d.name,
                "probabilities": d.distribution[0]["probabilities"],
                "labels": d.distribution[0].get("labels"),
                "reasons": d.distribution[0].get("reasons"),
            }
            for did, d in AGENT_DIMENSIONS
        ])
    # -------------------------------------------------------------- #
    # Step 3: Generate initial questions (with choices)
    # AGENT_QUESTIONS
    # -------------------------------------------------------------- #
    AGENT_QUESTIONS = QUESTIONS()
    class QUESTION_SCHEME(BaseModel):
        reason: NON_EMPTY_STRING
        text: NON_EMPTY_STRING
        choices: conlist(NON_EMPTY_STRING, min_length=2, max_length=config.max_choices_per_question)
    class RESPONSE_SCHEME(BaseModel):
        questions: conlist(QUESTION_SCHEME, min_length=config.num_initial_questions, max_length=config.num_initial_questions)
    response = await inference(
        model_config=config.agent_model_config,
        system_prompt=prompts.INITIAL_QUESTIONS_SYSTEM_PROMPT,
        user_prompt=prompts.INITIAL_QUESTIONS_USER_PROMPT.format(
            ambiguous_prompt=ambiguous_prompt,
            meta_context=meta_context,
            dimensions_with_values=json.dumps([{
                "name": dimension.name,
                "values": dimension.values
            } for _, dimension in AGENT_DIMENSIONS], ensure_ascii=False),
            num_initial_questions=config.num_initial_questions,
            max_choices_per_question=config.max_choices_per_question
        ),
        scheme=RESPONSE_SCHEME
    )
    for question in response.questions:
        AGENT_QUESTIONS.add_question(text=question.text, reason=question.reason, choices=question.choices, round=0)

    if logger:
        logger.log_json("INIT_QUESTIONS", [{"id": qid, "text": q.text, "reason": q.reason, "choices": q.choices} for qid, q in AGENT_QUESTIONS])
    if recorder:
          for qid, q in AGENT_QUESTIONS:
              recorder.append_question(qid, q)
    # -------------------------------------------------------------- #
    # Step 4: Generate likelihood tables for each (question, user, dimension)
    # AGENT_USERS, AGENT_LIKELIHOOD_TABLES
    # -------------------------------------------------------------- #
    AGENT_USERS = USERS()
    for i, user_info in enumerate(users_info):
        AGENT_USERS.add_user(private_info=user_info["private_info"], public_info=user_info["public_info"], name=user_info.get("name", f"user_{i}"))
    AGENT_LIKELIHOOD_TABLES = LIKELIHOOD_TABLES()

    tasks = []
    for user_id, user in AGENT_USERS:
        for question_id, question in AGENT_QUESTIONS:
            for dimension_id, dimension in AGENT_DIMENSIONS:
                tasks.append(one_likelihood_table(user_id, user.public_info, question_id, question.text, question.choices, dimension_id, dimension.name, dimension.values, prompts, ambiguous_prompt, meta_context, config.agent_model_config, None, use_verifier=config.use_verifier, verifier_log=AGENT_VERIFIER_LOG))
    results = await asyncio.gather(*tasks)

    for result in results:
        parse_and_store_likelihood_tables(result, AGENT_DIMENSIONS, AGENT_QUESTIONS, AGENT_LIKELIHOOD_TABLES, config)

    if logger:
        if config.use_verifier:
            logger.log_json("INIT_VERIFIER_STATS", _verifier_stats(AGENT_VERIFIER_LOG))
        logger.log_json("INIT_LIKELIHOODS", [
            {
                "question_id": key[0], "user_id": key[1], "dimension_id": key[2],
                "probabilities": table["probabilities"],
                "labels": table.get("labels"),
                "reasons": table.get("reasons"),
            }
            for key, table in AGENT_LIKELIHOOD_TABLES
        ])
    if recorder:
        for key, table in AGENT_LIKELIHOOD_TABLES:
            recorder.save_dim_likelihood(key[0], key[1], key[2], table)

    # -------------------------------------------------------------- #
    # Step 5: Build BeliefState from marginals
    # belief, state_likelihoods, dimensions_ids_ordered
    # -------------------------------------------------------------- #
    dimensions_ids_ordered = sorted(AGENT_DIMENSIONS.dimensions.keys())
    marginals = [AGENT_DIMENSIONS[id].latest_distribution for id in dimensions_ids_ordered]
    AGENT_BELIEF = BeliefState.from_marginals(marginals)

    # state-level likelihood table for each (question, user) pair
    # is using AGENT_LIKELIHOOD_TABLES
    def build_state_likelihood(question_id: int, user_id: int) -> torch.Tensor:
        dimensions_log_likelihoods = [
            torch.log(AGENT_LIKELIHOOD_TABLES[(question_id, user_id, dimension_id)]) for dimension_id in dimensions_ids_ordered
        ]
        log_likelihoods = compute_state_log_likelihood(dimensions_log_likelihoods, AGENT_BELIEF.dimension_sizes)
        return torch.exp(log_likelihoods)   # (num_states, num_question_choices)

    # key (question_id, user_id) -> (num_states, num_question_choices)
    AGENT_STATE_LIKELIHOODS: Dict[Tuple[int, int], torch.Tensor] = {}
    for question_id, question in AGENT_QUESTIONS:
        for user_id, user in AGENT_USERS:
            AGENT_STATE_LIKELIHOODS[(question_id, user_id)] = build_state_likelihood(question_id, user_id)

    if logger:
        logger.log_json("INIT_BELIEF", {
            "entropy": AGENT_BELIEF.entropy(),
            "max_prob": AGENT_BELIEF.max_prob(),
            "log_joint": AGENT_BELIEF.log_joint,
            "marginals": {did: AGENT_BELIEF.marginal(i) for i, did in enumerate(dimensions_ids_ordered)},
        })
    if recorder:
        recorder.save_round(0, "INIT", AGENT_BELIEF, dimensions_ids_ordered)

    # -------------------------------------------------------------- #
    # Step 5.1: Build answer-level likelihood tables for each dimension
    # -------------------------------------------------------------- #
    AGENT_ANSWER_LIKELIHOODS: List[torch.Tensor] = []
    # AGENT_ANSWER_LIKELIHOODS[d] must always correspond to dimensions_ids_ordered[d] (same index = same axis in log_joint)
    if possible_answers is not None:
        tasks = [
            one_answer_likelihood_table(
                dimension_name=AGENT_DIMENSIONS[dim_id].name,
                dimension_values=AGENT_DIMENSIONS[dim_id].values,
                possible_answers=possible_answers,
                ambiguous_prompt=ambiguous_prompt,
                meta_context=meta_context,
                model_config=config.agent_model_config,
                labels2probs=config.labels2probs,
                prompts=prompts
            )
            for dim_id in dimensions_ids_ordered
        ]
        AGENT_ANSWER_LIKELIHOODS = list(await asyncio.gather(*tasks))
    if logger and possible_answers is not None:
        logger.log_json("INIT_ANSWER_LIKELIHOODS", [
            {
                "dimension_id": dim_id,
                "dimension_name": AGENT_DIMENSIONS[dim_id].name,
                "dimension_values": AGENT_DIMENSIONS[dim_id].values,
                "possible_answers": possible_answers,
                "probabilities": AGENT_ANSWER_LIKELIHOODS[i],
            }
            for i, dim_id in enumerate(dimensions_ids_ordered)
        ])
    if recorder and possible_answers is not None:
        for i, dim_id in enumerate(dimensions_ids_ordered):
            recorder.save_answer_likelihood(dim_id, AGENT_ANSWER_LIKELIHOODS[i])

    # -------------------------------------------------------------- #
    # Step 6: Run the interaction loop
    # -------------------------------------------------------------- #
    asked_pairs = set()
    conversation_log = []
    converged = False
    # round 0 is the initialization round
    # change : max_ask_rounds
    current_round = 1
    num_asked = 0
    while ((current_round <= config.max_rounds) and (num_asked < config.max_ask_rounds)):
        if logger:
            logger.log_json("ROUND_START", {
                "round": current_round,
                "entropy": AGENT_BELIEF.entropy(),
                "max_prob": AGENT_BELIEF.max_prob(),
                "num_states": AGENT_BELIEF.num_states(),
            })

        # -------------------------------------------------------------- #
        # Step 6.0: Convergence criterion 
        # -------------------------------------------------------------- #
        converged2 = bool(np.mean([AGENT_BELIEF.marginal(i).max().item() >= 1 - config.alpha for i in range(len(dimensions_ids_ordered))]) >= config.beta)
        answer_probs = None
        if possible_answers is not None:
            answer_probs = compute_answer_probs(AGENT_BELIEF.log_joint, AGENT_ANSWER_LIKELIHOODS)
            converged3 = answer_probs.max().item() >= 1 - config.alpha
        else:
            converged3 = converged2

        converged = converged3
        if logger and possible_answers is not None:
            logger.log_json("CONVERGENCE_CHECK", {
                "round": current_round,
                "converged": bool(converged),
                "answer_probs": answer_probs,
            })
        
        if converged:
            break
    
        # Select the (question, user) pair with highest MI between belief and state-level likelihoods
        pair_keys_list = []
        likelihoods_list = []
        # we add mask to batch_compute_MI because question can have different number of choices.
        num_choices_list = []
        for question_id, question in AGENT_QUESTIONS:
            for user_id, user in AGENT_USERS:
                if (question_id, user_id) in asked_pairs:
                    continue
                pair_keys_list.append((question_id, user_id))
                likelihoods_list.append(AGENT_STATE_LIKELIHOODS[(question_id, user_id)])
                num_choices_list.append(AGENT_STATE_LIKELIHOODS[(question_id, user_id)].shape[-1])
        
        num_pairs = len(pair_keys_list)
        if num_pairs > 0 :
            # (num_states,), (num_pairs, num_states, num_question_choices) -> (num_pairs,)
            max_choices = max(num_choices_list)
            padded_likelihoods_list = []
            masks = []
            for likelihood, num_choices in zip(likelihoods_list, num_choices_list):
                pad_size = max_choices - num_choices
                if pad_size > 0:
                    padded = torch.nn.functional.pad(likelihood, (0, pad_size), value=0.)
                else:
                    padded = likelihood
                padded_likelihoods_list.append(padded)
                mask = torch.zeros(max_choices, dtype=torch.bool)
                mask[:num_choices] = True
                masks.append(mask)
            # (S,), (N, S, max_choices), (N, max_choices) -> (N,)
            mi_values = batch_compute_MI(AGENT_BELIEF.flatten_log_probs(), torch.stack(padded_likelihoods_list, dim=0), torch.stack(masks, dim=0))
            best_idx = int(torch.argmax(mi_values).item())
            best_mi = mi_values[best_idx].item()
            best_question_id, best_user_id = pair_keys_list[best_idx]
        else:
            best_question_id, best_user_id, best_mi, mi_values = None, None, 0., []


        # -------------------------------------------------------------- #
        # Step 6.1: Either EXPAND or ASK the best pair
        # -------------------------------------------------------------- #
        current_entropy = AGENT_BELIEF.entropy()
        target_entropy = compute_target_entropy(config.alpha, size=AGENT_BELIEF.num_states())

        expand_threshold = config.expand_multiplier * best_mi * (config.max_rounds - current_round)
        gap = max(0, current_entropy - target_entropy)
        condition = gap > expand_threshold

        if logger:
            logger.log_json("MI_SELECTION", {
                "round": current_round,
                "best_question_id": best_question_id,
                "best_user_id": best_user_id,
                "best_mi": best_mi,
                "current_entropy": current_entropy,
                "target_entropy": target_entropy,
                "gap": gap,
                "expand_threshold": expand_threshold,
                "action": "EXPAND" if gap > expand_threshold else "ASK",
                "all_mi_values": {f"q{qid}_u{uid}": mi_values[i].item() for i, (qid, uid) in enumerate(pair_keys_list)},
            })
        
        # -------------------------------------------------------------- #
        # Step 6.2: EXPAND
        # -------------------------------------------------------------- #
        can_expand = AGENT_BELIEF.num_states() * config.max_num_values_per_dim < config.max_total_states
        if num_pairs == 0 and not can_expand:
            if logger:
                logger.log_json("EARLY_STOP", {"reason": "no_pairs_and_state_space_maxed", "round": current_round})
            break

        if ((num_pairs==0) or condition) and can_expand:
            # EXPAND
            # Generate new dimension (new_dimension)
            class RESPONSE_SCHEME(BaseModel):
                reason: NON_EMPTY_STRING
                name: NON_EMPTY_STRING
                values: conlist(NON_EMPTY_STRING, min_length=2, max_length=config.max_num_values_per_dim)
            response = await inference(
                model_config=config.agent_model_config,
                system_prompt=prompts.EXPAND_DIMENSION_SYSTEM_PROMPT,
                user_prompt=prompts.EXPAND_DIMENSION_USER_PROMPT.format(
                    ambiguous_prompt=ambiguous_prompt,
                    meta_context=meta_context,
                    past_dimensions = json.dumps([{"name": dimension.name} for _, dimension in AGENT_DIMENSIONS], ensure_ascii=False),
                    conversation_log=json.dumps(conversation_log, ensure_ascii=False),
                    max_num_values_per_dim=config.max_num_values_per_dim
                ),
                scheme=RESPONSE_SCHEME
            )
            new_dimension = AGENT_DIMENSIONS.add_dimension(name=response.name, reason=response.reason, values=response.values, round=current_round)
            
            # Generate priors for new dimension
            async def one_prior_new_dimension(dimension_id: int, dimension_name: str, dimension_value: str) -> Tuple[int, str, str, str, str]:
                class RESPONSE_SCHEME(BaseModel):
                    reason: NON_EMPTY_STRING
                    label: Literal["likely", "unlikely", "neutral"]
                response = await inference(
                    model_config=config.agent_model_config,
                    system_prompt=prompts.EXPAND_DIMENSION_PRIORS_SYSTEM_PROMPT,
                    user_prompt=prompts.EXPAND_DIMENSION_PRIORS_USER_PROMPT.format(
                        ambiguous_prompt=ambiguous_prompt,
                        meta_context=meta_context,
                        dimension_name=dimension_name,
                        dimension_value=dimension_value,
                        conversation_log=json.dumps(conversation_log, ensure_ascii=False)
                    ),
                    scheme=RESPONSE_SCHEME
                )
                return (dimension_id, dimension_name, dimension_value, response.label, response.reason)
            tasks = []
            for dimension_value in new_dimension.values:
                tasks.append(one_prior_new_dimension(new_dimension.id, new_dimension.name, dimension_value))
            results = await asyncio.gather(*tasks)
            distribution = []
            reasons = []
            labels = []
            total_prob = 0.0
            for _, dimension_name, dimension_value, label, reason in results:
                prob = config.labels2probs[label]
                total_prob += prob
                distribution.append(prob)
                reasons.append(reason)
                labels.append(label)
            # (num_values,)
            new_marginal = torch.tensor(distribution) / total_prob
            AGENT_DIMENSIONS.update_distribution(dim_id=new_dimension.id, distribution=new_marginal, round=current_round, reasons=reasons, labels=labels)
            # Expand state with new dimension 
            AGENT_BELIEF.expand_dimension(new_marginal)
            dimensions_ids_ordered.append(new_dimension.id)
            if logger:
                logger.log_json("EXPAND_NEW_DIM", {
                    "round": current_round,
                    "dimension_id": new_dimension.id,
                    "name": new_dimension.name,
                    "reason": new_dimension.reason,
                    "values": new_dimension.values,
                    "prior_probabilities": new_marginal,
                    "prior_labels": labels,
                    "prior_reasons": reasons,
                })
            if recorder:
                recorder.append_dimension(new_dimension.id, new_dimension)
            if possible_answers is not None:
                new_dim_answer_likelihoods = await one_answer_likelihood_table(
                    dimension_name=new_dimension.name,
                    dimension_values=new_dimension.values,
                    possible_answers=possible_answers,
                    ambiguous_prompt=ambiguous_prompt,
                    meta_context=meta_context,
                    model_config=config.agent_model_config,
                    labels2probs=config.labels2probs,
                    prompts=prompts
                )
                AGENT_ANSWER_LIKELIHOODS.append(new_dim_answer_likelihoods)
                if logger :
                    logger.log_json("EXPAND_ANSWER_LIKELIHOODS", {
                        "round": current_round,
                        "dimension_id": new_dimension.id,
                        "dimension_name": new_dimension.name,
                        "dimension_values": new_dimension.values,
                        "possible_answers": possible_answers,
                        "probabilities": new_dim_answer_likelihoods,
                    })
                if recorder and possible_answers is not None:
                    recorder.save_answer_likelihood(new_dimension.id, new_dim_answer_likelihoods)

            # Compute likelihood tables for old questions over the new dimension
            old_question_ids = [question_id for question_id, _ in AGENT_QUESTIONS]
            tasks = []
            for question_id in old_question_ids:
                old_question = AGENT_QUESTIONS[question_id]
                for user_id, user in AGENT_USERS:
                    tasks.append(one_likelihood_table(user_id, user.public_info, question_id, old_question.text, old_question.choices, new_dimension.id, new_dimension.name, new_dimension.values, prompts, ambiguous_prompt, meta_context, config.agent_model_config, conversation_log, use_verifier=config.use_verifier, verifier_log=AGENT_VERIFIER_LOG))
            _verifier_log_before_expand_old = len(AGENT_VERIFIER_LOG)
            results = await asyncio.gather(*tasks)
            for result in results:
                parse_and_store_likelihood_tables(result, AGENT_DIMENSIONS, AGENT_QUESTIONS, AGENT_LIKELIHOOD_TABLES, config)
            if logger and config.use_verifier:
                logger.log_json("EXPAND_OLD_Q_VERIFIER_STATS", {"round": current_round, **_verifier_stats(AGENT_VERIFIER_LOG[_verifier_log_before_expand_old:])})
            if recorder:
                for result in results:
                    key = (result["question_id"], result["user_id"], result["dimension_id"])
                    recorder.save_dim_likelihood(*key, AGENT_LIKELIHOOD_TABLES.likelihood_tables[key])
            # Rebuild state-level likelihoods for old questions 
            for question_id in old_question_ids:
                for user_id, _ in AGENT_USERS:

                    # Full re-build (costly)
                    AGENT_STATE_LIKELIHOODS[(question_id, user_id)] = build_state_likelihood(question_id, user_id)

                    # Incremental update (might introduce some errors)
                    # old_state_likelihood_table = AGENT_STATE_LIKELIHOODS[(question_id, user_id)] # (num_old_states, num_question_choices)
                    # new_dimension_likelihood_table = AGENT_LIKELIHOOD_TABLES[(question_id, user_id, new_dimension.id)] # (num_new_dimension_values, num_question_choices)
                    # # (num_old_states, 1, num_question_choices) * (1, num_new_dimension_values, num_question_choices) -> (num_old_states, num_new_dimension_values, num_question_choices)
                    # new_state_likelihood_table = old_state_likelihood_table.unsqueeze(1) * new_dimension_likelihood_table.unsqueeze(0)
                    # num_question_choices = old_state_likelihood_table.shape[1]
                    # # (num_old_states*num_new_dimension_values, num_question_choices)
                    # new_state_likelihood_table = new_state_likelihood_table.reshape(-1, num_question_choices)
                    # AGENT_STATE_LIKELIHOODS[(question_id, user_id)] = new_state_likelihood_table / torch.sum(new_state_likelihood_table, dim=1, keepdim=True)

            # Generate new questions
            high_entropy_dimension_ids = sorted(
                AGENT_DIMENSIONS.dimensions.keys(),
                key=lambda dimension_id: compute_entropy(AGENT_DIMENSIONS.dimensions[dimension_id].latest_distribution),
                reverse=True
            )[:config.num_top_uncertainty_dims]
            class QUESTION_SCHEME(BaseModel):
                reason: NON_EMPTY_STRING
                text: NON_EMPTY_STRING
                choices: conlist(NON_EMPTY_STRING, min_length=2, max_length=config.max_choices_per_question)
            class RESPONSE_SCHEME(BaseModel):
                questions: conlist(QUESTION_SCHEME, min_length=1, max_length=config.max_new_questions_per_round)
            response = await inference(
                model_config=config.agent_model_config,
                system_prompt=prompts.EXPAND_QUESTIONS_SYSTEM_PROMPT,
                user_prompt=prompts.EXPAND_QUESTIONS_USER_PROMPT.format(
                    ambiguous_prompt=ambiguous_prompt,
                    meta_context=meta_context,
                    new_dimension_with_values=json.dumps({
                        "name": new_dimension.name,
                        "values": new_dimension.values
                    }, ensure_ascii=False),
                    high_uncertainty_dimensions_with_values=json.dumps([{
                        "name": AGENT_DIMENSIONS.dimensions[dimension_id].name, 
                        "values": AGENT_DIMENSIONS.dimensions[dimension_id].values
                    } for dimension_id in high_entropy_dimension_ids], ensure_ascii=False),
                    conversation_log=json.dumps(conversation_log, ensure_ascii=False),
                    max_choices_per_question=config.max_choices_per_question,
                    max_new_questions_per_round=config.max_new_questions_per_round
                ),
                scheme=RESPONSE_SCHEME
            )
            new_question_ids = []
            for question in response.questions:
                new_question_id = AGENT_QUESTIONS.add_question(text=question.text, reason=question.reason, choices=question.choices, round=current_round)
                new_question_ids.append(new_question_id)

            if logger:
                logger.log_json("EXPAND_NEW_QUESTIONS", {
                    "round": current_round,
                    "questions": [
                        {"id": qid, "text": AGENT_QUESTIONS[qid].text, "reason": AGENT_QUESTIONS[qid].reason, "choices": AGENT_QUESTIONS[qid].choices}
                        for qid in new_question_ids
                    ],
                })
            if recorder:
                for qid in new_question_ids:
                    recorder.append_question(qid, AGENT_QUESTIONS[qid])

            # Generate likelihood tables for new questions over all dimensions (old ones + new one)
            tasks = []
            for new_question_id in new_question_ids:
                new_question = AGENT_QUESTIONS[new_question_id]
                for user_id, user in AGENT_USERS:
                    for dimension_id, dimension in AGENT_DIMENSIONS:
                        tasks.append(one_likelihood_table(user_id, user.public_info, new_question_id, new_question.text, new_question.choices, dimension_id, dimension.name, dimension.values, prompts, ambiguous_prompt, meta_context, config.agent_model_config, conversation_log, use_verifier=config.use_verifier, verifier_log=AGENT_VERIFIER_LOG))
            _verifier_log_before_expand_new = len(AGENT_VERIFIER_LOG)
            results = await asyncio.gather(*tasks)
            for result in results:
                # can use current_questions because result is only for new questions
                parse_and_store_likelihood_tables(result, AGENT_DIMENSIONS, AGENT_QUESTIONS, AGENT_LIKELIHOOD_TABLES, config)
            if logger and config.use_verifier:
                logger.log_json("EXPAND_NEW_Q_VERIFIER_STATS", {"round": current_round, **_verifier_stats(AGENT_VERIFIER_LOG[_verifier_log_before_expand_new:])})
            if recorder:
                for result in results:
                    key = (result["question_id"], result["user_id"], result["dimension_id"])
                    recorder.save_dim_likelihood(*key, AGENT_LIKELIHOOD_TABLES.likelihood_tables[key])
                

            # Build state-level likelihoods for new questions
            for new_question_id in new_question_ids:
                for user_id, _ in AGENT_USERS:
                    AGENT_STATE_LIKELIHOODS[(new_question_id, user_id)] = build_state_likelihood(new_question_id, user_id)

            if logger:
                logger.log_json("EXPAND_BELIEF", {
                    "round": current_round,
                    "entropy": AGENT_BELIEF.entropy(),
                    "max_prob": AGENT_BELIEF.max_prob(),
                    "num_states": AGENT_BELIEF.num_states(),
                })
            if recorder:
                recorder.save_round(current_round, "EXPAND", AGENT_BELIEF, dimensions_ids_ordered, extra={"new_dimension_id": new_dimension.id, "new_question_ids": new_question_ids})

        # -------------------------------------------------------------- #
        # Step 6.3: ASK the best pair
        # -------------------------------------------------------------- #
        else:
            # ASK
            asked_pairs.add((best_question_id, best_user_id))
            user_answer = await get_user_answer(
                model_config=config.user_model_config,
                user_info=AGENT_USERS.get_user_private_info(best_user_id),
                question=AGENT_QUESTIONS[best_question_id].text,
                choices=AGENT_QUESTIONS[best_question_id].choices,
                prompts=prompts,
            )
            if logger:
                logger.log_json("ASK_USER_ANSWER", {
                    "round": current_round,
                    "question_id": best_question_id,
                    "user_id": best_user_id,
                    "question_text": AGENT_QUESTIONS[best_question_id].text,
                    "user_name": AGENT_USERS[best_user_id].name,
                    "user_answer": user_answer["answer"],
                    "user_reason": user_answer["reason"],
                })
            # still not logging user_answer["reason"]
            conversation_log.append({
                "question_text": AGENT_QUESTIONS[best_question_id].text,
                "user_name": AGENT_USERS[best_user_id].name,
                "user_answer": user_answer["answer"]
            })
            scores_and_reasons = await score_natural_language_answer(
                model_config=config.agent_model_config,
                question=AGENT_QUESTIONS[best_question_id].text,
                choices=AGENT_QUESTIONS[best_question_id].choices,
                user_answer=user_answer["answer"],
                labels2probs=config.labels2probs,
                prompts=prompts
            )
            if logger:
                logger.log_json("ASK_SCORES", {
                    "round": current_round,
                    "question_id": best_question_id,
                    "choices": AGENT_QUESTIONS[best_question_id].choices,
                    "scores": scores_and_reasons["scores"],
                    "reasons": scores_and_reasons["reasons"],
                })
            current_likelihoods = AGENT_STATE_LIKELIHOODS[(best_question_id, best_user_id)] # (S, C)
            current_scores = scores_and_reasons["scores"] # (C,)
            # (S, C) * (1,C) -> (S,C) -> (S,)
            #log_update = torch.log(torch.sum(current_likelihoods * current_scores.unsqueeze(0), dim=1))
            log_update = torch.logsumexp(torch.log(current_likelihoods) + torch.log(current_scores.unsqueeze(0)), dim=1)
            # Need to check if this reshape is correct
            AGENT_BELIEF.bayesian_update(log_update.reshape(AGENT_BELIEF.log_joint.shape))
            # update marginal distributions in dimensions -> needed in expand when using high entropy dimensions
            for i, dimension_id in enumerate(dimensions_ids_ordered):
                marginal = AGENT_BELIEF.marginal(i)
                AGENT_DIMENSIONS.update_distribution(dim_id=dimension_id, distribution=marginal, round=current_round)
            if logger:
                logger.log_json("ASK_POSTERIOR", {
                    "round": current_round,
                    "entropy": AGENT_BELIEF.entropy(),
                    "max_prob": AGENT_BELIEF.max_prob(),
                    "log_update": log_update,
                    "marginals": {
                        did: AGENT_BELIEF.marginal(i)
                        for i, did in enumerate(dimensions_ids_ordered)
                    },
                })
            if recorder:
                recorder.save_round(current_round, "ASK", AGENT_BELIEF, dimensions_ids_ordered, 
                                    extra={
                                            "asked_question_id": best_question_id,
                                            "asked_user_id": best_user_id,
                                            "user_answer": user_answer["answer"],
                                            "scores": scores_and_reasons["scores"].clone(),
                                            "log_update": log_update.clone(),
                                        })
            num_asked += 1
        current_round += 1

    if logger:
        final_answer_probs = compute_answer_probs(AGENT_BELIEF.log_joint, AGENT_ANSWER_LIKELIHOODS) if possible_answers is not None else None
        logger.log_json("LOOP_END", {
            "rounds_used": current_round,
            "converged": converged,
            "final_answer_probs": final_answer_probs,
            "final_entropy": AGENT_BELIEF.entropy(),
            "final_max_prob": AGENT_BELIEF.max_prob(),
            "total_questions_asked": len(asked_pairs),
            "total_dimensions": len(AGENT_DIMENSIONS.dimensions),
            "total_questions_generated": len(AGENT_QUESTIONS.questions),
            "conversation_log": conversation_log,
            **({"verifier_summary": _verifier_stats(AGENT_VERIFIER_LOG)} if config.use_verifier else {}),
        })
    # -------------------------------------------------------------- #
    # Step 7: Generate final answer
    # -------------------------------------------------------------- #
    map_indices = AGENT_BELIEF.map_state_indices()
    map_state = {
        AGENT_DIMENSIONS[dimensions_ids_ordered[i]].name : AGENT_DIMENSIONS[dimensions_ids_ordered[i]].values[idx]
        for i, idx in enumerate(map_indices)
    }
    if possible_answers is None:
        class RESPONSE_SCHEME(BaseModel):
            reason: NON_EMPTY_STRING
            final_answer: NON_EMPTY_STRING
        final_answer = await inference(
            model_config=config.agent_model_config,
            system_prompt=prompts.FINAL_ANSWER_SYSTEM_PROMPT,
            user_prompt=prompts.FINAL_ANSWER_WITHOUT_CHOICES_USER_PROMPT.format(
                ambiguous_prompt=ambiguous_prompt,
                meta_context=meta_context,
                conversation_log=json.dumps(conversation_log, ensure_ascii=False),
                map_state=json.dumps(map_state, ensure_ascii=False)
            ),
            scheme=RESPONSE_SCHEME
        )
        if logger:
            logger.log_json("FINAL_ANSWER", {
                "map_state": map_state,
                "map_prob": AGENT_BELIEF.max_prob(),
                "reason": final_answer.reason,
                "answer": final_answer.final_answer,
            })
        return final_answer.final_answer
    else:
        ids2choices = {f"choice_{i}": choice for i, choice in enumerate(possible_answers)}
        class RESPONSE_SCHEME(BaseModel):
            reason: NON_EMPTY_STRING
            final_answer_id: Literal[tuple(ids2choices.keys())]
        final_answer = await inference(
            model_config=config.agent_model_config,
            system_prompt=prompts.FINAL_ANSWER_SYSTEM_PROMPT,
            user_prompt=prompts.FINAL_ANSWER_WITH_CHOICES_USER_PROMPT.format(
                ambiguous_prompt=ambiguous_prompt,
                meta_context=meta_context,
                conversation_log=json.dumps(conversation_log, ensure_ascii=False),
                map_state=json.dumps(map_state, ensure_ascii=False),
                possible_answers_with_ids=json.dumps([{"id": cid, "value": choice} for cid, choice in ids2choices.items()], ensure_ascii=False)
            ),
            scheme=RESPONSE_SCHEME
        )
        if logger:
            logger.log_json("FINAL_ANSWER", {
                "map_state": map_state,
                "map_prob": AGENT_BELIEF.max_prob(),
                "reason": final_answer.reason,
                "answer": ids2choices[final_answer.final_answer_id]
            })
        return ids2choices[final_answer.final_answer_id]