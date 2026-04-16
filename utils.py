import asyncio
from typing import Dict, Any, Literal
from pydantic import BaseModel

from agent import LLMConfig, inference, NON_EMPTY_STRING

EQUIVALENCE_SYSTEM_PROMPT = "You are a helpful evaluator"

EQUIVALENCE_USER_PROMPT = """We are evaluating answers to the question "{question}"
Here are two possible answers:
Possible Answer 1: {text1}
Possible Answer 2: {text2}
Does Possible Answer 1 semantically entail Possible Answer 2? Respond with entailment, contradiction, or neutral."""

class EvaluationScheme(BaseModel):
    thought: NON_EMPTY_STRING
    evaluation: Literal["entailment", "contradiction", "neutral"]

async def check_implication(context: str, response1: str, response2: str, model_config: LLMConfig) -> Dict[str, Any]:
    user_prompt = EQUIVALENCE_USER_PROMPT.format(
        question=context,
        text1=response1,
        text2=response2,
    )
    response = await inference(
        model_config=model_config,
        system_prompt=EQUIVALENCE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        scheme=EvaluationScheme,
    )
    equiv_map = {"entailment": 2, "neutral": 1, "contradiction": 0}
    return {"equiv": equiv_map[response.evaluation], "thought": response.thought}

async def compute_equivalence(context: str, response1: str, response2: str, model_config: LLMConfig, strict_entailment: bool = False, both: bool = False) -> Dict[str, Any]:
    implication_1_full, implication_2_full = await asyncio.gather(
        check_implication(context, response1, response2, model_config),
        check_implication(context, response2, response1, model_config),
    )
    implication_1 = implication_1_full["equiv"]
    implication_2 = implication_2_full["equiv"]
    shared = {"left2rightImplication": implication_1_full["thought"], "right2leftImplication": implication_2_full["thought"]}
    if not both:
        if strict_entailment:
            equiv = (implication_1 == 2) and (implication_2 == 2)
        else:
            implications = [implication_1, implication_2]
            equiv = (0 not in implications) and ([1, 1] != implications)
        return {"equiv": equiv, **shared}
    else:
        return {
            "equiv_strict": (implication_1 == 2) and (implication_2 == 2),
            "equiv_nonstrict": (0 not in [implication_1, implication_2]) and ([1, 1] != [implication_1, implication_2]),
            **shared,
        }
