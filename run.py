import os
import sys

import argparse
import asyncio
import json
from pathlib import Path
import shutil

from agent import run_agent, AgentConfig, LLMConfig, RunLogger
from data import ARBenchDC, ARBenchSP, ICraftMD, Dataset

OUTPUT_DIR = './'

DATASETS = {
    'ar-bench-dc': lambda: ARBenchDC(name='ar-bench-dc', path='./data/arbench/dc/test.json'),
    'ar-bench-sp': lambda: ARBenchSP(name='ar-bench-sp', path='./data/arbench/sp/test.json'),
    'icraft-md':   lambda: ICraftMD(name='icraft-md',    path='./data/mediQ/all_craft_md.jsonl', use_key_facts=True),
}

async def run_observation(
    observation_id: int,
    dataset: Dataset,
    agent_config: AgentConfig,
    logs_dir: str,
    traj_dir: str,
    scores_dir: Path,
) -> dict:
    score_file = scores_dir / f'run_{observation_id}.json'
    if score_file.exists():
        with open(score_file) as f:
            return json.load(f)

    observation = dataset.get_observation(observation_id)
    logger = RunLogger(logs_path=f'{logs_dir}/run_{observation_id}.log')
    result = await run_agent(
        ambiguous_prompt=observation.ambiguous_prompt,
        meta_context=observation.meta_context,
        possible_answers=observation.possible_answers,
        users_info=observation.users_info,
        config=agent_config,
        logger=logger,
        prompts=dataset.get_prompt_set(),
        trajectory_dir=f'{traj_dir}/run_{observation_id}/',
    )
    if dataset.get_name() == 'ar-bench-sp':
        score = await dataset.score_answer(
            predicted_answer=result,
            correct_answer=observation.correct_answer,
            ambiguous_prompt=observation.ambiguous_prompt,
            model_config=agent_config.agent_model_config,
        )
    else:
        score = await dataset.score_answer(result, observation.correct_answer)
    output = {
        'observation_id': observation_id,
        'predicted': result,
        'correct': observation.correct_answer,
        'score': score,
    }
    with open(score_file, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return output

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_api_key',              type=str,   default=None)
    parser.add_argument('--agent_vllm_base_url',         type=str,   default=None)
    parser.add_argument('--user_vllm_base_url',          type=str,   default=None)
    parser.add_argument('--obs_start',                   type=int,   default=0)
    parser.add_argument('--obs_end',                     type=int,   default=-1)  # -1 = run to end
    parser.add_argument('--dataset_name',                default='icraft-md', choices=list(DATASETS.keys()))
    parser.add_argument('--max_rounds',                  type=int,   required=True)
    parser.add_argument('--max_ask_rounds',              type=int,   required=True)
    parser.add_argument('--alpha',                       type=float, required=True)
    parser.add_argument('--num_initial_dims',            type=int,   required=True)
    parser.add_argument('--num_initial_questions',       type=int,   required=True)
    parser.add_argument('--agent_model_name',            type=str,   required=True)
    parser.add_argument('--agent_model_class',           type=str,   required=True)
    parser.add_argument('--user_model_name',             type=str,   required=True)
    parser.add_argument('--user_model_class',            type=str,   required=True)
    parser.add_argument('--temperature',                 type=float, required=True)
    parser.add_argument('--top_p',                       type=float, required=True)
    parser.add_argument('--max_tokens',                  type=int,   required=True)
    parser.add_argument('--beta',                        type=float, required=True)
    parser.add_argument('--max_num_values_per_dim',      type=int,   required=True)
    parser.add_argument('--max_choices_per_question',    type=int,   required=True)
    parser.add_argument('--max_new_questions_per_round', type=int,   required=True)
    parser.add_argument('--num_top_uncertainty_dims',    type=int,   required=True)
    parser.add_argument('--expand_multiplier',           type=float, required=True)
    parser.add_argument('--use_verifier', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.openai_api_key is not None:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if args.agent_vllm_base_url is not None:
        os.environ["VLLM_AGENT_BASE_URL"] = args.agent_vllm_base_url
    if args.user_vllm_base_url is not None:
        os.environ["VLLM_USER_BASE_URL"] = args.user_vllm_base_url

    combo_key = (
        f'agent={args.agent_model_name.split("/")[-1]}'
        f'_user={args.user_model_name.split("/")[-1]}'
        f'_temp={args.temperature}_top_p={args.top_p}_max_tokens={args.max_tokens}'
        f'_rounds={args.max_rounds}_ask={args.max_ask_rounds}'
        f'_alpha={args.alpha}_beta={args.beta}'
        f'_dims={args.num_initial_dims}_qs={args.num_initial_questions}'
        f'_maxvals={args.max_num_values_per_dim}_choices={args.max_choices_per_question}'
        f'_newqs={args.max_new_questions_per_round}_topudims={args.num_top_uncertainty_dims}'
        f'_expand={args.expand_multiplier}_verifier={args.use_verifier}'
    )
    results_dir = Path(OUTPUT_DIR + f'./results/{args.dataset_name}/{combo_key}')
    logs_dir    = Path(OUTPUT_DIR + f'./logs/{args.dataset_name}/{combo_key}')
    traj_dir    = Path(OUTPUT_DIR + f'./trajectories/{args.dataset_name}/{combo_key}')
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    dataset = DATASETS[args.dataset_name]()
    n = dataset.get_num_observations()
    obs_start = args.obs_start
    obs_end   = n if args.obs_end < 0 else min(args.obs_end, n)

    agent_config = AgentConfig(
        agent_model_config=LLMConfig(
            model_name=args.agent_model_name,
            model_class=args.agent_model_class,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        ),
        user_model_config=LLMConfig(
            model_name=args.user_model_name,
            model_class=args.user_model_class,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        ),
        max_rounds=args.max_rounds,
        max_ask_rounds=args.max_ask_rounds,
        alpha=args.alpha,
        beta=args.beta,
        num_initial_dims=args.num_initial_dims,
        max_num_values_per_dim=args.max_num_values_per_dim,
        num_initial_questions=args.num_initial_questions,
        max_choices_per_question=2 if args.dataset_name == 'ar-bench-sp' else args.max_choices_per_question,
        max_new_questions_per_round=args.max_new_questions_per_round,
        num_top_uncertainty_dims=args.num_top_uncertainty_dims,
        expand_multiplier=args.expand_multiplier,
        use_verifier=args.use_verifier,
    )

    MAX_ATTEMPTS = 10
    results = []
    for i in range(obs_start, obs_end):
        score_file_pre = results_dir / f'run_{i}.json'
        if not score_file_pre.exists():
            stale_log  = logs_dir / f'run_{i}.log'
            stale_traj = traj_dir  / f'run_{i}'
            if stale_log.exists():  stale_log.unlink()
            if stale_traj.exists(): shutil.rmtree(stale_traj)

        for attempt in range(MAX_ATTEMPTS):
            try:
                result = await run_observation(i, dataset, agent_config, str(logs_dir), str(traj_dir), results_dir)
                results.append(result)
                break
            except Exception as e:
                print(f"[observation {i}] attempt {attempt + 1}/{MAX_ATTEMPTS} failed: {e}")
                log_file = logs_dir / f'run_{i}.log'
                traj_subdir = traj_dir / f'run_{i}'
                score_file = results_dir / f'run_{i}.json'
                if log_file.exists():    log_file.unlink()
                if traj_subdir.exists(): shutil.rmtree(traj_subdir)
                if score_file.exists():  score_file.unlink()
        else:
            print(f"[observation {i}] all {MAX_ATTEMPTS} attempts failed, skipping.")

if __name__ == '__main__':
    asyncio.run(main())
