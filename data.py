import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from prompts.base import PromptSet
from prompts.dc import PROMPT_SET as DC_PROMPT_SET
from prompts.sp import PROMPT_SET as SP_PROMPT_SET
from prompts.medical import PROMPT_SET as MEDICAL_PROMPT_SET
from utils import compute_equivalence
from agent import LLMConfig

@dataclass
class OBSERVATION:
    ambiguous_prompt: str
    meta_context: str
    correct_answer: str
    users_info: List[Dict[str, Any]]
    users_public_info: List[Dict[str, str]]
    users_private_info: List[Dict[str, str]]
    id : Optional[str] = None
    possible_answers: Optional[List[str]] = None
    key_questions: Optional[List[str]] = None
    key_facts: Optional[List[str]] = None
    dataset_name: Optional[str] = None

    def __str__(self) -> str:
        obs_str = f"OBSERVATION ID: {self.id}\n" + 100* '-' + '\n'
        obs_str += f"Ambiguous Prompt: {self.ambiguous_prompt}\n" + 100* '-' + '\n'
        obs_str += f"Meta Context: {self.meta_context}\n" + 100* '-' + '\n'
        obs_str += f"Possible Answers: {self.possible_answers}\n" + 100* '-' + '\n'
        obs_str += f"Correct Answer: {self.correct_answer}\n" + 100* '-' + '\n'
        obs_str += f"Key Questions: {self.key_questions}\n" + 100* '-' + '\n'
        obs_str += f"Key Facts: {self.key_facts}\n" + 100* '-' + '\n'
        obs_str += f"Users Information:\n" + 50* '.' + '\n'
        for user_info in self.users_info:
            obs_str += f"- User ID: {user_info['id']}\n" + 50* '.' + '\n'
            obs_str += f"- User Name: {user_info['name']}\n" + 50* '.' + '\n'
            obs_str += f"- Public Info: {user_info['public_info']}\n" + 50* '.' + '\n'
            obs_str += f"- Private Info: {user_info['private_info']}\n" + 50* '.' + '\n'
        return obs_str + 100* '-' + '\n'

class Dataset:
    def __init__(self, name: str, path: str):
        assert name in ['ar-bench-dc', 'ar-bench-sp', 'icraft-md', 'imedqa']
        self.name = name
        self.path = path
        self.observations = []
        self.dataset = None 
    
    def get_ambiguous_prompt(self, row: Dict) -> str:
        raise NotImplementedError()
    def get_meta_context(self, row: Dict) -> str:
        raise NotImplementedError()
    def get_possible_answers(self, row: Dict) -> Optional[List[str]]:
        raise NotImplementedError()
    def get_correct_answer(self, row: Dict) -> str:
        raise NotImplementedError()
    def get_name(self) -> str:
        return self.name
    def get_users_info(self, row: Dict) -> List[Dict[str, Any]]:
        private_info = self.get_users_private_info(row)
        public_info = self.get_users_public_info(row)
        assert len(private_info) == len(public_info)
        users_info = []
        for i in range(len(private_info)):
            assert private_info[i]['id'] == public_info[i]['id']
            users_info.append({
                "id": private_info[i]['id'],
                "name": private_info[i]['id'],
                "private_info": private_info[i]['private_info'],
                "public_info": public_info[i]['public_info'] 
            })
        return users_info
    def get_users_public_info(self, row: Dict) -> List[Dict[str, str]]:
        raise NotImplementedError()
    def get_users_private_info(self, row: Dict) -> List[Dict[str, str]]:
        raise NotImplementedError()
    def get_key_questions(self, row: Dict) -> Optional[List[str]]:
        raise NotImplementedError()
    def get_key_facts(self, row: Dict) -> Optional[List[str]]:
        raise NotImplementedError()
    def get_prompt_set(self) -> PromptSet:
        raise NotImplementedError()
    async def score_answer(self, predicted_answer: str, correct_answer: str, **kwargs) -> Any:
        raise NotImplementedError()
    def get_observations(self) -> None:
        assert self.dataset is not None
        for i in range(len(self.dataset)):
            row = self.dataset[i]
            users_public_info=self.get_users_public_info(row)
            users_private_info=self.get_users_private_info(row)
            assert len(users_public_info) == len(users_private_info)  
            obs = OBSERVATION(
                ambiguous_prompt=self.get_ambiguous_prompt(row),
                meta_context=self.get_meta_context(row),
                possible_answers=self.get_possible_answers(row),
                correct_answer=self.get_correct_answer(row),
                users_info=self.get_users_info(row),
                users_public_info=self.get_users_public_info(row),
                users_private_info=self.get_users_private_info(row),
                key_questions=self.get_key_questions(row),
                key_facts=self.get_key_facts(row),
                id=str(i),
                dataset_name=self.name,
            )
            self.observations.append(obs)
    def get_observation(self, idx: int) -> OBSERVATION:
        assert 0 <= idx < len(self.observations)
        return self.observations[idx]
    def get_num_observations(self) -> int:
        return len(self.observations)
    def __len__(self) -> int:
        return len(self.observations)

class ARBenchDC(Dataset):
    def __init__(self, name: str, path: str):
        super().__init__(name, path)
        assert self.path.endswith('.json')
        with open(self.path, 'r') as f:
            self.dataset = json.load(f)
        self.prompt_set = DC_PROMPT_SET
        self.get_observations()

    def get_ambiguous_prompt(self, row: Dict) -> str:
        prompt = "Who is the real murderer in this case among the following suspects: "
        prompt += ', '.join([suspect['name'] for suspect in row['suspects']])
        prompt += "?"
        return prompt
    
    def get_meta_context(self, row: Dict) -> str:
        context = f"The case background is:\n"
        context += f"Time : {row['time']}\n"
        context += f"Location : {row['location']}\n"
        context += f"Victim :\n -name: {row['victim']['name']}\n -introduction: {row['victim']['introduction']}\n -cause of death: {row['victim']['cause_of_death']}\n -murder weapon: {row['victim']['murder_weapon']}\n"
        context += f"The investigation focuses on {len(row['suspects'])} suspects, one of whom is the true murderer:"
        for i,suspect in enumerate(row['suspects']):
            context += f"-suspect {i+1}:"
            context += f" -name: {suspect['name']}\n -introduction: {suspect['introduction']}\n"
        return context

    def get_possible_answers(self, row: Dict) -> Optional[List[str]]:
        return [suspect['name'] for suspect in row['suspects']]
    
    def get_correct_answer(self, row: Dict) -> str:
        for suspect in row['suspects']:
            if suspect['is_murderer']:
                return suspect['name']
        raise ValueError("No murderer found.")
    
    def get_users_public_info(self, row: Dict) -> List[Dict[str, str]]:
        users_public_info = []
        for i,suspect in enumerate(row['suspects']):
            users_public_info.append({
                "id": suspect['name'],
                "public_info": f"-name: {suspect['name']}\n -introduction: {suspect['introduction']}"
            })
        return users_public_info

    def get_users_private_info(self, row: Dict) -> List[Dict[str, str]]:
        users_private_info = []
        for i,suspect in enumerate(row['suspects']):
            users_private_info.append({
                "id": suspect['name'],
                "private_info": f"-name: {suspect['name']}\n -task: {suspect['task']}\n -story: {suspect['story']}"
            })
        return users_private_info

    def get_prompt_set(self) -> PromptSet:
        return self.prompt_set
    
    async def score_answer(self, predicted_answer: str, correct_answer: str, **kwargs) -> Any:
        return 1.0 if predicted_answer.strip().lower() == correct_answer.strip().lower() else 0.0
    
    def get_key_questions(self, row: Dict) -> Optional[List[str]]:
        return row['key_question']
        
    def get_key_facts(self, row: Dict) -> Optional[List[str]]:
        return None

class ARBenchSP(Dataset):
    def __init__(self, name: str, path: str):
        super().__init__(name, path)
        assert self.path.endswith('.json')
        with open(self.path, 'r') as f:
            self.dataset = json.load(f)
        self.prompt_set = SP_PROMPT_SET
        self.get_observations()

    def get_ambiguous_prompt(self, row: Dict) -> str:
        return f'Explain the following puzzle: {row["surface"]}'
    
    def get_meta_context(self, row: Dict) -> str:
        return "The ambiguous prompt you are given is a situation puzzle."
        
    def get_possible_answers(self, row: Dict) -> Optional[List[str]]:
        return None
    
    def get_correct_answer(self, row: Dict) -> str:
        return row['bottom']
    
    def get_users_public_info(self, row: Dict) -> List[Dict[str, str]]:
        return [
            {
                "id": "user_1",
                "public_info": f"This user knows the explanation of the puzzle and will respond accordingly."
            }
        ]
    
    def get_users_private_info(self, row: Dict) -> List[Dict[str, str]]:
        return [
            {
                "id": "user_1",
                "private_info": f"The puzzle is: {row['surface']}.\n Its explanation is: {row['bottom']}"
            }
        ]
    
    def get_prompt_set(self) -> PromptSet:
        return self.prompt_set
    
    def get_key_questions(self, row: Dict) -> Optional[List[str]]:
        return row['key_question']
    
    async def score_answer(self, predicted_answer: str, correct_answer: str, **kwargs) -> Any:
        assert 'model_config' in kwargs, "model_config must be provided in kwargs for scoring answers in ARBenchSP dataset."
        assert 'ambiguous_prompt' in kwargs, "ambiguous_prompt must be provided in kwargs for scoring answers in ARBenchSP dataset."
        equivalence = await compute_equivalence(
            context=f"Question: {kwargs.get('ambiguous_prompt')}",
            response1=predicted_answer,
            response2=correct_answer,
            model_config=kwargs.get('model_config'),
            both=True
        )
        equiv_strict = equivalence['equiv_strict'] 
        equiv_nonstrict = equivalence['equiv_nonstrict']
        left2rightImplication = equivalence['left2rightImplication']
        right2leftImplication = equivalence['right2leftImplication']
        return {
            "equiv_strict": equiv_strict,
            "equiv_nonstrict": equiv_nonstrict,
            "left2rightImplication": left2rightImplication,
            "right2leftImplication": right2leftImplication
        }
    
    def get_key_facts(self, row: Dict) -> Optional[List[str]]:
        return None

class MediQ(Dataset):
    def __init__(self, name: str, path: str, use_key_facts: bool = False):
        super().__init__(name, path)
        assert self.path.endswith('.jsonl')
        with open(self.path, 'r') as f:
            self.dataset = [json.loads(line) for line in f]
        self.use_key_facts = use_key_facts
        self.prompt_set = MEDICAL_PROMPT_SET
        self.get_observations()

    def get_ambiguous_prompt(self, row: Dict) -> str:
        ambiguous_prompt = row['question']+"\n"
        ambiguous_prompt += ', '.join(list(row['options'].values())) 
        return ambiguous_prompt
    
    def get_meta_context(self, row: Dict) -> str:
        context = f"Patient information:\n"
        context += f"- Age : {row['patient']['age']}\n"
        context += f"- Gender : {row['patient']['gender']}\n"
        if len(row['context']) > 0:
            context += f"- Chief complaint : {row['context'][0]}"
        return context 
    
    def get_possible_answers(self, row: Dict) -> Optional[List[str]]:
        return list(row['options'].values())
    
    def get_correct_answer(self, row: Dict) -> str:
        return row['answer']
    
    def get_users_public_info(self, row: Dict) -> List[Dict[str, str]]:
        patient_info = f" - Age : {row['patient']['age']}\n - Gender : {row['patient']['gender']}\n"
        if len(row['context']) > 0:
            patient_info += f"- Chief complaint : {row['context'][0]}"
        return [
            {
                'id': 'user_1',
                'public_info': f"This user is the actual patient. Their information is :\n" + patient_info
            }
        ]
    
    def get_users_private_info(self, row: Dict) -> List[Dict[str, str]]:
        patient_info = f" - Age : {row['patient']['age']}\n - Gender : {row['patient']['gender']}\n"
        if len(row['context']) > 0:
            patient_info += f"- Chief complaint : {row['context'][0]}"
        user_private_info = f"The patient information is :\n" + patient_info + f"\nTheir medical record is :\n" + "\n".join([f"{info}" for info in row['facts']])
        if self.use_key_facts:
            user_private_info += f"\n If asked any question, ALWAYS respond with facts from your medical record ONLY. Do NOT USE any other information."
        return [
            {
                'id': 'user_1',
                'private_info': user_private_info
            }
        ]
    
    def get_prompt_set(self) -> PromptSet:
        return self.prompt_set
    
    def get_key_questions(self, row: Dict) -> Optional[List[str]]:
        return None
    
    async def score_answer(self, predicted_answer: str, correct_answer: str, **kwargs) -> Any:
        return 1.0 if predicted_answer.strip().lower() == correct_answer.strip().lower() else 0.0
    
    def get_key_facts(self, row: Dict) -> Optional[List[str]]:
        return row['facts']

class ICraftMD(MediQ):
    def __init__(self, name: str, path: str, use_key_facts: bool = False):
        super().__init__(name, path, use_key_facts=use_key_facts)
        