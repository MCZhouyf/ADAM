import copy
import os
import re
import sys
import time
import threading

import openai
import Adam.util_info
from concurrent.futures import ThreadPoolExecutor, as_completed
from env.bridge import VoyagerEnv
from env.process_monitor import SubprocessMonitor
from typing import Dict
from Adam.skill_loader import skill_loader
from Adam.module_utils import *
from Adam.infer_API import get_response, get_local_response
from Adam.MLLM_API import get_image_description

lock = threading.Lock()


def has_expected_action_progress(action, added_items):
    if action == "gatherWoodLog":
        return any(str(item).endswith("_log") for item in added_items)
    if action == "gatherStone":
        return "cobblestone" in added_items
    return bool(added_items)


def get_latest_save_marker(step_result):
    if not step_result:
        return None
    events = step_result
    if not isinstance(events, list):
        return None
    latest = None
    for event in events:
        if not isinstance(event, (list, tuple)) or len(event) != 2:
            continue
        event_name, payload = event
        if not isinstance(payload, dict):
            continue
        direct_marker = payload.get("saveMarker")
        if direct_marker:
            latest = direct_marker
        marker = payload.get("onSave")
        if marker:
            latest = marker
    return latest


def get_latest_inventory(step_result, default=None):
    if default is None:
        default = {}
    if not isinstance(step_result, list):
        return default
    latest = default
    for event in step_result:
        if not isinstance(event, (list, tuple)) or len(event) != 2:
            continue
        _, payload = event
        if isinstance(payload, dict) and isinstance(payload.get("inventory"), dict):
            latest = payload["inventory"]
    return latest


class ADAM:
    def __init__(
            self,
            mc_port: int = None,
            azure_login: Dict[str, str] = None,
            game_server_port: int = 3000,
            local_llm_port: int = 6000,
            local_mllm_port: int = 7000,
            game_visual_server_port: int = 9000,
            env_request_timeout: int = 180,
            env_wait_ticks: int = 10,
            max_infer_loop_num: int = 2,
            infer_sampling_num: int = 2,
            max_llm_answer_num: int = 2,
            max_try=2,
            prompt_folder_path: str = r'prompts',
            tmp_image_path: str = 'game_image',
            llm_model_type: str = 'gpt-4-turbo-preview',
            use_local_llm_service: bool = False,
            openai_api_key: str = '',
            load_ckpt_path: str = '',
            auto_load_ckpt: bool = False,
            parallel: bool = False,
            reset_position: Dict[str, float] = None,
            track_player: bool = False,
    ):
        self.env = VoyagerEnv(
            mc_port=mc_port,
            azure_login=azure_login,
            server_port=game_server_port,
            request_timeout=env_request_timeout,
            visual_server_port=game_visual_server_port
        )
        self.default_server_port = game_server_port
        self.local_llm_port = local_llm_port
        self.local_mllm_port = local_mllm_port
        self.parallel = parallel
        self.reset_position = reset_position
        self.track_player = track_player
        if parallel:
            self.env_vector = {game_server_port: self.env}
            for i in range(1, max([infer_sampling_num, max_try])):
                self.env_vector[game_server_port + i] = VoyagerEnv(
                    mc_port=mc_port,
                    azure_login=azure_login,
                    server_port=game_server_port + i,
                    request_timeout=env_request_timeout,
                )
        self.env_wait_ticks = env_wait_ticks
        self.max_infer_loop_num = max_infer_loop_num
        self.infer_sampling_num = infer_sampling_num
        self.tmp_image_path = tmp_image_path
        self.dataset_path = U.f_mkdir(os.path.abspath(os.path.dirname(__file__)), "causal_datasets", llm_model_type)
        U.f_mkdir(self.dataset_path, 'causal_result')
        U.f_mkdir(self.dataset_path, 'llm_steps_log')
        U.f_mkdir(self.dataset_path, 'log_data')
        self.ckpt_path = U.f_mkdir(self.dataset_path, 'ckpt', get_time())
        with open(prompt_folder_path + '/LLM_CD_prompt.txt', 'r') as prompt_file:
            self.CD_prompt = prompt_file.read()
        with open(prompt_folder_path + '/planner_prompt.txt', 'r') as prompt_file:
            self.planner_prompt = prompt_file.read()
        with open(prompt_folder_path + '/actor_prompt.txt', 'r') as prompt_file:
            self.actor_prompt = prompt_file.read()
        self.max_try = max_try
        self.max_llm_answer_num = max_llm_answer_num
        self.llm_model_type = llm_model_type
        self.use_local_llm_service = use_local_llm_service
        self.visual_api_started = False
        self.visual_api_monitor = None
        self.record = None
        self.loop_record = None
        # Observation Item Space S
        self.observation_item_space = []
        self.unlocked_actions = ['A']
        self.use_dynamic_planning = True
        self.planned_actions = []
        # Learned causal subgraph is represented as {action : [[causes],[effects]]}
        self.learned_causal_subgraph = {}
        self.learned_items = set()
        self.goal = ([], [])
        self.goal_item_letters = translate_item_name_list_to_letter(self.goal[0])
        self.memory = []
        self.openai_api_key = openai_api_key.strip()
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")
        if load_ckpt_path:
            self.load_state(load_ckpt_path)
        if auto_load_ckpt:
            self.auto_load_state()
        openai.api_key = openai_api_key
        if os.environ.get("OPENAI_BASE_URL"):
            base_url = os.environ["OPENAI_BASE_URL"].strip()
            if base_url and not base_url.endswith("/"):
                base_url += "/"
            openai.base_url = base_url

    def get_llm_answer(self, prompt):
        if self.use_local_llm_service:
            response_text = get_local_response(prompt, self.local_llm_port)
        else:
            response_text = get_response(
                prompt,
                self.llm_model_type,
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
            )
        return response_text

    def check_llm_answer(self, prompt_text):
        for _ in range(self.max_llm_answer_num):
            try:
                response_text = self.get_llm_answer(prompt_text)
                self.loop_record["llm_answer_content"] = response_text
                match = re.search(r'{([^{}]*)}', response_text, re.DOTALL)
                if not match:
                    raise ValueError(f"No brace-wrapped answer found in: {response_text!r}")
                extracted_response = match.group(1).replace(" ", "").strip()
                parts = extracted_response.split(";")
                if len(parts) != 2:
                    raise ValueError(
                        f"Expected '{{cause;effect}}' format, got: {extracted_response!r}"
                    )
                cause, effect = parts
            except Exception as e:
                print("\033[91mLLM inference failed:" + str(e) + '\033[0m')
                continue
            if cause == '':
                cause = []
            else:
                cause = cause.split(",")
            if effect == '':
                effect = []
            else:
                effect = effect.split(",")
            if check_len_valid(cause) and check_len_valid(effect):
                self.loop_record["llm_answer_checks_num"] = _ + 1
                self.loop_record["llm_answer_success"] = True
                self.loop_record["llm_answer_record"].append([cause, effect])
                return True, cause, effect
        return False, None, None

    def init_record_structure(self, action_name):
        return {
            "loop_num": 0,
            "infer_sampling_num": self.infer_sampling_num,
            "successful": False,
            "action_type": action_name,
            "loop_list": [],
        }

    def update_available_knowledge(self, item_key):
        self.learned_items.update([item_key])
        if item_key in Adam.util_info.unlock.keys():
            self.unlocked_actions.extend(Adam.util_info.unlock[item_key])

    def update_material_dict(self, end_item):
        current_max_key = max(Adam.util_info.material_names_dict.keys(), key=key_cmp_func)
        for item in end_item.keys():
            item = rename_item(item)
            if item not in Adam.util_info.material_names_dict.values():
                current_max_key = generate_next_key(current_max_key)
                Adam.util_info.material_names_dict[current_max_key] = item
                Adam.util_info.material_names_rev_dict[item] = current_max_key
            item_key = Adam.util_info.material_names_rev_dict[item]
            if item_key not in self.observation_item_space:
                self.observation_item_space.append(item_key)

    def save_state(self):
        state = {
            'observation_item_space': self.observation_item_space,
            'unlocked_actions': self.unlocked_actions,
            'learned_causal_subgraph': self.learned_causal_subgraph,
            'learned_items': list(self.learned_items),
            'memory': self.memory,  # serve as log
            'goal': self.goal,
            'goal_item_letters': self.goal_item_letters,
            'planned_actions': self.planned_actions,
            'material_names_dict': Adam.util_info.material_names_dict,
            'material_names_rev_dict': Adam.util_info.material_names_rev_dict
        }
        filepath = U.f_join(self.ckpt_path, get_time() + '.json')
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=4)

    def load_state(self, filepath):
        with open(filepath, 'r') as f:
            state = json.load(f)
        self.observation_item_space = state['observation_item_space']
        self.unlocked_actions = state['unlocked_actions']
        self.learned_causal_subgraph = state['learned_causal_subgraph']
        self.learned_items = set(state['learned_items'])
        self.goal = tuple(state['goal'])
        self.goal_item_letters = state['goal_item_letters']
        self.planned_actions = state.get('planned_actions', [])
        Adam.util_info.material_names_dict = state['material_names_dict']
        Adam.util_info.material_names_rev_dict = state['material_names_rev_dict']

    def auto_load_state(self):
        ckpt = U.f_listdir(self.dataset_path, 'ckpt', full_path=True, recursive=True)
        if ckpt:
            self.load_state(ckpt[-1])

    def get_causal_graph(self):
        return '\n'.join([f"Action: {key}; Cause: {value[0]}; Effect {value[1]}" for key, value in
                          self.learned_causal_subgraph.items()])

    def format_item_key(self, item_key):
        return translate_item_letter_to_name(item_key)

    def format_action_key(self, action_key):
        return translate_action_letter_to_name(action_key)

    def format_causal_graph_for_display(self):
        lines = []
        for action_key, value in self.learned_causal_subgraph.items():
            causes, effects = value
            action_name = self.format_action_key(action_key)
            cause_names = [self.format_item_key(item) for item in causes]
            effect_names = [self.format_item_key(item) for item in effects]
            lines.append(
                f"Action: {action_name}; Cause: {cause_names}; Effect: {effect_names}"
            )
        return '\n'.join(lines)

    def format_plan_for_display(self, plan):
        return " -> ".join(self.format_action_key(action_key) for action_key in plan)

    def format_item_plan_for_display(self, plan):
        return " -> ".join(self.describe_plan_items(plan))

    def print_plan_summary(self, header="Final learned plan summary"):
        print(header)
        print("LLM action plan: " + self.format_plan_for_display(self.planned_actions))
        print(
            "LLM item plan: goal "
            + ", ".join(self.goal[0])
            + " needs "
            + self.format_item_plan_for_display(self.planned_actions)
        )
        print("Actions and Dependencies:")
        for dependency_line in self.get_action_dependency_lines(self.planned_actions):
            print(dependency_line)

    def describe_plan_items(self, plan):
        item_chain = []
        action_to_effect = {
            "A": "log",
            "B": "planks",
            "C": "crafting_table",
            "D": "sticks",
            "E": "fence",
            "F": "fence_gate",
            "G": "wooden_axe",
            "H": "wooden_hoe",
            "I": "wooden_shovel",
            "J": "wooden_sword",
            "K": "wooden_pickaxe",
            "L": "coal",
            "M": "cobblestone",
            "N": "stone_axe",
            "O": "stone_hoe",
            "P": "stone_shovel",
            "Q": "stone_sword",
            "R": "stone_pickaxe",
            "S": "furnace",
            "T": "raw_iron",
            "U": "iron_ingot",
            "V": "iron_axe",
            "W": "iron_hoe",
            "X": "iron_shovel",
            "Y": "iron_sword",
            "Z": "iron_pickaxe",
            "AA": "raw_gold",
            "AB": "diamond",
            "AC": "diamond_axe",
            "AD": "diamond_hoe",
            "AE": "diamond_pickaxe",
            "AF": "diamond_shovel",
            "AG": "diamond_sword",
            "AH": "dirt",
            "AI": "sand",
            "AJ": "gold_ingot",
            "AK": "golden_axe",
            "AL": "golden_hoe",
            "AM": "golden_pickaxe",
            "AN": "golden_shovel",
            "AO": "golden_sword",
        }
        for action in plan:
            item_name = action_to_effect.get(action)
            if item_name and item_name not in item_chain:
                item_chain.append(item_name)
        return item_chain

    def get_action_mapping_prompt(self):
        return '\n'.join(
            f"{name}"
            for letter, name in Adam.util_info.action_names_dict.items()
        )

    def get_action_dependency_lines(self, plan):
        dependency_lines = []
        for action_key in plan:
            action_name = self.format_action_key(action_key)
            known_entry = self.learned_causal_subgraph.get(action_key)
            if known_entry:
                causes, effects = known_entry
                cause_names = [self.format_item_key(item) for item in causes] or ["nothing"]
                effect_names = [self.format_item_key(item) for item in effects] or ["unknown"]
            else:
                effect_names = [item for item in self.describe_plan_items([action_key])] or ["unknown"]
                cause_names = ["to be learned"]
            dependency_lines.append(
                f"{action_name}: requires {', '.join(cause_names)} -> produces {', '.join(effect_names)}"
            )
        return dependency_lines

    def get_goal_fallback_plan(self):
        goal_items = {rename_item(item) for item in self.goal[0]}
        if "iron_ingot" in goal_items:
            return ["A", "B", "D", "C", "K", "M", "R", "S", "T", "U"]
        if "raw_iron" in goal_items:
            return ["A", "B", "D", "C", "K", "M", "R", "T"]
        if "stone_pickaxe" in goal_items:
            return ["A", "B", "D", "C", "K", "M", "R"]
        if "cobblestone" in goal_items:
            return ["A", "B", "D", "C", "K", "M"]
        if "wooden_pickaxe" in goal_items:
            return ["A", "B", "D", "C", "K"]
        if "stick" in goal_items:
            return ["A", "B", "D"]
        if "crafting_table" in goal_items:
            return ["A", "B", "C"]
        if "planks" in goal_items:
            return ["A", "B"]
        if "log" in goal_items:
            return ["A"]
        return ["A"]

    def normalize_planned_action(self, token):
        token = token.strip().strip("'\"`[](){}")
        if not token:
            return None
        upper_token = token.upper()
        if upper_token in Adam.util_info.action_names_dict:
            return upper_token
        if token in Adam.util_info.action_names_rev_dict:
            return Adam.util_info.action_names_rev_dict[token]
        for action_name, action_letter in Adam.util_info.action_names_rev_dict.items():
            if action_name.lower() == token.lower():
                return action_letter
        return None

    def parse_learning_plan(self, response_text):
        if not response_text:
            return []
        candidates = []
        brace_matches = re.findall(r'{([^{}]*)}', response_text, re.DOTALL)
        source_text = brace_matches[-1] if brace_matches else response_text
        for token in re.split(r'[\s,;>\-\n]+', source_text):
            action_letter = self.normalize_planned_action(token)
            if action_letter and action_letter not in candidates:
                candidates.append(action_letter)
        return candidates

    def ensure_valid_learning_plan(self, plan):
        valid_plan = []
        for action_letter in plan:
            if action_letter in Adam.util_info.action_names_dict and action_letter not in valid_plan:
                valid_plan.append(action_letter)
        if not valid_plan:
            valid_plan = self.get_goal_fallback_plan()
        elif "A" not in valid_plan:
            valid_plan.insert(0, "A")
        return valid_plan

    def plan_learning_path(self, reason=None):
        fallback_plan = self.get_goal_fallback_plan()
        reason_text = reason or "Initial planning request."
        prompt = f"""
You are planning the action-learning order for ADAM in Minecraft.

Goal items: {self.goal[0]}
Goal environmental factors: {self.goal[1]}
Planning reason: {reason_text}
Current planned actions: {self.planned_actions or "None"}
Known learned causal graph:
{self.format_causal_graph_for_display() or "None"}

Available actions:
{self.get_action_mapping_prompt()}

Return only one brace-wrapped ordered action path, using exact action names.
The path must include every action needed to produce intermediate ingredients before they are used.
Do not assume a later crafting action can run unless its ingredient-producing actions already appear earlier.
If the previous action failed, treat that as evidence that the current recipe/order may be wrong and return a corrected plan.
Example format: {{gatherWoodLog, craftPlanks, craftCraftingTable}}
"""
        try:
            response_text = self.get_llm_answer(prompt)
            print('\033[94m' + '-' * 20 + 'Learning Planner' + '-' * 20 + '\n' + response_text + '\033[0m')
            plan = self.parse_learning_plan(response_text)
        except Exception as e:
            print(f"Learning planner failed, using fallback plan: {e}")
            plan = fallback_plan
        plan = self.ensure_valid_learning_plan(plan)
        if not plan:
            plan = fallback_plan
        print(
            "Dynamic learning plan: "
            + self.format_plan_for_display(plan)
        )
        print(
            "LLM item plan: goal "
            + ", ".join(self.goal[0])
            + " needs "
            + self.format_item_plan_for_display(plan)
        )
        print("Actions and Dependencies:")
        for dependency_line in self.get_action_dependency_lines(plan):
            print(dependency_line)
        self.planned_actions = plan
        self.unlocked_actions = list(dict.fromkeys(self.unlocked_actions + plan))
        return plan

    def wait_for_inventory_progress(self, env, start_item, max_checks=3, wait_seconds=1):
        last_result = None
        last_end_item = start_item
        last_consumed_items = []
        last_added_items = []
        last_save_marker = None

        for check_index in range(max_checks):
            time.sleep(wait_seconds)
            last_result = env.step('')
            time.sleep(wait_seconds)
            last_end_item = get_latest_inventory(last_result, last_end_item)
            last_consumed_items, last_added_items = get_item_changes(start_item, last_end_item)
            last_save_marker = get_latest_save_marker(last_result)
            print(
                f"Inventory check {check_index + 1}/{max_checks}: "
                f"start_item={start_item}, end_item={last_end_item}, save_marker={last_save_marker}"
            )
            if last_added_items or last_save_marker:
                return (
                    last_result,
                    last_end_item,
                    last_consumed_items,
                    last_added_items,
                    last_save_marker,
                )

        return (
            last_result,
            last_end_item,
            last_consumed_items,
            last_added_items,
            last_save_marker,
        )

    def sample_action_once(self, env, action):
        options = {"inventory": {}, "mode": "hard"}
        if self.reset_position:
            options["position"] = self.reset_position
        if self.track_player:
            options["track_player"] = True
        if action != "gatherWoodLog":
            for material in self.observation_item_space:
                options["inventory"] = get_inventory_number(options["inventory"], material)
        reset_result = env.reset(options=options)
        time.sleep(1)
        start_item = get_latest_inventory(reset_result)
        print(f"Action {action} start_item={start_item}")
        action_result = env.step(skill_loader(action))
        end_item = get_latest_inventory(action_result, start_item)
        consumed_items, added_items = get_item_changes(start_item, end_item)
        save_marker = get_latest_save_marker(action_result)
        print(
            f"Action {action} immediate_result: "
            f"start_item={start_item}, end_item={end_item}, save_marker={save_marker}"
        )
        if not added_items and not save_marker:
            _, end_item, consumed_items, added_items, save_marker = self.wait_for_inventory_progress(
                env, start_item
            )
        if action == "gatherWoodLog":
            if save_marker == "wood_log_gathered:expected_item_picked":
                print(
                    f"Action {action} confirmed by save marker: "
                    f"start_item={start_item}, end_item={end_item}, save_marker={save_marker}, added_items={added_items}"
                )
                with lock:
                    recorder(start_item, end_item, consumed_items, added_items, action, self.dataset_path)
                    self.update_material_dict(end_item)
                env.close()
                time.sleep(1)
                return True
            if save_marker in {"wood_log_gathered:side_drop_only", "wood_log_gathered:no_progress"}:
                print(
                    f"Action {action} did not confirm expected log pickup. "
                    f"Final start_item={start_item}, end_item={end_item}, "
                    f"save_marker={save_marker}, added_items={added_items}"
                )
                env.close()
                time.sleep(1)
                return False
        if not added_items:
            print(
                f"Action {action} produced no added items from the current local world state. "
                f"Final start_item={start_item}, end_item={end_item}, save_marker={save_marker}"
            )
            env.close()
            time.sleep(1)
            return False
        if not has_expected_action_progress(action, added_items):
            print(
                f"Action {action} added items {added_items}, but none match the expected progress type. "
                f"Final start_item={start_item}, end_item={end_item}, save_marker={save_marker}"
            )
            env.close()
            time.sleep(1)
            return False
        print(
            f"Action {action} inventory progress detected: "
            f"start_item={start_item}, end_item={end_item}, added_items={added_items}, save_marker={save_marker}"
        )
        with lock:
            recorder(start_item, end_item, consumed_items, added_items, action, self.dataset_path)
            self.update_material_dict(end_item)
        env.close()
        time.sleep(1)
        return True

    # Interaction module, sampling and recording
    def sampling_and_recording_action(self, action):
        if self.parallel:
            success_count = 0
            while success_count < self.infer_sampling_num:
                with ThreadPoolExecutor(max_workers=self.infer_sampling_num) as executor:
                    futures = []
                    for idx in range(self.infer_sampling_num):
                        futures.append(
                            executor.submit(self.sample_action_once, self.env_vector[self.default_server_port + idx],
                                            action))
                        time.sleep(0.5)
                    results = [future.result() for future in futures]
                success_count += results.count(True)
            return True
        else:
            for i in range(self.infer_sampling_num):
                print(f'Sampling {i + 1} started')
                if not self.sample_action_once(self.env, action):
                    print(
                        f"Action {action} did not produce progress from the current local setup. "
                        f"Continuing so the visible in-game search result remains inspectable."
                    )
                    return False
            return True

    def causal_verification_once(self, env, options_orig, action, effect_item):
        try:
            print(f'Verification of action {action}, inventory: {options_orig["inventory"]}')
            if self.reset_position:
                options_orig = copy.deepcopy(options_orig)
                options_orig["position"] = self.reset_position
            if self.track_player:
                options_orig = copy.deepcopy(options_orig)
                options_orig["track_player"] = True
            reset_result = env.reset(options=options_orig)
            time.sleep(1)
            start_item = get_latest_inventory(reset_result)
            print(f"Verification start_item={start_item}")
            env.step(skill_loader(action))
            time.sleep(1)
            result = env.step('')
            time.sleep(1)
            end_item = get_latest_inventory(result, start_item)
            consumed_items, added_items = get_item_changes(start_item, end_item)
            verification_success = check_in_material(added_items, effect_item)
            if action == "gatherStone" and effect_item in {"M", "cobblestone"}:
                verification_success = "cobblestone" in added_items
            print(f"Verification end_item={end_item}")
            print(f"Verification added_items={added_items}")
            print(f"Verification success={verification_success}")
            with lock:
                recorder(start_item, end_item, consumed_items, added_items, action, self.dataset_path)
            env.close()
            time.sleep(1)
            return verification_success
        except Exception as e:
            print(f"Error during causal verification: {e}")
            return False

    # Causal model module verification method
    def causal_verification(self, options_orig, action, effect_item):
        if self.parallel:
            with ThreadPoolExecutor(max_workers=self.max_try) as executor:
                futures = []
                for idx in range(self.max_try):
                    futures.append(
                        executor.submit(self.causal_verification_once, self.env_vector[self.default_server_port + idx],
                                        options_orig, action, effect_item))
                    time.sleep(0.5)
                results = [future.result() for future in as_completed(futures)]
            if any(results):
                return True
            return False
        else:
            for i in range(self.max_try):
                if self.causal_verification_once(self.env, options_orig, action, effect_item):
                    return True
        return False

    # Causal model module: LLM-based CD and Intervention-based CD
    def causal_learning(self, action):
        record_json_path = U.f_join(self.dataset_path, 'log_data', action + '.json')
        for loop_index in range(self.max_infer_loop_num):
            self.record["loop_num"] += 1
            self.loop_record = {"loop_id": loop_index + 1,
                                "llm_answer_record": [],
                                "llm_answer_checks_num": self.max_llm_answer_num,
                                "llm_answer_success": False,
                                "llm_answer_verification_success": False,
                                }

            print(f'Start action {action}')
            if self.sampling_and_recording_action(action) is False:
                self.record["loop_list"].append(self.loop_record)
                print(f"Sampling for {action} failed; skipping causal inference for this loop.")
                continue

            if not os.path.exists(record_json_path):
                self.record["loop_list"].append(self.loop_record)
                print(f"No valid sampling record for {action}; skipping causal inference for this loop.")
                continue

            with open(record_json_path, 'r') as file:
                data = json.load(file)
            if not data:
                self.record["loop_list"].append(self.loop_record)
                print(f"Sampling record for {action} is empty; skipping causal inference for this loop.")
                continue
            CD_prompt = copy.deepcopy(self.CD_prompt)
            dict_string = '\n'.join(
                [f"'{key}': '{Adam.util_info.material_names_dict[key]}'" for key in self.observation_item_space])
            CD_prompt = CD_prompt.replace("{mapping}", dict_string, 1)
            for i, item in enumerate(data[(-self.infer_sampling_num):], start=1):
                initial_items = ', '.join(item['Start item'])
                consumed_items = ', '.join(item['Consumed items'])
                added_items = ', '.join(item['Added items'])
                sampling_result = f"{i}. Initial items: {initial_items}; Consumed items: {consumed_items}; Added items: {added_items}\n"
                CD_prompt += sampling_result
            CD_prompt += "\nYour inference:\n"

            flag, cause, effect = self.check_llm_answer(CD_prompt)
            if not flag:
                self.record["loop_list"].append(self.loop_record)
                print('LLM inference failed')
                continue
            print(
                "Causal assumption: Cause:"
                + str([self.format_item_key(item) for item in cause])
                + ", Effect:"
                + str([self.format_item_key(item) for item in effect])
            )
            self.loop_record["cause_llm"] = cause
            self.loop_record['effect_llm'] = effect
            for effect_item in effect:
                options_orig = {"inventory": {}, "mode": "hard"}
                for item in cause:
                    options_orig["inventory"] = get_inventory_number(options_orig["inventory"], item)
                try:
                    if not self.causal_verification(options_orig, action, effect_item):
                        self.record["loop_list"].append(self.loop_record)
                        break
                except Exception as e:
                    print("Error: ", str(e))
                    break
                self.loop_record["llm_answer_verification_success"] = True

                # Implement do() operation for each variable in cause
                items_to_remove = []
                for item in cause:
                    options_modified = copy.deepcopy(options_orig)
                    item_name = rename_item_rev(translate_item_letter_to_name(item))
                    del options_modified["inventory"][item_name]
                    if self.causal_verification(options_modified, action, effect_item):
                        options_orig = options_modified
                        items_to_remove.append(item)

                self.loop_record['items_to_remove'] = items_to_remove
                self.loop_record['items_to_remove_length'] = len(items_to_remove)
                for item in items_to_remove:
                    cause.remove(item)

                print('Causal relation found!')
                print('Cause:', [self.format_item_key(item) for item in cause])
                print('Effect:', self.format_item_key(effect_item))
                self.loop_record["cause_found"] = cause
                self.loop_record["effect_found"] = effect_item
                with open(U.f_join(self.dataset_path, 'causal_result', action + '.json'), 'w') as json_file:
                    json.dump([cause, effect_item], json_file)
                self.record["successful"] = True
                self.record["loop_list"].append(self.loop_record)
                llm_steps_path = U.f_join(self.dataset_path, 'llm_steps_log', action + '.json')
                try:
                    with open(llm_steps_path, 'r') as file:
                        try:
                            logs = json.load(file)
                        except json.JSONDecodeError:
                            logs = []
                except FileNotFoundError:
                    logs = []
                logs.append(self.record)
                with open(llm_steps_path, 'w') as file:
                    json.dump(logs, file, indent=4)
                action_key = translate_action_name_to_letter(action)
                if action_key not in self.learned_causal_subgraph:
                    self.learned_causal_subgraph[action_key] = [cause, [effect_item]]
                else:
                    self.learned_causal_subgraph[action_key][1].append(effect_item)
                self.update_available_knowledge(effect_item)
                self.save_state()
            return True
        return False

    def planner(self, current_inventory):
        inventory_name_and_num = copy.deepcopy(current_inventory)
        current_inventory = translate_item_name_list_to_letter(current_inventory)
        not_obtained_items = [item for item in self.goal_item_letters if item not in current_inventory]
        planner_prompt = copy.deepcopy(self.planner_prompt)
        replacements = {
            "{goal}": ', '.join(translate_item_name_list_to_letter(self.goal[0])),
            "{mapping}": str(Adam.util_info.material_names_dict),
            "{current inventory}": ', '.join(current_inventory),
            "{inventory name and num}": str(inventory_name_and_num),
            "{lacked inventory}": ', '.join(not_obtained_items),
            "{causal graph}": self.get_causal_graph(),
        }
        for key, value in replacements.items():
            planner_prompt = planner_prompt.replace(key, value, 1)
        subtask = self.get_llm_answer(planner_prompt)
        print('\033[94m' + '-' * 20 + 'Planner' + '-' * 20 + '\n' + subtask + '\033[0m')
        return subtask

    def actor(self, subtask, perception):
        max_attempts = 3
        attempts = 0
        while attempts < max_attempts:
            try:
                actor_prompt = copy.deepcopy(self.actor_prompt)
                replacements = {
                    "{causal graph}": self.get_causal_graph(),
                    "{available actions}": ', '.join(self.unlocked_actions),
                    "{goal items}": ', '.join(translate_item_name_list_to_letter(self.goal[0])),
                    "{environmental factors}": ', '.join(self.goal[1]),
                    "{memory}": self.get_memory(),
                    "{subtasks}": subtask,
                    "{perception}": perception,
                }
                for key, value in replacements.items():
                    actor_prompt = actor_prompt.replace(key, value, 1)
                action_response = self.get_llm_answer(actor_prompt)
                print('\033[32m' + '-' * 20 + 'Actor' + '-' * 20 + '\n' + action_response + '\033[0m')
                action = translate_action_letter_to_name(re.search(r'{(.*?)}', action_response).group(1))
                break
            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts}: An error occurred - {e}")
                if attempts == max_attempts:
                    return 'moveForward'
        return action

    def update_memory(self, action_letter, consumed_items, added_items, environment_description):
        self.memory.append([action_letter, consumed_items, added_items, environment_description])

    def get_memory(self):
        recent_memory = self.memory[-3:]
        formatted_prompt = []

        for entry in recent_memory:
            action_letter, consumed_items, added_items, environment_description = entry
            formatted_entry = f"Action: {action_letter}\n" \
                              f"Consumed Items: {', '.join(translate_item_name_list_to_letter(consumed_items))}\n" \
                              f"Added Items: {', '.join(translate_item_name_list_to_letter(added_items))}\n" \
                              f"Environment: {environment_description}\n" \
                              "----"
            formatted_prompt.append(formatted_entry)

        return f"The most recent {len(recent_memory)} records\n----\n" + "\n".join(formatted_prompt)

    def controller(self):
        # initial Minecraft instance
        options = {"mode": "hard"}
        if self.reset_position:
            options["position"] = self.reset_position
        if self.track_player:
            options["track_player"] = True
        self.env.reset(options=options)
        result = self.env.step('')
        if not self.visual_api_started:
            self.run_visual_API()
        while True:
            environment_description = get_image_description(local_mllm_port=self.local_mllm_port)
            latest_payload = result[0][1]
            if all(item in translate_item_name_list_to_letter(latest_payload['inventory'].keys()) for item in
                   self.goal_item_letters):
                subtask = 'Achieve the environmental factors.'
            else:
                subtask = self.planner(latest_payload['inventory'])
            action = self.actor(subtask, environment_description)
            print('Action:', action)
            result = self.env.step(skill_loader(action))
            start_item = result[0][1]['inventory']
            result = self.env.step('')
            end_item = result[0][1]['inventory']
            print('Inventory now:', str(result[0][1]['inventory']))
            print('Voxels around:', str(result[0][1]['voxels']))
            consumed_items, added_items = get_item_changes(start_item, end_item)
            recorder(start_item, end_item, consumed_items, added_items, action, self.dataset_path)
            self.update_material_dict(end_item)
            self.update_memory(action, consumed_items, added_items, environment_description)
            if self.check_goal_completed(result):
                print("Controller goal completed. Stopping controller loop.")
                return

    def check_goal_completed(self, result):
        return all(item in translate_item_name_list_to_letter(result[0][1]['inventory'].keys()) for item in
                   self.goal_item_letters) and all(item in result[0][1]['voxels'] for item in self.goal[1])

    def learn_new_actions(self, candidate_actions=None):
        actions = candidate_actions or self.unlocked_actions
        for action in reversed(actions):
            if action not in self.learned_causal_subgraph.keys():
                self.record = self.init_record_structure(action)
                return self.causal_learning(translate_action_letter_to_name(action))
        return False

    def learn_planned_actions(self):
        if not self.planned_actions:
            self.plan_learning_path()
        learned_any = False
        for action in self.planned_actions:
            if action in self.learned_causal_subgraph:
                continue
            action_name = translate_action_letter_to_name(action)
            self.record = self.init_record_structure(action)
            print(f"Learning planned action {action_name}")
            if self.causal_learning(action_name):
                learned_any = True
                if action not in self.learned_causal_subgraph:
                    print(
                        f"Planned action {action_name} sampled but no verified causal edge was added."
                    )
                if all(item in self.learned_items for item in self.goal_item_letters):
                    break
            else:
                print(f"Planned action {action_name} failed; will retry after replanning/fallback.")
                return learned_any, action
        return learned_any, None

    def explore(self, goal_item, goal_environment):
        self.goal = (goal_item, goal_environment)
        self.goal_item_letters = translate_item_name_list_to_letter(self.goal[0])
        self.plan_learning_path()
        failed_learning_rounds = 0
        while not all(item in self.learned_items for item in self.goal_item_letters):
            learned_any, failed_action = self.learn_planned_actions()
            if learned_any:
                failed_learning_rounds = 0
                continue
            failed_learning_rounds += 1
            reason = (
                f"Previous plan made no progress. Failed action: {failed_action}. "
                f"No-progress rounds: {failed_learning_rounds}. "
                "Analyze whether the current action order is missing prerequisites or includes irrelevant actions, "
                "then return a corrected ordered action plan."
            )
            print("Dynamic learning plan made no progress; asking LLM to analyze and replan.")
            self.plan_learning_path(reason=reason)

        self.print_plan_summary()
        print("Learning phase finished. Entering controller(). Post-controller supplemental learning is disabled.")
        try:
            self.controller()
        finally:
            self.stop_visual_API()
        return

    def stop_visual_API(self):
        if self.visual_api_monitor is not None:
            self.visual_api_monitor.stop()
            self.visual_api_monitor = None
        self.visual_api_started = False

    def run_visual_API(self):
        if self.visual_api_started:
            return
        python_executable = sys.executable
        script_path = os.path.join(os.getcwd(), 'Adam', "visual_API.py")
        commands = [python_executable, script_path]
        visual_env = os.environ.copy()
        visual_env["PYTHONUNBUFFERED"] = "1"
        visual_env["ADAM_VISUAL_API_URL"] = f"http://127.0.0.1:{self.env.visual_server_port}"
        visual_env["ADAM_VISUAL_IMAGE_DIR"] = os.environ.get(
            "ADAM_VISUAL_IMAGE_DIR",
            os.path.join("Adam", "game_image"),
        )
        monitor = SubprocessMonitor(
            commands=commands,
            name="VisualAPIMonitor",
            ready_match=r"Visual API Ready",
            log_path="logs",
            callback_match=r"Error",
            callback=lambda: print("Error detected in subprocess!"),
            finished_callback=lambda: print("Subprocess has finished."),
            env=visual_env,
        )
        monitor.run()
        self.visual_api_monitor = monitor
        self.visual_api_started = True
