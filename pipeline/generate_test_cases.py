"""
Test case generation stage
"""

import os
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
import itertools

from .base_pipeline import BasePipeline, process_with_strategy
from core.data_formats import TestCase, PipelineConfig
from core.unified_registry import UNIFIED_REGISTRY
from utils.logging_utils import log_with_context
from .resource_policy import policy_for_test_case_generation


class TestCaseGenerator(BasePipeline):
    """Test case generator"""

    def __init__(self, config: PipelineConfig):
        super().__init__(config, stage_name="test_case_generation")
        self.attack_configs = config.test_case_generation

    def load_behaviors(self) -> List[Dict]:
        """Load harmful behavior list"""

        # Define data parsing function
        def parse_behavior(item: Dict) -> Dict:
            """Parse behavior data item"""
            case_id = item.get("id")
            if not isinstance(item, dict):
                self.logger.warning(
                    f"Data item {case_id} is not a dictionary format: {type(item)}"
                )
                return None

            # Check if image_path field exists
            image_path = item.get("image_path", "")
            original_prompt = item.get("original_prompt", "")

            if not image_path:
                self.logger.warning(f"Data item {case_id} is missing image_path field")
                return None

            if not original_prompt:
                self.logger.warning(
                    f"Data item {case_id} is missing original_prompt field"
                )
                return None

            return {
                "case_id": case_id,
                "original_prompt": original_prompt,
                "image_path": image_path,
                "metadata": item,
            }

        # Get behavior file path
        behaviors_file = self.attack_configs.get("input", {}).get("behaviors_file")
        if not behaviors_file:
            self.logger.error("Harmful behavior file not specified")
            return []

        file_paths = [Path(behaviors_file)]

        # Use unified data loading method (automatically filter None values)
        behaviors = self.load_data_files(
            data_type="behavior data",
            file_paths=file_paths,
            data_parser=parse_behavior,
        )

        if not behaviors:
            self.logger.error("No valid behavior data found")

        return behaviors

    def get_behaviors_count(self) -> int:
        """Get behavior data count"""
        behaviors = self.load_behaviors()
        return len(behaviors)

    @log_with_context("Generate single test data")
    def generate_single_test_case(
        self,
        original_prompt: Dict,
        image_path: str,
        case_id: int,
        attack_name: str,
        attack_config: Dict[str, Any],
        image_save_dir: Path,
    ) -> Dict[str, Any]:
        """Generate single test case (for fine-grained parallel processing)"""
        attack = UNIFIED_REGISTRY.create_attack(
            attack_name, attack_config, output_image_dir=str(image_save_dir)
        )

        if attack is None:
            self.logger.error(
                f"Attack method '{attack_name}' failed to initialize. "
                f"This is likely due to missing dependencies or import errors. "
                f"Check logs above for details. Skipping this attack."
            )
            return None

        test_case = attack.generate_test_case(
            original_prompt,
            image_path,
            case_id,
        )
        return test_case

    def generate_single_attack_test_cases(
        self,
        attack_name: str,
        attack_config: Dict[str, Any],
        behaviors: List[Dict],
        batch_size: int = 10,
        output_file_path: Path = None,
        image_save_dir: Path = None,
    ) -> List[TestCase]:
        """Generate test cases using a single attack method (supports batch and parallel processing)"""
        try:
            self.logger.info(
                f"Starting to generate test cases using {attack_name} (batch size: {batch_size})"
            )

            combinations = []
            if behaviors:
                for behavior_item in behaviors:
                    image_path = behavior_item.get("image_path")
                    original_prompt = behavior_item.get("original_prompt")
                    case_id = behavior_item.get("case_id")
                    if image_path and original_prompt:
                        combinations.append((case_id, original_prompt, image_path))

                self.logger.info(f"Will generate {len(combinations)} test cases")

            if not combinations:
                self.logger.warning(
                    f"No available combinations (attack: {attack_name})"
                )
                return []

            # Create attack instance to check load_model attribute
            attack = UNIFIED_REGISTRY.create_attack(
                attack_name, attack_config, output_image_dir=str(image_save_dir)
            )

            if attack is None:
                self.logger.error(
                    f"Attack method '{attack_name}' failed to initialize. "
                    f"This is likely due to missing dependencies or import errors. "
                    f"Check logs above for details. Skipping this attack."
                )
                return []

            # Unified resource policy (single source of truth)
            policy = policy_for_test_case_generation(
                attack_config, default_max_workers=self.config.max_workers
            )
            self.logger.info(
                f"Resource policy for attack={attack_name}: strategy={policy.strategy}, max_workers={policy.max_workers} ({policy.reason})"
            )

            if policy.strategy == "batched":
                # Attacks that need to load local models: batch processing, reuse the same attack instance, max_workers=1
                self.logger.info(
                    f"Attack method {attack_name} needs to load local model, using batch processing (reusing attack instance)"
                )
                return self._generate_test_cases_batched(
                    attack, combinations, batch_size, output_file_path, attack_name
                )
            else:
                # Attacks that don't need to load local models: parallel processing
                self.logger.info(
                    f"Attack method {attack_name} doesn't need to load local model, using multi-threaded parallel processing"
                )
                return self._generate_test_cases_parallel(
                    attack_name,
                    attack_config,
                    combinations,
                    batch_size,
                    output_file_path,
                    image_save_dir,
                    max_workers_override=policy.max_workers,
                )

        except Exception as e:
            self.logger.error(f"Failed to generate test cases using {attack_name}: {e}")
            return []

    def _generate_test_cases_batched(
        self,
        attack,
        combinations: List[Tuple[int, str, str]],
        batch_size: int,
        output_file_path: Path,
        attack_name: str,
    ) -> List[TestCase]:
        """Batch generate test cases, reuse the same attack instance (for attacks that need to load local models)"""
        from .base_pipeline import batch_save_context
        from tqdm import tqdm

        test_cases = []
        total_batches = (len(combinations) + batch_size - 1) // batch_size

        with batch_save_context(
            pipeline=self,
            output_filename=output_file_path,
            batch_size=batch_size,
            total_items=len(combinations),
            desc=f"Generate test cases ({attack_name})",
        ) as save_manager:
            for batch_idx in range(0, len(combinations), batch_size):
                batch = combinations[batch_idx : batch_idx + batch_size]
                batch_num = batch_idx // batch_size + 1

                self.logger.debug(
                    f"Processing batch {batch_num}/{total_batches}, contains {len(batch)} items"
                )

                # Process each item in current batch (sequential processing, reuse attack instance)
                for case_id, original_prompt, image_path in batch:
                    try:
                        test_case = attack.generate_test_case(
                            original_prompt,
                            image_path,
                            case_id,
                        )
                        if test_case:
                            save_manager.add_result(test_case.to_dict())
                            test_cases.append(test_case)
                    except Exception as e:
                        self.logger.error(
                            f"Failed to generate test case (attack: {attack_name}, image: {image_path}): {e}"
                        )

                self.logger.debug(f"Batch {batch_num}/{total_batches} completed")

        self.logger.info(f"{attack_name} generated {len(test_cases)} test cases")
        return test_cases

    def _generate_test_cases_parallel(
        self,
        attack_name: str,
        attack_config: Dict[str, Any],
        combinations: List[Tuple[int, str, str]],
        batch_size: int,
        output_file_path: Path,
        image_save_dir: Path,
        max_workers_override: int | None = None,
    ) -> List[TestCase]:
        """Parallel generate test cases (for attacks that don't need to load local models)"""

        # Prepare processing function
        def process_combination(item: Tuple[int, str, str]) -> Dict[str, Any]:
            case_id, original_prompt, image_path = item
            try:
                test_case = self.generate_single_test_case(
                    original_prompt,
                    image_path,
                    case_id,
                    attack_name,
                    attack_config,
                    image_save_dir,
                )
                return test_case.to_dict()
            except Exception as e:
                self.logger.error(
                    f"Failed to generate test case (attack: {attack_name}, image: {image_path}): {e}"
                )
                return None

        # Use strategy processing and batch saving
        results_dicts = process_with_strategy(
            items=combinations,
            process_func=process_combination,
            pipeline=self,
            output_filename=output_file_path,
            batch_size=batch_size,
            max_workers=max_workers_override,  # Use policy-decided value (or default)
            strategy_name="parallel",
            desc=f"Generate test cases ({attack_name})",
        )

        # Convert to TestCase objects
        test_cases = []
        for result_dict in results_dicts:
            if result_dict:
                try:
                    test_case = TestCase.from_dict(result_dict)
                    test_cases.append(test_case)
                except Exception as e:
                    self.logger.warning(f"Failed to parse test case dictionary: {e}")

        self.logger.info(f"{attack_name} generated {len(test_cases)} test cases")
        return test_cases

    def run(self, **kwargs) -> List[TestCase]:
        """Run test case generation, supports real-time batch saving"""
        if not self.validate_config():
            return []

        # Get batch size parameter (priority: kwargs parameter, then configuration parameter)
        batch_size = kwargs.get("batch_size", self.config.batch_size)
        self.logger.info(
            f"Starting test case generation stage (batch size: {batch_size})"
        )

        # Load data
        behaviors = self.load_behaviors()
        if not behaviors:
            self.logger.error(
                "Behavior data loading failed, cannot generate test cases"
            )
            return []

        self.logger.info(f"Loaded {len(behaviors)} data items")

        # Get attack methods to run
        attack_names = self.attack_configs.get("attacks", [])
        if not attack_names:
            self.logger.error("Attack methods not specified")
            return []

        # Process all attack methods
        pending_attacks = []
        for attack_name in attack_names:
            attack_config = self.attack_configs.get("attack_params", {}).get(
                attack_name, {}
            )
            task_config = {
                "attack_name": attack_name,
                "attack_config": attack_config,
                "behaviors_count": len(behaviors),
            }
            task_id = f"{attack_name}_{self.get_task_hash(task_config)}"
            pending_attacks.append((attack_name, attack_config, task_id))

        self.logger.info(f"Need to process {len(pending_attacks)} attack methods")

        # Check if each attack method has generated complete test cases
        completed_attacks = []
        pending_attacks_to_process = []

        for attack_name, attack_config, task_id in pending_attacks:
            # Generate separate filename for each attack method
            image_save_dir, output_file_path = self._generate_filename(
                "test_case_generation",
                attack_name=attack_name,
                target_model_name=attack_config.get("target_model_name", None),
            )

            # Calculate expected test case count for this attack method
            expected_count = self._calculate_expected_test_cases(behaviors)

            # Check existing test case files
            existing_test_cases = self.load_results(output_file_path)

            if len(existing_test_cases) == expected_count:
                self.logger.info(
                    f"Attack method {attack_name} has complete test cases: {len(existing_test_cases)}/{expected_count}"
                )
                completed_attacks.append(
                    (attack_name, attack_config, task_id, output_file_path)
                )
            else:
                self.logger.info(
                    f"Attack method {attack_name} needs to generate test cases: {len(existing_test_cases)}/{expected_count}"
                )
                pending_attacks_to_process.append(
                    (
                        attack_name,
                        attack_config,
                        task_id,
                        image_save_dir,
                        output_file_path,
                        expected_count,
                    )
                )

        # If all attack methods are completed, directly load existing results
        if not pending_attacks_to_process:
            self.logger.info("All attack methods completed, loading existing results")
            all_test_cases = []
            for (
                attack_name,
                attack_config,
                task_id,
                output_file_path,
            ) in completed_attacks:
                existing_results = self.load_results(output_file_path)
                for item in existing_results:
                    try:
                        test_case = TestCase.from_dict(item)
                        all_test_cases.append(test_case)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to parse test case ({attack_name}): {e}"
                        )
                self.logger.info(
                    f"Loaded {len(existing_results)} test cases from {output_file_path}"
                )

            self.logger.info(f"Total loaded {len(all_test_cases)} test cases")
            return all_test_cases

        self.logger.info(
            f"Need to process {len(pending_attacks_to_process)} attack methods"
        )

        all_test_cases = []

        # First load test cases from completed attack methods
        for attack_name, attack_config, task_id, output_file_path in completed_attacks:
            existing_results = self.load_results(output_file_path)
            for item in existing_results:
                try:
                    test_case = TestCase.from_dict(item)
                    all_test_cases.append(test_case)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to parse test case ({attack_name}): {e}"
                    )
            self.logger.info(
                f"Loaded {len(existing_results)} test cases from {output_file_path}"
            )

        for (
            attack_name,
            attack_config,
            task_id,
            image_save_dir,
            output_file_path,
            expected_count,
        ) in pending_attacks_to_process:
            try:
                test_cases = self.generate_single_attack_test_cases(
                    attack_name,
                    attack_config,
                    behaviors,
                    batch_size,
                    output_file_path,
                    image_save_dir,
                )

                if test_cases:
                    all_test_cases.extend(test_cases)

                    self.logger.info(
                        f"{attack_name} completed, generated {len(test_cases)} test cases, saved to {output_file_path}, current total {len(all_test_cases)} test cases"
                    )

                else:
                    self.logger.warning(
                        f"{attack_name} did not generate any test cases"
                    )

            except Exception as e:
                self.logger.error(f"{attack_name} execution failed: {e}")

        if all_test_cases:
            self.logger.info(
                f"Test case generation completed, generated {len(all_test_cases)} test cases in total"
            )
        else:
            self.logger.warning("No test cases generated")

        return all_test_cases

    def _calculate_expected_test_cases(
        self,
        behaviors: List[Dict],
    ) -> int:
        """Calculate expected test case count for this attack method"""
        if not behaviors:
            return 0

        count = 0
        for behavior_item in behaviors:
            image_path = behavior_item.get("image_path")
            original_prompt = behavior_item.get("original_prompt")
            if image_path and original_prompt:
                count += 1

        return count

    def validate_config(self) -> bool:
        """Validate configuration"""
        if not super().validate_config():
            return False

        if not self.attack_configs.get("input").get("behaviors_file"):
            self.logger.error("Input behavior file not specified")
            return False

        # Check if behavior data contains image_path field
        behaviors_file = self.attack_configs.get("input").get("behaviors_file")
        has_image_path_in_data = False

        if behaviors_file:
            try:
                import json
                from pathlib import Path

                behaviors_path = Path(behaviors_file)
                if behaviors_path.exists():
                    with open(behaviors_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Check if data contains image_path field
                    if isinstance(data, list) and len(data) > 0:
                        first_item = data[0]
                        if isinstance(first_item, dict) and "image_path" in first_item:
                            has_image_path_in_data = True
                            self.logger.info(
                                "Behavior data contains image_path field, will use these paths to load images"
                            )
                        else:
                            self.logger.warning(
                                "Behavior data does not contain image_path field, will not be able to load images"
                            )
            except Exception as e:
                self.logger.warning(f"Error checking behavior data format: {e}")

        # Now require behavior data to contain image_path field
        if not has_image_path_in_data:
            self.logger.error(
                "Behavior data does not contain image_path field, currently only supports data format with image_path field"
            )
            return False

        if not self.attack_configs.get("attacks"):
            self.logger.error("Attack methods not specified")
            return False

        return True
