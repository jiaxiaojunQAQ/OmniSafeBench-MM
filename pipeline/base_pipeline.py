"""
Pipeline base class
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Union, Callable, Iterator
import os
import json
import logging
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from contextlib import contextmanager

from core.data_formats import PipelineConfig
from .batch_strategies import BatchStrategyFactory, BatchProcessingStrategy


class BasePipeline(ABC):
    """Pipeline base class"""

    def __init__(self, config: PipelineConfig, stage_name: str = None):
        self.config = config
        self.logger = self._setup_logger()
        if stage_name:
            stage_dir = self._get_stage_dir_name(stage_name)
            self.output_dir = Path(config.system["output_dir"]) / stage_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_stage_dir_name(self, stage_name: str) -> str:
        """Convert stage name to directory name"""
        stage_mapping = {
            "test_case_generation": "test_cases",
            "response_generation": "responses",
            "evaluation": "evaluations",
        }
        return stage_mapping.get(stage_name, stage_name)

    def _generate_filename(self, stage_name: str, **context) -> str:
        """Generate filename based on stage and context

        Args:
            stage_name: Stage name
            **context: Context information, such as attack method, model, defense method, etc.

        Returns:
            Generated filename
        """
        if stage_name:
            stage_dir = self._get_stage_dir_name(stage_name)
            output_dir = Path(self.config.system["output_dir"]) / stage_dir
            output_dir.mkdir(parents=True, exist_ok=True)

        if stage_name == "test_case_generation":
            # Test case filename: Each attack method is saved to its own folder
            attack_name = context.get("attack_name")
            target_model_name = context.get("target_model_name", None)

            # Create attack method folder
            attack_dir = output_dir / attack_name
            attack_dir.mkdir(parents=True, exist_ok=True)

            # Determine image subfolder name
            if target_model_name:
                image_subdir_name = target_model_name
            else:
                image_subdir_name = "images"

            # Create image subfolder
            image_dir = attack_dir / image_subdir_name
            image_dir.mkdir(parents=True, exist_ok=True)

            if target_model_name:
                return (
                    image_dir,
                    attack_dir / f"target_model_{target_model_name}_test_cases.jsonl",
                )
            else:
                return image_dir, attack_dir / "test_cases.jsonl"

        elif stage_name == "response_generation":
            # Response filename: Each model+defense method combination is saved separately
            attack_name = context.get("attack_name")
            model_name = context.get("model_name")
            defense_name = context.get("defense_name")
            target_model_name = context.get("target_model_name", None)
            parts = []
            if attack_name:
                parts.append(f"attack_{attack_name}")
            if target_model_name:
                parts.append(f"target_model_{target_model_name}")
            if model_name:
                parts.append(f"model_{model_name}")

            if parts:
                # Create defense method folder
                defense_dir = output_dir / defense_name
                defense_dir.mkdir(parents=True, exist_ok=True)
                return None, defense_dir / f"{'_'.join(parts)}.jsonl"
            else:
                raise

        elif stage_name == "evaluation":
            # Evaluation filename: Each attack method+model+defense method combination is saved separately
            attack_name = context.get("attack_name")
            model_name = context.get("model_name")
            defense_name = context.get("defense_name")
            evaluator_name = context.get("evaluator_name", None)
            target_model_name = context.get("target_model_name", None)
            parts = []
            if attack_name:
                parts.append(f"attack_{attack_name}")
            if target_model_name:
                parts.append(f"target_model_{target_model_name}")
            if model_name:
                parts.append(f"model_{model_name}")
            if defense_name:
                parts.append(f"defense_{defense_name}")
            # IMPORTANT: evaluator dimension must be part of the filename to avoid collisions
            if evaluator_name:
                parts.append(f"evaluator_{evaluator_name}")
            if parts:
                return None, output_dir / f"{'_'.join(parts)}.jsonl"
            else:
                raise
        else:
            raise

    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger(self.__class__.__name__)
        # Do not add handler, use root logger configuration
        # Set propagate=True to let logs propagate to root logger
        logger.propagate = True
        # Do not set handler, let root logger handle output
        logger.setLevel(logging.INFO)
        return logger

    def load_results(self, filepath: Path) -> List[Dict]:
        """Load results from file

        Args:
            filepath: File path
        """
        if not filepath.exists():
            return []
        return self._load_json_or_jsonl(filepath)

    def _load_json_or_jsonl(self, filepath: Path) -> List[Dict]:
        """Load a JSON list file (.json) or JSON Lines file (.jsonl) into a list of dicts."""
        # Prefer suffix hint; otherwise sniff by first non-whitespace char.
        suffix = filepath.suffix.lower()
        try:
            if suffix == ".jsonl":
                return self._load_jsonl(filepath)
            if suffix == ".json":
                return self._load_json_list(filepath)
        except Exception:
            # If suffix-based attempt fails, fall back to sniffing
            pass

        # Sniff format
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.lstrip()
                if not stripped:
                    continue
                if stripped.startswith("["):
                    return self._load_json_list(filepath)
                return self._load_jsonl(filepath)
        return []

    def _load_json_list(self, filepath: Path) -> List[Dict]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"Unsupported JSON format {filepath}: expected list, got {type(data).__name__}"
            )
        return data

    def _load_jsonl(self, filepath: Path) -> List[Dict]:
        items: List[Dict] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"JSONL parsing failed {filepath}: line {line_no}: {e.msg}"
                    ) from e
                if obj is None:
                    continue
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Unsupported JSONL item {filepath}: line {line_no}: expected dict, got {type(obj).__name__}"
                    )
                items.append(obj)
        return items

    def load_data_files(
        self,
        data_type: str,
        config_key: str = None,
        file_paths: List[Path] = None,
        file_finder: Callable[[], List[Path]] = None,
        data_parser: Callable[[Dict], Any] = None,
    ) -> List[Any]:
        """Unified data file loading method

        Args:
            data_type: Data type name (for logging), such as "test cases", "model responses", "behavior data"
            config_key: Input file key name in config, such as "input_test_cases", "input_responses"
            file_paths: Directly specified file path list (priority)
            file_finder: File finder function that returns list of file paths to load
            data_parser: Data parser function that converts JSON dict to data object

        Returns:
            List of loaded data objects
        """
        all_data = []

        # Priority: use directly specified file paths
        if file_paths:
            for file_path in file_paths:
                try:
                    if file_path.exists():
                        data = self._load_single_data_file(file_path, data_parser)
                        all_data.extend(data)
                        self.logger.info(f"Loaded {len(data)} {data_type} from {file_path}")
                    else:
                        self.logger.warning(f"{data_type} file does not exist: {file_path}")
                except Exception as e:
                    self.logger.error(
                        f"Failed to load {data_type} file {file_path}: {e}",
                        exc_info=True,
                    )
            if all_data:
                self.logger.info(f"Total loaded {len(all_data)} {data_type}")
                return all_data

        # Second: use files specified in config
        if config_key:
            # Try to get from various stage configurations
            config = None
            for attr_name in [
                "attack_configs",
                "response_configs",
                "evaluation_configs",
            ]:
                if hasattr(self, attr_name):
                    config = getattr(self, attr_name)
                    if isinstance(config, dict) and config_key in config:
                        break

            if config and isinstance(config, dict):
                input_file = config.get(config_key)
                if input_file:
                    try:
                        file_path = Path(input_file)
                        if file_path.exists():
                            data = self._load_single_data_file(file_path, data_parser)
                            all_data.extend(data)
                            self.logger.info(
                                f"Loaded {len(data)} {data_type} from config-specified file {file_path}"
                            )
                            return all_data
                        else:
                            self.logger.error(
                                f"Config-specified {data_type} file does not exist: {file_path} (absolute path: {file_path.absolute()})"
                            )
                            return []
                    except Exception as e:
                        self.logger.error(
                            f"Failed to process config-specified {data_type} file path ({input_file}): {e}",
                            exc_info=True,
                        )
                        return []

        # Finally: use file finder function to auto-find
        if file_finder:
            try:
                found_files = file_finder()
                for file_path in found_files:
                    try:
                        if file_path.exists():
                            data = self._load_single_data_file(file_path, data_parser)
                            all_data.extend(data)
                            self.logger.info(f"Loaded {len(data)} {data_type} from {file_path}")
                        else:
                            self.logger.warning(f"{data_type} file does not exist: {file_path}")
                    except Exception as e:
                        self.logger.error(
                            f"Failed to load {data_type} file {file_path}: {e}",
                            exc_info=True,
                        )
            except Exception as e:
                self.logger.error(
                    f"Failed to execute file finder function: {e}",
                    exc_info=True,
                )

        if all_data:
            self.logger.info(f"Total loaded {len(all_data)} {data_type}")
        else:
            self.logger.warning(f"No {data_type} loaded")

        return all_data

    def _load_single_data_file(
        self, file_path: Path, data_parser: Callable[[Dict], Any] = None
    ) -> List[Any]:
        """Load single data file

        Args:
            file_path: File path
            data_parser: Data parser function that converts JSON dict to data object

        Returns:
            List of data objects
        """
        try:
            # Support both JSON list and JSONL
            data_list = self._load_json_or_jsonl(file_path)

            parsed_data = []
            parse_failures = 0
            for idx, item in enumerate(data_list):
                try:
                    if data_parser:
                        parsed_item = data_parser(item)
                        # Filter out None values (failed parsing data)
                        if parsed_item is not None:
                            parsed_data.append(parsed_item)
                        else:
                            parse_failures += 1
                            self.logger.debug(
                                f"Data item {idx} parsed to None, filtered (file: {file_path})"
                            )
                    else:
                        parsed_data.append(item)
                except Exception as e:
                    parse_failures += 1
                    item_id = item.get("id") if isinstance(item, dict) else idx
                    self.logger.warning(
                        f"Failed to parse data item (file: {file_path}, index: {idx}, ID: {item_id}): {type(e).__name__}: {e}",
                        exc_info=False,
                    )

            if parse_failures > 0:
                self.logger.warning(
                    f"File {file_path} has {parse_failures} data items that failed to parse, skipped"
                )

            return parsed_data
        except FileNotFoundError:
            self.logger.error(
                f"File does not exist: {file_path} (absolute path: {file_path.absolute()})"
            )
            return []
        except PermissionError:
            self.logger.error(f"No permission to read file: {file_path}")
            return []
        except Exception as e:
            self.logger.error(
                f"Failed to load data file {file_path}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return []

    def save_results_incrementally(self, results: List[Dict], filepath: Path) -> str:
        """Incrementally save results to file"""
        # filepath = self.output_dir / filename
        # Default to JSONL if suffix is .jsonl; otherwise keep JSON list behavior.
        if filepath.suffix.lower() == ".jsonl":
            return self._append_results_jsonl(results, filepath)
        return self._save_results_json_list_dedup(results, filepath)

    def _append_results_jsonl(self, results: List[Dict], filepath: Path) -> str:
        """Append results to a JSONL file with basic dedup by test_case_id (read ids once)."""
        existing_ids: Set[str] = set()
        if filepath.exists():
            try:
                for item in self._load_jsonl(filepath):
                    rid = item.get("test_case_id")
                    if rid:
                        existing_ids.add(rid)
            except Exception as e:
                self.logger.warning(f"Failed to read existing JSONL for dedup: {filepath}: {e}")

        new_count = 0
        with open(filepath, "a", encoding="utf-8") as f:
            for result in results:
                rid = result.get("test_case_id")
                if rid and rid in existing_ids:
                    continue
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                new_count += 1
                if rid:
                    existing_ids.add(rid)

        self.logger.info(
            f"Incremental JSONL append completed: {new_count} new results, file={filepath}"
        )
        return str(filepath)

    def _save_results_json_list_dedup(self, results: List[Dict], filepath: Path) -> str:
        """Legacy JSON list incremental save (atomic rewrite with dedup)."""
        existing_results = []
        if filepath.exists():
            try:
                existing_results = self._load_json_list(filepath)
                self.logger.info(f"Loaded {len(existing_results)} existing results")
            except Exception as e:
                self.logger.warning(f"Failed to load existing results: {e}")

        results_by_id: Dict[str, Dict] = {}
        for result in existing_results:
            result_id = result.get("test_case_id")
            if result_id:
                results_by_id[result_id] = result
            else:
                results_by_id[f"__no_id_{id(result)}"] = result

        new_count = 0
        for result in results:
            result_id = result.get("test_case_id")
            if result_id:
                if result_id not in results_by_id:
                    new_count += 1
                results_by_id[result_id] = result
            else:
                results_by_id[f"__no_id_{id(result)}"] = result
                new_count += 1

        all_results = list(results_by_id.values())
        temp_file = filepath.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            temp_file.rename(filepath)
            self.logger.info(
                f"Incremental save completed: {new_count} new results, total {len(all_results)} results"
            )
        except Exception as e:
            self.logger.error(f"Incremental save failed: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise
        return str(filepath)

    def save_single_result(self, result: Dict, filename: str) -> str:
        """Save single result to file (append mode)

        Args:
            result: The result to save
            filename: File path (can be absolute or relative to output_dir)

        Returns:
            str: Path to the saved file
        """
        # Handle both absolute and relative paths
        filepath = Path(filename)
        if not filepath.is_absolute():
            filepath = self.output_dir / filename

        # Atomic write: write to temporary file first
        temp_file = filepath.with_suffix(".tmp")

        try:
            # If file exists, load existing results
            existing_results = []
            if filepath.exists():
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        existing_results = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load existing results: {e}")

            # Use dictionary to deduplicate, ensure each test_case_id only keeps one result
            result_id = result.get("test_case_id")
            if result_id:
                # Build deduplication dictionary
                results_by_id = {}
                for existing in existing_results:
                    existing_id = existing.get("test_case_id")
                    if existing_id:
                        results_by_id[existing_id] = existing
                    else:
                        results_by_id[f"__no_id_{id(existing)}"] = existing

                # Add or replace new result
                results_by_id[result_id] = result

                # Convert back to list
                existing_results = list(results_by_id.values())
            else:
                # No test_case_id, directly append
                existing_results.append(result)

            # Write to temporary file
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(existing_results, f, ensure_ascii=False, indent=2)

            # Rename temporary file to official file
            temp_file.rename(filepath)

            self.logger.debug(f"Saved single result: {result_id or 'Unknown ID'}")

        except Exception as e:
            self.logger.error(f"Failed to save single result: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise

        return str(filepath)

    def get_task_hash(self, task_config: Dict) -> str:
        """Get hash value of task configuration for task identification"""
        config_str = json.dumps(task_config, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:16]

    def get_progress_bar(self, total: int, desc: str) -> tqdm:
        """Get progress bar"""
        return tqdm(total=total, desc=desc, ncols=100)

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Run pipeline"""
        pass

    def validate_config(self) -> bool:
        """Validate configuration"""
        # Basic validation
        if not self.output_dir:
            self.logger.error("Output directory is not set")
            return False
        return True


class BatchSaveManager:
    """Batch save manager for real-time result saving

    Features:
    1. Automatically save when specified number of results are collected
    2. Thread-safe, supports multi-threaded environment
    3. Supports graceful shutdown (save remaining results)
    4. Supports progress tracking
    """

    def __init__(
        self,
        pipeline: BasePipeline,
        output_filename: Path,
        batch_size: int = None,
        flush_on_exit: bool = True,
    ):
        """
        Initialize batch save manager

        Args:
            pipeline: BasePipeline instance
            output_filename: Output file path
            batch_size: Batch size, save once when this many results are collected, defaults to batch_size in config
            flush_on_exit: Whether to automatically save remaining results on exit
        """
        self.pipeline = pipeline
        self.output_filename = output_filename
        self.batch_size = batch_size if batch_size is not None else pipeline.config.batch_size
        self.flush_on_exit = flush_on_exit

        # Result buffer
        self.buffer: List[Dict] = []
        self.total_saved = 0
        self.lock = Lock()

        # Progress tracking
        self.progress_bar = None

    def add_result(self, result: Dict) -> None:
        """Add result to buffer, automatically save if batch size is reached"""
        with self.lock:
            self.buffer.append(result)

            # Check if save is needed
            if len(self.buffer) >= self.batch_size:
                self._flush_unlocked()

    def add_results(self, results: List[Dict]) -> None:
        """Batch add results"""
        with self.lock:
            self.buffer.extend(results)

            # If result count exceeds batch size, may need multiple saves
            while len(self.buffer) >= self.batch_size:
                batch = self.buffer[: self.batch_size]
                self.buffer = self.buffer[self.batch_size :]
                self._save_batch(batch)

    def _flush_buffer(self) -> None:
        """Save all results in buffer (thread-safe public method)"""
        with self.lock:
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        """Internal flush method, caller must hold lock.

        This is the actual implementation that flushes the buffer.
        It must only be called while holding self.lock.
        """
        if not self.buffer:
            return
        batch = self.buffer.copy()
        self.buffer.clear()
        self._save_batch(batch)

    def _save_batch(self, batch: List[Dict]) -> None:
        """Save a batch of results

        Note: This method may be called while holding self.lock.
        Any exception handling must NOT attempt to acquire the lock again.
        """
        try:
            # Use incremental save method
            self.pipeline.save_results_incrementally(batch, self.output_filename)

            self.total_saved += len(batch)

            # Update progress bar
            if self.progress_bar:
                self.progress_bar.update(len(batch))

            self.pipeline.logger.debug(
                f"Batch save: {len(batch)} results, total {self.total_saved}"
            )

        except Exception as e:
            self.pipeline.logger.error(f"Batch save failed: {e}")
            # Put failed results back into buffer
            # IMPORTANT: Do NOT acquire lock here - caller already holds it
            # (see _flush_unlocked and add_results methods)
            self.buffer.extend(batch)
            raise

    def flush(self) -> None:
        """Force save all remaining results in buffer"""
        self._flush_buffer()

    def close(self) -> None:
        """Close manager, save remaining results"""
        if self.flush_on_exit:
            self.flush()

        if self.progress_bar:
            self.progress_bar.close()

    def set_progress_bar(self, progress_bar: tqdm) -> None:
        """Set progress bar for progress tracking"""
        self.progress_bar = progress_bar

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit, automatically save remaining results"""
        self.close()
        return False  # Do not suppress exceptions


@contextmanager
def batch_save_context(
    pipeline: BasePipeline,
    output_filename: Path,
    batch_size: int = None,
    total_items: int = None,
    desc: str = "Processing",
) -> Iterator[BatchSaveManager]:
    """
    Batch save context manager

    Args:
        pipeline: BasePipeline instance
        output_filename: Output file path
        batch_size: Batch size, defaults to batch_size in config
        total_items: Total number of items (for progress bar)
        desc: Progress bar description

    Yields:
        BatchSaveManager instance
    """
    if batch_size is None:
        batch_size = pipeline.config.batch_size

    manager = BatchSaveManager(
        pipeline=pipeline, output_filename=output_filename, batch_size=batch_size
    )

    # Set progress bar
    if total_items is not None:
        progress_bar = pipeline.get_progress_bar(total_items, desc)
        manager.set_progress_bar(progress_bar)

    try:
        yield manager
    finally:
        manager.close()


def process_with_strategy(
    items: List[Any],
    process_func: Callable[[Any], Dict],
    pipeline: BasePipeline,
    output_filename: Path,
    batch_size: int = None,
    max_workers: int = None,
    desc: str = "Processing",
    strategy_name: str = None,
    strategy_kwargs: Dict[str, Any] = None,
) -> List[Dict]:
    """
    Process items with specified batch processing strategy and save results in real-time

    Args:
        items: List of items to process
        process_func: Function to process single item, returns result dictionary
        pipeline: BasePipeline instance
        output_filename: Output file path
        batch_size: Batch size, defaults to batch_size in config
        max_workers: Maximum number of worker threads, defaults to max_workers in config
        desc: Progress bar description
        strategy_name: Batch processing strategy name, if None uses batch_strategy in config
        strategy_kwargs: Additional parameters to pass to strategy

    Returns:
        List of all processing results
    """
    if batch_size is None:
        batch_size = pipeline.config.batch_size
    if max_workers is None:
        max_workers = pipeline.config.max_workers

    # Determine strategy name (default is parallel)
    if strategy_name is None:
        strategy_name = getattr(
            pipeline.config, "batch_strategy", None
        ) or pipeline.config.system.get("batch_strategy", "parallel")

    # Prepare strategy parameters
    if strategy_kwargs is None:
        strategy_kwargs = {}

    # All strategies need batch_size parameter
    strategy_kwargs.setdefault("batch_size", batch_size)
    # max_workers is passed through process method, not as initialization parameter
    strategy_kwargs.pop("max_workers", None)

    strategy = BatchStrategyFactory.create_strategy(strategy_name, **strategy_kwargs)

    pipeline.logger.info(f"Using batch processing strategy: {strategy.get_name()}")

    all_results = []

    with batch_save_context(
        pipeline=pipeline,
        output_filename=output_filename,
        batch_size=batch_size,
        total_items=len(items),
        desc=desc,
    ) as save_manager:
        # Define wrapper function to add results to save manager
        def wrapped_process_func(item):
            try:
                result = process_func(item)
                if result is not None:
                    save_manager.add_result(result)
                return result
            except Exception as e:
                pipeline.logger.error(f"Failed to process item {item}: {e}")
                return None

        # Use strategy to process items
        results = strategy.process(
            items=items,
            process_func=wrapped_process_func,
            max_workers=max_workers,
            logger=pipeline.logger,
        )

        # Filter out None results
        all_results = [r for r in results if r is not None]

    return all_results


def parallel_process_with_batch_save(
    items: List[Any],
    process_func: Callable[[Any], Dict],
    pipeline: BasePipeline,
    output_filename: Path,
    batch_size: int = None,
    max_workers: int = None,
    desc: str = "Processing",
) -> List[Dict]:
    """
    Backward-compatible helper for tests/older code.

    This is equivalent to `process_with_strategy(..., strategy_name="parallel")`.
    """
    return process_with_strategy(
        items=items,
        process_func=process_func,
        pipeline=pipeline,
        output_filename=output_filename,
        batch_size=batch_size,
        max_workers=max_workers,
        desc=desc,
        strategy_name="parallel",
    )
