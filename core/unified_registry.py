"""
Unified registry system - merged version, removed redundant code
"""

from typing import Dict, Any, Optional, Type, List, TYPE_CHECKING
import logging
import importlib
from typing import Set
from pathlib import Path
import yaml

# NOTE:
# - `core.base_classes` is the canonical place for component contracts (BaseAttack/BaseModel/BaseDefense/BaseEvaluator).
# - This registry module should NOT re-export Base* at runtime; keep imports here type-checking only.

if TYPE_CHECKING:
    from .base_classes import BaseAttack, BaseModel, BaseDefense, BaseEvaluator


class UnifiedRegistry:
    """Unified registry, only retains lazy mapping/explicit registration, no longer uses decorator paths"""

    def __init__(self):
        self.attack_registry: Dict[str, Type["BaseAttack"]] = {}
        self.model_registry: Dict[str, Type["BaseModel"]] = {}
        self.defense_registry: Dict[str, Type["BaseDefense"]] = {}
        self.evaluator_registry: Dict[str, Type["BaseEvaluator"]] = {}

        self.logger = logging.getLogger(__name__)
        # Imported module cache to avoid duplicate imports
        self._imported_modules: Set[str] = set()
        # Lazy mapping cache (names only). This should not trigger heavy imports.
        self._lazy_mappings_cache: Optional[Dict[str, Dict[str, Any]]] = None

    def _get_lazy_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Get lazy import mappings from config/plugins.yaml (cached)."""
        if self._lazy_mappings_cache is not None:
            return self._lazy_mappings_cache
        self._lazy_mappings_cache = self._load_plugins_yaml()
        return self._lazy_mappings_cache

    def _load_plugins_yaml(self) -> Dict[str, Dict[str, Any]]:
        """Load plugin mappings from `config/plugins.yaml`."""
        # Repo root = parent of `core/`
        repo_root = Path(__file__).resolve().parent.parent
        plugins_file = repo_root / "config" / "plugins.yaml"
        if not plugins_file.exists():
            raise FileNotFoundError(
                f"config/plugins.yaml not found at {plugins_file}. This project requires config/plugins.yaml."
            )

        with open(plugins_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        plugins = data.get("plugins") or {}
        mappings: Dict[str, Dict[str, Any]] = {}
        for section in ("attacks", "models", "defenses", "evaluators"):
            section_map = plugins.get(section) or {}
            converted: Dict[str, Any] = {}
            for name, spec in section_map.items():
                if not isinstance(spec, (list, tuple)) or len(spec) != 2:
                    raise ValueError(
                        f"Invalid plugin spec for {section}.{name}: expected [module, class], got {spec!r}"
                    )
                module_path, class_name = spec
                converted[name] = (module_path, class_name)
            mappings[section] = converted

        # Ensure all sections exist
        mappings.setdefault("attacks", {})
        mappings.setdefault("models", {})
        mappings.setdefault("defenses", {})
        mappings.setdefault("evaluators", {})
        return mappings

    def register_attack(self, name: str, attack_class: Type["BaseAttack"]) -> None:
        """Register attack method"""
        if name in self.attack_registry:
            self.logger.warning(
                f"Attack method '{name}' already exists, will be overwritten"
            )
        self.attack_registry[name] = attack_class
        self.logger.debug(f"Registered attack method: {name}")

    def register_model(self, name: str, model_class: Type["BaseModel"]) -> None:
        """Register model"""
        if name in self.model_registry:
            self.logger.warning(f"Model '{name}' already exists, will be overwritten")
        self.model_registry[name] = model_class
        self.logger.debug(f"Registered model: {name}")

    def register_defense(self, name: str, defense_class: Type["BaseDefense"]) -> None:
        """Register defense method"""
        if name in self.defense_registry:
            self.logger.warning(
                f"Defense method '{name}' already exists, will be overwritten"
            )
        self.defense_registry[name] = defense_class
        self.logger.debug(f"Registered defense method: {name}")

    def register_evaluator(
        self, name: str, evaluator_class: Type["BaseEvaluator"]
    ) -> None:
        """Register evaluator"""
        if name in self.evaluator_registry:
            self.logger.warning(
                f"Evaluator '{name}' already exists, will be overwritten"
            )
        self.evaluator_registry[name] = evaluator_class
        self.logger.debug(f"Registered evaluator: {name}")

    def _get_component(
        self,
        name: str,
        component_type: str,  # "attacks", "models", "defenses", "evaluators"
    ) -> Optional[Type]:
        """Generic component getter with lazy loading.

        Args:
            name: Component name
            component_type: One of "attacks", "models", "defenses", "evaluators"

        Returns:
            Component class or None if not found
        """
        type_info = {
            "attacks": ("attack_registry", "attack method"),
            "models": ("model_registry", "model"),
            "defenses": ("defense_registry", "defense method"),
            "evaluators": ("evaluator_registry", "evaluator"),
        }

        if component_type not in type_info:
            raise ValueError(f"Invalid component_type: {component_type}")

        registry_attr, type_name = type_info[component_type]
        registry = getattr(self, registry_attr)

        # Check cache first
        if name in registry:
            return registry[name]

        # Try lazy loading from plugins.yaml
        try:
            mappings = self._get_lazy_mappings()
            if name in mappings[component_type]:
                module_path, class_name = mappings[component_type][name]
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                # Cache for future access
                registry[name] = cls
                self.logger.debug(
                    f"Successfully imported {type_name} from mapping: {name}"
                )
                return cls
        except (ImportError, AttributeError) as e:
            self.logger.warning(
                f"Failed to import {type_name} '{name}': {e}. "
                f"This usually means missing dependencies. "
                f"Module: {module_path}, Class: {class_name}"
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Unknown error occurred while importing {type_name} '{name}': {e}"
            )
            return None

        self.logger.warning(f"{type_name.capitalize()} '{name}' is not defined in mapping")
        return None

    def get_attack(self, name: str) -> Optional[Type["BaseAttack"]]:
        """Get attack method class"""
        return self._get_component(name, "attacks")

    def get_model(self, name: str) -> Optional[Type["BaseModel"]]:
        """Get model class"""
        return self._get_component(name, "models")

    def get_defense(self, name: str) -> Optional[Type["BaseDefense"]]:
        """Get defense method class"""
        return self._get_component(name, "defenses")

    def get_evaluator(self, name: str) -> Optional[Type["BaseEvaluator"]]:
        """Get evaluator class"""
        return self._get_component(name, "evaluators")

    def create_attack(
        self, name: str, config: Dict[str, Any] = None, output_image_dir: str = None
    ) -> Optional["BaseAttack"]:
        """Create attack method instance

        Args:
            name: Attack method name
            config: Configuration dictionary
            output_image_dir: Output image directory path
        """
        attack_class = self.get_attack(name)
        if attack_class:
            try:
                cfg = dict(config or {})
                cfg.setdefault("_registry_name", name)
                return attack_class(config=cfg, output_image_dir=output_image_dir)
            except Exception as e:
                self.logger.error(f"Failed to create attack method '{name}': {e}")
        return None

    def create_model(
        self, name: str, config: Dict[str, Any] = None
    ) -> Optional["BaseModel"]:
        """Create model instance"""
        # Get provider information from configuration
        if config is None:
            config = {}

        provider = config.get("provider", name)
        model_class = self.get_model(provider)

        if model_class:
            try:
                # Prefer factory method if provided by the model class.
                # This removes provider-specific branching from the registry.
                if hasattr(model_class, "from_config") and callable(
                    getattr(model_class, "from_config")
                ):
                    # NOTE: models are instantiated via `models.*` which has its own config parsing.
                    # We keep `_registry_name` injection for symmetry, but it is optional for models.
                    cfg = dict(config or {})
                    cfg.setdefault("_registry_name", name)
                    return model_class.from_config(name=name, config=cfg)

                # ---- Backward-compatible fallback ----
                cfg = dict(config or {})
                cfg.setdefault("_registry_name", name)
                model_name = cfg.get("model_name", name)
                api_key = cfg.get("api_key", "")
                base_url = cfg.get("base_url", None)
                try:
                    return model_class(
                        model_name=model_name, api_key=api_key, base_url=base_url
                    )
                except TypeError:
                    # Some providers don't take base_url
                    return model_class(model_name=model_name, api_key=api_key)
            except Exception as e:
                self.logger.error(
                    f"Failed to create model '{name}' (provider: {provider}): {e}"
                )
        else:
            self.logger.error(f"Model provider not found: {provider}")
        return None

    def create_defense(
        self, name: str, config: Dict[str, Any] = None
    ) -> Optional["BaseDefense"]:
        """Create defense method instance"""
        defense_class = self.get_defense(name)
        if defense_class:
            try:
                cfg = dict(config or {})
                cfg.setdefault("_registry_name", name)
                return defense_class(config=cfg)
            except Exception as e:
                self.logger.error(f"Failed to create defense method '{name}': {e}")
        return None

    def create_evaluator(
        self, name: str, config: Dict[str, Any] = None
    ) -> Optional["BaseEvaluator"]:
        """Create evaluator instance"""
        evaluator_class = self.get_evaluator(name)
        if evaluator_class:
            try:
                cfg = dict(config or {})
                cfg.setdefault("_registry_name", name)
                return evaluator_class(config=cfg)
            except Exception as e:
                self.logger.error(f"Failed to create evaluator '{name}': {e}")
        return None

    def list_attacks(self) -> List[str]:
        """List all registered attack methods"""
        names = set(self.attack_registry.keys())
        names.update(self._get_lazy_mappings().get("attacks", {}).keys())
        return sorted(names)

    def list_models(self) -> List[str]:
        """List all registered models"""
        names = set(self.model_registry.keys())
        names.update(self._get_lazy_mappings().get("models", {}).keys())
        return sorted(names)

    def list_defenses(self) -> List[str]:
        """List all registered defense methods"""
        names = set(self.defense_registry.keys())
        names.update(self._get_lazy_mappings().get("defenses", {}).keys())
        return sorted(names)

    def list_evaluators(self) -> List[str]:
        """List all registered evaluators"""
        names = set(self.evaluator_registry.keys())
        names.update(self._get_lazy_mappings().get("evaluators", {}).keys())
        return sorted(names)

    def validate_attack(self, name: str) -> bool:
        """Validate if attack method exists"""
        if name in self.attack_registry:
            return True
        return name in self._get_lazy_mappings().get("attacks", {})

    def validate_model(self, name: str) -> bool:
        """Validate if model exists"""
        if name in self.model_registry:
            return True
        return name in self._get_lazy_mappings().get("models", {})

    def validate_defense(self, name: str) -> bool:
        """Validate if defense method exists"""
        if name in self.defense_registry:
            return True
        return name in self._get_lazy_mappings().get("defenses", {})

    def validate_evaluator(self, name: str) -> bool:
        """Validate if evaluator exists"""
        if name in self.evaluator_registry:
            return True
        return name in self._get_lazy_mappings().get("evaluators", {})

    def get_component_summary(self) -> Dict[str, List[str]]:
        """Get component summary"""
        return {
            "attacks": self.list_attacks(),
            "defenses": self.list_defenses(),
            "models": self.list_models(),
            "evaluators": self.list_evaluators(),
        }

    def initialize_components(self) -> Dict[str, List[str]]:
        """Initialize all components and return summary (based on lazy mapping, dynamically imported when accessed)"""
        summary = self.get_component_summary()
        self.logger.info("Component initialization completed")
        self.logger.info(f"Available attack methods: {len(summary['attacks'])}")
        self.logger.info(f"Available defense methods: {len(summary['defenses'])}")
        self.logger.info(f"Available models: {len(summary['models'])}")
        self.logger.info(f"Available evaluators: {len(summary['evaluators'])}")

        return summary


# Global registry instance
UNIFIED_REGISTRY = UnifiedRegistry()
