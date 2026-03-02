from pathlib import Path
from unittest.mock import patch

from core.data_formats import PipelineConfig
from pipeline.generate_test_cases import TestCaseGenerator as CaseGenerator


def test_run_uses_per_attack_image_dir(tmp_path):
    config = PipelineConfig(
        system={"output_dir": str(tmp_path)},
        test_case_generation={
            "input": {"behaviors_file": "unused_in_test.json"},
            "attacks": ["jood", "hades"],
            "attack_params": {"jood": {}, "hades": {}},
        },
        batch_size=1,
    )
    generator = CaseGenerator(config)

    behaviors = [{"case_id": 1, "original_prompt": "p", "image_path": "img.jpg"}]
    captured_image_dirs = {}

    def fake_generate_filename(stage_name, **context):
        attack_name = context["attack_name"]
        image_dir = tmp_path / attack_name / "images"
        output_file = tmp_path / attack_name / "test_cases.jsonl"
        return image_dir, output_file

    def fake_generate_single_attack_test_cases(
        attack_name,
        attack_config,
        behaviors,
        batch_size,
        output_file_path,
        image_save_dir,
    ):
        captured_image_dirs[attack_name] = Path(image_save_dir)
        return []

    with patch.object(generator, "validate_config", return_value=True), patch.object(
        generator, "load_behaviors", return_value=behaviors
    ), patch.object(generator, "_calculate_expected_test_cases", return_value=1), patch.object(
        generator, "load_results", return_value=[]
    ), patch.object(generator, "_generate_filename", side_effect=fake_generate_filename), patch.object(
        generator,
        "generate_single_attack_test_cases",
        side_effect=fake_generate_single_attack_test_cases,
    ):
        generator.run()

    assert captured_image_dirs["jood"] == tmp_path / "jood" / "images"
    assert captured_image_dirs["hades"] == tmp_path / "hades" / "images"
    assert captured_image_dirs["jood"] != captured_image_dirs["hades"]
