"""Full-mode export profile for the single_model engine."""

from razordl.core.export.full_project import export_full_project_for_engine


def export_full_project(preset_pkg_dir: str, project_dir: str, razordl_root: str) -> list[str]:
    return export_full_project_for_engine(
        preset_pkg_dir,
        project_dir,
        razordl_root,
        engine_name="single_model",
        engine_files=(
            "main.py",
            "trainer.py",
            "workgroup.py",
            "lm_workgroup.py",
            "config.py",
            "dataset.py",
        ),
    )
