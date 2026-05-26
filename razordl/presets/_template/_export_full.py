"""NEW full-mode project export.

[不改] 一行 re-export — 引擎层拥有所有 full-mode 导出逻辑。
如果你的 preset 使用了不同的 engine (如 multi_model)，改 import source 即可。
"""

from razordl.core.engine.single_model._export_profile import export_full_project  # noqa: F401
