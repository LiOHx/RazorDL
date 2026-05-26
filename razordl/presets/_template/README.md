## How to add a new preset

1. Copy the template:
   ```bash
   cp -r razordl/presets/_template razordl/presets/<new_name>
   ```

2. Edit files marked `[改]`:
   - `workgroup.py` — your loss class + WorkGroup (core difference)
   - `__init__.py` — rename NEWConfig/NEWWorkGroup/DistNEWLoss to your names
   - `default_config.yaml` — preset name, description, extra params
   - `_export.py` — rename loss class in string replacements

3. Files marked `[不改]` can usually be left as-is:
   - `_export_full.py` — re-exports the engine profile
   - `requirements.txt` — auto-synced (see below)

4. Set up requirements:
   ```bash
   touch razordl/presets/<new_name>/requirements.txt
   python scripts/sync_requirements.py
   ```
   If your preset needs extra dependencies, add to `pyproject.toml`:
   ```toml
   [project.optional-dependencies]
   <new_name> = ["extra-package>=1.0"]
   ```

5. The CLI auto-discovers the new preset. No CLI changes needed.

## Using a different engine

If your preset uses a different engine (e.g. `multi_model`):
- Change `_export_full.py` to import from that engine's `_export_profile.py`
- Create the engine's `_export_profile.py` following the single_model template
