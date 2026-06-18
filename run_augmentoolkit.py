# Imports the function named by the config file, and runs it with the arguments inside the config specified

import importlib
import asyncio
import sys
import traceback
import yaml
import argparse
from pathlib import Path
import json
import io


# ==============================================================================
# FORCE UTF-8 FOR ALL I/O
# where the default encoding is not UTF-8. It prevents UnicodeEncodeError
# when printing characters that are not in the default system codepage.
# ==============================================================================
if getattr(sys.stdout, "encoding", None) != 'utf-8' and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if getattr(sys.stderr, "encoding", None) != 'utf-8' and hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
# ==============================================================================

from resolve_path import resolve_path


def load_function_from_path(function_path):
    """Dynamically import a function from a module path string"""
    module_path, function_name = function_path.rsplit(".", 1)
    # Ensure module path uses dots for importlib
    module_import_path = module_path.replace("/", ".")
    module = importlib.import_module(module_import_path)
    return getattr(module, function_name)


def flatten_config(config, no_flatten_keys=None):
    """Flatten config while preserving specified nested structures"""
    flattened = {}
    no_flatten = set(no_flatten_keys or [])

    for key, value in config.items():
        if key in no_flatten:
            flattened[key] = value
            continue

        if isinstance(value, dict):
            nested_flat = flatten_config(value, no_flatten_keys)
            for nested_key, nested_value in nested_flat.items():
                if nested_key in flattened:
                    raise ValueError(f"Key conflict: '{nested_key}'")
                flattened[nested_key] = nested_value
        else:
            if key in flattened:
                raise ValueError(f"Key conflict: '{key}'")
            flattened[key] = value
    print("FLATTENED")
    print(flattened)
    return flattened


# ==============================================================================
# Global LLM overrides
# ------------------------------------------------------------------------------
# Optional root-level file that lets you set the LLM connection details
# (base_url, api_key, models, mode) for EVERY pipeline in one place, instead of
# editing the repeated *_base_url / *_small_model / ... fields in every block of
# every config. When enabled, these values OVERRIDE whatever the pipeline config
# specified. Ideal for pointing all datagen at a single local backend (e.g.
# Ollama's OpenAI-compatible endpoint at http://localhost:11434/v1).
# ==============================================================================
GLOBAL_LLM_CONFIG_PATH = Path(__file__).parent / "llm_config.yaml"

# Maps a flattened-key suffix to the override field that controls it. Note we
# match `_small_mode`/`_large_mode` rather than `_mode` so we never clobber
# unrelated keys like `completion_mode`.
_LLM_OVERRIDE_SUFFIXES = {
    "_base_url": "base_url",
    "_api_key": "api_key",
    "_small_model": "small_model",
    "_large_model": "large_model",
    "_small_mode": "mode",
    "_large_mode": "mode",
}


def load_global_llm_overrides(path=GLOBAL_LLM_CONFIG_PATH):
    """Load the optional root-level global LLM override config.

    Returns a dict with any of base_url / api_key / small_model / large_model /
    mode, or an empty dict if the file is missing, unparseable, or disabled.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as e:
        print(f"Warning: could not parse global LLM config {path}: {e}. Ignoring it.")
        return {}

    overrides = data.get("llm_overrides", {}) or {}
    if not overrides.get("enabled", False):
        return {}

    # `model` is a shorthand that fills both small and large when those are
    # not set individually (handles the present-but-null YAML case too).
    if overrides.get("model") is not None:
        if overrides.get("small_model") is None:
            overrides["small_model"] = overrides["model"]
        if overrides.get("large_model") is None:
            overrides["large_model"] = overrides["model"]
    return overrides


def apply_llm_overrides(flattened_config, overrides=None):
    """Override per-block LLM connection settings with global values.

    For every flattened key ending in a recognized suffix (e.g. `_base_url`,
    `_small_model`), replace its value with the corresponding global override
    when one is set. No-op if no global overrides are configured.
    """
    if overrides is None:
        overrides = load_global_llm_overrides()
    if not overrides:
        return flattened_config

    applied = {}
    for key in list(flattened_config.keys()):
        for suffix, field in _LLM_OVERRIDE_SUFFIXES.items():
            if key.endswith(suffix) and overrides.get(field) is not None:
                flattened_config[key] = overrides[field]
                applied[key] = overrides[field]
                break
    if applied:
        print(
            f"Applied global LLM overrides (llm_config.yaml) to {len(applied)} "
            f"field(s): {sorted(applied)}"
        )
    return flattened_config


super_config_path = Path(__file__).parent / "super_config.yaml"
try:
    with open(super_config_path, "r", encoding="utf-8") as f:
        super_config = yaml.safe_load(f)
    path_aliases = super_config.get(
        "path_aliases", {}
    )  # Get aliases, default to empty dict if not present

except FileNotFoundError:
    print(f"Error: Super config file not found at {super_config_path}")
    sys.exit(1)
# Load super config
except yaml.YAMLError as e:
    print(f"Error parsing super config file {super_config_path}: {e}")
    sys.exit(1)


def run_pipeline(node, config, override_fields=None):
    if override_fields is None:
        override_fields = {}
    # Resolve node and config paths using aliases
    resolved_node_path = resolve_path(node, path_aliases)
    # print(f"DEBUG: Resolved node path: {resolved_node_path}")

    # Handle optional config key and resolve its path if present
    config_path_str = config
    resolved_config_path_str = (
        resolve_path(config_path_str, path_aliases) if config_path_str else None
    )
    # print(f"DEBUG: Resolved config path: {resolved_config_path_str}")

    resolved_config_path_str = (
        resolved_config_path_str
        if resolved_config_path_str and resolved_config_path_str.endswith(".yaml")
        else (
            resolved_config_path_str + ".yaml"
            if resolved_config_path_str
            else resolved_config_path_str
        )
    )
    # Load pipeline-specific config if a path is provided
    config = {}
    if resolved_config_path_str:
        # Resolve config path relative to the script's directory
        script_dir = Path(__file__).parent
        config_path = (script_dir / resolved_config_path_str).resolve()
        # print(f"DEBUG: Final resolved config path: {config_path}")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = (
                    yaml.safe_load(f) or {}
                )  # Ensure config is at least an empty dict
        except FileNotFoundError:
            print(
                f"Warning: Config file not found at {config_path}. Proceeding without it."
            )
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}")
            # Decide if you want to raise the error or continue
            # raise e
            print("Proceeding with empty configuration for this pipeline.")
    # print("DEBUG: CONFIG")
    # print(config)
    run_pipeline_config(
        config=config,
        resolved_node_path=resolved_node_path,
        override_fields=override_fields,
    )


def run_pipeline_config(
    config, resolved_node_path, override_fields=None
):  # the second half of run_pipeline, extracted so that it is easier to use in isolation as an api.
    if override_fields is None:
        override_fields = {}
    # Merge pipeline config from super_config with loaded config file parameters
    # Parameters defined directly in the pipeline entry override those in the loaded config file.
    # Parameters passed in as overrides override those in either.

    # Flatten the nested configuration
    no_flatten_keys = config.get(
        "no_flatten", []
    )  # Get no_flatten from loaded/merged config
    flattened_config = flatten_config(config, no_flatten_keys=no_flatten_keys)
    # Apply global LLM overrides (llm_config.yaml) before CLI overrides, so an
    # explicit --override-json still takes highest precedence.
    flattened_config = apply_llm_overrides(flattened_config)
    flattened_config.update(override_fields)

    # Import the target function using the resolved node path
    try:
        function = load_function_from_path(resolved_node_path)
    except (ImportError, AttributeError, ValueError) as e:
        print(f"Error loading function from node path '{resolved_node_path}': {e}")
        print(f"Skipping pipeline: {resolved_node_path}")  # Use name if available
        return  # Skip this pipeline if function cannot be loaded

    if asyncio.iscoroutinefunction(function):
        print(f"Running async pipeline: {resolved_node_path}")

        # print("DEBUG: Flattened config")
        # print(flattened_config)

        try:
            asyncio.run(function(**flattened_config))
        except Exception as e:
            print(f"Error running async pipeline {resolved_node_path}: {e}")
            traceback.print_exc()
            raise
            # Optionally re-raise or handle error reporting
    else:
        print(f"Running sync pipeline: {resolved_node_path}")
        try:
            function(**flattened_config)
        except Exception as e:
            print(f"Error running sync pipeline {resolved_node_path}: {e}")
            # Optionally re-raise or handle error reporting
            raise

    print(f"Completed pipeline: {resolved_node_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Augmentoolkit pipelines.")
    parser.add_argument(
        "--node",
        type=str,
        help="Path (potentially aliased) to the pipeline node function (e.g., 'pipelines/my_pipeline.run').",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path (potentially aliased) to the pipeline-specific configuration YAML file.",
    )
    parser.add_argument(
        "--override-json",
        type=str,
        help="JSON string of parameters to override pipeline config.",
    )

    args = parser.parse_args()

    # Always load path aliases
    # path_aliases are already loaded globally before main()

    override_params = {}
    if args.override_json:
        try:
            override_params = json.loads(args.override_json)
            if not isinstance(override_params, dict):
                print(
                    "Error: --override-json must be a valid JSON object (dictionary)."
                )
                sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing --override-json: {e}")
            sys.exit(1)

    # Decide execution path based on provided arguments
    if args.node:
        print(f"Running single pipeline specified via command line arguments:")
        print(f"  Node: {args.node}")
        print(f"  Config: {args.config}")
        print(f"  Overrides: {override_params}")
        run_pipeline(args.node, args.config, override_params)
    else:
        print("Running pipelines defined in super_config.yaml 'pipeline_order'.")
        # Run pipelines in order specified in super_config
        pipelines_to_run = super_config.get("pipeline_order", [])
        if not pipelines_to_run:
            print(
                "No pipelines specified in 'pipeline_order' and no specific pipeline provided via args. Exiting."
            )
            return

        for pipeline in pipelines_to_run:
            if not isinstance(pipeline, dict) or "node" not in pipeline:
                print(
                    f"Warning: Skipping invalid pipeline entry in super_config: {pipeline}. Must be a dictionary with a 'node' key."
                )
                continue
            # Merge super_config parameters with any CLI overrides (though CLI overrides usually imply single pipeline run)
            pipeline_params = pipeline.get("parameters", {})
            # Note: If running from super_config, CLI overrides are *not* typically used per-pipeline.
            # The logic here prioritizes the CLI override if BOTH --override-json and super_config parameters exist,
            # but this scenario is less common when running the whole sequence.
            # If you intended CLI overrides to *only* apply when --node is used, keep override_params empty here.
            # If CLI overrides should *globally* apply even to super_config runs, update pipeline_params:
            # pipeline_params.update(override_params) # Uncomment this if CLI overrides should apply globally

            print(f"Running pipeline from super_config:")
            print(f"  Node: {pipeline['node']}")
            print(f"  Config: {pipeline.get('config')}")  # Use .get for optional config
            print(f"  Parameters: {pipeline_params}")

            run_pipeline(
                pipeline["node"], pipeline.get("config"), pipeline_params
            )  # Use .get for config key


if __name__ == "__main__":
    main()
