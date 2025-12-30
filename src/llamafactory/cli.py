# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import random
import subprocess
import sys
from enum import Enum, unique
from pathlib import Path

import yaml

from . import launcher
from .api.app import run_api
from .chat.chat_model import run_chat
from .eval.evaluator import run_eval
from .extras import logging
from .extras.env import VERSION, print_env
from .extras.misc import get_device_count, is_env_enabled, use_ray
from .train.tuner import export_model, run_exp
from .webui.interface import run_web_demo, run_web_ui


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   llamafactory-cli api -h: launch an OpenAI-style API server       |\n"
    + "|   llamafactory-cli chat -h: launch a chat interface in CLI         |\n"
    + "|   llamafactory-cli eval -h: evaluate models                        |\n"
    + "|   llamafactory-cli export -h: merge LoRA adapters and export model |\n"
    + "|   llamafactory-cli train -h: train models                          |\n"
    + "|   llamafactory-cli webchat -h: launch a chat interface in Web UI   |\n"
    + "|   llamafactory-cli webui: launch LlamaBoard                        |\n"
    + "|   llamafactory-cli version: show version info                      |\n"
    + "-" * 70
)

WELCOME = (
    "-" * 58
    + "\n"
    + f"| Welcome to LLaMA Factory, version {VERSION}"
    + " " * (21 - len(VERSION))
    + "|\n|"
    + " " * 56
    + "|\n"
    + "| Project page: https://github.com/hiyouga/LLaMA-Factory |\n"
    + "-" * 58
)

logger = logging.get_logger(__name__)


def _should_disable_torchrun() -> bool:
    r"""
    Checks if torchrun should be automatically disabled based on the training configuration.
    
    Returns True if:
    1. No deepspeed config is specified AND device count > 1
       (DDP without deepspeed will cause each GPU to load the full model, leading to OOM for large models)
    2. fsdp is not configured
    
    In these cases, single-process training with device_map="auto" is preferred.
    """
    if len(sys.argv) < 2:
        return False
    
    config_file = sys.argv[1]
    if not (config_file.endswith(".yaml") or config_file.endswith(".yml") or config_file.endswith(".json")):
        # Parse command line arguments to check for deepspeed
        return "--deepspeed" not in sys.argv and "--fsdp" not in sys.argv
    
    try:
        config_path = Path(config_file).absolute()
        if config_file.endswith(".json"):
            config = json.loads(config_path.read_text())
        else:
            config = yaml.safe_load(config_path.read_text())
        
        if not isinstance(config, dict):
            return False
        
        # Check if deepspeed or fsdp is configured
        has_deepspeed = config.get("deepspeed") is not None
        has_fsdp = config.get("fsdp") is not None and config.get("fsdp") != ""
        
        # If neither deepspeed nor fsdp is configured, disable torchrun for multi-GPU
        # to allow device_map="auto" for model parallelism
        if not has_deepspeed and not has_fsdp:
            return True
        
        return False
    except Exception:
        # If we can't parse the config, don't auto-disable
        return False


@unique
class Command(str, Enum):
    API = "api"
    CHAT = "chat"
    ENV = "env"
    EVAL = "eval"
    EXPORT = "export"
    TRAIN = "train"
    WEBDEMO = "webchat"
    WEBUI = "webui"
    VER = "version"
    HELP = "help"


def main():
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP
    if command == Command.API:
        run_api()
    elif command == Command.CHAT:
        run_chat()
    elif command == Command.ENV:
        print_env()
    elif command == Command.EVAL:
        run_eval()
    elif command == Command.EXPORT:
        export_model()
    elif command == Command.TRAIN:
        force_torchrun = is_env_enabled("FORCE_TORCHRUN")
        disable_torchrun = is_env_enabled("DISABLE_TORCHRUN")
        
        # Auto-disable torchrun if no deepspeed/fsdp is configured for multi-GPU training
        # This allows device_map="auto" for model parallelism instead of DDP
        if not disable_torchrun and not force_torchrun and get_device_count() > 1:
            if _should_disable_torchrun():
                logger.info_rank0(
                    "Auto-disabling torchrun: no deepspeed/fsdp configured. "
                    "Using device_map='auto' for model parallelism. "
                    "Set FORCE_TORCHRUN=1 to override."
                )
                disable_torchrun = True
                os.environ["DISABLE_TORCHRUN"] = "1"  # Set env var for parser.py check
        
        if not disable_torchrun and (force_torchrun or (get_device_count() > 1 and not use_ray())):
            master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
            master_port = os.getenv("MASTER_PORT", str(random.randint(20001, 29999)))
            logger.info_rank0(f"Initializing distributed tasks at: {master_addr}:{master_port}")
            process = subprocess.run(
                (
                    "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                    "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
                )
                .format(
                    nnodes=os.getenv("NNODES", "1"),
                    node_rank=os.getenv("NODE_RANK", "0"),
                    nproc_per_node=os.getenv("NPROC_PER_NODE", str(get_device_count())),
                    master_addr=master_addr,
                    master_port=master_port,
                    file_name=launcher.__file__,
                    args=" ".join(sys.argv[1:]),
                )
                .split()
            )
            sys.exit(process.returncode)
        else:
            run_exp()
    elif command == Command.WEBDEMO:
        run_web_demo()
    elif command == Command.WEBUI:
        run_web_ui()
    elif command == Command.VER:
        print(WELCOME)
    elif command == Command.HELP:
        print(USAGE)
    else:
        raise NotImplementedError(f"Unknown command: {command}.")


if __name__ == "__main__":
    main()
