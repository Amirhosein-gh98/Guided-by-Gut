"""
This file is largely borrowed from search-and_learn (https://github.com/huggingface/search-and-learn)
"""

import logging

import torch
from vllm import LLM

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts, cot, dvts4
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score
import time
import warnings


# warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "dvts4": dvts4, #GG
    "best_of_n": best_of_n,
    "cot": cot
}


def main():
    import os
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    approach_fn = APPROACHES[config.approach]

    num_gpus = torch.cuda.device_count()
    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=32768,
        seed=config.seed,
        tensor_parallel_size=num_gpus,
    )

    if config.use_prm:
        prm = load_prm(config)
    else:
        prm = None

    dataset = get_dataset(config)
    dataset = dataset.map(
        approach_fn,
        batched=True,
        batch_size=config.search_batch_size,
        fn_kwargs={"config": config, "llm": llm, "prm": prm},
        desc="Running search",
        load_from_cache_file=False,
    )

    dataset = score(dataset, config)

    save_dataset(dataset, config)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    start_time = time.time()
    main()
    duration = time.time() - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    print(f"Duration: {hours} hours and {minutes} minutes")
