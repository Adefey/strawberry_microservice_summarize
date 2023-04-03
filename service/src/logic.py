import os
import re

from tqdm import tqdm
import logging
import random

import torch

from transformers import pipeline

logging.basicConfig(format="%(asctime)s %(message)s", handlers=[logging.FileHandler(
    f"/home/logs/summarize_log_model.txt", mode="w", encoding="UTF-8")], datefmt="%I:%M:%S %p", level=logging.INFO)


class NeuralNetwork:
    def __init__(self, group_id=0):
        self.pipe = pipeline(model="cointegrated/rut5-base-absum")

    def generate(self, hint):
        logging.info(f"generating (summarizing) for {hint}")
        num_beams = random.randint(4, 8)
        repetition_penalty = random.uniform(5.0, 15.0)
        result = self.pipe(hint, num_beams=num_beams,
                           repetition_penalty=repetition_penalty)[0]["summary_text"]
        logging.info(f"generating (summarizing) for {hint}: {result}")
        return result
