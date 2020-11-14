import random
import torch
import torch.nn as nn
from torch import optim
from datasets import readLang, readLangs, SOS_token, EOS_token, MAX_LENGTH
from models import EncoderRNN, AttenDecoderRNN
from utils import timeSince
from utils import normalizeString
import time
import sys, getopt


def get_random_sentence():
    lang1 = "en"
    lang2 = "cn"
    path = "./mt/data/en-cn.txt"
    input_lang, output_lang, pairs = readLangs(lang1, lang2, path)
    pair = random.choice(pairs)
    return pair[0]

if __name__ == "__main__":
    try:
        get_random_sentence()
    except Exception as e:
        print(-1)
