# -*- coding: utf-8 -*-
import random
import torch
import torch.nn as nn
from torch import optim
from mt_datasets import readLang, readLangs, SOS_token, EOS_token, MAX_LENGTH
from models import EncoderRNN, AttenDecoderRNN
from utils import timeSince
from utils import normalizeString
import time
import sys, getopt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = MAX_LENGTH + 1

lang1 = "en"
lang2 = "cn"
path = "./mt/data/en-cn.txt"
input_lang, output_lang, pairs = readLangs(lang1, lang2, path)




def listTotensor(input_lang, data):
    indexes_in = [input_lang.word2index[word] for word in data.split(" ")]
    indexes_in.append(EOS_token)
    input_tensor = torch.tensor(indexes_in,
                                dtype=torch.long,
                                device=device).view(-1, 1)
    return input_tensor

def tensorsFromPair(pair):
    input_tensor = listTotensor(input_lang, pair[0])
    output_tensor = listTotensor(output_lang, pair[1])
    return (input_tensor, output_tensor)





# train_sen_pairs = [
#     random.choice(pairs) for i in range(n_iters)
# ]
# training_pairs = [
#     tensorsFromPair(train_sen_pairs[i]) for i in range(n_iters)
# ]


def get_translation(input_str):
    hidden_size = 256
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttenDecoderRNN(hidden_size,
                              output_lang.n_words,
                              max_len=MAX_LENGTH,
                              dropout_p=0.1).to(device)

    encoder.load_state_dict(torch.load("./mt/models/encoder_1000000.pth", map_location='cpu'))
    decoder.load_state_dict(torch.load("./mt/models/decoder_1000000.pth", map_location='cpu'))


    input_tensor = listTotensor(input_lang, normalizeString(input_str))

    # input_tensor, output_tensor = training_pairs[i]
    encoder_hidden = encoder.initHidden()
    input_len = input_tensor.size(0)
    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    for ei in range(input_len):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_hidden = encoder_hidden
    decoder_input = torch.tensor([[SOS_token]], device=device)
    use_teacher_forcing = True if random.random() < 0.5 else False
    decoder_words = []
    for di in range(MAX_LENGTH):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        topV, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        if topi.item() == EOS_token:
            decoder_words.append("<EOS>")
            break
        else:
            decoder_words.append(output_lang.index2word[topi.item()])
    decoder_words = "".join(decoder_words[:-1])
    # f = open('output/' + file_name, 'wb')
    # f.write(decoder_words.encode("utf-8"))
    # f.close()
    return decoder_words

if __name__ == "__main__":
    argvs = sys.argv
    input_str = argvs[1]
    try:
        get_translation(input_str)
    except Exception as e:
        print(-1)