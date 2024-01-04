import os
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--train_input', default=None, required=False, type=str, help="Please specify a train_input text file")
parser.add_argument('--vocab_size', default=None, required=False, type=int, help="Please specify vocab size for training process")
parser.add_argument('--file_name', default=None, required=False, type=str, help="Please specify the file name for final output")
parser.add_argument('--user_defined_symbols_dir', default=None, required=False, type=str, help="Please specify a file of user defined symbols")
parser.add_argument('--llama_tokenizer_dir', default=None, type=str, required=True, help="Please specify the path for llama tokenizer")
parser.add_argument('--output_sp_dir', default='/sp_tokenizer', type=str, required=True, help="Please specify the directory name for sentencepiece type tokenizer")
parser.add_argument('--output_hf_dir', default='/hf_tokenizer', type=str, required=True, help="Please specify the directory name for huggingface type tokenizer")

args = parser.parse_args()

# add user defined symbols
if args.user_defined_symbols_dir is not None:
    word_list = []
    if args.user_defined_symbols_dir != "":
        with open(args.user_defined_symbols_dir,'r',encoding = 'utf-8') as fp:
            data = json.load(fp)
            word_list = data["list"]

spm.SentencePieceTrainer.train(input=args.train_input, model_prefix=args.file_name, vocab_size=args.vocab_size, user_defined_symbols=word_list, model_type='bpe')

llama_tokenizer_dir = args.llama_tokenizer_dir
language_sp_model_file = args.file_name + ".model"

# load llama tokenizer
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
language_sp_model = spm.SentencePieceProcessor()
language_sp_model.Load(language_sp_model_file)

llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
language_spm = sp_pb2_model.ModelProto()
language_spm.ParseFromString(language_sp_model.serialized_model_proto())

# Add tokens to original LLama tokenizer
llama_spm_tokens_set=set(p.piece for p in llama_spm.pieces)
for p in language_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
print(f"New model pieces: {len(llama_spm.pieces)}")

output_sp_dir = './temp/' + args.output_sp_dir
output_hf_dir = './temp/' + args.output_hf_dir

# save sentencepiece type
os.makedirs(output_sp_dir, exist_ok=True)
with open(output_sp_dir+'/merged_tokenizer.model', 'wb') as f:
    f.write(llama_spm.SerializeToString())

# save huggingface type
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir+'/merged_tokenizer.model')
tokenizer.save_pretrained(output_hf_dir)