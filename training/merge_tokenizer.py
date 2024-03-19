import argparse
import json
import os

import sentencepiece
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Input text file")
parser.add_argument("--vocab_size", type=int, required=True, help="Extra vocab size for input text")
parser.add_argument("--output_dir", type=str, required=True, help="Directory name to output tokenizer")
parser.add_argument("--tokenizer_name_or_path", type=str, required=True, help="The path for original tokenizer")
parser.add_argument(
    "--user_defined_symbols_dir", default=None, required=False, type=str, help="File of user defined symbols"
)

args = parser.parse_args()

# add user defined symbols
if args.user_defined_symbols_dir is not None:
    word_list = []
    if args.user_defined_symbols_dir != "":
        with open(args.user_defined_symbols_dir, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            word_list = data["list"]

os.makedirs(args.output_dir, exist_ok=True)
model_prefix = os.path.join(args.output_dir, "tokenizer")
sentencepiece.SentencePieceTrainer.train(
    input=args.input,
    model_prefix=model_prefix,
    vocab_size=args.vocab_size,
    user_defined_symbols=word_list,
    model_type="bpe",
)

# load original tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, legacy=False, use_fast=False)
language_sp_model = sentencepiece.SentencePieceProcessor()
language_sp_model.Load(model_prefix + ".model")

spm = sp_pb2_model.ModelProto()
spm.ParseFromString(tokenizer.sp_model.serialized_model_proto())
language_spm = sp_pb2_model.ModelProto()
language_spm.ParseFromString(language_sp_model.serialized_model_proto())

# Add tokens to original tokenizer
spm_tokens_set = set(p.piece for p in spm.pieces)
for p in language_spm.pieces:
    piece = p.piece
    if piece not in spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        spm.pieces.append(new_p)
print(f"New model pieces: {len(spm.pieces)}")

# save sentencepiece type
output_sp_dir = os.path.join(args.output_dir, "merged_tokenizer.model")
with open(output_sp_dir, "wb") as f:
    f.write(spm.SerializeToString())

# save huggingface type
tokenizer = tokenizer.__class__(vocab_file=output_sp_dir, legacy=False)
tokenizer.save_pretrained(args.output_dir)
