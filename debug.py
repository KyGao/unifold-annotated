#@title Input protein sequence(s), then hit `Runtime` -> `Run all`
import os
import re
import hashlib
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union, Any, Optional

from unifold.data import residue_constants, protein
from unifold.msa.utils import divide_multi_chains

MIN_SINGLE_SEQUENCE_LENGTH = 16
MAX_SINGLE_SEQUENCE_LENGTH = 1000
MAX_MULTIMER_LENGTH = 1000

output_dir_base = "./prediction"
os.makedirs(output_dir_base, exist_ok=True)


def clean_and_validate_sequence(
    input_sequence: str, min_length: int, max_length: int) -> str:
  """Checks that the input sequence is ok and returns a clean version of it."""
  # Remove all whitespaces, tabs and end lines; upper-case.
  clean_sequence = input_sequence.translate(
      str.maketrans('', '', ' \n\t')).upper()
  aatypes = set(residue_constants.restypes)  # 20 standard aatypes.
  if not set(clean_sequence).issubset(aatypes):
    raise ValueError(
        f'Input sequence contains non-amino acid letters: '
        f'{set(clean_sequence) - aatypes}. AlphaFold only supports 20 standard '
        'amino acids as inputs.')
  if len(clean_sequence) < min_length:
    raise ValueError(
        f'Input sequence is too short: {len(clean_sequence)} amino acids, '
        f'while the minimum is {min_length}')
  if len(clean_sequence) > max_length:
    raise ValueError(
        f'Input sequence is too long: {len(clean_sequence)} amino acids, while '
        f'the maximum is {max_length}. You may be able to run it with the full '
        f'Uni-Fold system depending on your resources (system memory, '
        f'GPU memory).')
  return clean_sequence


def validate_input(
    input_sequences: Sequence[str],
    min_length: int,
    max_length: int,
    max_multimer_length: int) -> Tuple[Sequence[str], bool]:
  """Validates and cleans input sequences and determines which model to use."""
  sequences = []

  for input_sequence in input_sequences:
    if input_sequence.strip():
      input_sequence = clean_and_validate_sequence(
          input_sequence=input_sequence,
          min_length=min_length,
          max_length=max_length)
      sequences.append(input_sequence)

  if len(sequences) == 1:
    print('Using the single-chain model.')
    return sequences, False

  elif len(sequences) > 1:
    total_multimer_length = sum([len(seq) for seq in sequences])
    if total_multimer_length > max_multimer_length:
      raise ValueError(f'The total length of multimer sequences is too long: '
                       f'{total_multimer_length}, while the maximum is '
                       f'{max_multimer_length}. Please use the full AlphaFold '
                       f'system for long multimers.')
    print(f'Using the multimer model with {len(sequences)} sequences.')
    return sequences, True

  else:
    raise ValueError('No input amino acid sequence provided, please provide at '
                     'least one sequence.')

def add_hash(x,y):
    return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]


sequence_1 = 'LILNLRGGAFVSNTQITMADKQKKFINEIQEGDLVRSYSITDETFQQNAVTSIVKHEADQLCQINFGKQHVVCTVNHRFYDPESKLWKSVCPHPGSGISFLKKYDYLLSEEGEKLQITEIKTFTTKQPVFIYHIQVENNHNFFANGVLAHAMQVSI'  #@param {type:"string"}
sequence_2 = ''  #@param {type:"string"}
sequence_3 = ''  #@param {type:"string"}
sequence_4 = ''  #@param {type:"string"}

use_templates = True #@param {type:"boolean"}
msa_mode = "MMseqs2" #@param ["MMseqs2","single_sequence"]

input_sequences = [sequence_1, sequence_2, sequence_3, sequence_4]

jobname = 'unifold_colab' #@param {type:"string"}

basejobname = "".join(input_sequences)
basejobname = re.sub(r'\W+', '', basejobname)
target_id = add_hash(jobname, basejobname)

# Validate the input.
sequences, is_multimer = validate_input(
    input_sequences=input_sequences,
    min_length=MIN_SINGLE_SEQUENCE_LENGTH,
    max_length=MAX_SINGLE_SEQUENCE_LENGTH,
    max_multimer_length=MAX_MULTIMER_LENGTH)

descriptions = ['> '+target_id+' seq'+str(ii) for ii in range(len(sequences))]

if is_multimer:
    divide_multi_chains(target_id, output_dir_base, sequences, descriptions)
    
s = []
for des, seq in zip(descriptions, sequences):
    s += [des, seq]

unique_sequences = []
[unique_sequences.append(x) for x in sequences if x not in unique_sequences]

if len(unique_sequences)==1:
    homooligomers_num = len(sequences)
else:
    homooligomers_num = 1
    
with open(f"{jobname}.fasta", "w") as f:
    f.write("\n".join(s))