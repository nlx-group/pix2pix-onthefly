# pix2pix-onthefly
Repository for the paper Leveraging LLMs for On-the-fly Instruction Guided Image Editing EPIA 2024

## Test Data
Download MagicBrush test set from here: https://osu-nlp-group.github.io/MagicBrush/

Extract to folder "test/"

Install the requirements ("test/reqs.txt") in a separate environment

run python to_hf_dataset.py

## Clone pix2pix-zero

https://github.com/pix2pixzero/pix2pix-zero.git

## Evaluation
Install the requirements from "reqs.txt"

run python eval_clip.py

## Please Cite
```
@inproceedings{santos2024leveraging,
  title={Leveraging LLMs for On-the-fly Instruction Guided Image Editing},
  author={Santos, Rodrigo  and Silva, Jo{\~a}o and Branco, Ant{\'o}nio},
  booktitle={EPIA Conference on Artificial Intelligence},
  year={2024},
  organization={Springer}
}
```
