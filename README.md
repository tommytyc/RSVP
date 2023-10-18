# RSVP: Customer Intent Detection via Agent Response Contrastive and Generative Pre-Training (EMNLP 2023 Findings)
Official implementation for our EMNLP 2023 Findings paper "RSVP: Customer Intent Detection via Agent Response Contrastive and Generative Pre-Training".

## Setup
- Build environment
```bash
pip3 install requirements.txt
```

- Put data in `data/`

## Run
- Overall running command
```bash
python3 RSVP.py --train_path [train_data_path] --dev_path [dev/valid_data_path] --test_path [test_data_path] --label_map_path [label_map_path] --task [task_name] 
```
- For example, TwACS
```bash
python3 RSVP.py --train_path data/airlines_train.csv --dev_path data/airlines_valid.csv --test_path data/airlines_test.csv --label_map_path data/airline_label_map.pkl --task airline
```
- The `task` include `airline`, `woz`, and `sgd`. 