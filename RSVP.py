from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, EncoderDecoderModel
from dataset import AirlineDataset, WozDataset
from model import BertClassifier
from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from aim import Run
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def contrastive_loss(args, q, p, device):
    batch_size = q.size(0)
    mask = torch.eye(batch_size, dtype=torch.bool).to(device)
    q_norm = q / q.norm(dim=1)[:, None]
    p_norm = p / p.norm(dim=1)[:, None]
    sim_mat = torch.mm(q_norm, p_norm.transpose(0,1))
    nominator = (mask * torch.exp(sim_mat / args.temp)).sum(dim=1)
    denominator = (~mask * torch.exp(sim_mat / args.temp)).sum(dim=1) + 1e-6 + nominator
    all_losses = -torch.log(nominator / denominator)
    loss = torch.sum(all_losses) / batch_size
    return loss

def prepare_data(task, train_path, dev_path, label_map_path, batch_size):
    if task == 'airline':
        train_data = AirlineDataset(train_path, label_map_path)
        dev_data = AirlineDataset(dev_path, label_map_path)
    elif task == 'woz' or task == 'sgd':
        train_data = WozDataset(train_path, label_map_path)
        dev_data = WozDataset(dev_path, label_map_path)
    else:
        raise NotImplementedError(f'No such task: {task}')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_data, batch_size=1, shuffle=False)
    return train_data, train_dataloader, dev_data, dev_dataloader

def prepare_test_data(task, test_path, label_map_path, batch_size):
    if task == 'airline':
        test_data = AirlineDataset(test_path, label_map_path)
    elif task == 'woz' or task == 'sgd':
        test_data = WozDataset(test_path, label_map_path)
    else:
        raise NotImplementedError(f'No such task: {task}')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    return test_data, test_dataloader

def train_resp_source(args, tokenizer, model):
    colorprint('Training Response Source Model', 'yellow')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_data, train_loader, dev_data, dev_loader = prepare_data(args.task, args.train_path, args.dev_path,
                                                                  args.label_map_path, args.resp_source_batch_size)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    model.train()
    with tqdm(range(args.resp_source_epoch), desc='Epoch') as tepoch:
        for epoch in tepoch:
            total_loss = 0
            for idx, data in enumerate(train_loader):
                pid, uttr, response, label, str_label = data[0], list(data[1]), list(data[2]), data[3].to(device), list(data[4])
                uttr_tokens = tokenizer(uttr, padding=True, truncation=True, max_length=args.max_seq_len, return_tensors="pt").to(device)
                resp_tokens = tokenizer(response, padding=True, truncation=True, max_length=args.max_seq_len, return_tensors="pt").to(device)
                optimizer.zero_grad()
                uttr_embeddings = model(**uttr_tokens)[1]
                resp_embeddings = model(**resp_tokens)[1]
                loss = contrastive_loss(args, uttr_embeddings, resp_embeddings, device)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            tepoch.set_postfix(loss=total_loss/len(train_loader))

    best_model_path = f"model/RSVP_trial_resp_source_{args.task}_{args.seed}.pt"
    torch.save(model.state_dict(), best_model_path)
    return model

def train_resp(args, tokenizer, lm_layer):
    colorprint('Training Response Generation', 'yellow')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data, train_loader, dev_data, dev_loader = prepare_data(args.task, args.train_path, args.dev_path,
                                                                  args.label_map_path, args.resp_batch_size)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('roberta-base', 'roberta-base').to(device)
    model.encoder = lm_layer
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    optimizer = AdamW(model.parameters(), lr=args.lr)

    model.train()
    with tqdm(range(args.resp_epoch), desc='Epoch') as tepoch:
        for epoch in tepoch:
            total_loss = 0
            for idx, data in enumerate(train_loader):
                pid, uttr, response, label, str_label = data[0], list(data[1]), list(data[2]), data[3].to(device), list(data[4])

                resp_tokens = tokenizer(response, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len).to(device).input_ids  # in the language of your choice
                resp_tokens[resp_tokens == tokenizer.pad_token_id] = -100
                uttr_tokens = tokenizer(uttr, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len).to(device)
                loss = model(**uttr_tokens, labels=resp_tokens).loss
                loss /= args.accum_iter
                total_loss += loss.item()
                loss.backward()
                if (idx + 1) % args.accum_iter == 0 or idx == len(train_loader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()

            tepoch.set_postfix(loss=total_loss / len(train_loader))
            best_model_path = f"model/RSVP_trial_resp_{args.task}_{args.seed}.pt"
            # torch.save(model.transformer.state_dict(), best_model_path)
            torch.save(model.encoder.state_dict(), best_model_path)
    return model.encoder

def train_cls(args, tokenizer, lm_layer):
    run = Run()
    run['hparams'] = vars(args)
    colorprint('Training CLS', 'yellow')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data, train_loader, dev_data, dev_loader = prepare_data(args.task, args.train_path, args.dev_path,
                                                                  args.label_map_path, args.cls_batch_size)
    model = BertClassifier(num_labels=train_data.get_labels_num()).to(device)
    model.transformer = lm_layer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    ce_criterion = nn.CrossEntropyLoss()
    best_acc = 0
    with tqdm(range(args.cls_epoch), desc='Epoch') as tepoch:
        for epoch in tepoch:
            total_loss = 0
            for idx, data in enumerate(train_loader):
                model.train()
                pid, uttr, response, label, str_label = data[0], list(data[1]), list(data[2]), data[3].to(device), list(data[4])
                uttr_tokens = tokenizer(uttr, padding=True, truncation=True, max_length=args.max_seq_len, return_tensors="pt").to(device)
                logits1, uttr_sent_emb = model(uttr_tokens)
                q1 = F.dropout(uttr_sent_emb, p=0.1, training=True)
                q2 = F.dropout(uttr_sent_emb, p=0.1, training=True)
                loss = args.lamda * contrastive_loss(args, q1, q2, device) + ce_criterion(logits1.float(), label.float())                
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            model.eval()
            top1_preds, top3_preds, top5_preds, labels = [], [], [], []
            for data in dev_loader:
                pid, uttr, response, label, str_label = data[0], list(data[1]), list(data[2]), data[3].to(device), list(data[4])
                input_text = tokenizer(uttr, padding=True, truncation=True, max_length=args.max_seq_len, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits, _ = model(input_text)
                    hard_pred = torch.argmax(logits, dim=-1).cpu().numpy()[0]
                    top3_pred = torch.topk(logits, 3, dim=-1)[1].cpu().numpy()[0]
                    top5_pred = torch.topk(logits, 5, dim=-1)[1].cpu().numpy()[0]
                    label = torch.argmax(label, dim=-1).cpu().numpy()[0]
                top1_preds.append(hard_pred)
                top3_preds.append(top3_pred)
                top5_preds.append(top5_pred)
                labels.append(label)
            precision, recall, f1, acc = compute_metric(labels, top1_preds)
            top3_acc = compute_topk_acc(labels, top3_preds)
            top5_acc = compute_topk_acc(labels, top5_preds)
            mrr3 = mean_reciprocal_rank(labels, top3_preds)
            mrr5 = mean_reciprocal_rank(labels, top5_preds)
            tepoch.set_postfix(loss=total_loss / len(train_loader), acc=acc, mrr3=mrr3, mrr5=mrr5)
            run.track({
                "loss": total_loss/len(train_loader),
                "acc": acc,
                "mrr3": mrr3,
                "mrr5": mrr5
            }, epoch=epoch)
            if acc > best_acc:
                best_acc = acc
                print(f"Saving model best acc = {best_acc}")
                best_model_path = f"model/RSVP_trial_{best_acc}_{args.task}_{args.seed}.pt"
                torch.save(model.state_dict(), best_model_path)
    return model, best_model_path

def train_multilabel_cls(args, tokenizer, lm_layer):
    colorprint('Training Multi-label CLS', 'yellow')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data, train_loader, dev_data, dev_loader = prepare_data(args.task, args.train_path, args.dev_path,
                                                                  args.label_map_path, args.cls_batch_size)
    model = BertClassifier(num_labels=train_data.get_labels_num()).to(device)
    model.transformer = lm_layer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    bce_criterion = nn.BCEWithLogitsLoss()
    best_f1 = 0
    print("Parameters to train:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    with tqdm(range(args.cls_epoch), desc='Epoch') as tepoch:
        for epoch in tepoch:
            total_loss = 0
            for idx, data in enumerate(train_loader):
                model.train()
                pid, uttr, response, label, str_label = data[0], list(data[1]), list(data[2]), data[3].to(device), list(data[4])
                uttr_tokens = tokenizer(uttr, padding=True, truncation=True, max_length=args.max_seq_len, return_tensors="pt").to(device)
                logits1, uttr_sent_emb = model(uttr_tokens)
                q1 = F.dropout(uttr_sent_emb, p=0.1, training=True)
                q2 = F.dropout(uttr_sent_emb, p=0.1, training=True)
                loss = args.lamda * contrastive_loss(args, q1, q2, device) + bce_criterion(logits1.float(), label.float())
                # loss = bce_criterion(logits1.float(), label.float())
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            model.eval()
            preds, labels = [], []
            for data in dev_loader:
                pid, uttr, response, label, str_label = data[0], list(data[1]), list(data[2]), data[3].to(device), list(data[4])
                input_text = tokenizer(uttr, padding=True, truncation=True, max_length=args.max_seq_len, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits, _ = model(input_text)
                    # hard_pred = torch.argmax(logits, dim=-1).cpu().numpy()[0]
                    hard_pred = (logits > 0.5).cpu().numpy()[0]
                    label = label.cpu().numpy()[0]
                preds.append(hard_pred)
                labels.append(label)
            precision, recall, f1, acc = compute_metric(labels, preds, f1_average='samples')
            tepoch.set_postfix(loss=total_loss / len(train_loader), f1=f1, acc=acc)
            if f1 > best_f1:
                best_f1 = f1
                print(f"Saving model best f1 = {best_f1}")
                best_model_path = f"model/RSVP_trial_{best_f1}_{args.task}_{args.seed}.pt"
                torch.save(model.state_dict(), best_model_path)
    return model, best_model_path

def train(args):
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModel.from_pretrained('roberta-base').to(device)
    model = train_resp_source(args, tokenizer, model)
    model.load_state_dict(torch.load(f'model/RSVP_trial_resp_source_{args.task}_{args.seed}.pt', map_location=device))
    model = train_resp(args, tokenizer, model)
    model.load_state_dict(torch.load(f'model/RSVP_trial_resp_{args.task}_{args.seed}.pt', map_location=device))
    
    if args.task == 'airline':
        model, best_model_path = train_cls(args, tokenizer, model)
    else:
        model, best_model_path = train_multilabel_cls(args, tokenizer, model)
    return best_model_path
        
def test(args):
    colorprint(f"Seed {args.seed}, Testing stage. Loading model from {args.best_model_path}", 'yellow')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_data, test_loader = prepare_test_data(args.task, args.test_path,
                                               args.label_map_path, args.cls_batch_size)
    model = BertClassifier(num_labels=test_data.get_labels_num()).to(device)
    model.load_state_dict(torch.load(args.best_model_path, map_location=device))
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    model.eval()
    if args.task == 'airline':
        top1_preds, top3_preds, top5_preds, labels = [], [], [], []
        for data in test_loader:
            pid, uttr, response, label, str_label = data[0], list(data[1]), list(data[2]), data[3].to(device), list(data[4])
            input_text = tokenizer(uttr, padding=True, truncation=True, max_length=args.max_seq_len, return_tensors="pt").to(device)
            with torch.no_grad():
                logits, _ = model(input_text)
                hard_pred = torch.argmax(logits, dim=-1).cpu().numpy()[0]
                top3_pred = torch.topk(logits, 3, dim=-1)[1].cpu().numpy()[0]
                top5_pred = torch.topk(logits, 5, dim=-1)[1].cpu().numpy()[0]
                label = torch.argmax(label, dim=-1).cpu().numpy()[0]
            top1_preds.append(hard_pred)
            top3_preds.append(top3_pred)
            top5_preds.append(top5_pred)
            labels.append(label)
        precision, recall, f1, acc = compute_metric(labels, top1_preds)
        top3_acc = compute_topk_acc(labels, top3_preds)
        top5_acc = compute_topk_acc(labels, top5_preds)
        mrr3 = mean_reciprocal_rank(labels, top3_preds)
        mrr5 = mean_reciprocal_rank(labels, top5_preds)
        colorprint(f'acc: {acc}, top3_acc: {top3_acc}, top5_acc: {top5_acc}, f1: {f1}, mrr3: {mrr3}, mrr5: {mrr5}', 'green')
    else:
        preds, labels = [], []
        for data in test_loader:
            pid, uttr, response, label, str_label = data[0], list(data[1]), list(data[2]), data[3].to(device), list(data[4])
            input_text = tokenizer(uttr, padding=True, truncation=True, max_length=args.max_seq_len, return_tensors="pt").to(device)
            with torch.no_grad():
                logits, _ = model(input_text)
                hard_pred = (logits > 0.5).cpu().numpy()[0]
                label = label.cpu().numpy()[0]
            preds.append(hard_pred)
            labels.append(label)
        precision, recall, f1, acc = compute_metric(labels, preds, f1_average='samples')
        colorprint(f'acc: {acc}, f1: {f1}', 'green')
    
    # plot_confusion_matrix(labels, top1_preds)

def hp_search(args):
    # bss = [16, 12, 8, 4]
    bss = [0.2, 0.4, 0.6, 0.8]
    for bs in bss:
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        model = AutoModel.from_pretrained('roberta-base').to(device)
        
        args = vars(args)
        # args['resp_source_batch_size'] = bs
        args['lamda'] = bs
        args = Namespace(**args)
        
        model = train_resp_source(args, tokenizer, model)
        model.load_state_dict(torch.load(f'model/RSVP_trial_resp_source_{args.task}_{args.seed}.pt', map_location=device))
        model = train_resp(args, model)
        model.load_state_dict(torch.load(f'model/RSVP_trial_resp_{args.task}_{args.seed}.pt', map_location=device))
        model, best_model_path = train_cls(args, tokenizer, model)
        
        args = vars(args)
        args['best_model_path'] = best_model_path
        args = Namespace(**args)
        
        test(args)

if __name__ == "__main__":
    from argparse import Namespace
    args = set_arg()
    for seed in SEEDLIST:
        set_seed(seed)
        best_model_path = train(args)
        args = vars(args)
        args['best_model_path'] = best_model_path
        args = Namespace(**args)
        test(args)
        # hp_search(args)
