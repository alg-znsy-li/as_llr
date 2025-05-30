import argparse
import os
import traceback

import deepspeed
from modelscope import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DistributedSampler

from dataset import data_processing, MyDataset, MyDatasetWithDataAug
from layers import add_lora, append_llama_pro_block_group
from deepspeed.utils import log_dist
from tqdm import tqdm

from optimizer import WarmupExponentialLR


class TaskLayer(nn.Module):
    def __init__(self, hidden_size, embedding_size=128, dropout_rate=0.5):
        super(TaskLayer, self).__init__()
        self.custom_linear_0 = nn.Linear(in_features=hidden_size, out_features=embedding_size, dtype=torch.bfloat16)
        self.custom_linear_dropout = nn.Dropout(p=dropout_rate)
        self.custom_linear_activation = nn.GELU()
        self.custom_linear_1 = nn.Linear(in_features=embedding_size, out_features=1, dtype=torch.bfloat16)

    def forward(self, output):
        output = self.custom_linear_0(output)
        output = self.custom_linear_dropout(output)
        output = self.custom_linear_activation(output)
        output_f = self.custom_linear_1(output)
        return output_f, output


class LanguageModelWithLinear(nn.Module):
    def __init__(self, pretrained_model_name, lora_r, lora_alpha, llama_pro_group_size=-1, 
    k=128, dropout_rate=0.5, cl_tau=0, pissa=False):
        super(LanguageModelWithLinear, self).__init__()
        self.cl_tau = cl_tau
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='sum')

        self.pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            use_cache=False
        )
        log_dist(message=f"{self.pretrained_model}", ranks=[0])

        log_dist(message=f"*******LanguageModelWithLinear llama_pro group size:{llama_pro_group_size}")
        if llama_pro_group_size > 0:
            append_llama_pro_block_group(self.pretrained_model, llama_pro_group_size)

        else:
            if lora_r > 0:
                add_lora(self.pretrained_model, lora_r, lora_alpha, pissa)
        self.custom_linear = TaskLayer(self.pretrained_model.config.hidden_size, k, dropout_rate)

        log_dist(message="*******LanguageModelWithLinear COMPLETE", ranks=[0])

    def cl_loss_fn(self, x, b, tau=1):
        M = x @ x.T
        x_n = torch.norm(x, dim=-1, keepdim=True)
        V = x_n @ x_n.T
        M = M[:b, b:]
        V = V[:b, b:]
        M = (M / V / tau).exp()
        M = M / M.sum(-1, keepdim=True)
        return -(torch.diagonal(M.log())).sum()

    def forward(self, input_ids, attention_mask, bce_target, ce_target):
        b = input_ids.shape[0]
        if self.cl_tau > 0 and b > 1:
            input_ids = input_ids.repeat(2, 1)
            attention_mask = attention_mask.repeat(2, 1)
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        x, x_hidden = self.custom_linear(last_hidden_state)
        bce = x[:b].squeeze(-1)
        ce = outputs.logits[:b, -1, :]
        cls = bce
        logits = ce
        bce_loss = self.bce_loss_fn(bce, bce_target)
        ce_loss = self.ce_loss_fn(ce, ce_target)
        loss = bce_loss + ce_loss
        if self.cl_tau > 0 and b > 1:
            cl_loss = self.cl_loss_fn(x_hidden, b, self.cl_tau)
            loss += cl_loss
        return loss, cls, logits

def eval(args,pretrained_model_name):
    ii = args.ii
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    llama_pro_group_size = args.llama_pro_group_size
    k = args.k
    dropout_rate = args.dropout_rate
    max_len = args.max_len

    world_size = int(os.environ.get("WORLD_SIZE"))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"
    new_model = LanguageModelWithLinear(pretrained_model_name, lora_r, lora_alpha, llama_pro_group_size, k, dropout_rate, args.cl_tau, args.pissa)
    if rank == 0:
        print(new_model)
    log_dist(message="HERE ---- -3", ranks=[0])
    train_data, test_data = data_processing(world_size=world_size, path=args.data_file)
    log_dist(message="HERE ---- -2", ranks=[0])
    dataset1 = MyDataset(train_data, tokenizer, max_len)
    log_dist(message="HERE ---- -1", ranks=[0])

    parameters = filter(lambda p: p.requires_grad, new_model.parameters())
    params_g1 = list()
    params_g2 = list()
    for param in new_model.named_parameters():
        if param[1].requires_grad:
            if param[0].startswith("custom_linear"):
                print(f"{param[0]} in group1")
                params_g1.append(param[1])
            else:
                if args.llama_pro_group_size <= 0:
                    if param[0].find("lora") == -1:
                        param[1].requires_grad = False
                    else:
                        print(f"{param[0]} in group2")
                        params_g2.append(param[1])
                else: 
                    if param[0].find('mlp.down_proj') >=0 or param[0].find('self_attn.o_proj') >= 0:
                        params_g1.append(param[1])
                        print(f"{param[0]} in group1")
                    else:
                        params_g2.append(param[1]) 
                        print(f"{param[0]} in group2")

    log_dist(message="HERE ---- 0", ranks=[0])
    lr2 = args.lr / 20.0
    optimizer_ = torch.optim.AdamW([
        {
            "params": params_g1, 'lr': args.lr, "name": "task"
        },
        {
            "params": params_g2, 'lr': lr2, "name": "llm"
        }
    ], lr=args.lr)

    log_dist(message=f"HERE ---- 1{args.warmup_step_num}", ranks=[0])

    lr_scheduler_ = WarmupExponentialLR(optimizer=optimizer_, warmup_step=args.warmup_step_num, gamma=0.999)
    log_dist(message="HERE ---- 2", ranks=[0])
    log_dist(message="begin create deepspeed model", ranks=[0])
    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
            args=args, model=new_model, optimizer=optimizer_, lr_scheduler=lr_scheduler_,
            model_parameters=parameters, training_data=dataset1,
        )
    model_engine.load_checkpoint(load_dir=args.out_dir,
                                  tag=f"tag-ff-{ii}")
    model_engine.eval()
    evaluate(test_data=test_data, model_engine=model_engine, tokenizer=tokenizer,
             max_len=max_len, ii=ii)

def train(args,pretrained_model_name):
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    llama_pro_group_size = args.llama_pro_group_size
    k = args.k
    dropout_rate = args.dropout_rate
    max_len = args.max_len

    world_size = int(os.environ.get("WORLD_SIZE"))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"
    new_model = LanguageModelWithLinear(pretrained_model_name, lora_r, lora_alpha, llama_pro_group_size, k, dropout_rate, args.cl_tau, args.pissa)
    if rank == 0:
        print(new_model)
    log_dist(message="HERE ---- -3", ranks=[0])
    train_data, test_data = data_processing(world_size=world_size, path=args.data_file)
    log_dist(message="HERE ---- -2", ranks=[0])
    dataset1 = MyDatasetWithDataAug(train_data, tokenizer, max_len, args.data_aug_n)
    log_dist(message="HERE ---- -1", ranks=[0])

    parameters = filter(lambda p: p.requires_grad, new_model.parameters())
    params_g1 = list()
    params_g2 = list()
    for param in new_model.named_parameters():
        if param[1].requires_grad:
            if param[0].startswith("custom_linear"):
                print(f"{param[0]} in group1")
                params_g1.append(param[1])
            else:
                if args.llama_pro_group_size <= 0:
                    if param[0].find("lora") == -1:
                        param[1].requires_grad = False
                    else:
                        print(f"{param[0]} in group2")
                        params_g2.append(param[1])
                else: 
                    if param[0].find('mlp.down_proj') >=0 or param[0].find('self_attn.o_proj') >= 0:
                        params_g1.append(param[1]) 
                        print(f"{param[0]} in group1")
                    else:
                        params_g2.append(param[1])
                        print(f"{param[0]} in group2")

    log_dist(message="HERE ---- 0", ranks=[0])
    lr2 = args.lr / 20.0
    optimizer_ = torch.optim.AdamW([
        {
            "params": params_g1, 'lr': args.lr, "name": "task"
        },
        {
            "params": params_g2, 'lr': lr2, "name": "llm"
        }
    ], lr=args.lr)

    log_dist(message=f"HERE ---- 1{args.warmup_step_num}", ranks=[0])

    lr_scheduler_ = WarmupExponentialLR(optimizer=optimizer_, warmup_step=args.warmup_step_num, gamma=0.999)
    log_dist(message="HERE ---- 2", ranks=[0])
    log_dist(message="begin create deepspeed model", ranks=[0])
    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
            args=args, model=new_model, optimizer=optimizer_, lr_scheduler=lr_scheduler_,
            model_parameters=parameters, training_data=dataset1,
        )
    for param in model_engine.named_parameters():
        if param[1].requires_grad:
            log_dist(message=f"parameters:{param[0]}", ranks=[0])
    log_dist(message="end create deepspeed model", ranks=[0])


    def get_model_engine_device(model_engine):
        return f"cuda:{model_engine.local_rank}"

    rank = get_model_engine_device(model_engine=model_engine)

    y_true_tensor = torch.tensor([]).to(rank)
    y_pred_tensor = torch.tensor([]).to(rank)
    ii = 0
    log_dist(message=f"total training set size: {len(train_dataloader)}", ranks=[0])
    for epoch in range(args.epochs):
        log_dist(message=f"Epoch {epoch + 1}/{args.epochs}", ranks=[0])
        for batch_idx, (input_ids, attention_mask, label, label_str) in enumerate(train_dataloader):

            input_ids = input_ids.to(rank)
            attention_mask = attention_mask.to(rank)
            label = label.to(rank)
            label_str = label_str.to(rank)

            loss, cls, logits = model_engine(input_ids, attention_mask, label, label_str)
            cls = torch.sigmoid(cls)
            y_head = cls.detach()
            y_true_tensor = torch.hstack((y_true_tensor, label))
            y_pred_tensor = torch.hstack((y_pred_tensor, y_head))

            if ii % 10 == 0:
                log_dist(message=f"\n\n\nLoss:{loss}\t\t\t\t--------Step:{ii}", ranks=[0])
                for ele in optimizer.param_groups:
                    log_dist(message=f"{ele['name']}train_step-{ii}\t\tlast_lr-{ele['lr']}",ranks=[0])



            if (ii+1) % 100 == 0:

                log_dist(message=f"{y_true_tensor.detach().cpu()}", ranks=[0])
                log_dist(message=f"{y_pred_tensor.detach().cpu()}", ranks=[0])
                auc = roc_auc_score(y_true_tensor.cpu().float().numpy()[-256:],
                                    y_pred_tensor.cpu().float().numpy()[-256:])

                log_dist(message=f"\n\n\nAUC:{auc}\t\t\t\t--------Step:{ii}", ranks=[0])
                if model_engine.local_rank == 0:
                    if model_engine.monitor.tb_monitor is not None:
                        model_engine.monitor.tb_monitor.summary_writer.add_scalar("train_auc", auc, ii+1)

            model_engine.backward(loss)
            model_engine.step()
            ii += 1
            if ii % args.ckpt_interval == 0:
                n = len(test_data) // 10
                test_data1 = test_data.sample(n, random_state=42).reset_index(drop=True)

                evaluate(test_data=test_data1, model_engine=model_engine, tokenizer=tokenizer, max_len=max_len, ii=ii)


                client_dict = {
                    "loss": loss
                }

                model_engine.save_checkpoint(save_dir=args.out_dir,
                                    client_state=client_dict,
                                    tag=f"tag-ff-{ii}",
                                    save_latest=True)

    client_dict = {
        "loss": loss
    }

    model_engine.save_checkpoint(save_dir=args.out_dir,
                                client_state=client_dict,
                                tag=f"tag-ff",
                                save_latest=True)

    evaluate(test_data=test_data, model_engine=model_engine, tokenizer=tokenizer,
             max_len=max_len, ii=len(train_dataloader)*args.epochs+1)





def evaluate(test_data, model_engine, tokenizer, max_len, ii):
    log_dist(message=f"\n\n\n ----------- Evaluation -------------- \n\n\n", ranks=[0])
    rank = model_engine.local_rank
    y_true_test = torch.tensor([]).to(rank)
    y_pred_test = torch.tensor([]).to(rank)
    dataset_test = MyDataset(test_data, tokenizer, max_len)

    log_dist(message=f"Evaluation Steps 1 is {len(dataset_test)}", ranks=[0])
    dataset_sampler = DistributedSampler(dataset_test)
    dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  sampler=dataset_sampler)
    log_dist(message=f"Evaluation Steps 2 is {len(dataloader_test)}", ranks=[0])
    model_engine.eval()
    for batch_idx, (input_ids, attention_mask, label, label_str) in enumerate(dataloader_test):
        input_ids = input_ids.to(rank)
        attention_mask = attention_mask.to(rank)
        label = label.to(rank)
        label_str = label_str.to(rank)

        with torch.no_grad():

            loss, cls, logits = model_engine(input_ids, attention_mask, label, label_str)
            cls = torch.sigmoid(cls)
            y_head = cls.detach()
            y_true_test = torch.hstack((y_true_test, label))
            y_pred_test = torch.hstack((y_pred_test, y_head))
        if batch_idx % 10 == 0:
            log_dist(message=f"eval_step-{batch_idx}---loss:{loss}", ranks=[0])

    y_true_test_all_gather = [torch.zeros_like(y_true_test) for i in range(model_engine.world_size)]
    torch.distributed.all_gather(y_true_test_all_gather, y_true_test)

    y_pred_test_all_gather = [torch.zeros_like(y_pred_test) for i in range(model_engine.world_size)]
    torch.distributed.all_gather(y_pred_test_all_gather, y_pred_test)
    y_true = torch.cat(tensors=y_true_test_all_gather, dim=0)
    y_score = torch.cat(tensors=y_pred_test_all_gather, dim=0)
    log_dist(message=f"y_true:{y_true.detach().cpu().numpy()}", ranks=[0])
    log_dist(message=f"y_score:{y_score.detach().cpu().numpy()}", ranks=[0])

    AUC = roc_auc_score(y_true=y_true.detach().cpu().numpy(),
                        y_score=y_score.detach().cpu().float().numpy())

    log_dist(message=f"\n\n\nEvaluation AUC:{AUC}", ranks=[0])
    if model_engine.local_rank == 0:
        if model_engine.monitor.tb_monitor is not None:
            model_engine.monitor.tb_monitor.summary_writer.add_scalar("eval_auc", AUC, ii + 1)

    model_engine.train()
    log_dist(message="----------------Evaluation Completed!!!----------------", ranks=[0])

def add_argument():
    parser = argparse.ArgumentParser(description="qwen")

    parser.add_argument("--out_dir", default="/chj/caoyuji/lillm/",
                        type=str, help="where to save ckpt")
    parser.add_argument("--data_file",
                        default="/lpai/data/content_and_feat_baseline_0418.snappy.parquet",
                        type=str, help="train data file")
    parser.add_argument("--pretrained_model",
                        default="/lpai/llm_repo/qwen/Qwen1___5-1___8B",
                        type=str, help="base LLM")


    parser.add_argument("--batch_size", default=32,
                        type=int, help="mini-batch size (default: 32)")

    parser.add_argument("--epochs", default=30,
                        type=int, help="number of total epochs (default: 30) 训练轮数")

    parser.add_argument("--lr", default=1.0E-4,
                        type=float, help="learning rate")

    parser.add_argument("--ckpt_interval", default=10240,
                        type=int, help="save checkpoint at a given interval")

    parser.add_argument("--warmup_step_num", default=4000,
                        type=int, help="warmup step num")

    parser.add_argument("--local_rank", default=-1,
                        type=int, help="local rank passed from distributed launcher ")

    parser.add_argument("--log-interval", default=20,
                        type=int, help="output logging information at a given interval")
            



    parser.add_argument("--lora_r", default=16, type=int, help="lora rank")
    parser.add_argument("--lora_alpha", default=1.0, type=float, help="lora_alpha")

    parser.add_argument('--pissa', default=False, action='store_true',help="是否使用pissa 初始化lora")

    parser.add_argument("--max_len", default=7600, type=int, help="max_len")
    parser.add_argument("--dropout_rate", default=0.5, type=float, help="dropout rate")
    parser.add_argument("--k", default=128, type=int, help="hidden size of task layers")
    parser.add_argument("--cl_tau", default=0, type=float, help="contrastive learning loss tau")

    parser.add_argument('--llama_pro_group_size',
                           default=0, type=int, help="llama pro group size, 0 means no llama")

    parser.add_argument('--do_eval', default=False, action='store_true',help=""
                                                                              "only eval")


    parser.add_argument('--ii', default=0, type=int,help="tag_step")
    parser.add_argument('--data_aug_n', default=0, type=int, help="数据增强的程度")

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args

if __name__ == """__main__""":
    args = add_argument()
    if args.do_eval:
        eval(args, pretrained_model_name=args.pretrained_model)
        exit(0)
    train(args, pretrained_model_name=args.pretrained_model)



