import time
from modelscope import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from layers import add_lora

class TaskLayer(nn.Module):
    def __init__(self, hidden_size, embedding_size=128, dropout_rate=0.5):
        super(TaskLayer, self).__init__()
        self.custom_linear_0 = nn.Linear(in_features=hidden_size, out_features=embedding_size, dtype=torch.bfloat16)
        self.custom_linear_dropout = nn.Dropout(p=dropout_rate)
        self.custom_linear_activation = nn.GELU()
        self.custom_linear_1 = nn.Linear(in_features=embedding_size, out_features=1, dtype=torch.bfloat16)

    def forward(self, output):
        embedding = self.custom_linear_0(output)
        output = self.custom_linear_dropout(embedding)
        output = self.custom_linear_activation(output)
        output = self.custom_linear_1(output)
        return output, embedding
class LanguageModelWithLinear(nn.Module):
    def __init__(self, pretrained_model_name, lora_r, lora_alpha, k=128, dropout_rate=0.5):
        super(LanguageModelWithLinear, self).__init__()

        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='sum')

        self.pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            output_hidden_states=True,
            use_cache=True,
            # torch_dtype=torch.float16
        )
        if lora_r > 0:
            add_lora(self.pretrained_model, lora_r, lora_alpha, pissa = False)
        self.custom_linear = TaskLayer(self.pretrained_model.config.hidden_size, k, dropout_rate)

    def forward(self, input_ids, attention_mask, bce_target=None, ce_target=None):
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]

        x, embedding = self.custom_linear(last_hidden_state)
        bce = x.squeeze(-1)
        ce = outputs.logits[:, -1, :]
        cls = bce
        logits = ce
        if bce_target is None:
            return  cls, logits, embedding

        bce_loss = self.bce_loss_fn(bce, bce_target)
        ce_loss = self.ce_loss_fn(ce, ce_target)
        return bce_loss + ce_loss, cls, logits

class LLRModel:
    def __init__(self, model_path, checkpoint_path):
        self.model, self.tokenizer = self.load_model(model_path, checkpoint_path)
        self.max_token_length = 2048
    
    def load_model(self, model_path, checkpoint_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = '<|endoftext|>'
        tokenizer.padding_side = "left"
        lora_r, lora_alpha, k = 16, 1, 256
        model = LanguageModelWithLinear(model_path, lora_r, lora_alpha, k).to('cuda').to(dtype=torch.float32)
        model.load_state_dict(torch.load(checkpoint_path))
        model = model.eval()
        return model, tokenizer
    
    def predict(self, data):
        output = self.tokenizer([data], max_length=self.max_token_length - 1, padding='max_length',
                                truncation=True, return_tensors="pt")
        output.input_ids = torch.hstack((output.input_ids, torch.tensor([151643]).repeat(output.input_ids.shape[0], 1)))
        output.attention_mask = torch.hstack(
            (output.attention_mask, torch.tensor([1]).repeat(output.attention_mask.shape[0], 1)))
        with torch.no_grad():
            cls, _, embedding = self.model(output.input_ids.to('cuda'), output.attention_mask.to('cuda'))
            cls = torch.sigmoid(cls)
            score = cls.cpu().numpy().tolist()[0]
        return score, embedding[0].cpu().numpy().tolist()

llr_model = LLRModel(model_path='../ckpt/Qwen1___5-1___8B', checkpoint_path='../ckpt/best_model_2.bin')

import pandas as pd
data = pd.read_csv("../data/sample.csv")

data['last_content_create_times'] = data['last_content_create_times'].fillna('')
data['abstracts'] = data['abstracts'].fillna('')
data['feature_prompt'] = "\n\n1.跟进类特征: " + \
                     "\n\t1.1. 60天内总跟进次数: " + data.total_follow_up_cnt.fillna(-1).astype(int).astype(str) + \
                     "\n\t1.2. 60天内跟进频率: " + data.follow_frequence.round(4).astype(str) + \
      "\n\t1.3. 60天内上次跟进间隔: " + data.last_follow_interval.fillna(-1).astype(int).astype(str) + \
                     "\n\t1.4. 60天内上次的跟进意向: " + data.last_intention_level_id.fillna(-1).astype(int).astype(str) + \
                     "\n\t1.5. 60天内上次跟进的状态: " + data.last_follow_status_id.fillna(-1).astype(int).astype(str) + \
                     "\n\t1.6. 60天内上上次跟进间隔: " + data.last2_follow_interval.fillna(-1).astype(int).astype(str) + \
                     "\n\t1.7. 60天内上上次的跟进意向: " + data.last2_intention_level_id.fillna(-1).astype(int).astype(str) + \
                     "\n\n2. 通话类特征: " + \
                     "\n\t2.1. 60天内通话接通次数: " + data.call_conn_cnt.fillna(-1).astype(int).astype(str) + \
                     "\n\t2.2. 60天内通话接通频率: " + data.call_conn_frequence.round(4).astype(str) + \
                     "\n\t2.3. 60天内通话接通比例: " + data.call_conn_ratio.round(4).astype(str) + \
                     "\n\t2.4. 60天内最大通话时长(分钟): " + data.max_duration.round(2).astype(str) + \
                     "\n\t2.5. 60天内平均通话时长(分钟): " + data.avg_duration.round(2).astype(str) + \
                     "\n\t2.6. 60天内上次通话时长(分钟): " + data.last_call_duration.round(2).astype(str) + \
                     "\n\t2.7. 60天内上上次通话时长(分钟): " + data.last2_call_duration.round(2).astype(str) + \
                     "\n\n3. 试驾类特征: " + \
                     "\n\t3.1. 180天内试驾成功次数: " + data.attempt_drive_success_cnt.fillna(-1).astype(int).astype(str) + \
                     "\n\t3.2. 180天内试驾类型: " + data.attempt_drive_type_cnt.fillna(-1).astype(int).astype(str) + \
                     "\n\t3.3. 180天内试驾车型: " + data.attempt_drive_vehicle_type_cnt.fillna(-1).astype(int).astype(str) + \
                     "\n\n4. 工单-线索类特征: " + \
                     "\n\t4.1. 工单状态编码: " + data.ticket_status_id.fillna(-1).astype(int).astype(str) + \
                     "\n\t4.2. 创建工单次数: " + data.ticket_create_cnt.fillna(-1).astype(int).astype(str) + \
                     "\n\t4.3. 平均战败间隔天数: " + data.avg_defeat_interval.fillna(-1).astype(int).astype(str) + \
                     "\n\t4.4. 最近一次工单创建天数: " + data.last_ticket_create_interval.fillna(-1).astype(int).astype(str) + \
                     "\n\t4.5.  线索渠道: " + data.channel_code.fillna(-1).astype(int).astype(str) + \
                     "\n\n5. 专家类特征: " + \
                     "\n\t5.1. 历史待服务用户数: " + data.specialist_total_customer_num.fillna(-1).astype(int).astype(str) + \
                     "\n\t5.2. 历史锁单率: " + data.specialist_lock_rate.round(4).astype(str) + \
                     "\n\t5.3. 门店编码: " + data.specialist_store_code.astype(str) + \
                     "\n\t5.4. 省份ID: " + data.specialist_store_province_id.fillna(-1).astype(int).astype(str) + \
                     "\n\t5.5. 城市ID: " + data.specialist_store_city_id.fillna(-1).astype(int).astype(str) + \
                     "\n\t5.6. 进店线索转化率: " + data.specialist_into_store_ticket_lock_rate.round(2).astype(str) + \
                     "\n\t5.7. 专家等级: " + data.specialist_level.fillna(-1).astype(int).astype(str) + \
                     "\n\t5.8. 入职周期: " + data.specialist_join_level.fillna(-1).astype(int).astype(str) + \
                     "\n\t5.9. 试驾质量分: " + data.specialist_avg_drive_score.round(2).astype(str) 
def generat_corpus(data):
    recent_times = data['last_content_create_times'].split(',') if data['last_content_create_times'] else []
    recent_record = data['abstracts'].split('$%$') if data['abstracts'] else []
    return '你是一个销售专家，请你根据以下信息判断该客户是否有意愿在未来锁单？用户特征如下：'+ data['feature_prompt'] + "客户与销售专家的通话记录摘要如下，通话记录摘要按照距离现在从近到远的时间顺序排列："+',\n\n '.join([f'\"{time}: {record}\"' for time, record in zip(recent_times, recent_record)][::-1])
data['corpus'] = data.apply(generat_corpus,axis=1)

score, embedding = llr_model.predict(data['corpus'][0])

print(score)