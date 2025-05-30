import random

import pandas as pd
from torch.utils.data import Dataset
import torch

TRAIN_SAMPLE_BASE = 1
TEST_SAMPLE_BASE = 1
def data_processing(world_size, path = None):
    data = pd.read_pickle(path)
    data[['content', 'last_content', 'last2_content']] = data[['content', 'last_content', 'last2_content']].fillna("")
    data[['abstracts', 'last_content_create_times']] = data[['abstracts', 'last_content_create_times']].fillna("")
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
    data = data[['customer_account_id', 'dataset', 'corpus', 'lock_label']]
    data['lock_label_str'] = data['lock_label'].apply(lambda x: '是' if x == 1 else '否')
    train_data = data[data.dataset == 1].reset_index(drop=True)
    print("oringinal number of train samples", len(train_data))
    pos_train_data = train_data[train_data.lock_label == 1].reset_index(drop=True)
    train_data = [train_data] + [pos_train_data for _ in range(9)]
    train_data = pd.concat(train_data)
    n = int(len(train_data) / TRAIN_SAMPLE_BASE) 
    n = (n // world_size) * world_size
    print("\n\n\n number of train samples", n)
    train_data = train_data.sample(n, random_state=42).reset_index(drop=True)
    test_data = data[data.dataset == 0].reset_index(drop=True)
    n = int(len(test_data) / TEST_SAMPLE_BASE)  
    print("\n\n\n number of test samples", len(test_data))
    return train_data, test_data
    
class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_length):
        self.data = data.corpus.values
        self.label = data.lock_label.values
        self.label_str = data.lock_label_str.values
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    def __getitem__(self, index):
        output = self.tokenizer([self.data[index]],
                                max_length=self.max_token_length - 1,
                                padding='max_length',
                                truncation=True, return_tensors="pt")
        output.input_ids = torch.hstack((output.input_ids,
                                torch.tensor([self.tokenizer.pad_token_id]).repeat(output.input_ids.shape[0], 1)))
        output.attention_mask = torch.hstack(
            (output.attention_mask, torch.tensor([1]).repeat(output.attention_mask.shape[0], 1)))
        y = self.label[index]
        return (output.input_ids.squeeze(0),
                output.attention_mask.squeeze(0),
                torch.tensor(y,dtype=torch.bfloat16),
                self.tokenizer(self.label_str[index], return_tensors="pt", max_length=1, padding='max_length').input_ids[:, -1].squeeze(-1)
                )

    def __len__(self):
        return len(self.data)

class MyDatasetWithDataAug(Dataset):

    def __init__(self, data, tokenizer, max_token_length, n=0):
        self.data = data.corpus.values
        self.label = data.lock_label.values
        self.label_str = data.lock_label_str.values

        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

        self.nlp_neg_pool = set()
        self.nlp_pos_pool = set()

        self.data_new = list()
        self.label_str_new = list()
        self.label_new = list()

        self.mk_data_aug(n)

    def __getitem__(self, index):

        output = self.tokenizer([self.data_new[index]],
                                max_length=self.max_token_length - 1,
                                padding='max_length',
                                truncation=True, return_tensors="pt")
        output.input_ids = torch.hstack((output.input_ids,
                torch.tensor([self.tokenizer.pad_token_id]).repeat(output.input_ids.shape[0],1)))
        output.attention_mask = torch.hstack(
            (output.attention_mask, torch.tensor([1]).repeat(output.attention_mask.shape[0], 1)))

        return (output.input_ids.squeeze(0),
                output.attention_mask.squeeze(0),
                torch.tensor(self.label_new[index], dtype=torch.bfloat16),
                self.tokenizer(self.label_str_new[index],
                               return_tensors="pt",
                               max_length=1,
                               padding='max_length').input_ids[:, -1].squeeze(-1)
                )


    def __len__(self):
        return len(self.data_new)

    def get_nlp_features(self, i_str):
        beg_idx = i_str.find("通话记录按时间顺序是")
        return i_str[:beg_idx], i_str[beg_idx:]

    def mk_data_aug(self, n=0):

        for i in range(len(self.data)):
            x_ele = self.data[i]
            y_ele = self.label[i]
            ys_ele = self.label_str[i]
            x_tbl_ele, x_nlp_ele = self.get_nlp_features(i_str=x_ele)

            if y_ele == 1:
                for _ in range(n):
                    if len(self.nlp_neg_pool) > 24 :
                        x_nlp_ele_neg = random.sample(list(self.nlp_neg_pool),1)[0]
                        x_ele_new = x_tbl_ele + x_nlp_ele_neg

                        self.data_new.append(x_ele_new)
                        self.label_new.append(0)
                        self.label_str_new.append('否')

            self.data_new.append(x_ele)
            self.label_new.append(y_ele)
            self.label_str_new.append(ys_ele)

            if y_ele == 1:
                self.nlp_pos_pool.add(x_nlp_ele)
            else:
                self.nlp_neg_pool.add(x_nlp_ele)

            if len(self.nlp_pos_pool) > 1024*2:
                a = random.sample(list(self.nlp_pos_pool), 1024)
                self.nlp_pos_pool = set(a)
            if len(self.nlp_neg_pool) > 1024*2:
                a = random.sample(list(self.nlp_neg_pool), 1024)
                self.nlp_neg_pool = set(a)


