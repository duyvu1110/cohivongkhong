import torch, random, gc
from torch import nn, optim
from tqdm import tqdm
import json
import os
from transformers import AdamW
from utils.average_meter import AverageMeter
from utils.functions import formulate_gold
from utils.functions_absa import formulate_gold_absa
from utils.metric import metric, num_metric, overlap_metric, proportional_metric, binary_metric
from utils.metric_absa import metric_absa, proportional_metric_absa, binary_metric_absa
from datetime import datetime
from pdb import set_trace as stop
import shutil

class Trainer(nn.Module):
    def __init__(self, model, data, args):
        super().__init__()
        self.args = args
        self.model = model
        self.data = data
        self.tokenizer = self.args.tokenizer

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        component = ['encoder', 'decoder']
        grouped_params = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': args.weight_decay,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': 0.0,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           not any(nd in n for nd in no_decay) and component[1] in n],
                'weight_decay': args.weight_decay,
                'lr': args.decoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           any(nd in n for nd in no_decay) and component[1] in n],
                'weight_decay': 0.0,
                'lr': args.decoder_lr
            }
        ]
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(grouped_params)
        elif args.optimizer == 'AdamW':
            self.optimizer = AdamW(grouped_params)
        else:
            raise Exception("Invalid optimizer.")

    def train_model(self):
        best_f1 = 0
        if self.args.stage.lower() == "one":
            os.makedirs(os.path.join(self.args.output_path, 'ckpt-three')) 
        elif self.args.stage.lower() == "two":
            try:
                os.makedirs(os.path.join(self.args.output_path, 'ckpt-five'))
            except:
                print("文件地址已存在，请删除后重建")
                shutil.rmtree(os.path.join(self.args.output_path, 'ckpt-five')) # 删除已有路径下的所有文件
                os.makedirs(os.path.join(self.args.output_path, 'ckpt-five'))
                print("文件地址已重建")

        train_loader = self.data['train']
        # result = self.eval_model(self.data.test_loader)
        for epoch in range(self.args.max_epoch):
            # Train
            self.model.train()
            self.model.zero_grad()
            self.optimizer = self.lr_decay(self.optimizer, epoch, self.args.lr_decay)

            print("=== Epoch %d train ===" % epoch, flush=True) # flush=True 会在print结束之后,立即将内存中的东西显示到屏幕上,清空缓存
            avg_loss = AverageMeter()
            for batch_id, (input_ids, targets, _) in enumerate(tqdm(train_loader, desc=f'training on epoch {epoch}')):
                attention_mask = (input_ids != self.args.tokenizer.pad_token_id).long()
                loss, _ = self.model(input_ids, attention_mask, targets)
                avg_loss.update(loss.item(), 1)
                # Optimize
                loss.backward()
                if self.args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm) # old

                if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                if batch_id % 100 == 0 and batch_id != 0:
                    print("     Instance: %d; loss: %.4f" % (batch_id * self.args.batch_size, avg_loss.avg), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            # Validation
            print("=== Epoch %d Validation ===" % epoch)
            result = self.eval_model(self.data['dev'], process='dev')
            f1 = result['f1']
            # Test
            if f1 > best_f1:
                print("Achieving Best Result on Validation Set.", flush=True)
                # torch.save({'state_dict': self.model.state_dict()}, self.args.generated_param_directory + " %s_%s_epoch_%d_f1_%.4f.model" %(self.model.name, self.args.dataset_name, epoch, result['f1']))
                if self.args.stage == "one":
                    torch.save(self.model.state_dict(), open(os.path.join(self.args.output_path, 'ckpt-three', 'best.pt'), 'wb'))
                else:
                    torch.save(self.model.state_dict(), open(os.path.join(self.args.output_path, 'ckpt-five', 'best.pt'), 'wb'))
                best_f1 = f1
                best_result_epoch = epoch

            gc.collect()
            torch.cuda.empty_cache()
        print("Best result on validation set is %f achieving at epoch %d." % (best_f1, best_result_epoch), flush=True)
        if self.args.stage.lower() == "one":
            self.model.load_state_dict(torch.load(open(os.path.join(self.args.output_path, 'ckpt-three', 'best.pt'), 'rb')))
            file_name = 'params'+"_three.txt"
        elif self.args.stage.lower() == "two":
            self.model.load_state_dict(torch.load(open(os.path.join(self.args.output_path, 'ckpt-five', 'best.pt'), "rb")))
            file_name = 'params'+"_five.txt"
        # Test
        print("=== Final Test ===", flush=True)
        result = self.eval_model(self.data['test'], process='test')

        with open(os.path.join(self.args.output_path, file_name),"a") as f:
            print("================   Final Result   ====================", file=f)
            print("-------------- Exact Result --------------------------", file=f)
            print(result[0], file=f)
            print("-------------- Proportional Result --------------------------", file=f)
            print(result[1], file=f)
            print("-------------- Binary Result --------------------------", file=f)
            print(result[2], file=f)
            print("+++++++++++++++++++++++++++++++++++++", file=f)
            print("End time is {}".format(datetime.today().strftime("%Y-%m-%d-%H-%M-%S")), file=f)

        # test_f1 = result['f1'] 


    def eval_model(self, eval_loader, process):
        self.model.eval()
        # print(self.model.decoder.query_embed.weight)
        prediction, gold = {}, {}
        pred_texts = {}

        def get_text(input_ids, start_index, end_index):
            tokenizer = self.args.tokenizer
            text = tokenizer.decode(input_ids[start_index: end_index])
            return text.strip()

        whole_input_ids = []
        with torch.no_grad():
            batch_size = self.args.batch_size
            for batch_id, (input_ids, target, info) in enumerate(tqdm(eval_loader, f'evaluation')):
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
                whole_input_ids += input_ids.tolist()
                if self.args.stage == "one":
                    gold.update(formulate_gold_absa(target, info))
                    gen_triples = self.model.gen_triples_absa(input_ids, attention_mask, info)
                else:
                    gold.update(formulate_gold(target, info))
                    gen_triples = self.model.gen_triples(input_ids, attention_mask, info)

                prediction.update(gen_triples)

        # just output the predicted results
        # if self.args.stage == "one": # absa三元组抽取写入txt
        #     for k in prediction:
        #         pred_texts[k] = {
        #             'aspect': [get_text(whole_input_ids[k], x.aspect_start_index, x.aspect_end_index) for x in prediction[k]],
        #             'opinion': [get_text(whole_input_ids[k], x.opinion_start_index, x.opinion_end_index) for x in prediction[k]],
        #             'sentiment': [x.pred_rel for x in prediction[k]],                    
        #         }
        #     with open(os.path.join(self.args.output_path, 'preds_three.txt'), 'w', encoding='utf-8') as f:
        #         json.dump(pred_texts,f)

        # elif self.args.stage == "two": # 五元组抽取写入txt
        #     for k in prediction:
        #         pred_texts[k] = {
        #             'sub': [get_text(whole_input_ids[k], x.sub_start_index, x.sub_end_index) for x in prediction[k]],
        #             'obj': [get_text(whole_input_ids[k], x.obj_start_index, x.obj_end_index) for x in prediction[k]],
        #             'aspect': [get_text(whole_input_ids[k], x.aspect_start_index, x.aspect_end_index) for x in prediction[k]],
        #             'opinion': [get_text(whole_input_ids[k], x.opinion_start_index, x.opinion_end_index) for x in prediction[k]],
        #             'sentiment': [x.pred_rel for x in prediction[k]],
        #         }
        #     # write to file
        #     with open(os.path.join(self.args.output_path, 'preds_five.txt'), 'w', encoding='utf-8') as f:
        #         json.dump(pred_texts, f)
        
        if process=='dev':
            print("run dev", process)
            if self.args.stage.lower() == "one":
                return metric_absa(prediction, gold)
            elif self.args.stage.lower() == "two":
                return metric(prediction, gold)
        elif process=='test':
            print("run", process)
            if self.args.stage.lower() == "one":
                return metric_absa(prediction, gold), proportional_metric_absa(prediction, gold), binary_metric_absa(prediction, gold)
            else:
                return metric(prediction, gold), proportional_metric(prediction, gold), binary_metric(prediction, gold)


    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate):
        # lr = init_lr * ((1 - decay_rate) ** epoch)
        if epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (1 - decay_rate)
                # print(param_group['lr'])
        return optimizer
