from functools import partial
from vit import VisionTransformer
from xbert import BertConfig, BertModel
from xbert import BertConfig, BertForMaskedLM

import torch
from torch import nn
import torch.nn.functional as F


class ALBEF3class(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 num_category=None
                 ):
        super().__init__()

        self.num_label = num_category
        self.tokenizer = tokenizer
        self.distill = config['distill']
        self.loss_fct = nn.BCELoss

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=6, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertModel(config=bert_config)#, resumeadd_pooling_layer=False)
        self.label_encoder = BertModel(config=bert_config, add_pooling_layer=False)

        self.cls_head = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
            nn.ReLU(),
            # nn.Linear(self.text_encoder.config.hidden_size, 15)
            nn.Linear(self.text_encoder.config.hidden_size, self.num_label)
        )

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=6, num_heads=12,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            self.text_encoder_m = BertModel(config=bert_config, add_pooling_layer=False)
            self.label_encoder_m = BertModel(config=bert_config, add_pooling_layer=False)

            self.cls_head_m = nn.Sequential(
                nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.text_encoder.config.hidden_size, self.num_label)
            )

            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.label_encoder, self.label_encoder_m],
                                [self.cls_head, self.cls_head_m],
                                ]

            self.copy_params()
            self.momentum = 0.995

    def forward(self, text, label, one_hot_label, alpha=0, train=True):
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        # label_embeds = self.label_encoder(label.input_ids, attention_mask=label.attention_mask,
                                            # return_dict=True, mode='text').last_hidden_state

        if train:   #train mode
            output = self.text_encoder(input_ids,
                                       attention_mask=text.attention_mask,
                                    #    encoder_hidden_states=label_embeds,
                                    #    encoder_attention_mask=label.attention_mask,
                                       return_dict=True, mode='text'
                                       )
            prediction = self.cls_head(output.last_hidden_state[:, 0, :])   # torch.Size([4, 15])
            if self.distill:
                with torch.no_grad():
                    self._momentum_update()
                    # label_embeds_m = self.label_encoder_m(label.input_ids, attention_mask=label.attention_mask,
                                            # return_dict=True, mode='text').last_hidden_state   
                    output_m = self.text_encoder_m(input_ids,
                                                   attention_mask=text.attention_mask,
                                                #    encoder_hidden_states=label_embeds_m,
                                                #    encoder_attention_mask=label.attention_mask,
                                                   return_dict=True, mode='text'
                                                   )
                    prediction_m = self.cls_head_m(output_m.last_hidden_state[:, 0, :])
            
                loss=(1-alpha)*nn.BCELoss()(torch.sigmoid(prediction), one_hot_label) - alpha * torch.sum(
                    F.log_softmax(prediction, dim=1) * F.softmax(prediction_m, dim=1), dim=1).mean()
            else:
                loss=nn.BCELoss()(torch.sigmoid(prediction), one_hot_label)
            return loss

        else:   #eval mode
            output = self.text_encoder(input_ids,
                                        attention_mask=text.attention_mask,
                                        # encoder_hidden_states=label_embeds,
                                        # encoder_attention_mask=label.attention_mask,
                                        return_dict=True, mode='text'
                                        )
            prediction = self.cls_head(output.last_hidden_state[:, 0, :])
            return prediction


    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)