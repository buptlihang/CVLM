from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import re

from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers import ChineseCLIPVisionModel, ChineseCLIPImageProcessor, ChineseCLIPVisionConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from .utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX


def build_vision_projector(config):
    modules = [
        nn.Linear(config.adapter_hidden_size, config.hidden_size),
        nn.GELU(),
        nn.Linear(config.hidden_size, config.hidden_size)
    ]
    return nn.Sequential(*modules)


def build_vision_encoder(config):
    return ChineseCLIPVisionModel(
        ChineseCLIPVisionConfig.from_pretrained(config.vision_encoder))


def build_image_processor(config):
    return ChineseCLIPImageProcessor.from_pretrained(config.vision_encoder)


class CvlmConfig(LlamaConfig):
    model_type = "cvlm"


class CvlmModel(LlamaModel):
    config_class = CvlmConfig

    def __init__(self, config: LlamaConfig):
        super(CvlmModel, self).__init__(config)
        self.vision_encoder = build_vision_encoder(config)
        self.adapter = build_vision_projector(config)
        self.image_processor = build_image_processor(config)


class CvlmForCausalLM(LlamaForCausalLM):
    config_class = CvlmConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = CvlmModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_vision_encoder(self):
        return self.model.vision_encoder

    def extract_vision_feature(self, images):
        vision_features_outs = self.model.vision_encoder(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True)
        vision_features = vision_features_outs.hidden_states[-2][:, 1:].to(
            images.dtype)
        vision_features = self.model.adapter(vision_features)
        return vision_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds,
                             use_cache=use_cache,
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states,
                             return_dict=return_dict)
        logits = self.lm_head(outputs[0])

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      attention_mask=None,
                                      inputs_embeds=None,
                                      **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "images": kwargs.get("images", None),
        })
        return model_inputs

    def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask,
                                             past_key_values, labels, images):
        vision_encoder = self.get_vision_encoder()
        if vision_encoder is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_encoder is not None and images is not None and input_ids.shape[
                    1] == 1:
                attention_mask = torch.ones(
                    (attention_mask.shape[0],
                     past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            vision_features = self.extract_vision_feature(concat_images)
            split_sizes = [image.shape[0] for image in images]
            vision_features = torch.split(vision_features, split_sizes, dim=0)
            vision_features = [x.flatten(0, 1) for x in vision_features]
        else:
            vision_features = self.extract_vision_feature(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_vision_features = vision_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(
                    cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(
                    cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([
                    cur_input_embeds_1, cur_vision_features[0:0],
                    cur_input_embeds_2
                ],
                                             dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(
                cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_vision_features = vision_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                cur_new_input_embeds.append(self.get_model().embed_tokens(
                    cur_input_ids[:image_token_start]))
                cur_new_input_embeds.append(cur_vision_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(
                        torch.full((cur_vision_features.shape[0], ),
                                   IGNORE_INDEX,
                                   device=labels.device,
                                   dtype=labels.dtype))
                    cur_labels = cur_labels[image_token_start + 1:]
                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start + 1:]
                image_token_indices = torch.where(
                    cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(
                    self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds
            ]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (cur_new_embed,
                     torch.zeros((max_len - cur_new_embed.shape[0],
                                  cur_new_embed.shape[1]),
                                 dtype=cur_new_embed.dtype,
                                 device=cur_new_embed.device)),
                    dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (cur_new_label,
                         torch.full((max_len - cur_new_label.shape[0], ),
                                    IGNORE_INDEX,
                                    dtype=cur_new_label.dtype,
                                    device=cur_new_label.device)),
                        dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                        attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1], ),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] -
                         cur_new_labels.shape[0], ),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask,
                         new_attn_mask_pad_right),
                        dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0],
                     new_input_embeds.shape[1] - input_ids.shape[1]),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device)
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels
