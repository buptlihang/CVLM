import argparse
import torch
import os
import json
from tqdm import tqdm

from model.utils import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from model.utils import build_conversation, load_pretrained_model, disable_torch_init, get_model_name_from_path
from model.utils import tokenizer_image_token, process_images
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
from collections import defaultdict


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_gt(data_path):
    GT = {}
    for category in os.listdir(data_path):
        category_dir = os.path.join(data_path, category)
        if not os.path.isdir(category_dir):
            continue
        if os.path.exists(os.path.join(category_dir, 'images')):
            image_path = os.path.join(category_dir, 'images')
            qa_path = os.path.join(category_dir, 'questions_answers_YN')
        else:
            image_path = qa_path = category_dir
        assert os.path.isdir(image_path), image_path
        assert os.path.isdir(qa_path), qa_path
        for file in os.listdir(qa_path):
            if not file.endswith('.txt'):
                continue
            for line in open(os.path.join(qa_path, file)):
                question, answer = line.strip().split('\t')
                GT[(category, file, question)] = answer
    return GT


# Custom dataset class
class CustomDataset(Dataset):

    def __init__(self, questions, image_folder, tokenizer, image_processor,
                 model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = build_conversation()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder,
                                        image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor,
                                      self.model_config)[0]

        input_ids = tokenizer_image_token(prompt,
                                          self.tokenizer,
                                          IMAGE_TOKEN_INDEX,
                                          return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions,
                       image_folder,
                       tokenizer,
                       image_processor,
                       model_config,
                       batch_size=1,
                       num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer,
                            image_processor, model_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path)

    questions = [
        json.loads(q)
        for q in open(os.path.expanduser(args.question_file), "r")
    ]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_dir = os.path.expanduser(args.answers_dir)

    data_loader = create_data_loader(questions, args.image_folder, tokenizer,
                                     image_processor, model.config)
    pred_results = defaultdict(list)
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions),
                                                total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        stop_str = build_conversation().sep2
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16,
                                       device='cuda',
                                       non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids'
            )
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:],
                                         skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        category = idx.split('/')[0]
        file = idx.split('/')[-1].split(".")[0] + ".txt"
        pred_results[category].append((file, cur_prompt, outputs))

    GT = get_gt(args.image_folder)
    for category, cate_tups in pred_results.items():
        fw = open(os.path.join(answers_dir, f'{category}.txt'), 'w')
        for file, prompt, answer in cate_tups:
            prompt = prompt.replace('\n', ' ')
            if (category, file, prompt) not in GT:
                prompt = prompt.replace(' Please answer yes or no.',
                                        '  Please answer yes or no.')
            gt_ans = GT[category, file, prompt]
            tup = file, prompt, gt_ans, answer
            fw.write('\t'.join(tup) + '\n')
        fw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file",
                        type=str,
                        default="tables/question.jsonl")
    parser.add_argument("--answers-dir",
                        type=str,
                        default="./evaluation/MME/results")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
