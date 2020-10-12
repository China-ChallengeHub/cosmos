import os
import json
import tqdm
import logging

from io import open
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 swag_id,
                 context,
                 question,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label=None):
        self.swag_id = swag_id
        self.context = context
        self.question = question
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"swag_id: {self.swag_id}",
            f"ending_0: {self.endings[0]}",
            f"ending_1: {self.endings[1]}",
            f"ending_2: {self.endings[2]}",
            f"ending_3: {self.endings[3]}"
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


# 定义CommonsenseQA的工具处理类
class CosmosProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "valid.json")), "valid")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        return json.load(open(input_file, "r"))

    def _create_examples(self, rand_data, type):
        """Creates examples for the training and dev sets."""

        examples = []
        for cosmos_json in tqdm(rand_data):
            swag_id = cosmos_json["id"]
            label = ord(cosmos_json["label"]) - ord("0") if type in ["train", "valid"] else 0

            examples.append(
                InputExample(
                    swag_id=swag_id,
                    context=cosmos_json["context"],
                    question=cosmos_json["question"],
                    ending_0=cosmos_json["answer0"],
                    ending_1=cosmos_json["answer1"],
                    ending_2=cosmos_json["answer2"],
                    ending_3=cosmos_json["answer3"],
                    label=label
                ))
        return examples

    def _create_features(self,
                         examples,
                         tokenizer,
                         max_seq_length,
                         sep_token_extra):  # for roberta

        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        pad_token = tokenizer.pad_token

        features = []
        for example_index, example in enumerate(tqdm(examples)):

            choices_features = []
            for ending_index, ending in enumerate(example.endings):

                # tokens = ["<s>"] + context_tokens_choice + ["<s>"] + ["<s>"] + ending_tokens + ["<s>"]
                text_a = example.context + " " + example.question
                text_b = ending

                tokens_a = tokenizer.tokenize(text_a)
                tokens_b = tokenizer.tokenize(text_b)
                if sep_token_extra:
                    tokens_a = tokens_a + [sep_token]

                special_tokens_count = 4 if sep_token_extra else 3
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)

                tokens_a = tokens_a + [sep_token]
                tokens_b = tokens_b + [sep_token]
                tokens   = [cls_token] + tokens_a + tokens_b

                input_ids   = tokenizer.convert_tokens_to_ids(tokens)
                segment_ids = [0] * (len(tokens_a) + 1) + [1] * len(tokens_b)
                input_mask  = [1] * len(input_ids)

                assert len(input_ids) == len(segment_ids) == len(input_mask)

                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                # input_ids   += padding
                input_mask  += padding
                segment_ids += padding

                tokens = tokens + [pad_token] * (max_seq_length - len(input_ids))
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                assert len(input_ids)   == max_seq_length
                assert len(input_mask)  == max_seq_length
                assert len(segment_ids) == max_seq_length

                choices_features.append((tokens, input_ids, input_mask, segment_ids))

            label = example.label

            # if example_index < 2:
            #     logger.info("*** Example ***")
            #     logger.info("csqa_id: {}".format(example.swag_id))
            #     for choice_idx, (tokens, input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
            #         logger.info("choice: {}".format(choice_idx))
            #         logger.info("tokens: {}".format(' '.join(map(str, tokens))))
            #         logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
            #         logger.info("attention_mask: {}".format(' '.join(map(str, attention_mask))))
            #         logger.info("token_type_ids: {}".format(' '.join(map(str, token_type_ids))))
            #         logger.info("label: {}".format(label))

            features.append(
                InputFeatures(
                    example_id=example.swag_id,
                    choices_features=choices_features,
                    label=label
                )
            )
        return features
