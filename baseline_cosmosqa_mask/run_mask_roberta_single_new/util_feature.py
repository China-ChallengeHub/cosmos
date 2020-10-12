import os
import time
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import TensorDataset


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in tqdm(features)
    ]


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label):

        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids':   input_ids,
                'input_mask':  input_mask,
                'segment_ids': segment_ids,
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class CommonsenseFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features):
        self.example_id = example_id
        self.choices_features = [
            {
                'commonsense_mask': commonsense_mask
            }
            for commonsense_mask in choices_features
        ]


class DependencyFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features):
        self.example_id = example_id
        self.choices_features = [
            {
                'dependency_mask': dependency_mask
            }
            for dependency_mask in choices_features
        ]


class EntityFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features
                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'entity_mask': entity_mask
            }
            for entity_mask in choices_features
        ]


class SentimentFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features):
        self.example_id = example_id
        self.choices_features = [
            {
                'sentiment_mask': sentiment_mask
            }
            for sentiment_mask in choices_features
        ]


def convert(features, features_prior, mask_type):
    for feature, feature_commonsense in zip(features, features_prior):
        assert feature.example_id == feature_commonsense.example_id

    all_input_ids   = torch.tensor(select_field(features, 'input_ids'),   dtype=torch.long)
    all_input_mask  = torch.tensor(select_field(features, 'input_mask'),  dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    all_prior_mask  = torch.tensor(select_field(features_prior, f'{mask_type}_mask'), dtype=torch.int8)
    all_label_ids   = torch.tensor([f.label for f in features],           dtype=torch.int8)

    print("[TIME] --- time: {} ---, convert features finished".format(time.ctime(time.time())))
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_prior_mask, all_label_ids)
    return dataset


def read_features(args, train_prior_feature_path, dev_prior_feature_path, mask_type):
    casual_feature_dir = "../util_mask_prior/feature_roberta/casual_feature"

    print("[TIME] --- time: {} ---, read casual features".format(time.ctime(time.time())))
    cached_train_features_file = os.path.join(
        casual_feature_dir, "train_features_{}.pkl".format(args.max_seq_length))
    cached_dev_features_file   = os.path.join(
        casual_feature_dir,   "dev_features_{}.pkl".format(args.max_seq_length))

    with open(cached_train_features_file, "rb") as reader:
        train_features = pickle.load(reader)
    with open(cached_dev_features_file,   "rb") as reader:
        dev_features   = pickle.load(reader)

    print("[TIME] --- time: {} ---, read prior features".format(time.ctime(time.time())))
    with open(train_prior_feature_path, "rb") as reader:
        train_features_prior = pickle.load(reader)
    with open(dev_prior_feature_path, "rb") as reader:
        dev_features_prior   = pickle.load(reader)

    if args.debug:
        print("[DEBUG]: --- time: {} ---, convert dataset started".format(time.ctime(time.time())))
        train_features_prior = train_features_prior[:32]
        dev_features_prior   = dev_features_prior[:32]

        train_features = train_features[:32]
        dev_features   = dev_features[:32]

        train_dataset = convert(train_features, train_features_prior, mask_type)
        dev_dataset   = convert(  dev_features,   dev_features_prior, mask_type)

    else:
        # debug: _bp.pkl:
        #   未考虑到<s>与</s>之间的连接
        #   未考虑到一个句子中的用一个词出现多次的情况

        cached_train_dataset_file = "./train_dataset0_{}_{}.pkl".format(mask_type, args.max_seq_length)
        cached_dev_dataset_file   = "./dev_dataset0_{}_{}.pkl".format(mask_type, args.max_seq_length)

        # cached_train_dataset_file = "./train_dataset_{}_{}_bp.pkl".format(mask_type, args.max_seq_length)
        # cached_dev_dataset_file   = "./dev_dataset_{}_{}_bp.pkl".format(mask_type, args.max_seq_length)

        try:
            print("[TIME] --- time: {} ---, load dataset started".format(time.ctime(time.time())))
            train_dataset = torch.load(cached_train_dataset_file)
            dev_dataset   = torch.load(cached_dev_dataset_file)

        except:
            print("[TIME] --- time: {} ---, convert dataset started".format(time.ctime(time.time())))
            train_dataset = convert(train_features, train_features_prior, mask_type)
            dev_dataset   = convert(  dev_features,   dev_features_prior, mask_type)

            torch.save(train_dataset, cached_train_dataset_file)
            torch.save(  dev_dataset,   cached_dev_dataset_file)
    return train_dataset, dev_dataset


def read_commonsense_features(args):
    commonsense_feature_dir = "../util_mask_prior/feature_roberta/commonsense_feature"

    # 不考虑<s>与</s>之间的连接
    train_features_commonsense_file = os.path.join(
        commonsense_feature_dir, "train_features_commonsense0_{}.pkl".format(args.max_seq_length))
    dev_features_commonsense_file   = os.path.join(
        commonsense_feature_dir,   "dev_features_commonsense0_{}.pkl".format(args.max_seq_length))

    train_dataset, dev_dataset = read_features(
        args, train_features_commonsense_file, dev_features_commonsense_file, mask_type="commonsense")
    return train_dataset, dev_dataset


# dependency mask 考虑根节点以下的所有结点
def read_dependency_features(args):
    dependency_feature_dir = "../util_mask_prior/feature_roberta/dependency_feature"

    train_features_dependency_file = os.path.join(
        dependency_feature_dir, "train_features_dependency0_{}.pkl".format(args.max_seq_length))
    dev_features_dependency_file   = os.path.join(
        dependency_feature_dir, "dev_features_dependency0_{}.pkl".format(args.max_seq_length))

    train_dataset, dev_dataset = read_features(
        args, train_features_dependency_file, dev_features_dependency_file, mask_type="dependency")
    return train_dataset, dev_dataset


# spacy named entity detection
def read_entity_features(args):
    entity_feature_dir = "../util_mask_prior/feature_roberta/entity_feature"

    train_features_entity_file = os.path.join(
        entity_feature_dir, "train_features_entity0_{}.pkl".format(args.max_seq_length))
    dev_features_entity_file   = os.path.join(
        entity_feature_dir,   "dev_features_entity0_{}.pkl".format(args.max_seq_length))

    train_dataset, dev_dataset = read_features(
        args, train_features_entity_file, dev_features_entity_file, mask_type="entity")
    return train_dataset, dev_dataset


def read_sentiment_features(args):
    sentiment_feature_dir = "../util_mask_prior/feature_roberta/sentiment_feature"

    train_features_sentiment_file = os.path.join(
        sentiment_feature_dir, "train_features_sentiment0_{}.pkl".format(args.max_seq_length))
    dev_features_sentiment_file   = os.path.join(
        sentiment_feature_dir,   "dev_features_sentiment0_{}.pkl".format(args.max_seq_length))

    train_dataset, dev_dataset = read_features(
        args, train_features_sentiment_file, dev_features_sentiment_file, mask_type="sentiment")
    return train_dataset, dev_dataset
