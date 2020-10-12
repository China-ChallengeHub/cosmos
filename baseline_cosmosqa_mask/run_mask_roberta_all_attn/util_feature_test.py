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


def convert(features, features_commonsense, features_dependency, features_entity, features_sentiment):

    for feature, feature_commonsense, feature_dependency, feature_entity, feature_sentiment in zip(
            features, features_commonsense, features_dependency, features_entity, features_sentiment):
        assert feature.example_id == feature_commonsense.example_id == feature_dependency.example_id \
               == feature_entity.example_id == feature_sentiment.example_id

    print("[TIME] --- time: {} ---, convert dependency features".format(time.ctime(time.time())))
    # Convert to Tensors and build dataset
    all_input_ids   = torch.tensor(select_field(features, 'input_ids'),   dtype=torch.long)
    all_input_mask  = torch.tensor(select_field(features, 'input_mask'),  dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    all_label_ids   = torch.tensor([f.label for f in features], dtype=torch.long)

    print("[TIME] --- time: {} ---, convert commmonsense mask".format(time.ctime(time.time())))
    all_commonsense_mask = torch.tensor(select_field(features_commonsense, 'commonsense_mask'), dtype=torch.int8)

    print("[TIME] --- time: {} ---, convert dependency mask".format(time.ctime(time.time())))
    all_dependency_mask = torch.tensor(select_field(features_dependency, 'dependency_mask'), dtype=torch.int8)

    print("[TIME] --- time: {} ---, convert entity mask".format(time.ctime(time.time())))
    all_entity_mask = torch.tensor(select_field(features_entity, 'entity_mask'), dtype=torch.int8)

    print("[TIME] --- time: {} ---, convert sentiment mask".format(time.ctime(time.time())))
    all_sentiment_mask = torch.tensor(select_field(features_sentiment, 'sentiment_mask'), dtype=torch.int8)

    print("[TIME] --- time: {} ---, convert TensorDataset".format(time.ctime(time.time())))
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_commonsense_mask, all_dependency_mask,
                            all_entity_mask, all_sentiment_mask, all_label_ids)
    return dataset


def _read_features(args):
    casual_feature_dir      = "../util_mask_prior/feature_roberta/casual_feature"
    commonsense_feature_dir = "../util_mask_prior/feature_roberta/commonsense_feature"
    dependency_feature_dir  = "../util_mask_prior/feature_roberta/dependency_feature"
    entity_feature_dir      = "../util_mask_prior/feature_roberta/entity_feature"
    sentiment_feature_dir   = "../util_mask_prior/feature_roberta/sentiment_feature"

    cached_test_features_commonsense_file = os.path.join(
        commonsense_feature_dir, "test_features_commonsense0_{}.pkl".format(args.max_seq_length)
    )
    cached_test_features_dependency_file = os.path.join(
        dependency_feature_dir, "test_features_dependency0_{}.pkl".format(args.max_seq_length)
    )
    cached_test_features_entity_file = os.path.join(
        entity_feature_dir, "test_features_entity_{}.pkl".format(args.max_seq_length)
    )
    cached_test_features_sentiment_file = os.path.join(
        sentiment_feature_dir, "test_features_sentiment0_{}.pkl".format(args.max_seq_length)
    )
    cached_test_features_file = os.path.join(
        casual_feature_dir, "test_features_{}.pkl".format(args.max_seq_length)
    )

    print("[TIME] --- time: {} ---, read commonsense features".format(time.ctime(time.time())))
    with open(cached_test_features_commonsense_file, "rb") as reader:
        test_features_commonsense = pickle.load(reader)

    print("[TIME] --- time: {} ---, read dependency features".format(time.ctime(time.time())))
    with open(cached_test_features_dependency_file, "rb") as reader:
        test_features_dependency = pickle.load(reader)

    print("[TIME] --- time: {} ---, read entity features".format(time.ctime(time.time())))
    with open(cached_test_features_entity_file, "rb") as reader:
        test_features_entity = pickle.load(reader)

    print("[TIME] --- time: {} ---, read sentiment features".format(time.ctime(time.time())))
    with open(cached_test_features_sentiment_file, "rb") as reader:
        test_features_sentiment = pickle.load(reader)

    print("[TIME] --- time: {} ---, read features".format(time.ctime(time.time())))
    with open(cached_test_features_file, "rb") as reader:
        test_features = pickle.load(reader)

    if args.debug:
        _test_features = (
            test_features_commonsense[:31], test_features_dependency[:31], test_features_entity[:31],
            test_features_sentiment[:31], test_features[:31]
        )
    else:
        _test_features = (
            test_features_commonsense, test_features_dependency, test_features_entity,
            test_features_sentiment, test_features
        )

    return _test_features


def read_features_test(args):

    if args.debug:
        _test_features = _read_features(args)

        test_features_commonsense, test_features_dependency, test_features_entity, test_features_sentiment, test_features = _test_features

        test_dataset = convert(test_features, test_features_commonsense, test_features_dependency,
                               test_features_entity, test_features_sentiment)

    else:
        cached_test_dataset_file = "./test_dataset0_{}.pkl".format(args.max_seq_length)

        try:
            print("[TIME] --- time: {} ---, load dataset".format(time.ctime(time.time())))
            test_dataset = torch.load(cached_test_dataset_file)

        except:
            _test_features = _read_features(args)

            test_features_commonsense, test_features_dependency, test_features_entity, test_features_sentiment, test_features = _test_features

            print("[TIME] --- time: {} ---, convert dataset".format(time.ctime(time.time())))
            test_dataset = convert(test_features, test_features_commonsense, test_features_dependency,
                                   test_features_entity, test_features_sentiment)

            torch.save(test_dataset, cached_test_dataset_file)

    return test_dataset
