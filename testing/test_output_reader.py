from tzkg.data_processing import read_rules_to_df, read_triplets_prediction_to_df
import os

test_pytest_file_dir = os.path.dirname(__file__)

pred_mln_path = os.path.join(test_pytest_file_dir, "../test_output/record/0/pred_mln.txt")
rules_path = os.path.join(test_pytest_file_dir, "../test_output/record/0/rule.txt")

entities_dict_path = os.path.join(test_pytest_file_dir, "../test_output/train_test_data_dir/entities.dict")
relations_dict_path = os.path.join(test_pytest_file_dir, "../test_output/train_test_data_dir/relations.dict")

def test_read_triplets():
    df = read_triplets_prediction_to_df(pred_mln_path, entities_dict_path, relations_dict_path)
    print(df.head())

def test_read_rules():
    df = read_rules_to_df(rules_path, relations_dict_path)
    print(df.head())