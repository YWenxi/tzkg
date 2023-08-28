from tzkg.inference import train
from tzkg.inference.config import testing_configs, override_config
from tzkg.datasets.utils import setup_in_one_step, setup_mainspace, setup_workspace
from tzkg.reasoners import mln

def _test_train_minitary():

    metadata = setup_in_one_step(
        source_file=testing_configs.source_file,
        train_test_data_dir=testing_configs.train_test_data_dir,
        main_path=testing_configs.main_path,
        iteration_id=0,
        name_space=testing_configs.name_space
    )

    override_config(testing_configs, metadata)

    train(testing_configs)

    mln(testing_configs.main_path, testing_configs.workspace_path, preprocess=False)


def _test_train_minitary_wiki():

    testing_configs.source_file = "test-data/test_new/records_new.rdf"
    testing_configs.train_test_data_dir = "test-data/test_new/"
    testing_configs.main_path = "test_output/test_new_record"

    setup_mainspace(testing_configs.main_path, testing_configs.train_test_data_dir)
    mln(testing_configs.main_path)
    testing_configs.workspace_path = setup_workspace(0, testing_configs.main_path)
    train(testing_configs)
    mln(testing_configs.main_path, testing_configs.workspace_path, preprocess=False, mln_iters=10)

def test_train_wiki_minitary():
    testing_configs.source_file = "test-data/wikimilitary/data.csv"
    testing_configs.train_test_data_dir = "test-data/wikimilitary/"
    testing_configs.main_path = "test_output/test_wikimilitary"

    metadata = setup_in_one_step(
        source_file=testing_configs.source_file,
        train_test_data_dir=testing_configs.train_test_data_dir,
        main_path=testing_configs.main_path,
        iteration_id=0,
        name_space=testing_configs.name_space
    )

    override_config(testing_configs, metadata)

    train(testing_configs)

    mln(testing_configs.main_path, testing_configs.workspace_path, preprocess=False)

if __name__=="__main__":
    # test_train_minitary_new()
    pass