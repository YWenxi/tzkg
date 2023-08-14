### Default configs for development test
from argparse import Namespace

def override_config(args, d: dict):
    for k, v in d.items():
        setattr(args, k, v)

class base_configs(Namespace):
    model_name = "TransE"

    cuda = False

    uni_weight = False
    regularization = 0.1
    learning_rate = 0.0001
    batch_size = 1024
    hidden_dim = 1000
    gamma = 24.0

    negative_sample_size = 256

    negative_adversarial_sampling = False
    adversarial_temperature = 1.0
    
    cpu_num = 10

    record = True
    
    do_train = True
    warm_up_steps = None
    max_steps = 10000
    log_steps = 100
    init_checkpoint = None
    save_checkpoint_steps = 2500

    do_valid = True
    valid_steps = 5000
    test_batch_size = 16
    test_log_steps = 1000
    topk = 100
    
    double_entity_embedding = True
    double_relation_embedding = True

    countries = False


class testing_configs(base_configs):
    source_file = "test-data/weapons/weapons.csv"
    train_test_data_dir = "test_output/train_test_data_dir"
    main_path = "test_output/record"
    name_space = "https://tzkg.cn/#"
    save_path = None

    max_steps = 10
    log_steps = 2
    save_checkpoint_steps = 2
    valid_steps = 5
    test_log_steps = 1000
    topk = 100