### Default configs for development test

def override_config(args, d: dict):
    vars(args).update(d)

class base_configs:
    model_name = "TransE"

    cuda = False

    uni_weight = False
    regularization = 0.1
    learning_rate = 0.0001
    batch_size = 1024
    hidden_dim = 1000
    gamma = 24.0

    negative_adversarial_sampling = False
    adversarial_temperature = 1.0
    
    cpu_num = 10

    record = True
    log_steps = 100

    init_checkpoint = None
    save_checkpoint_steps = 10000

    do_valid = True
    valid_steps = 50000
    
    double_entity_embedding = True
    double_relation_embedding = True


class testing_configs(base_configs):
    source_file = "../test-data/weapons/weapons.csv"
    train_test_data_dir = "../test-output/train_test_data_dir"
    main_path = "../test_output"
    name_space = "https://tzkg.cn/#"