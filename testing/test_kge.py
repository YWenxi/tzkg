import pytest
from torch import nn
import torch
from tzkg.reasoners import KGE
from .test_dataset import test_setup_in_one_step_and_pytorch_dataset as get_dataset
from tqdm import tqdm


class _args:
    cuda = False
    negative_adversarial_sampling = False
    uni_weight = False
    regularization = 0.1


class Test_KGE:
    
    TRANSE_CONFIG = {
        "model_name": "TransE",
        # "nentity": 10000,
        # "nrelation": 20000,
        "hidden_dim": 1001,
        "gamma": 24.0,
        "double_entity_embedding": True,
        "double_relation_embedding": True,
    }
    learning_rate = 0.001
    warm_upsteps = 10

    def _test_kge_template(self, configs: dict):
        train_iterator, nentity, nrelation, _ = get_dataset(["/root/TZ-tech/knowledge-reasoning-demo/test-data/weapons/weapons.csv"])
        kge_model = KGE(nentity=nentity, nrelation=nrelation, **configs)
        
        for module in kge_model.modules():
            if not isinstance(module, nn.Sequential):
                print(module)
        
        current_learning_rate = self.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )

        init_step = 0
        training_logs = []

        kge_model.train()
        for step in tqdm(range(init_step, self.warm_upsteps)):
            log = kge_model.train_step(kge_model, optimizer, train_iterator, _args)
            training_logs.append(log)


    def test_transE(self):
        self._test_kge_template((self.TRANSE_CONFIG))
    
