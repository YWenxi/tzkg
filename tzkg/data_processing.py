import json
import os
import pandas as pd
from rdflib import Graph, Namespace, Literal, URIRef
from typing import Union
from sklearn.model_selection import train_test_split
import warnings
from shutil import copy

__all__ = ["Transfer"]

def _save_to_file(data_dict, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for key, value in data_dict.items():
            file.write(f"{value}\t{key}\n")

def _get_entity_transfer_func(dictionary: dict):
    """Get the function to convert entity strings to id accoring to dictionary, or reversely.
        dictionary (dict): the reference for conversion. Used as `id = dictionary[entity_or_relation_string]`

    Return:
        ent_rel_to_id: the conversion function, which could be used in pd.DataFrame().apply()
    """
    def ent_rel_to_id(key: str|int):
        try:

            # add type conversions according input types
            if isinstance(key, str):
                return int(dictionary[key])
            if isinstance(key, int):
                return str(dictionary[key])
        except KeyError:
            warnings.warn(f"KeyError Catched: Some keys are not found, will return `pd.NA`."
                          "And this piece of data would be dropped later.", UserWarning)
            return pd.NA
    return ent_rel_to_id

def read_dict(dict_file: str) -> dict:
    """Read the `.dict` files: `entities.dict` and `relations.dict`

    Args:
        dict_file (str): file path.

    Returns:
        dict: an index-to-entity/relation dictionary
    """
    return pd.read_csv(dict_file, sep="\t", index_col=0, header=None).to_dict()[1]

def read_triplets_prediction_to_df(pred_output: str, entity_dict_file: str, relations_dict_file: str) -> pd.DataFrame:
    """Read the predicted triplets output after training

    Args:
        pred_output (str): prediction output file. Usually named as `pred_mln.txt`
        entity_dict_file (dict): entity dictionary file; usually `entity.dict`.
        relations_dict_file (dict): relation dictionary file; usually `relation.dict`.

    Returns:
        pd.DataFrame: Converted output files. With four column labels: `["subject", "predicate", "object", "score"]`
    """
    temp_df = pd.read_csv(pred_output, names=['subject', 'predicate', 'object', 'score'], header=None, sep='\t')
    ent_converter = _get_entity_transfer_func(read_dict(entity_dict_file))
    rel_converter = _get_entity_transfer_func(read_dict(relations_dict_file))
    converters = {
        "subject": ent_converter,
        "predicate": rel_converter,
        "object": ent_converter,
        "score": lambda x: x
    }
    return temp_df.transform(converters)

def read_rules_to_df(rule_output: str, relations_dict_file: str|None = None) -> pd.DataFrame:


    outs = []
    with open(rule_output, "r") as f:
        for line in f.readlines():
            tokens = line.split("\t")
            # for rule type other than `composition`, 
            # it has only one relation in r_premise, 
            # so we insert an empty position at index 3
            if tokens[0] != "composition":
                tokens.insert(3, None)
                tokens[1] = int(tokens[1])
                tokens[2] = int(tokens[2])
                tokens[-1] = float(tokens[-1])
            else:
                tokens[3] = int(tokens[3])
            outs.append(tokens)

    outs = pd.DataFrame(outs, columns=["type", "r_hypothesis", "r_premise_0", "r_premise_1", "weight"])
    
    if relations_dict_file is not None:
        rel_converter = _get_entity_transfer_func(read_dict(relations_dict_file))
        converters = {
            "type": lambda x: x,
            "r_hypothesis": rel_converter,
            "r_premise_0": rel_converter,
            "r_premise_1": rel_converter,
            "weight": lambda x: x
        }

        outs = outs.transform(converters)

    return outs


class Transfer:
    def __init__(self, data_dir: str, name_space: Union[None, str]=None) -> None:
        """Initialize the transfer tool.
            data_dir (str): The data file to upload.
            name_space: Common prefix for urlref; default `None`.
        """
        self.g = Graph()
        self.data_type = data_dir.split(".")[-1]
        if self.data_type == "csv":
            self.data = pd.read_csv(data_dir, names=['subject', 'predicate', 'object'])
        elif self.data_type == "txt":
            self.data = pd.read_csv(data_dir, sep="\t", names=['subject', 'predicate', 'object'])
        elif self.data_type in ["owl", "rdf"]:
            self.g.parse(data_dir, format="xml")
            self.data = []
            for s, p, o in self.g:
                self.data.append({"subject": s, "predicate": p, "object": o})
        elif self.data_type == "json":
            # with open(data_dir, 'r', encoding="utf-8-sig") as file:
            #     raw_data = file.read()
            #     cleaned_data = raw_data.replace("\n", "").replace("\t", "").replace("\r", "").replace(r"\'", "").replace(r'\"', "")
            #     self.data = json.loads(cleaned_data)
            with open(data_dir, 'r', encoding="utf-8-sig") as file:
                self.data = json.load(file)
        else:
            raise ImportError("suggested file type -> csv/txt/rdf/owl/json ...")

        if name_space:
            self.ns = Namespace(name_space)

    def _to_rdf(self, sub, output_name="test"):
        # 将节点转换为RDF三元组
        for item in self.data:
            start_node = item[sub]['start']
            end_node = item[sub]['end']
            relationship_segments = item[sub]['segments']

            # Create node URIs
            start_node_uri = self.ns['node_' + str(start_node['identity'])]
            end_node_uri = self.ns['node_' + str(end_node['identity'])]

            # Add start node labels and properties
            for label in start_node['labels']:
                label_uri = self.ns[label]
                self.g.add((start_node_uri, self.ns['hasLabel'], label_uri))
            
            for key, value in start_node['properties'].items():
                property_uri = self.ns[key]
                self.g.add((start_node_uri, property_uri, Literal(value)))

            # Add end node labels and properties
            for label in end_node['labels']:
                label_uri = self.ns[label]
                self.g.add((end_node_uri, self.ns['hasLabel'], label_uri))
            
            for key, value in end_node['properties'].items():
                property_uri = self.ns[key]
                self.g.add((end_node_uri, property_uri, Literal(value)))

            # Process relationship segments
            for segment in relationship_segments:
                relationship = segment['relationship']
                relationship_type = relationship['type']

                self.g.add((start_node_uri, self.ns[relationship_type], end_node_uri))

        # 将RDF图序列化为Turtle格式并保存到文件
        self.g.serialize(destination=f'{output_name}.rdf', format="xml")
        

    def csv_to_onto(self, out_name:str, out_format:str="rdf") -> None:
        """Convert `.csv` files to `xml`-like files with extention in {`.rdf`, `.csv`}.
        - out_name (str): filename for the output file without extension.
        - out_format (str): file extension for the output file.
        """
        if self.data_type == "txt":
            for line in self.data:
                subject, predicate, obj = line.strip().split("\t")
                subject = URIRef(subject)
                predicate = URIRef(predicate)
                self.g.add((subject, predicate, Literal(obj)))
        elif self.data_type == "csv":
            for _, row in self.data.iterrows():
                subject = URIRef(row["subject"])
                predicate = URIRef(row["predicate"])
                obj = Literal(row["object"])
                self.g.add((subject, predicate, obj))

        self.g.serialize(destination=f'{out_name}.{out_format}', format="xml")


    def _to_triples(self, out_name, out_format="csv", sep: str | None = None):
        """Output `self.data` to `{out_name}.{csv/txt}`.
        - out_name (str): filename for the output file without extension.
        - out_format (str): file extension for the output file, must be either `txt` or `csv`, otherwise raise `ValueError`.
        - sep (str): seperation character.
            - When save to ".txt", we set `sep="\\t"` as default.
            - When save to ".csv", we set `sep=","` as default.
        """
        if out_format not in ["txt", "csv"]:
            raise ValueError("out format expected to csv & txt.")
        
        df = pd.DataFrame(self.data)

        default_sep = {"csv": ",", "txt": "\t"}
        
        if sep is None:
            sep = default_sep[out_format]
        df.to_csv(f"{out_name}.{out_format}", sep=sep, header=None, index=False)


    def _to_trainds(self, out_name="test01", save=False, out_type="txt"):
        if self.data_type in ["csv", "txt"]:
            raw_data = self.data
        elif self.data_type in ["owl", "rdf"]:
            raw_data = pd.DataFrame(self.data)

        entities = pd.concat([raw_data['subject'], raw_data['object']]).unique()
        self.entity2id = {entity: idx for idx, entity in enumerate(entities)}

        relations = raw_data['predicate'].unique()
        self.relation2id = {relation: idx for idx, relation in enumerate(relations)}

        if save:
            # save_to_file(self.entity2id, f"{out_name}_entity2id.{out_type}")
            # save_to_file(self.relation2id, f"{out_name}_relation2id.{out_type}")
            _save_to_file(self.entity2id, f"{out_name}_entity2id.{out_type}")
            _save_to_file(self.relation2id, f"{out_name}_relation2id.{out_type}")

    def save_to_trainable_sets(
            self, 
            out_dir: str, 
            convert_relations = True, 
            convert_entities = True,
            data_split: list[float] = [0.8, 0.1, 0.1],
            random_state = 42
            ):
        """
        Save to the data directory in the format that could be used by mln.py and later on.
            out_dir (str): the directory name to save the files. If it is not exists, a new directory shall be made.
            convert_relations (bool): whether to convert relations into id in the triplets files to save; default `False`
            convert_entities (bool): whether to convert entities into id (both in subject and object) in the triplets files to save; default `True`.
            data_split (list[float]): plan for dataset split, should be a list of three or two floats, defining the portion of each split
                - example: [train_set, valid_set, test_set]
            random_state (int): random_state for dataset splits.
        """
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        if self.data_type in ["csv", "txt"]:
            raw_data = self.data
        elif self.data_type in ["owl", "rdf"]:
            raw_data = pd.DataFrame(self.data)
        
        # extract_dictionary
        if "entity2id" not in dir(self) or "relation2id" not in dir(self):
            self._to_trainds()
        
        # save dictionary files for future use
        _save_to_file(self.entity2id, os.path.join(out_dir, "entities.dict"))
        _save_to_file(self.relation2id, os.path.join(out_dir, "relations.dict"))

        # convert entities to id
        identity_func = lambda x: x
        converters = {
            "subject": identity_func,
            "predicate": identity_func,
            "object": identity_func
        }
        if convert_entities:
            converter = _get_entity_transfer_func(self.entity2id)
            converters.update({
                "subject": converter,
                "object": converter
            })
        
        # convert relations to id
        if convert_relations:
            converters.update({
                "predicate": _get_entity_transfer_func(self.relation2id)
            })

        transformed_data = raw_data.copy()
        if convert_relations or convert_entities:
            transformed_data = transformed_data.transform(converters, axis=0)

        transformed_data.dropna(inplace=True)

        # split
        train_df, val_test_df = train_test_split(transformed_data, train_size=data_split[0], random_state=random_state)
        val_df, test_df = train_test_split(val_test_df, train_size=data_split[1]/(1-data_split[0]), random_state=random_state)

        # save df
        assert isinstance(train_df, pd.DataFrame)
        sep = "\t"
        train_df.to_csv(os.path.join(out_dir, "train.txt"), sep=sep, header=None, index=False)
        test_df.to_csv(os.path.join(out_dir, "test.txt"), sep=sep, header=None, index=False)
        val_df.to_csv(os.path.join(out_dir, "valid.txt"), sep=sep, header=None, index=False)
        copy(os.path.join(out_dir, "train.txt"), os.path.join(out_dir, "train_augmented.txt"))


if __name__ == "__main__":
    # data_dir = "/root/TZ-tech/knowledge-reasoning-demo/test-data/records_new.rdf"
#     name_space = "http://tzzn.kg.cn/#"
    # trf = Transfer(data_dir, name_space="http://tzzn.kg.cn/#")
    # trf._to_rdf(sub='p')
    # trf.save_to_trainable_sets(out_dir="/root/TZ-tech/knowledge-reasoning-demo/test-data")
#     trf.csv_to_onto(out_name="weapons_test", out_format="rdf")
#     trf._to_trainds(out_name="weapons_test", save=True)

    data_dir = "/root/knowledge-reasoning-demo/test-data/minitary/cdmo.owl"
    # name_space = None
    trf = Transfer(data_dir, name_space=None)
    trf._to_triples(out_name="cdmo", out_format="txt")
    trf._to_trainds(out_name="cdmo", save=True, out_type="dict")

#     ##TODO: 需适配json各种表示格式?
#     # data_dir = "/Users/hechengda/Documents/codes/TZ-KG/data/ZJHistory.json"
#     # name_space = "http://tzzn.kg.cn/#"
#     # trf = Transfer(data_dir, name_space=name_space)
#     # trf.json_to_onto(out_name="history", out_format="rdf")
#     # trf._to_triples(out_name="weapons_test", out_format="txt")
#     # trf._to_trainds(out_name="weapons_test", save=True, out_type="dict")

