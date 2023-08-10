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
    def ent_rel_to_id(ent_rel: str):
        try:
            return int(dictionary[ent_rel])
        except KeyError:
            warnings.warn(f"KeyError Catched: Some keys are not found, will return `pd.NA`."
                          "And this piece of data would be dropped later.", UserWarning)
            return pd.NA
    return ent_rel_to_id


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
            self.data = pd.read_csv(data_dir, sep="\t")
        elif self.data_type in ["owl", "rdf"]:
            self.g.parse(data_dir, format="xml")
            self.data = []
            for s, p, o in self.g:
                self.data.append({"subject": s, "predicate": p, "object": o})
        elif self.data_type == "json":
            with open(data_dir, 'r', encoding="utf-8-sig") as file:
                raw_data = file.read()
                cleaned_data = raw_data.replace("\n", "").replace("\t", "").replace("\r", "").replace(r"\'", "").replace(r'\"', "")
                self.data = json.loads(cleaned_data)
        else:
            raise ImportError("suggested file type -> csv/txt/rdf/owl/json ...")

        if name_space:
            self.ns = Namespace(name_space)

    def json_transfer(self, out_name, out_format="rdf"):
        ##TODO: 将Json转换为RDF/三元组/训练数据
        pass
        

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
        if self.data_type == "csv":
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

        if self.data_type == "csv":
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


# if __name__ == "__main__":
#     data_dir = "/root/knowledge-reasoning-demo/test-data/minitary/weapons.csv"
#     name_space = "http://tzzn.kg.cn/#"
#     trf = Transfer(data_dir, name_space)
#     trf.csv_to_onto(out_name="weapons_test", out_format="rdf")
#     trf._to_trainds(out_name="weapons_test", save=True)

#     data_dir = "/root/knowledge-reasoning-demo/test-data/minitary/cdmo.owl"
#     # name_space = None
#     trf = Transfer(data_dir, name_space=None)
#     trf._to_triples(out_name="cdmo", out_format="txt")
#     trf._to_trainds(out_name="cdmo", save=True, out_type="dict")

#     ##TODO: 需适配json各种表示格式?
#     # data_dir = "/Users/hechengda/Documents/codes/TZ-KG/data/ZJHistory.json"
#     # name_space = "http://tzzn.kg.cn/#"
#     # trf = Transfer(data_dir, name_space=name_space)
#     # trf.json_to_onto(out_name="history", out_format="rdf")
#     # trf._to_triples(out_name="weapons_test", out_format="txt")
#     # trf._to_trainds(out_name="weapons_test", save=True, out_type="dict")

