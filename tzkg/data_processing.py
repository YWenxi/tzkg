import json

import pandas as pd
from rdflib import Graph, Namespace, Literal, URIRef
from typing import Union

__all__ = ["Transfer"]

def save_to_file(data_dict, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for key, value in data_dict.items():
            file.write(f"{value}\t{key}\n")


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
            save_to_file(self.entity2id, f"entity2id.{out_type}")
            save_to_file(self.relation2id, f"relation2id.{out_type}")



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

