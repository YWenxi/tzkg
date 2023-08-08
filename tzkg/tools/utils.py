import os
import json

import pandas as pd
from collections import OrderedDict
from rdflib import Graph, Namespace, URIRef, Literal

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def _readfile(data_dir, data_type="train", file_type="csv"):
    _path = os.path.join(data_dir, data_type)
    data = pd.read_csv(f"{_path}.{file_type}")

    return data.to_dict(orient="records")

def _handle_relation_data(relation_data):
    rels = OrderedDict()
    relation_data = sorted(relation_data, key=lambda i: int(i['index']))
    for d in relation_data:
        rels[d['relation']] = {
            'index': int(d['index']),
            'head_type': d['head_type'],
            'tail_type': d['tail_type'],
        }
    return rels

def _serialize(data, tokenizer):
    # tokenizer = BertTokenizer.from_pretrained(tokenizer)
    if isinstance(data, list):
        for d in data:
            sent = d['sentence'].strip()
            sent = sent.replace(d['head'], d['head_type'], 1).replace(d['tail'], d['tail_type'], 1)
            sent += '[SEP]' + d['head'] + '[SEP]' + d['tail']
            d['token2idx'] = tokenizer.encode(sent, add_special_tokens=True)
            d['seq_len'] = len(d['token2idx'])
    elif isinstance(data, dict):
        sent = data['sentence'].strip()
        sent = sent.replace(data['head'], data['head_type'], 1).replace(data['tail'], data['tail_type'], 1)
        sent += '[SEP]' + data['head'] + '[SEP]' + data['tail']
        data['token2idx'] = tokenizer.encode(sent, add_special_tokens=True)
        data['seq_len'] = len(data['token2idx'])
    return data

def textsplitter(text, split=True, chunk_size=512):
    if split:
        loader = UnstructuredFileLoader(text)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = 0
        )
        split_documents = text_splitter.split_documents(document)
        if len(split_documents) <= 1:
            return split_documents[0].page_content.split("\n\n")
        else:
            s1 = [i.page_content for i in split_documents]
            return " ".join(s1).split("\n\n")
    else:
        with open(text, 'r', encoding='utf-8') as file:
            data = file.read()
        
        return data.split("\n")
    
def file_merge(file_path, out_name="merged.txt", overwrite=False):
    out_path = os.path.join(file_path, out_name)

    if os.path.exists(out_path):
        if not overwrite:
            raise FileExistsError(f"The output file '{out_name}' already exists. Set 'overwrite=True' to overwrite the file.")
        else:
            os.remove(out_path)
            print(f"file {out_name} remove done!")

    txt_files = [f for f in os.listdir(file_path) if f.endswith('.txt')]
    with open(out_path, 'w') as outfile:
        for file in txt_files:
            with open(os.path.join(file_path, file)) as infile:
                # 将每个文件的内容写入输出文件
                for line in infile:
                    outfile.write(line)
                    

class Transfer:
    def __init__(self, data_dir, name_space=None) -> None:
        self.g = Graph()
        data_type = data_dir.split(".")[-1]
        if data_type == "csv":
            self.data = pd.read_csv(data_dir, names=['head', 'relation', 'tail'])
        elif data_type == "txt":
            self.data = pd.read_csv(data_dir, sep="\t")
        elif data_type in ["owl", "rdf"]:
            self.g.parse(data_dir, format="xml")
            self.data = []
            for s, p, o in self.g:
                self.data.append({"Subject": s, "Predicate": p, "Object": o})
        elif data_type == "json":
            with open(data_dir, 'r', encoding="utf-8-sig") as file:
                raw_data = file.read()
                cleaned_data = raw_data.replace("\n", "").replace("\t", "").replace("\r", "").replace(r"\'", "").replace(r'\"', "")
                self.data = json.loads(cleaned_data)
        else:
            raise ImportError("suggested file type -> csv/txt/rdf/owl/json ...")

        if name_space:
            self.ns = Namespace(name_space)

    def _to_onto(self, out_name, out_format="rdf"):
        # 将节点转换为RDF三元组
        for item in self.data:
            node = item['n']
            node_id = node['identity']
            node_labels = node['labels']
            node_properties = node['properties']

            # 创建节点的主题
            node_uri = self.ns['node_' + str(node_id)]

            # 添加节点标签（第一个标签作为谓词）
            for label in node_labels:
                label_uri = self.ns[label]
                self.g.add((node_uri, self.ns['hasLabel'], label_uri))

            # 添加节点的属性（键值对作为对象）
            for key, value in node_properties.items():
                property_uri = self.ns[key]
                self.g.add((node_uri, property_uri, Literal(value)))

        # 将RDF图序列化为Turtle格式并保存到文件
        self.g.serialize(destination=f'{out_name}.{out_format}', format="xml")

    def _to_triples(self, out_name, out_format="csv"):
        df = pd.DataFrame(self.data)
        if out_format == "csv":
            df.to_csv(out_name, index=False)
        elif out_format == "txt":
            txt_data = df.to_string(header=False, index=False)
            with open(out_name, "w") as txtfile:
                txtfile.write(txt_data)
        else:
            raise ValueError("out format expected to csv & txt.")

    def _to_trainds(self, out_name="test01", save=False):
        entities = pd.concat([self.data['head'], self.data['tail']]).unique()
        self.entity2id = {entity: idx for idx, entity in enumerate(entities)}

        relations = self.data['relation'].unique()
        self.relation2id = {relation: idx for idx, relation in enumerate(relations)}

        if save:
            save_to_file(self.entity2id, f"{out_name}.txt")
            save_to_file(self.relation2id, f"{out_name}.txt")


def save_to_file(data_dict, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for key, value in data_dict.items():
            file.write(f"{value}\t{key}\n")


if __name__ == "__main__":
    data_dir = "/root/TZ-tech/samples/military/records.json"
    name_space = "http://tzzn.kg.cn/#"
    trf = Transfer(data_dir, name_space)
    trf._process()
