from neo4j import GraphDatabase


class NeoRequest:
    def __init__(self, uri, username, password=None):
        # 初始化neo4j的驱动
        self._uri = uri
        self._username = username
        self._password = password
        self._driver = GraphDatabase.driver(uri, auth=(username, password))

    def _close(self):
        self._driver.close()
        
    def add_entity_relation(self, tx, head, head_type, relation, tail, tail_type):
        tx.run(
            f"MERGE (a:{head_type} {{name: $head}}) "
            f"MERGE (b:{tail_type} {{name: $tail}}) "
            f"MERGE (a)-[:{relation} {{name: $relation}}]->(b)",
            head=head, relation=relation, tail=tail)
            
    def _save(self, head, head_type, relation, tail, tail_type):
        with self._driver.session() as session:
            session.write_transaction(self.add_entity_relation, head, head_type, relation, tail, tail_type)

    def _request(self, obj, head=None, tail=None, relation=None, cypher=None):
        '''
        obj: Cypher查询语句, 例如   "MATCH (H)-[R]->(T) "
                                  "WHERE T.name = '日本' AND type(R) = '国籍'"
                                  "RETURN H, R, T"
        '''
        session = self._driver.session()
        result = session.run(obj)
        records = (result)
        return records


if __name__ == "__main__":
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "Abcd1234!"

    req = NeoRequest(uri, username, password)
    results = req._request(obj="relation", head="乡村爱情", tail="赵本山", relation=None)
    print("request results: ", results)

    req._close()
