import chromadb
from scrt import CHROMADB_HOST, CHROMADB_PORT

client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
print(client.list_collections())
