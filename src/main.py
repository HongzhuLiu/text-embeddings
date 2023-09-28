import json
import time

# Use tensorflow 1 behavior to match the Universal Sentence Encoder
# examples (https://tfhub.dev/google/universal-sentence-encoder/2).
import tensorflow_hub as hub
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


##### INDEXING #####

def index_data():
    print("Creating the 'posts' index.")
    # 检查索引是否存在，然后删除
    if client.indices.exists(index=INDEX_NAME):
        # client.indices.delete(index=INDEX_NAME)
        # print(f"Index '{INDEX_NAME}' deleted exists.")
        return
    else:
        print(f"Index '{INDEX_NAME}' does not exist.")

    with open(INDEX_FILE) as index_file:
        source = index_file.read().strip()
        index_settings = json.loads(source)
        client.indices.create(index=INDEX_NAME, body=index_settings)

    docs = []
    count = 0

    with open(DATA_FILE) as data_file:
        for line in data_file:
            line = line.strip()

            doc = json.loads(line)
            if doc["type"] != "question":
                continue

            docs.append(doc)
            count += 1

            if count % BATCH_SIZE == 0:
                index_batch(docs)
                docs = []
                print("Indexed {} documents.".format(count))

        if docs:
            index_batch(docs)
            print("Indexed {} documents.".format(count))

    client.indices.refresh(index=INDEX_NAME)
    print("Done indexing.")


def index_batch(docs):
    titles = [doc["title"] for doc in docs]
    title_vectors = embed_text(titles)

    requests = []
    for i, doc in enumerate(docs):
        request = doc
        request["_op_type"] = "index"
        request["_index"] = INDEX_NAME
        request["title_vector"] = title_vectors[i]
        requests.append(request)
    bulk(client, requests)


##### SEARCHING #####

def run_query_loop():
    while True:
        try:
            handle_query()
        except KeyboardInterrupt:
            return


def handle_query():
    query = input("Enter query: ")

    embedding_start = time.time()
    query_vector = embed_text([query])[0]
    print(query_vector)
    embedding_time = time.time() - embedding_start

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'title_vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    search_start = time.time()
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": script_query,
            "_source": {"includes": ["title", "body"]}
        }
    )
    print(response)
    search_time = time.time() - search_start

    print()
    print("{} total hits.".format(response["hits"]["total"]["value"]))
    print("embedding time: {:.2f} ms".format(embedding_time * 1000))
    print("search time: {:.2f} ms".format(search_time * 1000))
    for hit in response["hits"]["hits"]:
        print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
        print(hit["_source"])
        print()


##### EMBEDDING #####

def embed_text(textList):
    embedding_values = embed(textList).numpy()
    # # 打印嵌入向量值
    # for i, text in enumerate(textList):
    #     print(f"Text: {text}")
    #     print(f"Embedding Vector: {embedding_values[i]}")
    #     print()

    return embedding_values.tolist()


##### MAIN SCRIPT #####

if __name__ == '__main__':
    INDEX_NAME = "posts"
    INDEX_FILE = "/Users/lhz/IdeaProjects/text-embeddings/data/posts/index.json"

    DATA_FILE = "/Users/lhz/IdeaProjects/text-embeddings/data/posts/posts.json"
    BATCH_SIZE = 1000

    SEARCH_SIZE = 5

    GPU_LIMIT = 0.5

    print("Downloading pre-trained embeddings from tensorflow hub...")
    # embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/4")
    embed = hub.load("/Users/lhz/IdeaProjects/text-embeddings/data/universal-sentence-encoder_4")
    # 创建一个示例文本
    sample_text = ["This is a sample text.", "Another sample sentence."]
    embed_text(sample_text)

    print("Done.")

    client = Elasticsearch(["http://localhost:9200"])

    index_data()
    run_query_loop()

    print("Done.")
