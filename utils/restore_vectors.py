import logging
from sentence_transformers import SentenceTransformer
import weaviate
from config.settings import settings
WEAVIATE_URL = f"https://{settings.weaviate_host}"
CLASS_NAME = "Tables_Test"

logging.basicConfig(level=logging.INFO)

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def get_client():
    url = f"https://{settings.weaviate_host}"
                        
    client = weaviate.Client(url)
    logging.info("Connected to Weaviate")
    return client


def fetch_objects(client):
    result = (
        client.query
        .get(CLASS_NAME, ["content", "paper_id"])
        .with_additional(["id"])
        .with_limit(10000)
        .do()
    )

    return result["data"]["Get"][CLASS_NAME]


def backfill(client, objects):
    logging.info(f"Found {len(objects)} objects")

    for i, obj in enumerate(objects):
        uuid = obj["_additional"]["id"]
        text = obj.get("content")

        if not text:
            continue

        vector = model.encode(text).tolist()

        client.data_object.update(
            uuid=uuid,
            class_name=CLASS_NAME,
            data_object={
                "content": text,
                "paper_id": obj.get("paper_id")
            },
            vector=vector
        )

        if i % 50 == 0:
            logging.info(f"Processed {i}/{len(objects)}")

    logging.info("Backfill complete ✅")


def main():
    client = get_client()
    objects = fetch_objects(client)
    backfill(client, objects)


if __name__ == "__main__":
    main()