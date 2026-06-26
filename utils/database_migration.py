import requests
import time
import structlog
VULTR_URL = "vultr-url"
AZURE_URL = "azure-url"

BATCH_SIZE = 100

logger = structlog.get_logger(__name__)


def get_schema():
    r = requests.get(f"{VULTR_URL}/v1/schema", timeout=30)
    r.raise_for_status()
    return r.json()


def create_schema(schema):
    for class_obj in schema.get("classes", []):
        logger.info(f"Creating class: {class_obj['class']}")
        r = requests.post(f"{AZURE_URL}/v1/schema", json=class_obj, timeout=30)
        if r.status_code not in [200, 201]:
            logger.exception(r.text)


def get_objects(class_name):
    objects = []
    offset = 0

    while True:
        url = f"{VULTR_URL}/v1/objects?class={class_name}&limit=1000&offset={offset}"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()

        if "objects" not in data or len(data["objects"]) == 0:
            break

        objects.extend(data["objects"])
        offset += 1000

        print(f"Fetched {len(objects)} objects for {class_name}")

    return objects

def send_batch(objects):
    batch = {"objects": []}

    for obj in objects:
        batch["objects"].append({
            "class": obj.get("class"),
            "id": obj.get("id"),
            "properties": obj.get("properties", {}),
            "vector": obj.get("vector", None)
        })

        if len(batch["objects"]) >= BATCH_SIZE:
            push_batch(batch)
            batch = {"objects": []}

    if batch["objects"]:
        push_batch(batch)


def push_batch(batch):
    r = requests.post(f"{AZURE_URL}/v1/batch/objects", json=batch)
    if r.status_code not in [200, 201]:
        print("Batch error:", r.text)
    else:
        print(f"Inserted batch of {len(batch['objects'])}")


def migrate():
    print("Fetching schema from Vultr...")
    schema = get_schema()

    print("Creating schema on Azure...")
    create_schema(schema)

    classes = [c["class"] for c in schema.get("classes", [])]

    for cls in classes:
        print(f"\nMigrating class: {cls}")

        objects = get_objects(cls)
        print(f"Total objects: {len(objects)}")

        send_batch(objects)

        print(f"Finished class: {cls}")

        time.sleep(2)

    print("\nMigration completed!")


if __name__ == "__main__":
    migrate()