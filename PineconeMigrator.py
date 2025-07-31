import hashlib
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import islice
import pandas as pd
from pinecone import Pinecone
import multiprocessing
import threading
import unicodedata
import re

# --- Load Config.json and set globals ---
CONFIG_PATH = "config.json"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError("Missing config.json file")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

API_KEY = config["api_key"]
SOURCE_INDEX_NAME = config["source_index"]
TARGET_INDEX_NAME = config["target_index"]

CLOUD = config.get("target_cloud")
REGION = config.get("target_region")
SOURCE_NAMESPACE = config.get("source_namespace", "")
TARGET_NAMESPACE = config.get("target_namespace", "")
BATCH_SIZE = config.get("batch_size", 100)
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024  # 2MB limit
DELTA_MODE = config.get("delta_mode", False)
RESET_MIGRATION_FLAGS = config.get("reset_migration_flags", False)
VERIFY_ONLY = config.get("verify_only", False)
RUN_VALIDATION_AFTER_BATCH = config.get("run_validation_after_every_batch")
WRITE_TO_PARQUET = config.get("write_to_parquet")
USE_FILTER = config.get("use_filter")
FILTER_TO_USE = config.get("filter_to_use")
CREATE_TARGET = config.get("create_target")
PARQUET_WRITE_BATCH_SIZE = 10000  # or whatever size you want for Parquet chunks
NAMESPACE_FROM_METADATA_FIELD = config.get("namespace_from_metadata_field")
METADATA_FIELD_TO_NAMESPACE_FROM = config.get("metadata_field_to_namespace_from")
ITERATE_OVER_NAMESPACES = config.get("iterate_over_namespaces")
USE_SOURCE_AS_TARGET_NAMESPACE = config.get("use_source_as_target_namespace")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Pinecone Client ---
pc = Pinecone(api_key=API_KEY)
source_index = pc.Index(SOURCE_INDEX_NAME)

# for using multiple processors
MAX_WORKERS = min(32, (multiprocessing.cpu_count() or 1) * 2)

# handle keyboard input to early quit
stop_requested = False
stop_event = threading.Event()

#If iterate over namespace flag is true, use this to set if we need to move to the next namespace when no vectors are found
next_namespace_flag = False

#used to monitor for q-enter to stop running. This can be cleaned up later with UI improvements. Fine for now
def monitor_quit_key():
    try:
        while not stop_event.is_set():
            user_input = input("Press 'q' + Enter to stop after current batch: ")
            if user_input.strip().lower() == 'q':
                stop_event.set()
                print("✅ Graceful shutdown signal received.")
    except Exception as e:
        print(f"[ERROR] Quit listener error: {e}")

#start keyboard listing thread
listener_thread = threading.Thread(target=monitor_quit_key)
listener_thread.start()

#############################################################
#  Methods
#############################################################

"""
    Used to clean non-ascii printable characters in namespace generation from meta-data field
"""
def sanitize_namespace(name):
    # Normalize to closest ASCII equivalent
    normalized = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    # Replace spaces and other non-word characters with underscore
    return re.sub(r'\W+', '_', normalized)

"""
Feature: Stream vectors to Parquet with metadata update
Description:
   - Writes the current batch of vectors to a uniquely named Parquet file.
   - Updates the source index to set `migrated` and `migrated_at` flags.
Parameters:
   - vectors: Dictionary of vector ID -> Vector object
   - output_dir: Directory to write Parquet files
   - source_index: Pinecone Index object to update metadata
   - namespace: Pinecone namespace for source vectors
   - batch_num: Integer used to differentiate output files
"""
def write_vectors_to_parquet(vectors, output_dir, source_index, namespace):
    if output_dir == "":
        output_dir = "default-directory"
    os.makedirs(output_dir, exist_ok=True)
    records = []
    update_batch = []
    migrated_timestamp = datetime.utcnow().isoformat()

    for vid, vec in vectors.items():
        metadata = vec.metadata or {}
        metadata["migrated"] = True
        metadata["migrated_at"] = migrated_timestamp

        record = {
            "id": vid,
            "values": vec.values,
            "metadata": json.dumps(metadata)
        }

        update_record = {
            "id": vid,
            "values": vec.values,
            "metadata": metadata
        }
        records.append(record)
        update_batch.append(update_record)

    df = pd.DataFrame(records, columns=["id", "values", "metadata"])

    unique_str = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    parquet_path = os.path.join(output_dir, f"vectors_{unique_str}.parquet")

    df.to_parquet(parquet_path, index=False)
    logging.info(f"Wrote {len(records)} vectors to {parquet_path}")

    # Update source index with metadata
    for i in range(0, len(update_batch), 500):
        try:
            source_index.upsert(vectors=update_batch[i:i + 500], namespace=namespace)
            logging.info(f"Updated metadata for {len(update_batch[i:i + 500])} vectors")
        except Exception as e:
            logging.error(f"Metadata update failed: {e}")

"""
Helper to keep vector query from returning nothing
"""
def random_vector(dimension):
    return [random.uniform(-1.0, 1.0) for _ in range(dimension)]

"""
Feature: Metadata-based ID retrieval
Description: Returns vector IDs matching a metadata filter using Pinecone's query() method.
Parameters:
   - index: Pinecone index object
   - filter_condition: dict, e.g., {"migrated": {"$ne": True}}
   - namespace: optional namespace
   - top_k: max number of results to return
Returns: List of matching vector IDs
"""
def get_ids_with_metadata_filter(index, filter_condition, namespace=None, top_k=10000, dimensions=8):
    # dummy_vector = [0.0] * dimensions  # Dim must match; safe fallback if unknown

    dummy_vector = random_vector(
        dimensions)  # this keeps the similarity score from being 0 as can happen with an all 0 vector
    try:
        response = index.query(
            vector=dummy_vector,
            top_k=top_k,
            include_metadata=False,
            include_values=False,
            filter=filter_condition,
            namespace=namespace
        )
        return [match.id for match in response.matches]
    except Exception as e:
        logging.error(f"Query failed for metadata filter: {e}")
        return []

"""
Feature: Reset migration metadata
Description: Removes 'migrated' and 'migrated_at' flags from all vectors in the source index.
Parameters:
    - index: Pinecone Index object
    - namespace: Namespace string
"""
def reset_migrated_flags(index, namespace=None, batch_size=100):
    try:
        stats = source_index.describe_index_stats()
        dimensions = stats.dimension
        results = index.query(vector=random_vector(dimensions), top_k=batch_size, namespace=namespace, include_values=False, include_metadata=False, filter={"migrated": {"$exists": True}})

        while len(results.matches) > 0:
            ids = [match.id for match in results.matches]
            # limit must be between 1 and 100
            batch_size = len(ids)
            logging.info(
                f"Resetting migration flags for {batch_size} vectors in namespace '{namespace or '__default__'}'...")

            processed = 0
            batch_num = 1

            logging.info(f"Processing batch {batch_num}: {batch_size} IDs")

            if stop_event.is_set():
                break
            try:
                vectors = index.fetch(ids=ids, namespace=namespace).vectors
            except Exception as e:
                logging.error(f"Fetch error during reset on batch {batch_num}: {e}")
                batch_num += 1
                continue

            reset_batch = []
            for vid, vec in vectors.items():
                if vec.metadata:
                    vec.metadata.pop("migrated", None)
                    vec.metadata.pop("migrated_at", None)
                    reset_batch.append((vid, vec.values, vec.metadata))

            #if reset_batch:
            try:
                index.upsert(vectors=reset_batch, namespace=namespace)

                logging.info(f"Batch {batch_num}: Reset {len(reset_batch)} vectors")
            except Exception as e:
                logging.error(f"Upsert error during reset on batch {batch_num}: {e}")

            processed += len(reset_batch)
            batch_num += 1

            logging.info(f"Completed reset for {processed} vectors.")
            results = index.query(vector=random_vector(dimensions), top_k=batch_size, namespace=namespace, include_values=False, include_metadata=False, filter={"migrated": {"$exists": True}})
    except Exception as e:
        logging.error(f"Pagination error while listing vector IDs: {e}")

    print("Reset complete, enter q and Enter to exit")

"""
Feature: Retrieve all vector IDs with pagination
Description: Safely fetches all vector IDs from a Pinecone index using pagination.
Parameters:
    - index: Pinecone Index object
    - namespace: Optional namespace string
Returns: List of vector IDs
"""
def get_source_vector_ids(index, namespace=None):
    all_ids = []
    try:
        # limit must be between 1 and 100
        for batch in index.list(namespace=namespace, limit=100):
            if isinstance(batch, list):
                all_ids.extend(batch)
            else:
                all_ids.append(batch)
    except Exception as e:
        logging.error(f"Pagination error while listing vector IDs: {e}")
    return all_ids

"""
Feature: Filter non-migrated vectors
Description: Checks each vector's metadata to determine if it has not been marked as migrated.
Parameters:
    - index: Pinecone Index object
    - ids: List of vector IDs
    - namespace: Namespace string
Returns: List of vector IDs that have not been migrated
"""
def get_non_migrated_vector_ids(index, ids, namespace=None):
    non_migrated = []
    vectors = fetch_vectors(index, ids, namespace)
    for vid, vec in vectors.items():
        metadata = vec.metadata or {}
        if not metadata.get("migrated", False):
            non_migrated.append(vid)
    return non_migrated

"""
Feature: Fetch vectors from index
Description: Retrieves vectors from Pinecone in batches.
Parameters:
    - index: Pinecone Index object
    - ids: List of vector IDs
    - namespace: Namespace string
    - batch_size: Number of vectors to fetch per request
Returns: Dictionary of vector ID -> Vector object
"""
def fetch_vectors(index, ids, namespace=None, batch_size=100):
    from itertools import islice

    def chunked(iterable, size):
        it = iter(iterable)
        return iter(lambda: list(islice(it, size)), [])

    results = {}
    for chunk in chunked(ids, batch_size):
        try:
            res = index.fetch(ids=chunk, namespace=namespace)
            results.update(res.vectors)
        except Exception as e:
            logging.error(f"Fetch error on batch of size {len(chunk)}: {e}")
    return results


"""
Feature: Upsert and tag vectors
Description: Applies migration flags and upserts both to target and source indexes.
Parameters:
    - target_index: Pinecone target index object
    - vectors: Dictionary of vector ID -> Vector object
"""
def upsert_vectors_batch_namespace(target_index, vectors, source_namespace, target_namespace):
    current_batch = []
    current_size = 0
    migrated_timestamp = datetime.utcnow().isoformat()

    def batch_size(vec):
        return len(json.dumps({"id": vec[0], "values": vec[1], "metadata": vec[2]}).encode("utf-8"))

    def flush(batch, source_namespace, target_namespace):
        nonlocal current_batch, current_size
        if batch:
            try:
                target_index.upsert(vectors=batch, namespace=target_namespace)
                source_index.upsert(vectors=batch, namespace=source_namespace)
                logging.info(f"Upserted {len(batch)} vectors")
            except Exception as e:
                logging.error(f"Upsert error: {e}")
            current_batch = []
            current_size = 0

    for vid, v in vectors.items():
        if not v.metadata:
            v.metadata = {}
        v.metadata["migrated"] = True
        v.metadata["migrated_at"] = migrated_timestamp
        item = (vid, v.values, v.metadata)
        size = batch_size(item)
        if current_size + size > MAX_PAYLOAD_BYTES:
            flush(current_batch)
        current_batch.append(item)
        current_size += size

    flush(current_batch, source_namespace, target_namespace)


"""
Feature: Upsert and tag vectors
Description: Applies migration flags and upserts both to target and source indexes.
Parameters:
    - target_index: Pinecone target index object
    - vectors: Dictionary of vector ID -> Vector object
"""
def upsert_vectors_metadata_to_namespace(target_index, vectors, source_namespace, target_namespace):
    batches_by_namespace = {}
    sizes_by_namespace = {}
    migrated_timestamp = datetime.utcnow().isoformat()

    def batch_size(vec):
        return len(json.dumps({"id": vec[0], "values": vec[1], "metadata": vec[2]}).encode("utf-8"))

    def flush(namespace_batches):
        for ns, batch in namespace_batches.items():
            if not batch:
                continue
            try:
                target_index.upsert(vectors=batch, namespace=ns)
                source_index.upsert(vectors=batch, namespace=source_namespace)
                logging.info(f"Upserted {len(batch)} vectors to namespace '{ns}'")
            except Exception as e:
                logging.error(f"Upsert error in namespace '{ns}': {e}")
            namespace_batches[ns] = []
            sizes_by_namespace[ns] = 0

    for vid, v in vectors.items():
        if not v.metadata:
            v.metadata = {}
        v.metadata["migrated"] = True
        v.metadata["migrated_at"] = migrated_timestamp

        ns = sanitize_namespace(v.metadata.get(METADATA_FIELD_TO_NAMESPACE_FROM))

        if ns == '':
            ns = "__default__"

        if not ns:
            logging.warning(f"Vector {vid} missing namespace field '{METADATA_FIELD_TO_NAMESPACE_FROM}' in metadata; skipping.")
            continue

        item = (vid, v.values, v.metadata)
        size = batch_size(item)

        if ns not in batches_by_namespace:
            batches_by_namespace[ns] = []
            sizes_by_namespace[ns] = 0

        if sizes_by_namespace[ns] + size > MAX_PAYLOAD_BYTES:
            flush({ns: batches_by_namespace[ns]})

        batches_by_namespace[ns].append(item)
        sizes_by_namespace[ns] += size

    # Final flush of all remaining batches
    flush(batches_by_namespace)

"""
Feature: Vector checksum
Description: Generates a hash of vector contents for equality checks. Used for equality checking to see if the record has been updated
Parameters:
    - vec: Pinecone vector object
Returns: MD5 hash string
"""
def vector_hash(vec):
    serializable = {
        "values": vec.values,
        "metadata": vec.metadata
    }
    return hashlib.md5(json.dumps(serializable, sort_keys=True).encode()).hexdigest()
"""
Feature: Post-migration validation
Description: Verifies that source and target indexes contain identical vectors.
Parameters:
    - source_index: Pinecone source index object
    - target_index: Pinecone target index object
    - source_ids: List of vector IDs
Returns: Boolean indicating match status
"""
def indexes_are_identical(source_index, target_index, source_ids, source_namespace, target_namespace):
    target_vectors = fetch_vectors(index=target_index, ids=source_ids, namespace=target_namespace)
    source_vectors = fetch_vectors(index=source_index, ids=source_ids, namespace=source_namespace)

    missing_in_target = [vid for vid in source_ids if vid not in target_vectors]
    if missing_in_target:
        logging.warning(f"{len(missing_in_target)} vectors missing in target index: {missing_in_target[:5]}...")
        return False

    for vid in source_ids:
        src = source_vectors.get(vid)
        tgt = target_vectors.get(vid)
        if not src or not tgt or vector_hash(src) != vector_hash(tgt):
            logging.warning(f"Vector mismatch for ID {vid}")
            return False

    return True

"""
Feature: Infer vector dimension
Description: Fetches a sample vector to determine its dimensionality.
Parameters:
    - index: Pinecone Index object
Returns: Integer dimension value
"""
def infer_dimension_from_source(index):
    dim = index.describe_index_stats().dimension
    logging.info(f"Inferred dimension: {dim}")

    return dim

"""
Feature: Ensure index availability
Description: Verifies or creates the target index with appropriate config.
Parameters:
    - pc: Pinecone client
    - source_index: Pinecone source index
    - target_index_name: Name of target index
"""
def ensure_target_index(pc, target_index_name, dimensions):
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if target_index_name in existing_indexes:
        logging.info(f"Target index '{target_index_name}' already exists.")
        return

    logging.warning(f"Target index '{target_index_name}' does not exist.")

    try:
        pc.create_index(
            name=target_index_name,
            dimension=dimensions,
            metric="cosine",
            spec={"serverless": {"cloud": CLOUD, "region": REGION}}
        )
        logging.info(f"Created target index '{target_index_name}' in {REGION}.")
    except Exception as e:
        logging.error(f"Failed to create index: {e}")
        exit(1)

def process_batch(batch_ids, target_index, source_namespace, target_namespace):
    vectors = fetch_vectors(source_index, batch_ids, namespace=source_namespace)
    if WRITE_TO_PARQUET:
        write_vectors_to_parquet(
            vectors=vectors,
            source_index=source_index,
            namespace=source_namespace,
            output_dir=target_namespace
        )
        logging.info("30 second delay giving source time to recover")
        time.sleep(30) #since we're not doing a validate, we need to give the source meta-data a change to
        #update otherwise it will loop multiple times, creating unnecessary parquet files
    else:
        if NAMESPACE_FROM_METADATA_FIELD:
            #not yet implemented
            upsert_vectors_metadata_to_namespace(target_index=target_index, vectors=vectors, source_namespace=source_namespace, target_namespace=target_namespace)
        else:
            upsert_vectors_batch_namespace(target_index=target_index, vectors=vectors, source_namespace=source_namespace, target_namespace=target_namespace)

def batch_chunks(iterable, size):
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])

"""
Feature: Execute full migration loop
Description: Main driver function for delta/full migration, batching, and threading.
Parameters:
    - source_index: Pinecone source index object
"""
def migration_loop(source_index, batch_size_loop, dim, source_namespace, target_namespace):
    logging.info("Starting Pinecone Index Migration")
    all_ids = []

    if DELTA_MODE:
        if USE_FILTER:
            filter_to_use = json.loads(FILTER_TO_USE)
        else:
            filter_to_use = {"migrated": {"$ne": True}}

        # all_ids = get_non_migrated_vector_ids(source_index, all_ids, namespace=SOURCE_NAMESPACE)
        all_ids = get_ids_with_metadata_filter(
            index=source_index,
            filter_condition=filter_to_use,
            namespace=source_namespace,
            dimensions=dim
        )
        if not all_ids:
            logging.info("No new vectors to migrate.")

            if ITERATE_OVER_NAMESPACES:
                global next_namespace_flag
                next_namespace_flag = True

            return
        logging.info(f"Found {len(all_ids)} new vectors to migrate.")
    else:
        all_ids = get_source_vector_ids(source_index, namespace=source_namespace)
        logging.info(f"Preparing full migration of {len(all_ids)} vectors")

    #If we're writing to parquet, override the batch size to write out 10k records
    if WRITE_TO_PARQUET:
        batch_size_loop = len(all_ids)

    batches = list(batch_chunks(iterable=all_ids, size=batch_size_loop))
    start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_batch, batch, target_index, source_namespace, target_namespace) for batch in batches]
        for f in as_completed(futures):
            f.result()
    duration = time.time() - start
    logging.info(f"Migrated {len(all_ids)} vectors in {duration:.2f}s")

    # Retry verification with exponential backoff
    if RUN_VALIDATION_AFTER_BATCH:
        if NAMESPACE_FROM_METADATA_FIELD:
            logging.info("Comparison of namespace from meta-data is not implemented yet.")
        else:
            for attempt in range(3):
                wait_time = 5 * (attempt + 1)
                logging.info(f"Validation attempt {attempt + 1}/3... (waiting {wait_time}s before check)")
                time.sleep(wait_time)
                if indexes_are_identical(source_index=source_index, target_index=target_index, source_ids=all_ids, source_namespace=source_namespace, target_namespace=target_namespace):
                    logging.info("Indexes are fully synchronized. Migration complete.")
                    break
                else:
                    logging.warning("Migration completed, but indexes are not identical.")

#############################################################
#  Main
#############################################################
if __name__ == "__main__":
    source_namespace = SOURCE_NAMESPACE
    target_namespace = TARGET_NAMESPACE

    if VERIFY_ONLY:
        logging.info("Running in verification-only mode...")

        target_index = pc.Index(TARGET_INDEX_NAME)
        try:
            # limit must be between 1 and 100
            for batch in source_index.list(namespace=source_namespace, limit=100):

                if indexes_are_identical(source_index, target_index, batch):
                    logging.info("✅ Indexes are fully synchronized.")
                else:
                    logging.warning("❌ Indexes differ between source and target.")

                if stop_event.is_set():
                    break
        except Exception as e:
            logging.error(f"Pagination error while listing vector IDs: {e}")
    else:
        if RESET_MIGRATION_FLAGS:
            reset_migrated_flags(index=source_index, namespace=source_namespace)
        else:
            runloop = True

            # Use describe_index_stats to short-circuit if index is empty
            stats = source_index.describe_index_stats()
            namespaces_keys = list(source_index.describe_index_stats()["namespaces"].keys())
            #start at the first namespace if namespae iteration is turned on
            if ITERATE_OVER_NAMESPACES:
                source_namespace = namespaces_keys[0]

            namespace_stats = stats.namespaces.get(source_namespace, {})

            vector_count = namespace_stats.get("vector_count", 0)
            dimensions = stats.dimension

            #create a new index if the target index does not exist.
            if CREATE_TARGET:
                ensure_target_index(pc=pc, target_index_name=TARGET_INDEX_NAME, dimensions=dimensions)
            target_index = pc.Index(TARGET_INDEX_NAME)

            if vector_count == 0:
                logging.info(f"No vectors found in source namespace '{source_namespace}'. Nothing to migrate.")
            else:
                while runloop:
                    if ITERATE_OVER_NAMESPACES:
                        if next_namespace_flag:
                            i = namespaces_keys.index(source_namespace)
                            l = len(namespaces_keys)
                            if i >= (l - 1):
                                source_namespace = namespaces_keys[0]
                            else:
                                source_namespace = namespaces_keys[namespaces_keys.index(source_namespace) + 1]
                            next_namespace_flag = False
                            logging.info(f"Next Namespace: {source_namespace}")

                    if USE_SOURCE_AS_TARGET_NAMESPACE:
                        target_namespace = source_namespace

                    migration_loop(source_index=source_index, batch_size_loop=BATCH_SIZE, dim=dimensions, source_namespace=source_namespace, target_namespace=target_namespace)
                    DELTA_MODE = True

                    if stop_event.is_set():
                        runloop = False
                        break

listener_thread.join()
print("Graceful shutdown complete.")

