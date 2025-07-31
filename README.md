# Pinecone Index Migrator

A robust, production-grade Python utility for migrating vector data between Pinecone indexes — with support for metadata preservation, verification, optional Parquet export, namespace filtering, metadata-based filtering, and dynamic namespace routing.

## Overview

The **Pinecone Index Migrator** enables efficient and continuous migration from a **source index** to a **target index** in Pinecone. It supports full metadata copying, real-time delta polling for new vectors, and validation mechanisms to ensure data integrity. Whether you're replicating indexes, migrating across environments, or preparing for batch import to S3, this tool is built to streamline the process.

## Features

- **Automatic Target Index Creation**\
  Creates the target index using the vector dimension and metric of the source index.

- **Incremental Migration with Polling**\
  After completing the initial batch, the tool continuously polls for new vectors and migrates them, making it ideal for real-time or streaming ingestion scenarios.

- **Metadata Preservation**\
  All metadata from the source vectors is preserved during migration.

- **Parquet Export Support**\
  Optionally write out vectors and metadata to Parquet files, split by batch, for S3 import and external processing workflows.

- **Data Validation Modes**

  - Hash comparison to verify that source and target indexes are equivalent.
  - Batch-by-batch verification to ensure data integrity during transfer.

- **Source Metadata Stamping**\
  Source vectors are tagged with:

  - `"migrated": true`
  - `"migrated_at": <UTC timestamp>`\
    to allow for efficient delta-mode filtering on subsequent runs.

- **Namespace Filtering**\
  Supports filtering by `SOURCE_NAMESPACE` and `TARGET_NAMESPACE` to scope migration to a specific partition within the index.

- **Metadata Filtering**\
  Supports filtering vectors by metadata fields (e.g., `{ "migrated": { "$ne": true } }`) to limit processing to unprocessed or specific vectors.

- **Create Namespace in Target from Source Metadata Field**\
  Dynamically assigns target namespace based on a metadata field (e.g., `category`, `domain`, etc.) for logical partitioning.

- **Iterate Over All Source Namespaces**\
  Automatically migrates all records across multiple namespaces in the source index.

- **Propagate Namespace Mapping**\
  Option to use the same namespace name from the source in the target, when no override is provided.

## Configuration & Usage

```bash
python PineconeMigrator.py
```

All runtime behavior is controlled via a `config.json` file. Options include:

- `SOURCE_INDEX` / `TARGET_INDEX`
- `SOURCE_NAMESPACE` / `TARGET_NAMESPACE`
- `DELTA_MODE`: `true` to only migrate new or changed vectors
- `VERIFY_MODE`: `true` to run hash comparison after migration (note: this is slow, will use all ids to verify. has seperate verify batch during migration mode)
- `WRITE_TO_PARQUET`: `true` to write vectors to batch-based Parquet files instead of upserting into Pinecone
- `RESET_MODE`: `true` to clear all `migrated` flags in source index before beginning migration (note: this only works with source/destinations set in config.json)
- `FILTER_TO_USE`: Optional JSON string to define metadata filtering criteria
- `NAMESPACE_FROM_METADATA_FIELD`: If specified, uses this metadata field to determine target namespace dynamically
- `ITERATE_ALL_NAMESPACES`: Set `true` to scan and process all namespaces in the source index sequentially

### Metadata Example (on source vector after migration)

```json
{
  "migrated": true,
  "migrated_at": "2025-07-10T21:14:03.123Z"
}
```

## Output (Parquet Mode)

Parquet files are written in batches (default: 10,000 vectors per file) using timestamp-based filenames like:

```
/your-target-namespace/2025-07-10T21-14-03Z.parquet
```

Each file contains:

- `id`
- `values`
- All metadata fields stored as json in metadata column

## Notes

- The tool requires **write access to the source index**. A future SQLite-based mode is planned.
- Pinecone vector fetch and query limits (e.g., 10,000) are respected.
- Dyanmically figures size to keep under 2mb upsert limit
- Ideal for datasets up to tens of millions of vectors; for very large indexes, parallelism and memory usage should be tuned accordingly.
- Running standalone verify and reset source modified flags are very slow operations. Right now they get all IDs, which is a slow process. This will be optimized in the future.
- To gracefully exit polling mode, press `q` and then `Enter` — the tool will shut down cleanly after completing the current batch.
- Write to parquet will respect iterate over namespace and target=source. This will create a seperate directory for each namespace with parquet files under. 

## Author

John Ward\
Solutions Engineer — Pinecone

This application is **NOT** an official Pinecone product, and is not supported by Pinecone Systems. 

