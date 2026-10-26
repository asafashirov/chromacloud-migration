---
name: chroma-cloud-migration
description: Moves a Chroma Cloud collection to a Pinecone serverless index with the chromacloud-migration toolkit, reusing the existing embeddings. Runs preflight, index creation, export, conversion, load, and validation in order and stops at every failed check. Use when the user says "migrate off Chroma Cloud", "move my Chroma collection to Pinecone", "Chroma Cloud is shutting down", "export my Chroma Cloud data", or "replace Chroma with Pinecone". Also works on a local open-source Chroma directory. Not for moving Chroma into Postgres, and not for re-embedding data with a new model.
license: Apache-2.0
compatibility: Needs uv, network access to Chroma Cloud and Pinecone, a Pinecone API key, and Chroma Cloud credentials (API key, tenant, database). Tested with Python 3.13, pinecone 10.2.0, and chromadb 1.5.9.
metadata:
  author: Pinecone
  version: 1.0.0
  repository: https://github.com/pinecone-field/chromacloud-migration
  guide: https://www.pinecone.io/blog/migrate-chroma-cloud-to-pinecone/
---

# Chroma Cloud to Pinecone migration

Chroma Cloud stops accepting writes on November 22, 2026 and stops serving reads on December 22, 2026. This skill moves one collection at a time into a Pinecone document index during a planned write pause. It runs the six scripts in this repository and checks each step's output before going on. The collection's vectors are copied as they are. The app keeps its embedding model.

## Critical rules

Read these before running anything.

1. **The user pauses writes and flips traffic. You never do.** Step 5 and Step 9 change production. Stop before each one, say exactly what the user needs to do, and wait for them to confirm it's done.
2. **Exit code 1 means stop.** Every script exits 0 on success, 1 when a check fails, and 2 on a usage or precondition error. On anything other than 0, explain the finding and fix the cause before running the next step. Never continue past a failure.
3. **Never patch a failed load.** If a data check fails, fix the cause and reload into a new, empty namespace (for example `my-collection-v2`). An upsert-only pass for missing IDs can't remove records that were deleted in Chroma.
4. **Pass the same policy flags to `preflight.py` and `convert.py`.** These are `--array-policy`, `--big-int-policy`, `--text-mode`, and the field-name flags. Keep a note of the flags in use and repeat them.
5. **Don't print secrets.** Check that environment variables are set without echoing their values. Use `your-key` as a placeholder in anything you show.

## Before you start

Ask the user for anything you don't already know:

- The Chroma collection name, plus any other collections to move later.
- A Pinecone index name and a namespace name (convention: the collection name plus `-v1`).
- The cloud and region. AWS `us-east-1` works on every plan. Other regions need Standard or Enterprise.
- Whether the app filters with `where` or `where_document`. If it does, collect the real filters for the parity check (see Step 1).
- The function the app uses to embed search queries. Step 8 needs it to build golden queries.
- Today's date against November 22. After that date Chroma Cloud is already read-only (see "Past November 22" below).

Set up from the repository root:

```bash
git clone https://github.com/pinecone-field/chromacloud-migration.git
cd chromacloud-migration
uv sync
```

Check credentials without printing them:

```bash
for v in PINECONE_API_KEY CHROMA_API_KEY CHROMA_TENANT CHROMA_DATABASE; do
  printenv "$v" > /dev/null && echo "$v set" || echo "$v MISSING"
done
```

If one is missing, ask the user to `export` it in their terminal, or to put it in a `.env` file and run every script as `uv run --env-file .env <script>`. Don't assume your shell inherited it.

Every command below uses `--workdir migration_work`, `my_collection`, `my-index`, and `my-collection-v1`. Replace those with the user's names.

## Workflow

### Step 1: Preflight

```bash
uv run preflight.py --collection my_collection --workdir migration_work
```

Preflight reads the whole collection and writes `migration_work/preflight.json`. Findings come in four levels:

- **BLOCKING** findings must be fixed in Chroma before the write pause. Examples: all-zero vectors, mixed dimensions, metadata over 40 KB, numeric or boolean arrays.
- **TRANSFORM** findings are changes the conversion will make, such as new IDs, renamed metadata keys, and dropped `None` values. Explain each one and what the app has to change because of it (see `references/app-changes.md`).
- **WARNING** and **INFO** findings need no action, but tell the user about any warning.

Choose policies with the user rather than for them:

- `non_string_arrays`: Pinecone lists hold strings only. Either fix the arrays in Chroma or add `--array-policy stringify`, which turns `3` into `"3"`. Range filters on those fields stop working.
- Integers above 2^53: Pinecone would round them. `--big-int-policy stringify` stores them as strings.
- `metadata_over_40kb` from long documents: switch to the full-text-search variant in Step 3 now. The index schema can't change after creation.

If the app uses filters, write them to `app_filters.json` as a JSON list of `{"name": ..., "where": ...}` or `{"name": ..., "where_document": ...}` objects. Then rerun with `--app-filters app_filters.json`. Preflight reports which filters translate and which need redesign.

Once the user accepts the TRANSFORM findings, rerun with the same flags plus `--allow-transforms`. Exit 0 here is the gate for Step 2.

### Step 2: Rehearse and size the window

Run Steps 3, 6, 7, and 8 against a copy or a small collection first. A local open-source copy works: add `--chroma-path PATH` to the scripts that read Chroma. Time the export and the load. That time, plus validation, is the write-pause window the user needs to schedule. Don't estimate a duration from anything else.

### Step 3: Create the index

```bash
uv run create_index.py --index my-index --workdir migration_work
```

It reads the dimension and metric from `preflight.json`. Chroma defaults to `l2`, which becomes `euclidean`. Add `--cloud` and `--region` for other locations, and suggest `--deletion-protection enabled` for production. If the index already exists, the script only checks its schema and exits 1 on a mismatch.

For long documents, use the full-text-search variant for preflight, index creation, and conversion. Use one path, not both:

```bash
uv run preflight.py    --collection my_collection --workdir migration_work --text-mode fts --allow-transforms
uv run create_index.py --index my-index --workdir migration_work --full-text-search
uv run convert.py      --workdir migration_work --text-mode fts
```

### Step 4: Ship the switch, pointed at Chroma

The app needs one config flag that moves reads and writes from Chroma to Pinecone together. Help the user write it using `references/app-changes.md`. It ships now and stays pointed at Chroma until Step 9.

### Step 5: Pause writes (the user does this)

Stop and ask the user to set the app to read-only. Reads stay on Chroma. Once they confirm, rerun preflight with the exact flags from Step 1:

```bash
uv run preflight.py --collection my_collection --workdir migration_work --allow-transforms
```

### Step 6: Export

```bash
uv run export.py --collection my_collection --workdir migration_work
```

Exit 1 almost always means writes are still landing: the count changed during the export. Ask the user to confirm the pause, then rerun. An update that leaves the count unchanged can't be detected by this check. Only the pause prevents that. Keep `export.jsonl` as the archive copy.

### Step 7: Convert and load

```bash
uv run convert.py --workdir migration_work
uv run load.py    --index my-index --namespace my-collection-v1 --workdir migration_work
```

Give `convert.py` the same policy flags as preflight. If any record fails, it writes no `documents.jsonl` and keeps `transforms.jsonl.failed` for review. `load.py` needs a new or empty namespace and can resume with `--resume` after an interruption.

### Step 8: Validate

Build `golden.jsonl` from real user queries, embedded with the app's own query-embedding function. The toolkit doesn't embed text.

```python
import json
queries = ["how do refunds work", "reset my password"]   # real queries from the app
with open("golden.jsonl", "w") as f:
    for i, q in enumerate(queries):
        emb = my_app_embed(q)   # the app's query-embedding function
        f.write(json.dumps({"query_id": f"q{i}", "embedding": list(map(float, emb))}) + "\n")
```

```bash
uv run validate.py --collection my_collection --index my-index --namespace my-collection-v1 \
                   --workdir migration_work --golden-queries golden.jsonl
```

Add `--app-filters app_filters.json` if the app filters. Read `migration_work/validation_report.json` and report each check to the user by name. If a data check fails, follow critical rule 3. If only `app_filters` fails, the data is fine: fix the app's filter instead of reloading.

### Step 9: Decision gate, then flip (the user does this)

Summarize the validation report in a few lines and ask the user whether to flip. Rollback is still free at this point, because nothing has been written to Pinecone. If they flip, reads and writes move together through the Step 4 flag, then writes resume. Moving both at once means no write lands in a store the reads can't see.

### Step 10: Watch, then retire Chroma Cloud

After the flip, rolling back means replaying Pinecone writes into Chroma. Have the user watch errors, latency, and result quality for a while. Before December 22, they should remove the Chroma client code, rotate or delete the Chroma Cloud credentials, and archive `export.jsonl` with `field_map.json`, `id_map.jsonl`, and `transforms.jsonl`. All four files are needed to reverse the renames and ID mappings.

## Past November 22

Chroma Cloud is read-only from November 22, which enforces the write pause for you. Skip Step 5. Steps 6 through 9 still work until December 22, but the app's writes fail until it flips to Pinecone. Say this plainly if the user is starting late.

## Several collections

Run the whole workflow once per collection, each with its own workdir. Collections that share a dimension and distance metric can share one index, each in its own namespace.

## Reporting

After each step, tell the user the exit code, the numbers that matter from that step's JSON output (records counted, exported, converted, loaded, and checks passed), and what comes next. Keep each update short. The user is deciding whether to continue, not reading a log.

## Examples

**A small collection with no filters.** The user says "Chroma Cloud is shutting down, move my `support_docs` collection to Pinecone." Preflight reports a 1,536-dimension `cosine` collection with two TRANSFORM findings, both renamed keys (`text` becomes `meta_text_<hash>`). Explain the rename, rerun with `--allow-transforms`, create the index, and help the user add the switch. Stop for the write pause. Then export, convert, load, and validate with golden queries the user generates from their app. Report that every check passed and ask whether to flip.

**Long documents.** Preflight blocks with `metadata_over_40kb`. Explain that document text counts toward the 40 KB metadata limit by default. Offer the full-text-search variant, which also makes the text keyword-searchable. Rerun preflight with `--text-mode fts`, create the index with `--full-text-search`, and convert with `--text-mode fts`.

**Starting after the cutoff.** On November 25 the user asks for help. Chroma Cloud is already read-only. Skip the pause, run Steps 6 through 9 now, and tell them the app's writes are failing until the flip.

## Troubleshooting

See `references/troubleshooting.md` for each exit code, the common preflight findings, and the validation failures with their usual causes. The repository README has the full script reference.
