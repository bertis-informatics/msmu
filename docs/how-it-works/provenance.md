# Provenance

MSMU records supported public calls in `mdata.uns["_log"]`. The history travels with the
MuData object and its `.h5mu` file; no sidecar files are created. This replaces `_cmd`.
Existing `_cmd` content is neither read nor migrated.

```python
import msmu as mm

mdata = mm.read_diann("report.parquet")
mdata = mm.pp.log2_transform(mdata, modality="precursor")
log = mm.provenance.get_log(mdata)
last = log["events"][-1]
print(last["function_path"], last["parameters"])
print(log["environments"][last["environment_id"]])

mdata.write_h5mu("analysis.h5mu")
loaded = mm.read_h5mu("analysis.h5mu")
```

`get_log` returns a detached dictionary. Editing it does not alter the stored history.
Events are ordered by lineage sequence, then UTC start time (UUID breaks ties); `parents` and `head` identify
branches explicitly. Reading with `mm.read_h5mu` preserves the stored history and adds a
read event. Native `mudata.read_h5mu` preserves the stored history without adding an event.

## Recorded information

Each event includes a UUID, function name and qualified path, UTC start/end times, elapsed
computation time, parameters including defaults, an environment ID, input/output entity descriptors, and parent event IDs.
Only successfully completed calls create events. Events have neither `status` nor `error`;
hash entries retain their separate `status`.

Scalar parameters, strings, mappings and sequences are not truncated. Data objects (including
MuData, AnnData, pandas objects and arrays) are omitted from parameters, including nested
data entries and containers containing only data. Other option values and defaults are kept.
Input/output entities contain IDs, roles, types and optional hashes, without shape, column
or dimension summaries. File paths remain readable parameter values. Unsupported parameters and callables are described
and marked non-replayable. Function objects are not pickled or executed from the log.

Entities have independent UUIDs and argument/return roles. Input descriptors and hashes are
captured **before** computation, including for in-place mutations. A new observation receives
a new entity ID even when its content hash matches an earlier observation. Parents identify
inherited history, not proof that no unrecorded edits happened between calls.

Returned MuData objects receive the history. For `tl.run_de`, the history stays on the input
MuData, with a descriptor and optional hash of the returned `DeaResult`'s instance fields.
Failed calls propagate the original exception without adding an event or changing existing
history. Changes made to data before failure are not rolled back and are not logged.
Invalid pre-existing `_log` structures are rejected before computation; a failure while
saving a completed event emits a warning and preserves the computation's return value.

Nested decorated calls are represented by one outer public event. This avoids duplicate
records and repeated full-data hashes for internal implementation steps. Independent branches
keep separate histories; merging MuData combines histories by event ID without duplicating
shared ancestors. Logs are copied independently from data, including functions that copy
`.uns` shallowly. Console output and Python/MSMU logging messages are not captured or
stored in events; their normal output follows the caller's logging configuration.
Independent calls are not serialized by a global lock. Concurrent mutation of the same
MuData object is not supported.

## Optional content hashes

Hashing defaults to **off**. Environment and execution records remain enabled.

```python
# Persistent setting in the current execution context:
mm.provenance.set_options(hashing=True)
print(mm.provenance.get_options())  # {"hashing": True}
mdata = mm.pp.log2_transform(mdata, modality="precursor")
mm.provenance.set_options(hashing=False)

# Or enable hashing for a block and restore the previous setting afterwards:
with mm.provenance.options(hashing=True):
    mdata = mm.pp.scale_data(mdata, modality="precursor")

last = mm.provenance.get_log(mdata)["events"][-1]
print(last["inputs"][0]["hash"])

# Explicitly hash the current state without recording an event:
current_hash = mm.provenance.compute_hash(mdata)
```

A hash entry has status `disabled`, `completed`, or `unavailable`. Completed entries include
the algorithm (`sha256`), digest and hash duration. Unsupported types or
unreadable inputs produce `unavailable` with a reason; these must never be interpreted as
matching content. Explicit `compute_hash` calls raise an exception for unsupported content.

The MuData hash covers global and modality `obs`, `var`, aligned mappings, `uns`,
MuData axis maps, AnnData `.X`, named layers and `.raw`. Only each object's `_log` is excluded.
Numeric dtype, shape, values, axis labels/order, and metadata affect the digest. Numeric
arrays use canonical byte order and NaN payloads. Sparse arrays are copied to canonical CSR,
with sorted/combined entries, without densification. Explicit zero entries remain meaningful;
sparse and dense representations are distinct. Unordered categorical columns hash by values,
normalizing h5mu's automatic string-to-category conversion. Unused unordered categories and
the string storage backend are intentionally ignored; ordered categories retain order metadata.

Path objects and strings in `file`/`path` arguments are input file entities. Existing files
are streamed in 1 MiB blocks when hashing is enabled. HTTP(S)/FTP URLs retain their original
address. Shared tabular readers download each URL once per logged call into `BytesIO`, then
give the same seekable buffer to Polars or pandas. With hashing enabled, provenance hashes
those downloaded bytes, using the same file-content encoding as a local file. No second
download or disk file is needed. URL input hashes are finalized after the reader has consumed
the source; URLs not consumed through this shared loader remain `unavailable`.

Download buffers are retained until the logged call finishes and are closed on success or
failure. Peak memory can include all downloaded source bytes plus parsed tables. This is a
full download, not a remote partial-read optimization. Hashing off skips hash calculation but
uses the same loading path. Directories and unavailable files are not recursively fetched.
Other implicit external dependencies are outside
the captured inputs. Hashing is proportional to covered content and allocates sparse copies
and any required canonical arrays. It does not create checkpoints or data snapshots.

## Execution environment

Identical environment records are stored once and referenced by ID. Records include Python
version/implementation, OS release and architecture, names and versions of all installed
Python distributions visible to the running interpreter (including indirect dependencies), an available
MSMU checkout commit/dirty flag, NumPy error settings, numerical threadpool details when
available, and an allowlist of numerical thread/hash-seed environment variables. Arbitrary
environment variables, credentials and host/user identifiers are not collected.

Package/source metadata is a snapshot cached on the first recorded call in a Python process;
restart Python after changing installed packages or source code. Threadpool and numerical
settings are sampled on each call. Unavailable metadata is explicitly marked. Seeds supplied
as function arguments are recorded, but hidden RNG state, GPU state, package binaries and
complete environment reconstruction are not guaranteed.

## Coverage and interpretation

Automatic recording covers existing preprocessing and PCA/UMAP/correlation calls, plus
`split_tmt`, import readers including DELPI/h5mu, `io.add_quant`, `merge_mudata`, `tl.run_de`,
`tl.compute_precursor_isolation_purity`, and the MuData helpers `reindex_obs`, `attach_fasta`,
`map_fasta`, and `select_repr_protein`. The `normalize` alias uses the `normalise` event.
All eight MuData-based `pl.plot_*` functions also log to the input MuData. Figures are
returned normally and are not serialized or hashed in the log.

Export functions (`io.to_readable`, `io.write_csv`, `io.write_flashlfq_input`, and
`io.write_pin`) do not log events or calculate hashes. `write_pin` requires an output
filename and returns `None`.

Standalone DataFrame readers, `pl.plot_volcano`, standalone mzML purity calculation, scalar
utilities, configuration functions and direct pandas/NumPy/MuData edits are not automatically
recorded. They either have no MuData storage target or are outside processing history.
User functions accepting or returning MuData can opt into the
same boundary using `@mm.provenance.log`.

The model follows the core concepts of [W3C PROV-DM](https://www.w3.org/TR/prov-dm/):

| MSMU log entry | PROV interpretation |
| --- | --- |
| A function execution | Activity |
| An observed input/output state | Entity |
| Execution consuming an input entity | Usage (`used`) |
| Successful execution producing an output state | Generation (`wasGeneratedBy`) |
| MSMU software identity plus environment reference | Software agent and execution context |

The stored representation is MSMU's versioned JSON schema, not PROV-JSON/RDF interchange,
and does not claim PROV constraint validation or FAIR certification. History and hashes
support auditing observations. They do not reconstruct arbitrary pandas edits, store original
data, prove computational reproducibility, or provide replay.

## Storage

`_log` contains `schema_version`, `head`, `events`, and `environments`. Events and environments
are dictionaries from IDs to JSON strings. This avoids h5mu restrictions on heterogeneous
lists, missing values and user parameter names. `get_log` decodes those strings on demand.
No write/load monkey-patches or additional file formats are required.
