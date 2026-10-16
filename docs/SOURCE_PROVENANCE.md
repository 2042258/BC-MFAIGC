# Source Identity and Compatibility

## Two records, two meanings

Fresh federated runs retain `code_fingerprint.json`, which records the bytes at
its original seven source paths. After organization, some paths are import
adapters. Their hashes are real adapter hashes, not substituted historical hashes.

An additional `source_manifest.json` records all Python files under the selected
experiment's `BFL/` and `models/` directories and the complete installed
`bcmfaigc` package. This includes the actual shared model, data readers, transforms
and compatibility adapters. Paths are relative labels, not private machine paths.
The inventory never overwrites the seven-file fingerprint or an existing inventory.

`historical_reference_status: "not_evaluated"` means exactly that. An inventory is
not an assertion of historical source equality, numerical equivalence or paper
reproduction. It does not identify dataset bytes, binary dependencies or every
possible dynamically imported external plugin. Keep source immutable during a run.
Coverage is fresh-run startup only; a resumed run does not obtain new provenance
from this hook. Resume provenance requires a separate implementation and validation.

## Historical reference hashes remain immutable

The restored publication baseline already differs in byte hashes from the raw
historical Stage 2 references for Block.py, DatasetLoad.py, Device.py and main.py.
A read-only audit against the existing original experiment source found equal
statement/expression ASTs for all seven fingerprinted Stage 2 files. The recorded
byte/AST hashes are in `original_source_audit.json`. This audit identifies the
pre-organization baseline; it is not a blanket equivalence claim for subsequent
changes, nor proof of identical runtime context.

The historical Krum and Soft pipeline assertions and their reference hashes have
not been rewritten or relaxed. Those drivers require their frozen reference
artifacts and exact-source conditions. They are historical matched-run audits,
not generic launchers for reorganized implementations; they can reject this
candidate and no successful historical audit is claimed. Their new early source
preflight fails without weakening the original post-run assertions. Fresh source
runs use `bcmfaigc-train`; configuration/entrypoint locations are documented in
`EXPERIMENTS.md` and `PREPARATION.md`.

## Data and model compatibility

There is one DDPM implementation in `bcmfaigc/models/ddpm.py`. The registered
`unet.*` parameter names, diffusion buffers, construction order and arithmetic
are retained. `DenoisingNetwork` exposes the role of the network; `UNet` remains
its legacy pickle symbol. The four old `models/Models.py` files only re-export
shared classes so historical trusted model/network pickle paths can resolve.

CIFAR-10 reading is in `bcmfaigc/data/cifar10.py`, MNIST IDX helpers in `mnist.py`,
and Gaussian augmentation in `transforms.py`. `compat.DatasetLoad` adapts old
constructor/method keywords to `Cifar10Dataset`; it does not duplicate data loading.
Old array attribute names, dtypes, relative dataset root, NumPy shuffle/sort calls,
unsupported-name behavior and Gaussian transform RNG behavior remain unchanged.
The legacy relative data root is still interpreted from the experiment BFL working
directory. The new single-run command explicitly establishes `WORKSPACE/BFL/`
and leaves the relative dataset semantics intact.

Ad-hoc inline dataset demonstrations and commented-out, non-executable model
prototypes were removed from shared runtime modules. Useful MNIST helpers were
retained rather than discarded. Compatibility tests cover original import names,
array values/dtypes, RNG state, registered parameter keys and trusted old pickle
class paths. Do not unpickle untrusted files; this organization does not make
Python pickle safe.

## Trust-weight computation

`bcmfaigc/aggregation/trust_weighting.py` contains the single implementations of
`compute_validation_trust` and `median_trust_weight`. The former returns the same
(normalized score, trust weight) tuple as before. The four legacy modules export
these functions under their original names and keyword signatures. Nonfinite-score
fallbacks, parameter coercion/clamping order and invalid-weight median handling
are preserved. This module does not vote, reject an update or select Krum workers.
Those caller policies and frozen experiment parameters have not been changed.

## Local model operations

`training/local.py` owns optimizer creation, diffusion-batch normalization, DDPM
loss calculation, local gradient accumulation and denoising-loss evaluation.
The existing Device methods retain their signatures and delegate these operations.
Update-model evaluation/scoring is organized separately below; protocol and round
orchestration remain in their existing callers. Per-batch loss overrides are still looked
up on each batch rather than cached. Training uses the existing global Torch RNG;
validation keeps its existing seeded generator and root-mode restoration behavior.

The local and centralized evaluators are intentionally not merged: Device restores
its original root training mode, while the centralized reference evaluator leaves
the model in eval mode. Existing edge behavior is preserved, including NaN for an
empty validation loader and eval mode after a validation-seed setup failure before
the original try/finally boundary. These quirks are characterization constraints,
not newly recommended behavior. Frozen model-operation methods in
`tests/fixtures/legacy_local_training.py` are test references, never runtime code.

## Update-model evaluation and scoring

`bcmfaigc/validation.py` prepares a cloned reference model and its optimizer,
temporarily evaluates supplied model parameters with strict loading/restoration,
and computes the signed worker-minus-reference DDPM loss plus existing aggregation
metrics. The optimizer argument of reference preparation remains independent of
the node's constructor optimizer setting. Caller-owned timing stays outside these
functions. Per-call evaluator and aggregation-metric overrides remain active.

The legacy transaction validator retains signature checking, threshold voting,
malicious-validator vote flipping, aggregation eligibility, rewards, signing and
logging in their original order. Its local `worker_loss` name replaces a misleading
accuracy variable, without changing persisted keys or literal legacy log labels.
Frozen methods in `tests/fixtures/legacy_update_validation.py` are test-only
references, not a parallel runtime.

## Consolidated runtime and persistence

`federation/node.py` and `network.py` are the single active node/network engines.
Small immutable variant profiles preserve original attack dispatch, single versus
Multi-Krum support, logging and rejection-event differences. The complete round
loop remains in `simulation/rounds.py` to retain original loop-carried bindings,
iteration order and random calls. Initialization and persistence are separate;
no runtime AST/exec engine, generic locals bag or new RNG policy is introduced.

Canonical block/ledger/encoding implementations are shared. Direct versus block
aggregation decision algorithms stay separate because epoch overwrite, first-vote,
eligibility and reward-addition behavior differ. Krum math is isolated from node
mutation/reporting. Node-owned per-key assignment order remains outside reduction.

Trusted historical network snapshots are loaded with explicit legacy global-name
mapping in `federation/checkpoints.py`, including `models.Models` and all three
node/network compatibility profiles. A cold subprocess test constructs real old
module references and restores them without preloading legacy model adapters.
This fixes the independently identified cold-import blocker; it does not certify
every historical resume path, original snapshot folder naming, offline/fork state
or provenance on resumed runs. Pickle remains unsafe for untrusted input.

## Acceptance limits

The latest real-dependency CPU checkpoint runs 80 tests with zero skips, plus the
unchanged Soft 201-case and Krum 300-case checks. It covers exact local optimizer,
loss, model/gradient/RNG and transaction-visible comparisons, block serialization,
CLI contracts, cold legacy pickle names, configuration identity and collector
schemas. Forty original JSON configurations are byte-identical to the frozen
pre-restructure snapshot; tests retain those hashes and indexed selection IDs.

Actual fresh-process CPU simulations on tiny synthetic CIFAR-format files completed
seven two-round executions. Classic and extended plain-FedAvg checkpoint tensors
and role records match their old implementations exactly. Direct filtering,
blockchain coordination and strict single Krum completed functional smoke runs
with checkpoints, role metadata and snapshots. Functional smoke success is not
exact numerical old/new equivalence for those three paths. Separate bounded model
comparisons use shortened diffusion schedules; none is a paper-scale reproduction.

Earlier bounded extractions and the node consolidation received independent
read-only source review. The simulation review identified the cold legacy-pickle
blocker subsequently corrected and tested. Final persistence/layout follow-up
review is separately scoped to its frozen snapshot. Reviewers inspect source and
supplied execution evidence; they do not claim to have rerun it independently.

A separate isolated editable install in a prepared dependency environment passed
nine fresh-process help checks. Its installed public training command also
completed a two-round synthetic CPU run, preserving caller arguments and producing
a final checkpoint and actual source inventory. The installed analysis command
collected that run using manifest-provided tags. The initial environment setup
lacked ensurepip; verification reused existing dependencies through a scratch
virtual environment without modifying original projects. This is not validation
of a clean dependency installation. Help checks alone are startup evidence, not
training/FID evidence.
Full-data GPU training, real-image Inception/FID, exhaustive resume/offline/fork and
protocol-malice combinations remain unvalidated. Source inventories do not identify
dataset bytes, binary dependencies or JSON configuration bytes automatically.
Publication URLs, stable identifiers and rights/licensing still require owner
confirmation; no public-release readiness is asserted.
