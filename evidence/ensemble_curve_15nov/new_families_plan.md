# s271 step 1: how base families are trained and what was changed to add four (15 Nov)

Existing path: studies/13nov/ttt18.py (env SCHEME, PSEED, DRAW, CHAIN_NPZ, EXT) wraps studies/13nov/ttt7.py (run/stream/main),
which wraps studies/12nov/ttt5.py (ARCHS: family -> (cfg, test prior k), ORDER, make, chain_labels). make() calls
experiments/session_bn.make(cfg, d, seed): for `NAME_sbn` it builds run_cv.resolve(NAME) (an AlignedNet, MODS suffix rot05, pretrained
trunk ptm7r from runs/pretrain/<arch>_ea-m7r-s<seed>.pt) and switches its class to SessionNet, which swaps every BatchNorm for SessionBN
(swap_bn copies the pretrained BN state). Per fold of protocol.folds(d, "all_loso") it fits, streams each test session in the order of
runs/shuffle_perms.npz[SCHEME|PSEED|proto], and writes probs18/ttt18<SCHEME><PSEED>d<DRAW>_<fam>_base-all_loso-s<seed>.npz (proba, index).
Families: fb = fb_sbn_rot05 (k 2), fbl = fblight_ea_rot05_sbn (k 5), fbraw = fbcnet_raw_sbn (k 20), sinc = sincshallow_ea_ptm7r_rot05_sbn (k 20).
Task id = ORDER index + len(ORDER) * seed.

## Registered (diff about 20 lines)
- experiments/run_cv.py: PT regex now accepts eegnet|deep4|conformer|atcnet (their m7r trunks exist for seeds 0-5; CONFIGS _ea already existed).
- studies/12nov/ttt5.py: ARCHS conf/eegn/atc/deep4 = <arch>_ea_ptm7r_rot05_sbn, k 20 (as sinc, fbraw; not tuned); ORDER from env FAMS
  (default unchanged, so earlier task ids are unchanged).
- studies/13nov/ttt7.py: env PROTOS (default both), SMOKE_ARCH; guard `n_sbn > 0` after fit (a no-SBN run fails loudly).
- studies/13nov/ttt18.py: env BASE_ONLY=1 -> no gC5 arm (only the base posteriors §271 scores; tag unchanged).

## SessionBN, honestly
- EEGNet, ATCNet, Deep4: BatchNorm2d/1d only -> fully SessionBN (guard asserts a nonzero count; the count is printed? no: only asserted).
- EEGConformer: BatchNorm only in the patch-embedding block; the transformer uses LayerNorm. SessionBN covers the convolutional front only.
  This is a partial application and is reported as such. ATCNet's attention/TCN also has LayerNorm parts.
- SPDNet: no SPD module exists in the repo (models.TangentSpace is an LR on tangent vectors); it has no BatchNorm. Needs a new
  torch module (BiMap, ReEig, LogEig on causal per-window covariance, about 80+ lines plus a trainer in AlignedNet's API): not done here.

## Commands
Smoke (Mac): SCHEME=within PSEED=0 DRAW=1 BASE_ONLY=1 FAMS=conf,eegn,atc,deep4 SMOKE_ARCH=conf scripts/run_capped.sh LOG .venv/bin/python -I studies/13nov/ttt18.py smoke
Leonardo (from $WORK/ni26; per account, one array over TASK = family + 4*seed, seeds 0-2 -> ids 0-11, one job per scheme/draw):
SCHEME=within PSEED=<0|1> DRAW=<1|2> BASE_ONLY=1 PROTOS=all_loso FAMS=conf,eegn,atc,deep4 \
 scripts/cloud/leo_array.sh s271w<P>d<D> AIFAC_P02_1636|1675 HH:MM:SS "python -I studies/13nov/ttt18.py" "0-11"
(leo_array.sh: sbatch -p boost_usr_prod --gres=gpu:1 --cpus-per-task=8 --mem=60G; env is exported inside the --wrap, so prefix the env on the
sbatch call with --export=ALL or put it in CMD.)
