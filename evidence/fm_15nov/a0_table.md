# A0 (ledger s279): foundation-model weights, licence and rules (read-only, 15 Nov 2026 11:5x CET; nothing downloaded)
Sizes are bytes as listed by the HF/GitHub tree APIs; "n/s" = not stated on the official page.

## Rules (2026-competition/codabench/pages/terms.md Rule 4, 7, 8; participate.md; scientific_faq_bci.md)
- Rule 4: pre-training on any publicly available, redistributable dataset is allowed; sealed test split and closed clinical sets are not;
  "original dataset and model licenses remain in force"; every external corpus and a compute estimate must be declared in the method description.
- Rule 7: top 3 are rerun from the committed training pipeline and rescored within 2 sigma (a reproducible weight download path is needed).
- participate.md: weights ship inside the ZIP (15 GB profile storage); no download of any kind inside load_model/predict. The organisers' own
  start-kit has a frozen REVE probe (brain-bzh/reve-base), so pretrained foundation weights are within the intended scope.
- Verdict: all six are allowed by the rules if their licence permits the use and the corpus is declared. Weights must be bundled, not fetched at inference.

## Table (windows: ours are 41 ch, 4 s, 120 Hz = 480 samples; resampled to 200 Hz = 800 samples = 4 patches of 200)
| Model | Code licence | Weight licence | Weight file (size) | Channels / rate / window | Pretraining corpora | Mapping to ours / issues |
|---|---|---|---|---|---|---|
| LaBraM | MIT | n/s (only MIT code stated) | labram-base.pth 96,612,769 (github checkpoints/); vqnsp.pth 94,805,201; HF mirror braindecode/labram-pretrained model.safetensors 23,299,672 | named 10-20/10-10 channels via channel-order list; 200 Hz; 200-sample patches; 0.1-75 Hz BP + 50 Hz notch, uV | ~2,500 h, ~20 datasets, not named on README | needs 41 names in its vocabulary (check montage); 4 patches/channel, 41x4=164 tokens; used before (s71, 35.9 linear probe) |
| CBraMod | MIT (GitHub) | Apache-2.0 on HF (weighting666/CBraMod) | pretrained_weights.pth 19,775,842 | any channel count (example 22), patch 200 pts, 4 segments in example; rate n/s on page (200 Hz in paper) | "our pretraining dataset" (TUEG in paper, not named on page) | easiest fit: arbitrary channels, 4 patches of 200; used before (s71, 42.0) |
| EEGPT | Apache-2.0 | n/s | eegpt_mcae_58chs_4s_large4E.ckpt, figshare share link, size n/s | 58 fixed channels, 256 Hz, 4 s, patch 64 | "mixed dataset", not named on README (pretrain/readme.md) | 41 of 58 names must be a subset; resample 120 to 256 Hz (1024 samples); never run |
| BIOT | MIT | n/s | pretrained-models/ EEG-PREST-16-channels.ckpt 13,791,969; EEG-SHHS+PREST-18-channels.ckpt 13,794,194; EEG-six-datasets-18-channels.ckpt 13,794,371 | 16 or 18 bipolar channels, 200 Hz, 10 s (2000 samples) | MGH resting EEG, SHHS sleep, +TUAB/TUEV/CHB-MIT/IIIC (six-datasets) | poor fit: bipolar montage, 10 s vs our 4 s, 41 channels need a bipolar subset; weak candidate |
| Neuro-GPT | GPL-3.0 (copyleft; weights n/s) | n/s | pretrained_model/pytorch_model.bin 318,189,905 (HF wenhuic/Neuro-GPT) | n/s on README (22-ch MI downstream per paper, not verified) | TUH EEG (README datasets section); paper abstract: "large-scale data set" | 22-channel front end, GPL-3 code in a submission is a licence risk; low priority |
| REVE | repo licence n/s on HF page | REVE Responsible Use License v1.0 (research, commercial, education, personal use allowed; bans re-identification, surveillance, discrimination; keep licence/attribution/provenance on redistribution) | model.safetensors 276,773,456 (HF brain-bzh/reve-base); plus brain-bzh/reve-positions (position bank, size n/s); config/modeling .py 16 KB (trust_remote_code) | arbitrary electrodes via 3D positions from names; 200 Hz; 1 s patches, overlap 0.1 s; length multiple of 1 s | 60,000 h, 92 datasets, ~25,000 subjects; paper appendix lists MOABB MI sets (Cho2017, Lee2019MI, Schirrmeister2017, ...) and Dreyer 2023 ("Inria Large", Zenodo CC BY 4.0) | 4 s at 200 Hz fits exactly; needs the 41 electrode names; used before (s71: 44.2, random-init 42.8) |

## Data-overlap notes
- REVE pretrains on Dreyer 2023 (a different, earlier acquisition than competition PROTEUS nm000290, which is Dreyer 2026 per our contract). Whether PROTEUS
  recordings are in the 92 sets is not stated in the pages read (OpenNeuro list only partly read); the REVE paper says downstream-task recordings were removed.
- LaBraM, EEGPT, CBraMod: corpora not named on the official pages read; full overlap check needs the papers' appendices.
- BIOT/Neuro-GPT corpora are clinical (MGH, SHHS, TUH): TUH needs a data agreement, which only matters if we retrain, not for using released weights.

## Sources
- https://github.com/935963004/LaBraM ; https://api.github.com/repos/935963004/LaBraM/contents/checkpoints ; https://huggingface.co/api/models/braindecode/labram-pretrained/tree/main
- https://github.com/wjq-learning/CBraMod ; https://huggingface.co/weighting666/CBraMod ; https://huggingface.co/api/models/weighting666/CBraMod/tree/main
- https://github.com/BINE022/EEGPT
- https://github.com/ycq091044/BIOT ; https://api.github.com/repos/ycq091044/BIOT/contents/pretrained-models
- https://github.com/wenhui0206/NeuroGPT ; https://huggingface.co/api/models/wenhuic/Neuro-GPT/tree/main/pretrained_model ; https://arxiv.org/abs/2311.03764
- https://huggingface.co/brain-bzh/reve-base ; https://huggingface.co/api/models/brain-bzh/reve-base/tree/main ; https://arxiv.org/abs/2510.21585 (html appendix)
- Local: 2026-competition/codabench/pages/terms.md, participate.md, scientific_faq_bci.md, tracks/bci_decoding/solvers/reve_probe.py
