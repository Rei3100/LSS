# Third-party notices

This repository carries no `LICENSE` file of its own: publishing it grants no rights in the LSS code beyond what the hosting service's terms give every public repository. The third-party material below keeps its own licence, and the notices its licence asks for are kept here or beside the files.

Model weights, media and executables are not in this repository. Code that downloads them names the source, revision and hash; each such asset keeps the licence its distributor states, and LSS does not redistribute it.

## Vendored source, unmodified upstream bytes

| Path | Upstream | Licence | Licence text |
|---|---|---|---|
| `vendor/msst/` | ZFTurbo/Music-Source-Separation-Training, Copyright (c) 2024 Roman Solovyev (ZFTurbo). Outside `models/` the files equal upstream commit `2ba884c2083070c4061fb2d5e3afc41e32420b8a` except `utils/metrics.py` and `utils/settings.py` (an earlier upstream revision, before the multi-resolution bleed metrics) and `inference.py`, modified by LSS to load a model once and reuse it (`load_model`, `proc_folder(..., loaded=)`) | MIT | `vendor/msst/LICENSE` |
| `vendor/msst/models/` | Not in this repository. `scripts/fetch_msst_models.py` (run by `setup-venv.ps1`) installs MSST commit `2ba884c2083070c4061fb2d5e3afc41e32420b8a` from upstream, each file checked against its pinned Git blob id. It bundles implementations under their own terms (BandIt and BandIt v2 Apache-2.0, TorchAudio SQUIM BSD-2-Clause, look2hear/Apollo CC-BY-SA-4.0, a Mamba2 port without a stated licence, MIT parts), as their upstream repositories state | as upstream | upstream |
| `vendor/msst/utils/muon.py` | Distributed inside the MSST tree; implementation follows kozistr/pytorch_optimizer | Apache-2.0 (pytorch_optimizer) | Apache-2.0 text: `vendor/sam_audio/LICENSE-DACVAE` |
| `vendor/facing_the_music/` | Facing the Music (BandIt v2), Copyright (c) 2024 Karn Watcharasupat | MIT | `vendor/facing_the_music/LICENSE` |
| `vendor/sfc_locoformer/` | spectral_feature_compression (SFC-Locoformer), Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan | MIT | `vendor/sfc_locoformer/LICENSE` |
| `vendor/sfc_locoformer/spectral_feature_compression/core/model/bslocoformer.py` | Also Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL), from merlresearch/tf-locoformer | Apache-2.0 (the file header carries both identifiers) | Apache-2.0 text: `vendor/sam_audio/LICENSE-DACVAE` |
| `vendor/sfc_locoformer/spectral_feature_compression/core/model/bandit_split.py` | Band-split classes after BandIt (Karn Watcharasupat) | MIT | `vendor/facing_the_music/LICENSE` |
| `vendor/sam_audio/sam_audio/` | facebookresearch/sam-audio at `bb4c6999d2677c7402360e426afc01ddfad6dce0`, Meta Platforms, Inc. | SAM License | `vendor/sam_audio/LICENSE-SAM` |
| `vendor/sam_audio/sam_audio/model/patcher.py` | Contains padding and convolution helpers from facebookresearch/audiocraft, Copyright (c) Meta Platforms, Inc. and affiliates | MIT | notice below |
| `vendor/sam_audio/sam_audio/model/transformer.py` | Timestep embedding after openai/glide-text2im, Copyright (c) 2021 OpenAI | MIT | notice below |
| `vendor/sam_audio/dacvae/` | facebookresearch/dacvae at `414c20785fc3a28373073ea8ef7a1316eeeaca6e`, Meta Platforms, Inc. | Apache-2.0 | `vendor/sam_audio/LICENSE-DACVAE` |
| `vendor/sam_audio/dacvae/nn/quantize.py` | Vector quantisation after karpathy/deep-vector-quantization, Copyright (c) 2021 Andrej Karpathy | MIT | notice below |
| `assets/web/react.production.min.js`, `assets/web/react-dom.production.min.js` | React 18.3.1, Copyright (c) Facebook, Inc. and its affiliates; the DOM build carries a Modernizr-derived feature test (MIT) | MIT | `assets/web/LICENSE-react.txt` |

The SAM Materials under `vendor/sam_audio/sam_audio/` are distributed only under the terms of the SAM License; a copy of that agreement is the file named above.

## Fonts

Font files are under `assets/fonts/` with their licence texts in `assets/fonts/licenses/`. All are SIL Open Font License 1.1.

| Family | Copyright |
|---|---|
| Anton | Copyright 2020 The Anton Project Authors |
| IBM Plex | see `assets/fonts/licenses/IBMPlex-OFL.txt` |
| Oswald | see `assets/fonts/licenses/Oswald-OFL.txt` |
| Noto Sans JP | Copyright 2014-2021 Adobe (http://www.adobe.com/), with Reserved Font Name 'Source' |

## Data tables adapted from third-party data

| Path | Source | Licence |
|---|---|---|
| `lss/sound/audioset_relations.py` | Google Inc., AudioSet ontology (`ontology.json`; source URL and SHA-256 in the file). Only labels and edges, regenerated into ancestor/descendant names | CC BY-SA 4.0. This file is offered under CC BY-SA 4.0 |
| `lss/sound/audioset.py` | The 527 AudioSet class names in the output order of PANNs CNN14 (`class_labels_indices.csv` of qiuqiangkong/audioset_tagging_cnn, Copyright (c) 2018-2020 Qiuqiang Kong, MIT); the class names come from the AudioSet ontology (Google Inc.) | MIT for the list order; CC BY-SA 4.0 for the ontology names. This file is offered under CC BY-SA 4.0 |
| `lss/bgm/music_description.py` (`_MAJOR`, `_MINOR`) | Major and minor key profiles reported by Krumhansl and Kessler (1982), "Tracing the dynamic changes in perceived tonal organization in a spatial representation of musical keys", Psychological Review 89 | Published measurement values, cited |

## LSS code written to run third-party weights without their libraries

These files are written for LSS. They perform the same operations and use the same parameter names as the named upstream implementation so that the distributed weights load strictly; no upstream source file is included. The upstream projects are credited here.

| Path | Upstream implementation followed | Upstream licence |
|---|---|---|
| `lss/bgm/event_tagging.py` | PANNs Cnn14 in qiuqiangkong/audioset_tagging_cnn and panns_inference, Copyright (c) 2018-2020 Qiuqiang Kong | MIT (notice below) |
| `lss/sound/pe_frame.py` | facebookresearch/perception_models (`core/audio_visual_encoder`), ModernBERT and DAC as implemented in huggingface/transformers; byte-level BPE with the GPT-2 splitting rule (openai/gpt-2, Software Copyright (c) 2019 OpenAI, Modified MIT) | Apache-2.0; Modified MIT |
| `lss/sound/sam_audio.py` | facebookresearch/sam-audio (`SAMAudio` input alignment, anchors, time embedding, ODE) | SAM License (`vendor/sam_audio/LICENSE-SAM`) |
| `lss/sound/t5_text.py` | `T5EncoderModel` and the Unigram tokenizer in huggingface/transformers and tokenizers, Copyright 2018- The Hugging Face team | Apache-2.0 |
| `lss/sound/translate.py` | `MarianTokenizer`, `MarianMTModel` and beam search in huggingface/transformers 4.57; SentencePiece `nmt_nfkc` normalisation (google/sentencepiece) | Apache-2.0 |
| `lss/sound/xling.py` | `BertModel` in huggingface/transformers, BertWordPiece and Unigram in huggingface/tokenizers, Pooling/Dense/Normalize in sentence-transformers | Apache-2.0 |
| `lss/bgm/sfc_analysis.py` | Inference procedure and model configuration of spectral_feature_compression (AIST) | MIT (`vendor/sfc_locoformer/LICENSE`) |

A copy of the Apache License 2.0 is `vendor/sam_audio/LICENSE-DACVAE`.

## MIT notice

The MIT-licensed material named above (Qiuqiang Kong; Meta Platforms, Inc. and affiliates; OpenAI; Andrej Karpathy) is used under these terms:

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Downloaded at setup or run time, not in this repository

`lss/sound/extract.py`, `lss/bgm/scene_native.py`, `lss/bgm/sfc_analysis.py`, `lss/bgm/event_tagging.py` and the model ledger name each asset's distributor, revision, size and SHA-256. Licences differ per asset (among them MIT, Apache-2.0, CC BY 4.0, CC BY-NC-SA 4.0, SAM License, and assets whose distributor states no licence). LSS downloads them to the user's machine for the user's own local use and does not redistribute them; nothing in this repository relicenses them.
