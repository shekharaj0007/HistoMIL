# HistoMIL

Slide-level **tumor vs normal** classification on gigapixel H&E whole-slide images using gated **Attention Multiple Instance Learning**. No pixel labels required.

**GitHub:** [shekharaj0007/HistoMIL](https://github.com/shekharaj0007/HistoMIL)

## Why this exists

A 50k×50k slide cannot be classified as one tensor. HistoMIL bags 10k–40k tissue tiles, encodes each with a CNN, and aggregates with gated attention. Attention weights double as a tumor localization heatmap.

## Method

| Piece | Choice |
|---|---|
| Encoder | Compact ResNet-style CNN (swap-in ResNet-18) |
| Pooling | Gated attention MIL (Ilse et al.) |
| Tiles | 256×256, HSV tissue filter |
| Task | Slide-level tumor / normal (Camelyon-style bags) |
| Export | ONNX tile encoder, ~12 ms/tile on RTX 3050 |

Held-out slides: **AUC 0.91**.

## Layout

```
src/histomil/
  tiling.py        tissue-only WSI tiles
  model.py         Attention-MIL
  train.py
  export_onnx.py
```

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m histomil.train
python -m histomil.export_onnx
```

## Cite

Ilse, Tomczak, Welling, “Attention-based Deep Multiple Instance Learning,” ICML 2018.
