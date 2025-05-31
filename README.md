# SmartShot – Final Notebook Setup Guide

This README walks you (or any collaborator) through preparing the drive, populating it with images, configuring secrets, and running \`\`. Follow the steps in order – *you only need to run the ****Final**** notebook*.

---

## 1  Prerequisites

| Requirement                            | Why                                              | Notes                                                                                         |
| -------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| Google Drive (or mounted workspace)    | Storage for raw burst images & model checkpoints | Drive will be mounted at `/content/drive` or similar.                                         |
| Python 3.10+ kernel (Colab/Jupyter)    | Executes the notebook                            | Colab **Pro** is recommended for faster GPUs.                                                 |
| Approved Hugging Face account          | Required to access \`\`                          | Make sure you clicked **“Agree”** on the model card.                                          |
| Hugging Face access token (`HF_TOKEN`) | Authenticates notebook to pull the model         | Generate at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). |

---

## 2  Create the `bursts` folder

```bash
# From a notebook cell **before** running FINAL.ipynb
%bash
mkdir -p /content/drive/MyDrive/bursts
```

*Feel free to change the path, but keep it consistent everywhere.*

---

## 3  Upload your burst photos

1. Copy all candidate images into \`\`.
2. \*\*File format must be \*\*\`\` (uppercase extension is fine). Mixed formats will be ignored.
3. Recommended naming: `YYYYMMDD_HHMMSS_frame#.JPG` – but any names work.

---

## 4  Configure your Hugging Face token as a secret

In Colab / JupyterLab (with `jupyterlab_secrets`):

```python
import os
os.environ["HF_TOKEN"] = "<PASTE-YOUR-TOKEN-HERE>"
```

*or, if using Colab UI:*

1. **Settings ▷ Secrets**
2. Add key = `HF_TOKEN`, value = `<your token>`.
3. Toggle **“Use in notebook”** ✅.

> **Do *****not***** hard‑code the token in the repo.**

---

## 5  Run the Final notebook

Use `%run` so the path is explicit and compatible with CI:

```python
# Example (adjust the path to match your repo layout)
%run /content/drive/MyDrive/SmartShot/notebooks/FINAL.ipynb
```

The notebook will:

1. Mount the drive (if not already).
2. Scan `/bursts` for `.JPG` files.
3. Select the **K** best frames via `best_shot_selector`.
4. Apply edits & caption generation.
5. Save results to `/outputs` (created automatically).

---

## 6  Choosing how many photos to keep (K)

`best_shot_selector` defaults to **K = 10**. To change:

1. Open \`\`.
2. Locate the cell that defines `TOP_K = 10`.
3. Replace `10` with any positive integer.
4. *If **`K > 10`**, also update the slice in the post‑processing cell:*

   ```python
   selected = selected[:K]  # instead of selected[:10]
   ```

---

## 7  Troubleshooting

| Symptom                              | Fix                                                                                 |
| ------------------------------------ | ----------------------------------------------------------------------------------- |
| `FileNotFoundError: bursts/ …`       | Confirm the folder exists **before** running notebook & the path in `%run` matches. |
| Images skipped                       | Ensure files use `.JPG`/`.JPEG` extension (case‑insensitive).                       |
| `401 Unauthorized` from Hugging Face | Make sure `HF_TOKEN` is exported and **Use in notebook** is turned on.              |
| CUDA OOM                             | Lower batch size or switch to Colab Pro/High‑RAM session.                           |

---

## 8  Repository layout (reference)

```
SmartShot/
├─ notebooks/
│  └─ FINAL.ipynb
├─ bursts/                # <‑‑ you create + upload here
├─ outputs/               # auto‑generated
└─ README.md              # this file
```

---

## 9  Credits

*Notebook author:* Jeet Swadia
*Model:* `google/gemma-2b-it` courtesy of Google & Hugging Face.

Happy shooting! 📸
