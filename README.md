
# RadioTherapy Dose Prediction

A PyTorch/MONAI-based application for training and running latent diffusion models to predict radiation dose distributions from CT scans, with support for Gaussian quadrature over multiple energy levels.

---

##  Project Structure

```
RadioTherapy/
├── sourcecode/
│   ├── gui.py                   # PyQt5 GUI entry point
│   ├── system_manager.py        # Orchestrates training & inference
│   ├── data_management.py       # Dataset & DataLoader logic
│   ├── inference_module.py      # Gaussian‐quadrature inference pipeline
│   ├── training_pipeline.py     # Trainer classes for AE & diffusion
│   └── generative/              # Prebuilt architectures & inferers
├── testdata/                    # Sample input/output cubes by energy
├── Master_thesis_Hagedorn.pdf   # Hagedorn master’s thesis (Gaussian quadrature)
├── requirements.txt             # Python dependencies
└── README.md                    # You are here
```

---

##  Quickstart

1. **Clone repo**  
   ```bash
   git clone https://github.com/<your-username>/RadioTherapy.git
   cd RadioTherapy
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run GUI**  
   ```bash
   python sourcecode/gui.py
   ```

---

##  GUI Workflow

1. **Select a model checkpoint (only for inference)**  
   - Pretrained autoencoder + UNet `.ckpt` file.  
2. **Choose an energy folder**  
   - Point to one of the `testdata/<energy_ev>/` directories.  
3. **Train or infer**  
   - **Train** runs AE + diffusion training for that energy, logging “Expected input channels” and “UNet-Forward” debug info.  
   - **Calculate Dose** loads CT `.npy`, picks Gaussian quadrature points/weights from your GUI input, then runs inference over all energies and aggregates results.  

---

##  Core Components

### 1. `ParameterManager`
- Loads/stores list of quadrature energies & weights (dynamically from GUI selection).
- Passes them to `SystemManager`.

### 2. `SystemManager`
- **train_training()**  
  - Iterates each epoch for AE & diffusion, per selected energy.
- **run_inference(ct_path, checkpoint)**  
  - Preprocesses CT → tensor → runs `InferenceModule`.

### 3. `InferenceModule`
- **preprocess_ct()**  
  - Resizes CT to a fixed cube (default `64³`).
- **run_inference_over_energies()**  
  - Loops Gaussian quadrature energies & weights.
  - Calls `run_inference_conditioned_on_energy()` per energy.
  - Stacks & weighted‐sums dose outputs.
- **run_inference_conditioned_on_energy()**  
  - Concatenates a normalized energy channel to CT.
  - Encodes latent via AE, then samples with `LatentDiffusionInferer`.

---

##  Gaussian Quadrature

Based on Hagedorn’s Master’s Thesis (Chapter 4.4.3), we choose 8 optimal energy/weight pairs for integrating over the dose spectrum.  
- **Quadrature energies**: e.g. `[11.5, 20.0, …]` keV  
- **Weights**: corresponding integration weights  

The GUI lets you pick an **energy range**; we then compute the 8 quadrature points & weights on the fly and feed them into `InferenceModule`.

---

## 📈 Training & Debug

- **Autoencoder**  
  - Logs “Expected input channels: 2, Actual input channels: 1” if your AE-UNet coupling is misconfigured.
- **Diffusion**  
  - Prints `[DIFF DEBUG] UNet-Forward-Fehler: …` when dimensions mismatch in cross-attention.
- Common errors often stem from mismatched channel counts or missing `cross_attention_dim`.

---

[Figure:Main window for selecting energy folders, CT scans, and running training/inference.](graph/gui_example.jpeg)

---

[Figure:Example result dose distribution](graph/dose_example.png)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
