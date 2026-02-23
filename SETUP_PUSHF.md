# Push-F Demo Collection & Training Setup

## Local Setup (MacBook with conda)

### 1. Clone the repo
```bash
git clone https://github.com/bryandong24/reu_adaptation.git
cd reu_adaptation
```

### 2. Create conda environment
```bash
conda create -n pushf python=3.10 -y
conda activate pushf
```

### 3. Install dependencies
```bash
pip install pygame "pymunk<7" "numpy<2" shapely scikit-image opencv-python click "zarr<3" gym
pip install -e .
```

You do NOT need torch/CUDA locally. Only the environment + pygame are needed for demo collection.

### 4. Collect demos
```bash
python demo_pushf.py -o data/pushf/pushf_demo.zarr
```

- Hover your mouse near the **blue circle** to grab control
- Push the gray **F block** into the green target area
- Episode auto-ends on success
- **R** = retry episode, **Q** = quit, **Space** = pause
- Aim for **50-100 successful episodes**

### 5. Upload data to GCP
```bash
scp -r data/pushf/pushf_demo.zarr <your-gcp-user>@<gcp-ip>:~/reu_adaptation/data/pushf/
```

## Training on GCP

### 6. Train
```bash
source venv/bin/activate
python train.py --config-name=train_diffusion_unet_image_pushf_workspace
```
