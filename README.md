# VSL Synthesis - Vietnamese Sign Language Video Synthesis

## Overview

This project synthesizes smooth, continuous Vietnamese Sign Language (VSL) videos from individual word videos using diffusion models.

**Goal:** Take multiple individual sign language word videos and create a single, natural-looking sentence video with smooth transitions.

**Example:**
```
Input: 
- Video 1: "hello"
- Video 2: "my"
- Video 3: "name"
- Video 4: "is"
- Video 5: "Ram"

Output: 
- Single video of person signing "Hello my name is Ram" continuously
```

---

## Project Structure

```
vsl-synthesis/
├── src/
│   ├── synthesis/          # Main synthesis modules
│   │   ├── concatenation.py    # Sequence concatenation
│   │   ├── interpolation.py    # Spline/Linear interpolation
│   │   ├── render.py           # Video rendering
│   │   ├── pipeline.py         # Main synthesis pipeline
│   │   ├── mdm_adapter.py      # MDM integration
│   │   └── evaluation.py       # Metrics and evaluation
│   └── utils/              # Utility functions
├── models/
│   └── mdm/                # MDM pretrained model
├── data/
│   ├── words/              # Individual word videos/skeletons
│   └── sentences/          # Full sentence videos (for training)
├── outputs/
│   ├── baseline/           # Spline interpolation results
│   └── diffusion/          # Diffusion model results
├── tests/                  # Unit and integration tests
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
cd vsl-synthesis
pip install -r requirements.txt
```

### 2. Basic Usage (Phase 2 - Baseline)

```python
from src.synthesis.pipeline import synthesize_sentence

# Synthesize sentence from word videos
output = synthesize_sentence(
    word_videos=['data/words/hello.npy', 'data/words/my.npy', 'data/words/name.npy'],
    method='spline',
    output_path='outputs/baseline/sentence.mp4'
)
```

### 3. Using Diffusion Model (Phase 3)

```python
output = synthesize_sentence(
    word_videos=['data/words/hello.npy', 'data/words/my.npy'],
    method='diffusion',
    model_path='models/mdm/finetuned.pt',
    output_path='outputs/diffusion/sentence.mp4'
)
```

---

## 📋 Development Phases

### Phase 1: Planning (Completed)
- Research pretrained diffusion models
- Create implementation plan
- Setup project structure

### Phase 2: Baseline Implementation (Current)
- Implement skeleton concatenation
- Implement spline interpolation
- Create video rendering pipeline
- Establish baseline metrics

### Phase 3: Diffusion Model Integration
- Setup MDM (Motion Diffusion Model)
- Adapt for VSL skeleton format
- Fine-tune on VSL data
- Integrate into pipeline

### Phase 4: Evaluation
- Calculate metrics (FID, jerk, smoothness)
- Compare baseline vs diffusion
- User study

---

## Key Components

### Interpolation Methods
- **Linear:** Simple linear interpolation between poses
- **Spline:** Smooth cubic spline interpolation
- **Bezier:** Bezier curve interpolation
- **Diffusion:** MDM-based transition generation (Phase 3)

### Evaluation Metrics
- **FID (Fréchet Inception Distance):** Distribution similarity
- **Jerk:** Motion smoothness (lower is better)
- **Foot Skating:** Unnatural foot sliding detection
- **User Study:** Subjective naturalness rating

---

## Expected Results

| Method | FID ↓ | Jerk ↓ | Naturalness ↑ |
|--------|-------|--------|---------------|
| Linear | ~45 | ~0.82 | ~3.0/5 |
| Spline | ~32 | ~0.54 | ~3.5/5 |
| **Diffusion** | **~18** | **~0.21** | **~4.5/5** |

---

## Resources

- **MDM Paper:** [Human Motion Diffusion Model](https://arxiv.org/abs/2209.14916)
- **MDM GitHub:** [GuyTevet/motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model)
- **Related Project:** [vsl-recognition](../vsl-recognition) (Recognition task)

---

