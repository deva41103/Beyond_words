# Beyond_words

Beyond_words is a sign-language-to-text/web demo project that uses a trained deep-learning model to recognize sign-language gestures from video or webcam input. The repository contains a web application frontend/back-end, a PyTorch model file, helper scripts for uploading videos and preprocessing, and a Jupyter notebook demonstrating experiments.

> NOTE: This README was generated from the repository file listing. Before running, please inspect `app.py`, `model.py`, and related scripts to confirm exact dependency names, endpoints, and configuration values used by this project.

## Key features
- Real-time or recorded-video sign language recognition (model inference).
- Web application interface (Flask or similar) to upload or stream video and receive predicted words.
- Pretrained model checkpoint (`best_model.pth`) for inference.
- Jupyter notebook (`Beyond_word.ipynb`) for exploration and experiments.
- Supporting JSON files for vocabulary and metadata.

## Quick links (repository files)
- app.py — Main web application (server) script.
- app_updated_per_gloss.py — Updated application variant (per-gloss processing).
- model.py — Model architecture and helper functions.
- upload_video.py — Helper script for handling video uploads.
- new.py — Additional script (utility or experiment).
- best_model.pth — Trained PyTorch model weights (large binary file).
- Beyond_word.ipynb — Notebook with experiments and demonstrations.
- vocab1.json — Vocabulary mapping used by the model.
- sign_language_data.json — Additional structured metadata about the dataset or labels.
- wlasl100_dataset/ — Directory with dataset or dataset references (WLASL subset).
- keypoints1/ — Directory likely containing extracted keypoint files (pose/hand keypoints).
- templates/ and static/ — Web app frontend assets (HTML templates, JS/CSS).
- instance/ and __pycache__/ — runtime or cache directories.

## Requirements
The exact dependencies are not provided in the repository listing. Typical dependencies for this type of project include:
- Python 3.8+
- torch (PyTorch) and torchvision — for the model and inference
- Flask (or a similar WSGI framework) — for the web app
- OpenCV (cv2) — for video capture and preprocessing
- numpy, pandas — data handling
- (optional) mediapipe or other pose/hand keypoint extractors — if keypoints are used
- Any additional libraries referenced in `app.py` or `model.py`

Recommendation: create a virtual environment and install dependencies. If you add a `requirements.txt`, include versions you tested with.

## Installation (example)
1. Clone the repository
   - git clone https://github.com/deva41103/Beyond_words.git
   - cd Beyond_words

2. Create and activate a virtual environment
   - python -m venv venv
   - source venv/bin/activate  (macOS/Linux)
   - venv\Scripts\activate     (Windows)

3. Install dependencies (example)
   - pip install torch torchvision flask opencv-python numpy pandas

4. (Optional) Add a `requirements.txt` with pinned versions for reproducibility:
   - pip freeze > requirements.txt

5. Ensure model file is present:
   - `best_model.pth` should be in the repository root (already present in this repo). If you relocate it, update the path used in `app.py` or `model.py`.

## Running the web app (typical)
1. Inspect `app.py` to confirm how to start the server. Common ways:
   - python app.py
   - export FLASK_APP=app.py && flask run
   - python -m flask run

2. If `app.py` contains a `if __name__ == "__main__":` block, running `python app.py` will likely start the app.

3. By default, the app may run on http://127.0.0.1:5000 — open that URL in your browser.

4. Upload a video using the web UI (templates + upload) or use webcam capture if implemented.

Important: Inspect `app.py` before running to confirm the expected command, environment variables (e.g., host/port), and whether GPU support is optional.

## Using the notebook
- `Beyond_word.ipynb` contains exploratory analysis and example code for preprocessing, training, or inference.
- Open in Jupyter: jupyter notebook Beyond_word.ipynb or use VS Code / Jupyter Lab.
- The notebook may reference local paths (datasets, keypoints). Make sure paths match and required data is in place.

## Model & Data
- Model architecture: see `model.py` for class definitions and forward/inference logic.
- Checkpoint: `best_model.pth` contains trained weights (PyTorch format).
- Vocabulary: `vocab1.json` likely maps class indices to words/labels.
- Dataset directory: `wlasl100_dataset/` indicates the use of a subset of the WLASL dataset (or similar). Confirm data licensing and usage restrictions before redistributing.

If you need to run inference:
- Load model via code in `model.py` (or your own loader) and load `best_model.pth`.
- Ensure the same preprocessing pipeline is used as during training (keypoints, normalization, frame rate, resizing). Look at the notebook or `app.py`/`upload_video.py` for preprocessing steps.

## File structure (annotated)
- Beyond_word.ipynb — Experiment notebook
- app.py — Main web server
- app_updated_per_gloss.py — Alternate/updated server logic (per-gloss processing)
- model.py — Model architecture and helper functions
- upload_video.py — Upload handler (server-side)
- new.py — Utility or experimental script
- best_model.pth — Trained weights (PyTorch)
- vocab1.json — Vocabulary / label mapping
- sign_language_data.json — Metadata for sign language labels
- templates/ — HTML templates used by the web app
- static/ — static assets (CSS/JS/images)
- keypoints1/ — stored keypoint files (likely used for inference/preprocessing)
- wlasl100_dataset/ — dataset or dataset metadata and files

## Common tasks & examples

Load the model (example pseudocode)
```python
# Example (adapt to model.py interface)
import torch
from model import SignModel  # replace with actual class name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignModel(...)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()
```

Start the app (example)
```bash
python app.py
# or
flask run
```

Upload video
- Use the web UI or run `python upload_video.py <video-file>` if upload script supports CLI (inspect script to confirm).

## Troubleshooting
- Missing dependencies: read `app.py` and `model.py` to find imports, then pip install missing packages.
- GPU errors: ensure the correct CUDA-enabled PyTorch wheel is installed or run on CPU by setting map_location in torch.load.
- Large model file: `best_model.pth` is ~20MB+ (in the listing ~20,331,234 bytes). Ensure you have sufficient disk and memory.
- Missing dataset: If notebook or scripts expect the dataset in `wlasl100_dataset/`, acquire the data and place it there.

## Testing & Development
- Add unit tests and a CI workflow (GitHub Actions) to automate linting and test runs.
- Consider adding a `requirements.txt` and `environment.yml` for reproducibility.
- Add a small example video and a quick CLI script to test inference without the web UI.

## Contributing
- Fork the repo, create a feature branch, and open a pull request.
- Document any new dependencies in `requirements.txt`.
- Add tests for new features and update README with any new usage instructions.

## License & Attribution
- No license file was found in the repository listing. If you intend to share the project publicly, add a license (e.g., MIT, Apache 2.0) to clarify reuse terms.
- If you use external datasets such as WLASL, make sure to follow their license and citation requirements.

## Contact / Author
- Repository: https://github.com/deva41103/Beyond_words
- Author/owner: deva41103

---

If you’d like, I can:
- produce a ready-to-add README.md file (with the full text above) suitable for committing,
- generate a suggested requirements.txt by scanning the code for imports (I can do that next if you want).
