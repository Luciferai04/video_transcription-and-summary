## Installation
1. **Set Up Python Environment**:
   - Ensure you have Python 3.11 installed. You can install it via Homebrew:
     ```bash
     brew install python@3.11
     ```
   - Create and activate a virtual environment:
     ```bash
     /opt/homebrew/bin/python3.11 -m venv venv311
     source venv311/bin/activate
     ```
2. **Install System Dependencies**:
   - Install `ffmpeg` for audio extraction:
     ```bash
     brew install ffmpeg
     ```
   - Install `libsndfile` for `torchaudio`’s `soundfile` backend:
     ```bash
     brew install libsndfile
     ```
3. **Install Python Dependencies**:
   - Install project dependencies:
     ```bash
     pip install -r requirements.txt
     ```
4. **Install SpeechBrain for Noise Reduction**:
   - SpeechBrain provides advanced noise reduction with MetricGAN+. Install compatible versions:
     ```bash
     pip install torch==1.13.0 torchaudio==0.13.0 speechbrain==1.0.2
     ```
   - Verify PyTorch and MPS support (for M1/M2/M4 Macs):
     ```bash
     python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
     ```
     Should output `1.13.0`. MPS support will be `False` with `torch==1.13.0`.
   - Pre-download the MetricGAN+ model:
     ```bash
     python -c "from speechbrain.pretrained import SpectralMaskEnhancement; SpectralMaskEnhancement.from_hparams('speechbrain/metricgan-plus-voicebank')"
     ```
5. **Run the App**:
   ```bash
   streamlit run app.py