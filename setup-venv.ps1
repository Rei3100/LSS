# setup-venv.ps1
# D-2の順序を固定して実行する（順序が壊れるとGPUが使われないビルドが出る＝T309実測）。
# 使い方: powershell -File setup-venv.ps1

py -3.11 -m venv .venv
.venv\Scripts\pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install "audio-separator[gpu]"
.venv\Scripts\pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
.venv\Scripts\pip install torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python scripts\fetch_ffmpeg.py
if ($LASTEXITCODE -ne 0) { throw "bin\ffmpeg.exe の取得・検証に失敗しました" }
.venv\Scripts\python scripts\fetch_msst_models.py
if ($LASTEXITCODE -ne 0) { throw "vendor\msst\models の取得・検証に失敗しました" }

Write-Output "完了: .venv構築（順序=torch/torchaudio(cu128) -> audio-separator[gpu] -> torch/torchaudio/torchvision再固定(force-reinstall --no-deps)）"
Write-Output "次: .venv\Scripts\python scripts\verify_env.py でGPU認識を確認してください。"
