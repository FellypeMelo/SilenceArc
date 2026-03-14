$ProgressPreference = 'SilentlyContinue'
$version = "1.20.1"
$url = "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/$version"
$dest = "third_party\onnxruntime.zip" # Force .zip extension
$extractPath = "third_party\onnxruntime"

if (-not (Test-Path "third_party")) { New-Item -ItemType Directory -Path "third_party" }

echo "[INFO] Baixando ONNX Runtime DirectML v$version..."
Invoke-WebRequest -Uri $url -OutFile $dest

echo "[INFO] Extraindo..."
if (Test-Path $extractPath) { Remove-Item -Recurse -Force $extractPath }
Expand-Archive -Path $dest -DestinationPath $extractPath

echo "[SUCCESS] ONNX Runtime pronto em $extractPath"
if (Test-Path $dest) { Remove-Item $dest }
