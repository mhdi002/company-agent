# ProposalAgent — one-command install + launch (Windows PowerShell).
#
#   .\run.bat                       (or: powershell -ExecutionPolicy Bypass -File run.ps1)
#   .\run.bat -Port 9000 -Offline -Background -Update -Test -SmokeTrain -NoBrowser
#   .\run.bat -Stop
#
# Then: open the page → Telegram card → paste your bot token → press Start in the bot chat → press Start.
param(
  [string]$HostName = "",
  [int]$Port = 0,
  [switch]$Offline,
  [switch]$Background,
  [switch]$Update,
  [switch]$Test,
  [switch]$SmokeTrain,
  [switch]$NoBrowser,
  [switch]$Stop
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
function Say($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "!!  $m" -ForegroundColor Yellow }

if ($Stop) {
  if (Test-Path logs\app.pid) { Stop-Process -Id (Get-Content logs\app.pid) -ErrorAction SilentlyContinue; Remove-Item logs\app.pid; "stopped" }
  else { "not running" }
  exit 0
}
if ($Update) { Say "git pull"; try { git pull --ff-only } catch { Warn "git pull failed; continuing" } }

# ---------------------------------------------------------------- Python >= 3.10
function Find-Python {
  foreach ($c in @(@("py", "-3.12"), @("py", "-3.11"), @("py", "-3.10"), @("python"), @("python3"))) {
    try {
      $exe = $c[0]; $args_ = @(); if ($c.Count -gt 1) { $args_ = @($c[1]) }
      & $exe @args_ -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
      if ($LASTEXITCODE -eq 0) { return ,$c }
    } catch { }
  }
  return $null
}
$py = Find-Python
if (-not $py) {
  Say "Python 3.10+ not found - installing with winget"
  try { winget install -e --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements }
  catch { Write-Host "Install Python 3.11 from https://www.python.org/downloads/ (tick 'Add to PATH') and re-run."; exit 1 }
  $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
  $py = Find-Python
  if (-not $py) { Write-Host "Python installed; please open a new terminal and re-run run.bat"; exit 1 }
}
$pyExe = $py[0]; $pyArgs = @(); if ($py.Count -gt 1) { $pyArgs = @($py[1]) }

# ---------------------------------------------------------------- virtualenv + packages
if (-not (Test-Path .venv\Scripts\python.exe)) { Say "creating virtual environment .venv"; & $pyExe @pyArgs -m venv .venv }
$V = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$gpu = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
$hash = (Get-FileHash requirements.txt -Algorithm SHA256).Hash.Substring(0, 16) + $(if ($gpu) { "gpu" } else { "" })
$stamp = ".venv\.installed"
if (-not (Test-Path $stamp) -or (Get-Content $stamp) -ne $hash) {
  Say "installing Python packages (first run takes a few minutes)"
  & $V -m pip install --upgrade pip wheel | Out-Null
  & $V -c "import torch" 2>$null
  if ($LASTEXITCODE -ne 0) {
    if ($gpu) { Say "NVIDIA GPU detected -> CUDA build of PyTorch"; & $V -m pip install torch --index-url https://download.pytorch.org/whl/cu124 }
    else { Say "no GPU -> CPU build of PyTorch"; & $V -m pip install torch --index-url https://download.pytorch.org/whl/cpu }
    if ($LASTEXITCODE -ne 0) { & $V -m pip install torch }
  }
  & $V -m pip install -r requirements.txt
  if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed"; exit 1 }
  Set-Content -Path $stamp -Value $hash
} else { Say "Python packages up to date" }

# ---------------------------------------------------------------- Node + docx-js (optional)
if (Get-Command npm -ErrorAction SilentlyContinue) {
  if (-not (Test-Path node_modules\docx)) { Say "installing docx-js (npm)"; npm install --no-audit --no-fund --silent }
} else { Warn "Node.js not found - Word files use python-docx (install Node LTS: winget install OpenJS.NodeJS.LTS)" }

# ---------------------------------------------------------------- config
if (-not (Test-Path .env)) { Copy-Item .env.example .env; Say "created .env (connect the Telegram bot from the web page)" }
New-Item -ItemType Directory -Force -Path logs, outputs\proposals, outputs\records | Out-Null
$env:PYTHONUTF8 = "1"
if ($Offline) { $env:PA__SEARCH__PROVIDER = "offline"; $env:PA__RESEARCH__PROVIDER = "offline"; Say "offline demo mode" }
if ($Test) { Say "running tests"; & $V -m pytest -q }
if ($SmokeTrain) {
  Say "training the tiny model end-to-end (CPU)"
  & $V -m data.download --sample
  & $V -m data.clean.pipeline --offline-benchmarks
  & $V -m tokenizer.train --vocab-size 4000
  $env:PA__DATA__VAL_FRACTION = "0.05"; & $V -m data.clean.shard
  & $V -m training.pretrain --preset tiny --seq-len 512 --micro-bs 8 --accum 1 --max-steps 400 --warmup 40 --lr 2e-3 --min-lr 2e-4 --eval-every 200 --eval-batches 10 --sample-every 200 --save-every 200 --log-every 50 --out training/checkpoints/tiny
  & $V -m agent.synth.generate --companies 60 --k 4 --out data/sft
  & $V -m training.sft --init training/checkpoints/tiny/latest.pt --data data/sft --seq-len 1024 --micro-bs 8 --accum 1 --lr 1e-3 --warmup 30 --max-steps 600 --eval-every 200 --log-every 50 --out training/checkpoints/sft-tiny
  $env:PA__AGENT__CHECKPOINT = "training/checkpoints/sft-tiny/latest.pt"
}

# ---------------------------------------------------------------- launch
if (-not $HostName) { $HostName = & $V -c "from core.config import load_config; print(load_config()['app']['host'])" }
if ($Port -eq 0) { $Port = [int](& $V -c "from core.config import load_config; print(load_config()['app']['port'])") }
$urlHost = if ($HostName -eq "0.0.0.0") { "127.0.0.1" } else { $HostName }
$url = "http://${urlHost}:$Port/"
Say "ProposalAgent -> $url"
Write-Host "    1) Telegram card: paste your bot token (from @BotFather) -> open the bot -> press Start"
Write-Host "    2) Press Start in the app. The heartbeat keeps it running until you press Stop."

$sp = @{ FilePath = $V; ArgumentList = @("-m", "app", "--host", $HostName, "--port", "$Port"); PassThru = $true }
if ($Background) { $sp.WindowStyle = "Hidden"; $sp.RedirectStandardOutput = "logs\app.out"; $sp.RedirectStandardError = "logs\app.err" }
else { $sp.NoNewWindow = $true }
$proc = Start-Process @sp
Set-Content -Path logs\app.pid -Value $proc.Id
if (-not $NoBrowser) {
  for ($i = 0; $i -lt 60; $i++) { try { Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 1 | Out-Null; break } catch { Start-Sleep -Milliseconds 500 } }
  Start-Process $url
}
if ($Background) { Say "running in background (PID $($proc.Id)); stop with run.bat -Stop" }
else { Wait-Process -Id $proc.Id }
