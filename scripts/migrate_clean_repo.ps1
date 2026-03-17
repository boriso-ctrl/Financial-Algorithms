# migrate_clean_repo.ps1
# Creates a clean Financial-Algorithms repo with only production-essential files.
# Usage: .\scripts\migrate_clean_repo.ps1 -Destination "C:\Users\boris\Documents\GitHub\FA-Clean"

param(
    [Parameter(Mandatory=$true)]
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$Source = Split-Path -Parent (Split-Path -Parent $PSCommandPath)

if (Test-Path $Destination) {
    Write-Error "Destination '$Destination' already exists. Choose a new path or remove it first."
    return
}

Write-Host "Migrating essential files from:" -ForegroundColor Cyan
Write-Host "  $Source" -ForegroundColor White
Write-Host "To clean repo at:" -ForegroundColor Cyan
Write-Host "  $Destination" -ForegroundColor White
Write-Host ""

# --- Create directory structure ---
$dirs = @(
    "$Destination\src\financial_algorithms",
    "$Destination\tests",
    "$Destination\backtests",
    "$Destination\scripts",
    "$Destination\docs"
)
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Path $d -Force | Out-Null
}

# --- 1. Core package (entire src/financial_algorithms tree) ---
Write-Host "[1/7] Copying core package (src/financial_algorithms/)..." -ForegroundColor Green
Copy-Item -Recurse -Path "$Source\src\financial_algorithms\*" `
          -Destination "$Destination\src\financial_algorithms\" -Exclude "__pycache__","*.pyc"
# Remove any __pycache__ that snuck in
Get-ChildItem -Path "$Destination\src" -Directory -Recurse -Filter "__pycache__" |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# --- 2. Tests ---
Write-Host "[2/7] Copying tests/..." -ForegroundColor Green
Copy-Item -Path "$Source\tests\conftest.py"              -Destination "$Destination\tests\"
Copy-Item -Path "$Source\tests\test_backtest_smoke.py"   -Destination "$Destination\tests\"
Copy-Item -Path "$Source\tests\test_data.py"             -Destination "$Destination\tests\"
Copy-Item -Path "$Source\tests\test_demo_blend.py"       -Destination "$Destination\tests\"
Copy-Item -Path "$Source\tests\test_hft_smoke.py"        -Destination "$Destination\tests\"
Copy-Item -Path "$Source\tests\test_indicators_smoke.py" -Destination "$Destination\tests\"
Copy-Item -Path "$Source\tests\test_signal.py"           -Destination "$Destination\tests\"

# --- 3. Backtests ---
Write-Host "[3/7] Copying backtests/..." -ForegroundColor Green
Copy-Item -Path "$Source\backtests\*" -Destination "$Destination\backtests\" -Exclude "__pycache__","*.pyc"

# --- 4. Key scripts ---
Write-Host "[4/7] Copying selected scripts/..." -ForegroundColor Green
$scriptFiles = @(
    "search_combos.py",
    "demo_blend.py",
    "data_loader.py"
)
foreach ($f in $scriptFiles) {
    $path = "$Source\scripts\$f"
    if (Test-Path $path) {
        Copy-Item -Path $path -Destination "$Destination\scripts\"
    }
}
# Copy validate_*.py scripts
Get-ChildItem -Path "$Source\scripts" -Filter "validate_*.py" |
    Copy-Item -Destination "$Destination\scripts\"

# --- 5. Key docs ---
Write-Host "[5/7] Copying selected docs/..." -ForegroundColor Green
$docFiles = @("ARCHITECTURE.md", "PROJECT_ROADMAP.md", "RESULTS.md")
foreach ($f in $docFiles) {
    $path = "$Source\docs\$f"
    if (Test-Path $path) {
        Copy-Item -Path $path -Destination "$Destination\docs\"
    }
}

# --- 6. Root config files ---
Write-Host "[6/7] Copying root config files..." -ForegroundColor Green
$rootFiles = @("pyproject.toml", "python-requirements.txt", "README.md")
foreach ($f in $rootFiles) {
    $path = "$Source\$f"
    if (Test-Path $path) {
        Copy-Item -Path $path -Destination "$Destination\"
    }
}

# --- 7. Create .gitignore ---
Write-Host "[7/7] Creating .gitignore..." -ForegroundColor Green
@"
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.eggs/
*.egg
.venv*/
venv/
.env
*.log
.mypy_cache/
.ruff_cache/
.pytest_cache/
reports/results/
data/search_results/
"@ | Set-Content -Path "$Destination\.gitignore" -Encoding UTF8

# --- Summary ---
$fileCount = (Get-ChildItem -Path $Destination -Recurse -File).Count
Write-Host ""
Write-Host "===== Migration Complete =====" -ForegroundColor Cyan
Write-Host "  Files copied: $fileCount" -ForegroundColor White
Write-Host "  Location:     $Destination" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  cd '$Destination'"
Write-Host "  git init"
Write-Host "  python -m venv .venv"
Write-Host "  .venv\Scripts\Activate.ps1"
Write-Host "  pip install -e .[dev]"
Write-Host "  pytest"
Write-Host ""
