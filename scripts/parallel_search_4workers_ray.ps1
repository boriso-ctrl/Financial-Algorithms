# Parallel Ray-based search orchestrator
# Launches 4 Ray workers in parallel, each running 250k-sample Monte Carlo search
# Each worker writes to a separate JSON file, then aggregates results

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$PythonExe = "$ProjectRoot\.venv_py311\Scripts\python.exe"
$SearchScript = "$ProjectRoot\scripts\search_combos.py"
$DataDir = "$ProjectRoot\data\search_results"

# Ensure output directory exists
if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# Worker configuration
$Workers = @(
    @{ id = 1; seed = 42; output = "search_ray_1m_worker1_$Timestamp.json" },
    @{ id = 2; seed = 123; output = "search_ray_1m_worker2_$Timestamp.json" },
    @{ id = 3; seed = 456; output = "search_ray_1m_worker3_$Timestamp.json" },
    @{ id = 4; seed = 789; output = "search_ray_1m_worker4_$Timestamp.json" }
)

Write-Host "Ray Parallel Search Orchestrator" -ForegroundColor Cyan
Write-Host "  Project: $ProjectRoot" -ForegroundColor Gray
Write-Host "  Python: $PythonExe" -ForegroundColor Gray
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Workers: 4" -ForegroundColor Gray
Write-Host "  Samples per worker: 250,000 (total 1M)" -ForegroundColor Gray
Write-Host "  Execution: Ray distributed" -ForegroundColor Green
Write-Host "  Ray actors per worker: 4" -ForegroundColor Gray
Write-Host ""

# Launch all 4 workers in parallel
Write-Host "Launching workers..." -ForegroundColor Cyan
$Jobs = @()

foreach ($Worker in $Workers) {
    $OutputPath = Join-Path $DataDir $Worker.output
    
    $CommandLine = @(
        "scripts\search_combos.py",
        "--tickers", "AAPL", "MSFT", "AMZN",
        "--years", "3",
        "--max-combos", "250000",
        "--sampling-method", "random",
        "--seed", $($Worker.seed),
        "--num-workers", "4",
        "--use-ray",
        "--top-n", "10",
        "--refine-top-k", "5",
        "--refine-max-combos", "500",
        "--report-json", $OutputPath
    )
    
    Write-Host "  Worker $($Worker.id): seed=$($Worker.seed), output=$($Worker.output)"
    
    $Job = Start-Job -ScriptBlock {
        param($PythonExe, $ProjectRoot, $Args)
        Set-Location $ProjectRoot
        & $PythonExe @Args
    } -ArgumentList $PythonExe, $ProjectRoot, $CommandLine
    
    $Jobs += @{
        id = $Worker.id
        job = $Job
        output = $OutputPath
    }
}

Write-Host ""
Write-Host "All 4 workers launched. Waiting for completion..." -ForegroundColor Green
Write-Host ""

# Monitor progress
$StartTime = Get-Date
$CompletedCount = 0

while ($CompletedCount -lt 4) {
    $CompletedCount = 0
    foreach ($JobObj in $Jobs) {
        $state = $JobObj.job.State
        if ($state -eq "Completed" -or $state -eq "Failed") {
            $CompletedCount++
        }
    }
    
    $Elapsed = ((Get-Date) - $StartTime).TotalMinutes
    Write-Host "  Progress: $CompletedCount/4 workers complete | Elapsed: $([math]::Round($Elapsed, 1))m" -ForegroundColor Yellow
    
    if ($CompletedCount -lt 4) {
        Start-Sleep -Seconds 10
    }
}

Write-Host ""
Write-Host "Worker Status:" -ForegroundColor Cyan

# Collect results
$AllResults = @()
$SuccessCount = 0

foreach ($JobObj in $Jobs) {
    $Job = $JobObj.job
    $State = $Job.State
    
    if ($State -eq "Completed") {
        Write-Host "  Worker $($JobObj.id): [OK] COMPLETED" -ForegroundColor Green
        $SuccessCount++
        
        # Try to load JSON result
        if (Test-Path $JobObj.output) {
            $Json = Get-Content $JobObj.output | ConvertFrom-Json
            if ($Json -and $Json.Count -gt 0) {
                Write-Host "    Results: $($Json.Count) combos found" -ForegroundColor Gray
                $AllResults += $Json
            }
        }
    }
    else {
        Write-Host "  Worker $($JobObj.id): [FAIL] FAILED" -ForegroundColor Red
        $Output = Receive-Job -Job $Job -ErrorAction SilentlyContinue
        if ($Output) {
            Write-Host "    Error: $($Output | Select-Object -First 1)" -ForegroundColor Gray
        }
    }
}

Write-Host ""
Write-Host "Aggregation:" -ForegroundColor Cyan
Write-Host "  Successful workers: $SuccessCount/4" -ForegroundColor Gray
Write-Host "  Total results collected: $($AllResults.Count)" -ForegroundColor Gray

# Aggregate and rank by score
if ($AllResults.Count -gt 0) {
    $Ranked = $AllResults | Sort-Object { $_._score } -Descending | Select-Object -First 10
    
    $AggOutputPath = Join-Path $DataDir "search_ray_1m_aggregated_$Timestamp.json"
    $Ranked | ConvertTo-Json -Depth 10 | Out-File $AggOutputPath -Encoding UTF8
    
    Write-Host "  [OK] Top-10 aggregated results saved" -ForegroundColor Green
    Write-Host "  Output: $AggOutputPath" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Top Results (by Sharpe):" -ForegroundColor Cyan
    
    $Ranked | ForEach-Object -Begin { $i = 1 } -Process {
        $Sharpe = $_.metrics."Sharpe Ratio"
        $Return = $_.metrics."Total Return (%)"
        $DD = $_.metrics."Max Drawdown (%)"
        $Active = ($_.weights | Get-Member -MemberType NoteProperty).Count
        
        Write-Host "  $i. Sharpe=$Sharpe | Return=$Return | MaxDD=$DD | Active=$Active" -ForegroundColor Gray
        $i++
    }
}

Write-Host ""
Write-Host "Total elapsed time: $([math]::Round(((Get-Date) - $StartTime).TotalMinutes, 1))m" -ForegroundColor Cyan

# Cleanup jobs
Get-Job | Remove-Job -Force -ErrorAction SilentlyContinue
