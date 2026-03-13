# Parallel Monte Carlo search across 4 workers
# Each worker runs 250K samples with different seed
# Total: 1M samples across 15 indicators with 8 local processes per worker

$ErrorActionPreference = "Stop"

# Configuration
$apiKey = '985f143a-6709-4578-b2ee-4d5eaac01330'
$dataDir = 'C:/Users/boris/Documents/GitHub/Financial-Algorithms/simfin_data'
$pythonExe = 'C:/Users/boris/Documents/GitHub/Financial-Algorithms/.venv tradingalgo/Scripts/python.exe'
$searchScript = 'C:/Users/boris/Documents/GitHub/Financial-Algorithms/scripts/search_combos.py'
$tickers = 'AAPL', 'MSFT', 'AMZN'
$years = 3
$components = @(
    'ma_cross', 'sar_stoch', 'stoch_macd', 'rsi', 'bb_rsi', 'rsi_obv_bb', 'adx', 'cci_adx', 'williams_r', 'vwsma', 'macd', 'atr_trend', 'cmf', 'force_index', 'volume_osc'
)
$samplesPerWorker = 250000
$numWorkers = 8
$resultsDir = 'C:/Users/boris/Documents/GitHub/Financial-Algorithms/data/search_results'

Write-Host "=== Starting 4-Worker Parallel Monte Carlo Search ===" -ForegroundColor Cyan
Write-Host "Total samples: $($samplesPerWorker * 4) (1M)" -ForegroundColor Green
Write-Host "Workers per terminal: $numWorkers" -ForegroundColor Green
Write-Host "Components: $($components.Count)" -ForegroundColor Green
Write-Host ""

$jobs = @()
$startTime = Get-Date

# Launch 4 parallel jobs
for ($seed = 1; $seed -le 4; $seed++) {
    $outputFile = "$resultsDir/search_parallel_1m_worker_$seed.json"
    
    $scriptBlock = {
        param($seed, $samplesPerWorker, $numWorkers, $pythonExe, $searchScript, $tickers, $years, $components, $outputFile, $apiKey, $dataDir)
        
        $env:SIMFIN_API_KEY = $apiKey
        $env:SIMFIN_DATA_DIR = $dataDir
        
        $componentsStr = $components -join ' '
        $tickersStr = $tickers -join ' '
        
        Write-Host "Worker ${seed}: Starting (seed=$seed, samples=$samplesPerWorker)" -ForegroundColor Cyan
        $workerStart = Get-Date
        
        & $pythonExe $searchScript `
          --tickers $tickersStr `
          --years $years `
          --weight-grid 1,1.5,2 `
          --components $components `
          --sampling-method random `
          --sample-size $samplesPerWorker `
          --seed $seed `
          --num-workers $numWorkers `
          --report-json $outputFile
        
        $workerElapsed = (Get-Date) - $workerStart
        Write-Host "Worker ${seed}: Complete ($($workerElapsed.TotalSeconds) seconds)" -ForegroundColor Green
    }
    
    $job = Start-Job -ScriptBlock $scriptBlock -ArgumentList @($seed, $samplesPerWorker, $numWorkers, $pythonExe, $searchScript, $tickers, $years, $components, $outputFile, $apiKey, $dataDir) -Name "Worker_$seed"
    $jobs += $job
    Write-Host "Launched Worker $seed (Job ID: $($job.Id))" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "All workers launched. Waiting for completion..." -ForegroundColor Cyan
Write-Host ""

# Monitor progress
$completedCount = 0
while ($completedCount -lt 4) {
    $completedCount = ($jobs | Where-Object State -eq 'Completed').Count
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Progress: $completedCount/4 workers complete" -ForegroundColor Cyan
    Start-Sleep -Seconds 30
}

# Wait for all jobs to finish
$jobs | Wait-Job | Out-Null

$totalElapsed = (Get-Date) - $startTime

Write-Host ""
Write-Host "=== All Workers Complete ===" -ForegroundColor Green
Write-Host "Total elapsed time: $($totalElapsed.TotalMinutes) minutes ($($totalElapsed.TotalSeconds) seconds)" -ForegroundColor Green
Write-Host ""

# Aggregate results
Write-Host "Aggregating results from 4 workers..." -ForegroundColor Cyan

$allResults = @()
for ($seed = 1; $seed -le 4; $seed++) {
    $filePath = "$resultsDir/search_parallel_1m_worker_$seed.json"
    if (Test-Path $filePath) {
        $content = Get-Content $filePath -Raw | ConvertFrom-Json
        $allResults += $content
        Write-Host "  Loaded worker ${seed}: $($content.Count) results" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: Missing results from worker ${seed}" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Total results loaded: $($allResults.Count)" -ForegroundColor Green

# Sort by score and get top 10
$top10 = $allResults | Sort-Object -Property @{Expression={[float]$_._score}; Ascending=$false} | Select-Object -First 10

# Save aggregated top-10
$aggregatedFile = "$resultsDir/search_parallel_1m_aggregated_top10.json"
$top10 | ConvertTo-Json -Depth 10 | Set-Content $aggregatedFile

Write-Host "Saved aggregated top-10 to: $aggregatedFile" -ForegroundColor Green
Write-Host ""
Write-Host "=== Top 10 Results ===" -ForegroundColor Cyan

for ($i = 0; $i -lt $top10.Count; $i++) {
    $result = $top10[$i]
    $sharpe = [float]$result.metrics.'Sharpe Ratio'
    $sortino = [float]$result.metrics.'Sortino Ratio'
    $calmar = [float]$result.metrics.'Calmar Ratio'
    $ret = $result.metrics.'Total Return'
    $dd = $result.metrics.'Max Drawdown'
    $score = [float]$result._score
    
    Write-Host "[$($i+1)] Score=$([math]::Round($score, 2)) Sharpe=$([math]::Round($sharpe, 2)) Sortino=$([math]::Round($sortino, 2)) Calmar=$([math]::Round($calmar, 2)) Return=$ret MaxDD=$dd" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Parallel Search Summary ===" -ForegroundColor Cyan
Write-Host "Total samples: 1M (4 workers × 250k each)" -ForegroundColor Green
Write-Host "Elite time: $([math]::Round($totalElapsed.TotalMinutes, 1)) minutes" -ForegroundColor Green
Write-Host "Indicators: $($components.Count) price+volume" -ForegroundColor Green
Write-Host "Best Sharpe: $([math]::Round([float]$top10[0].metrics.'Sharpe Ratio', 2))" -ForegroundColor Green
Write-Host "Results saved to: $aggregatedFile" -ForegroundColor Green
