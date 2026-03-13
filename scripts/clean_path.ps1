Set-StrictMode -Version Latest
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$backupDir = Join-Path $env:USERPROFILE 'PathBackups'
if (-not (Test-Path $backupDir)) {
    New-Item -ItemType Directory -Path $backupDir | Out-Null
}
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
$userBackup = Join-Path $backupDir "user_Path_$ts.txt"
$machineBackup = Join-Path $backupDir "machine_Path_$ts.txt"
Set-Content -Path $userBackup -Value $userPath
Set-Content -Path $machineBackup -Value $machinePath

function Get-QuotedEntries([string]$path) {
    if ([string]::IsNullOrEmpty($path)) { return @() }
    return $path -split ';' | Where-Object { $_ -match '"' }
}

function Remove-Quotes([string]$path) {
    if ([string]::IsNullOrEmpty($path)) { return $path }
    return (($path -split ';') | ForEach-Object { $_.Replace('"', '') }) -join ';'
}

$userHasQuotes = Get-QuotedEntries $userPath
$machineHasQuotes = Get-QuotedEntries $machinePath

Write-Host "User Path entries with quotes:"
if ($userHasQuotes) {
    $userHasQuotes | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host '  (none)'
}

Write-Host "Machine Path entries with quotes:"
if ($machineHasQuotes) {
    $machineHasQuotes | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host '  (none)'
}

$cleanUser = Remove-Quotes $userPath
$cleanMachine = Remove-Quotes $machinePath

if ($cleanUser -ne $userPath) {
    [Environment]::SetEnvironmentVariable('Path', $cleanUser, 'User')
    Write-Host "Updated User Path (quotes removed). Backup saved to $userBackup"
} else {
    Write-Host 'User Path unchanged (no quotes found).'
}

try {
    if ($cleanMachine -ne $machinePath) {
        [Environment]::SetEnvironmentVariable('Path', $cleanMachine, 'Machine')
        Write-Host "Updated Machine Path (quotes removed). Backup saved to $machineBackup"
    } else {
        Write-Host 'Machine Path unchanged (no quotes found).'
    }
} catch {
    Write-Warning "Failed to update Machine Path (need admin?): $($_.Exception.Message)"
}
