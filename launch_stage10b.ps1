# Launch Stage 10B (10-round federated baseline) as a detached process.
# The process survives the launching shell/session ending.

$ErrorActionPreference = "Stop"

# Double-launch guard: refuse if a Stage 10B runner is already running.
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'run_stage10b' }
if ($existing) {
    Write-Output ("REFUSED: Stage 10B already running, PID(s): " +
        ($existing | ForEach-Object { $_.ProcessId }) -join ", ")
    exit 1
}

$p = Start-Process -FilePath "python" `
    -ArgumentList "scripts/run_stage10b.py" `
    -WorkingDirectory $PSScriptRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput "$PSScriptRoot\stage10b_log.txt" `
    -RedirectStandardError "$PSScriptRoot\stage10b_err.txt" `
    -PassThru

Start-Sleep -Seconds 3

if ($p.HasExited) {
    Write-Output ("FAILED: process exited immediately with code " + $p.ExitCode)
    exit 1
}

Set-Content -Path "$PSScriptRoot\stage10b_pid.txt" -Value $p.Id
Write-Output "STARTED: PID=$($p.Id)"
Write-Output "LOG:     $PSScriptRoot\stage10b_log.txt"
Write-Output "ERRLOG:  $PSScriptRoot\stage10b_err.txt"
Write-Output "PIDFILE: $PSScriptRoot\stage10b_pid.txt"
