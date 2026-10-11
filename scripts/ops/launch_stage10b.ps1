# Launch Stage 10B (10-round federated baseline) as a detached process.
# The process survives the launching shell/session ending.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # repository root (script lives in scripts/ops)

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
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput "$root\stage10b_log.txt" `
    -RedirectStandardError "$root\stage10b_err.txt" `
    -PassThru

Start-Sleep -Seconds 3

if ($p.HasExited) {
    Write-Output ("FAILED: process exited immediately with code " + $p.ExitCode)
    exit 1
}

Set-Content -Path "$root\stage10b_pid.txt" -Value $p.Id
Write-Output "STARTED: PID=$($p.Id)"
Write-Output "LOG:     $root\stage10b_log.txt"
Write-Output "ERRLOG:  $root\stage10b_err.txt"
Write-Output "PIDFILE: $root\stage10b_pid.txt"
