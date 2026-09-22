# Launch Stage 11 (federated Transformer+MMoE+SSL) as a detached process.
# The process survives the launching shell/session ending.

$ErrorActionPreference = "Stop"

# Double-launch guard: refuse if a Stage 11 runner is already running.
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'run_stage11' }
if ($existing) {
    Write-Output ("REFUSED: Stage 11 already running, PID(s): " +
        (($existing | ForEach-Object { $_.ProcessId }) -join ", "))
    exit 1
}

$p = Start-Process -FilePath "python" `
    -ArgumentList "scripts/run_stage11.py" `
    -WorkingDirectory $PSScriptRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput "$PSScriptRoot\stage11_log.txt" `
    -RedirectStandardError "$PSScriptRoot\stage11_err.txt" `
    -PassThru

# Give Python + CUDA/torch import (~90 s on this machine) time to initialize
# before the liveness check, so a healthy start is not misreported as failed.
Start-Sleep -Seconds 100

if ($p.HasExited) {
    Write-Output ("FAILED: process exited during startup (exit code " + $p.ExitCode + ")")
    Write-Output "See stage11_err.txt / stage11_log.txt for details."
    exit 1
}

Set-Content -Path "$PSScriptRoot\stage11_pid.txt" -Value $p.Id
Write-Output "STARTED: PID=$($p.Id)"
Write-Output "LOG:     $PSScriptRoot\stage11_log.txt"
Write-Output "ERRLOG:  $PSScriptRoot\stage11_err.txt"
Write-Output "PIDFILE: $PSScriptRoot\stage11_pid.txt"
