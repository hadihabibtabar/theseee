# Check whether the Stage 11 runner process is alive; print PID + start time.
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # repository root (script lives in scripts/ops)
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'run_stage11' }
if ($procs) {
    foreach ($p in $procs) {
        Write-Output ("RUNNING: PID=" + $p.ProcessId + " started " + $p.CreationDate)
    }
} else {
    Write-Output "NOT_RUNNING"
}
if (Test-Path "$root\stage11_pid.txt") {
    Write-Output ("PIDFILE: " + (Get-Content "$root\stage11_pid.txt" -Raw).Trim())
} else {
    Write-Output "PIDFILE: (none)"
}
