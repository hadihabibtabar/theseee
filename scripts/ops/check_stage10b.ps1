# Check whether the Stage 10B runner process is alive; print PID + CPU time.
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # repository root (script lives in scripts/ops)
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'run_stage10b' }
if ($procs) {
    foreach ($p in $procs) {
        Write-Output ("RUNNING: PID=" + $p.ProcessId + " started " + $p.CreationDate)
    }
} else {
    Write-Output "NOT_RUNNING"
}
if (Test-Path "$root\stage10b_pid.txt") {
    Write-Output ("PIDFILE: " + (Get-Content "$root\stage10b_pid.txt" -Raw).Trim())
} else {
    Write-Output "PIDFILE: (none)"
}
