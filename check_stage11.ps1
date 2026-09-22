# Check whether the Stage 11 runner process is alive; print PID + start time.
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'run_stage11' }
if ($procs) {
    foreach ($p in $procs) {
        Write-Output ("RUNNING: PID=" + $p.ProcessId + " started " + $p.CreationDate)
    }
} else {
    Write-Output "NOT_RUNNING"
}
if (Test-Path "$PSScriptRoot\stage11_pid.txt") {
    Write-Output ("PIDFILE: " + (Get-Content "$PSScriptRoot\stage11_pid.txt" -Raw).Trim())
} else {
    Write-Output "PIDFILE: (none)"
}
