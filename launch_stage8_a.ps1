# Detached launcher for the Stage 8 Experiment A run (Windows PowerShell).
$root = $PSScriptRoot
Set-Location $root
$stdout = Join-Path $root "stage8_experiment_a_log.txt"
$stderr = Join-Path $root "stage8_experiment_a_err.txt"
$p = Start-Process -FilePath "python" `
    -ArgumentList "-u", "scripts/run_stage8_experiment.py", "--experiment", "A" `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
    -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 15
Write-Output ("PID: " + $p.Id)
Write-Output ("HasExited: " + $p.HasExited)
if (Test-Path $stdout) { Write-Output "--- log tail ---"; Get-Content $stdout -Tail 8 }
if (Test-Path $stderr) { $e = Get-Content $stderr -Tail 3; if ($e) { Write-Output "--- stderr tail ---"; $e } }
