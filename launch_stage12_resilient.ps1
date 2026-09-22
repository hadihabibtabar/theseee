# Launch a Stage 12 federated cell (scripts/run_stage12_federated.py) as a
# detached process with automatic relaunch+resume.
#
# Mirrors launch_stage11_resilient.ps1 (same root cause: this machine's NVIDIA
# driver sporadically logs nvlddmkm Event ID 153 errors under sustained CUDA
# load, killing the CUDA context; an interrupted round replays deterministically
# from <prefix>_latest.pt, losing at most the interrupted round).
#
# BEHAVIOR:
#   * refuses to start if this cell's runner is already alive (double-launch
#     guard),
#   * if <prefix>_result.json exists -> the cell is already complete; exits 0
#     without touching anything,
#   * per-attempt relaunch policy:
#       - <prefix>_latest.pt exists     -> --resume
#       - best/final exist without it   -> STOP (unexpected state; inspect
#                                          manually; never auto-discards)
#       - otherwise                     -> fresh start
#   * does NOT retry integrity/fingerprint failures (VIOLATION:/FAILED: in the
#     log is not transient),
#   * preserves each failed attempt's log as <prefix>_attempt<N>_log/_err.txt,
#   * writes the python runner PID to <prefix>_pid.txt each attempt,
#   * sleeps -RetryDelaySeconds between attempts (driver/process cleanup).
#
# USAGE (blocks until the cell completes or gives up; start THIS SCRIPT detached):
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\launch_stage12_resilient.ps1 `
#       -Experiment stage12_fed_nossl_alpha_1_0_seed42

param(
    [string]$Experiment = "stage12_fed_nossl_alpha_1_0_seed42",
    [int]$MaxAttempts = 10,
    [int]$RetryDelaySeconds = 90
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$stageDir = Join-Path $root ("artifacts\federated\stage12\" + $Experiment)
$latest = Join-Path $stageDir ($Experiment + "_latest.pt")
$best   = Join-Path $stageDir ($Experiment + "_best.pt")
$final  = Join-Path $stageDir ($Experiment + "_final_round10.pt")
$result = Join-Path $stageDir ($Experiment + "_result.json")
$log    = Join-Path $root ($Experiment + "_log.txt")
$errlog = Join-Path $root ($Experiment + "_err.txt")
$pidfile = Join-Path $root ($Experiment + "_pid.txt")

if (Test-Path $result) {
    Write-Output "ALREADY_COMPLETE: $result exists; nothing to do."
    exit 0
}

# Double-launch guard (same convention as the Stage 11 launcher).
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'run_stage12_federated' -and $_.CommandLine -match $Experiment }
if ($existing) {
    Write-Output ("REFUSED: " + $Experiment + " already running, PID(s): " +
        (($existing | ForEach-Object { $_.ProcessId }) -join ", "))
    exit 1
}

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {

    if ($attempt -gt 1) {
        Move-Item -Force -ErrorAction SilentlyContinue `
            $log (Join-Path $root ($Experiment + "_attempt" + ($attempt - 1) + "_log.txt"))
        Move-Item -Force -ErrorAction SilentlyContinue `
            $errlog (Join-Path $root ($Experiment + "_attempt" + ($attempt - 1) + "_err.txt"))
        Start-Sleep -Seconds $RetryDelaySeconds
    }

    if (Test-Path $latest) {
        $argList = @("scripts/run_stage12_federated.py", "--experiment", $Experiment, "--resume")
        $mode = "resume"
    } elseif ((Test-Path $best) -or (Test-Path $final)) {
        Write-Output ("STOP: " + $Experiment + " best/final exist but the latest checkpoint " +
            "does not; inspect $stageDir manually. Nothing was discarded.")
        exit 1
    } else {
        $argList = @("scripts/run_stage12_federated.py", "--experiment", $Experiment)
        $mode = "fresh"
    }

    Write-Output ("ATTEMPT {0}/{1} ({2}) [{3}]" -f `
        $attempt, $MaxAttempts, $mode, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

    $p = Start-Process -FilePath "python" `
        -ArgumentList $argList `
        -WorkingDirectory $root `
        -WindowStyle Hidden `
        -RedirectStandardOutput $log `
        -RedirectStandardError $errlog `
        -PassThru

    Set-Content -Path $pidfile -Value $p.Id
    $p.WaitForExit()
    $code = $p.ExitCode
    Write-Output ("ATTEMPT {0} exited with code {1} [{2}]" -f `
        $attempt, $code, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

    if ($code -eq 0) {
        if (Test-Path $result) {
            Write-Output ("SUCCESS: " + $Experiment + " completed; result JSON written.")
        } else {
            Write-Output "STOP: runner exited 0 but the result JSON is missing; inspect the log."
        }
        exit 0
    }

    # Integrity/fingerprint failures are not transient: never auto-retry them.
    $logText = ""
    if (Test-Path $log) {
        $logText = Get-Content $log -Raw
    }
    if ($logText -match "VIOLATION:" -or $logText -match "STAGE 12 FAILED:") {
        Write-Output "STOP: runner reported an integrity/fingerprint failure; NOT retrying. See the log."
        exit 1
    }

    Write-Output ("Relaunching in {0} s (driver/process cleanup)..." -f $RetryDelaySeconds)
}

Write-Output ("GAVE_UP: no successful completion after {0} attempts. " +
    "Failed-attempt logs preserved as " + $Experiment + "_attempt*_log.txt." -f $MaxAttempts)
exit 1
