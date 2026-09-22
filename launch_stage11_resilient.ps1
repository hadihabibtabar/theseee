# Launch Stage 11 (federated Transformer+MMoE+SSL) as a detached process with
# automatic relaunch+resume.
#
# WHY THIS EXISTS (root cause of the 2026-09-20 crash):
#   This machine's NVIDIA driver sporadically logs nvlddmkm Event ID 153 errors
#   under sustained CUDA load; every observed CUDA crash episode coincides with
#   such an event burst (Stage 8 runs on Sep 17/19, Stage 10B round 9 on
#   Sep 19 ~20:18 local, and the Stage 11 round-1/client_00 crash at
#   01:42-01:43 local on Sep 20). The event kills the CUDA context and the
#   runner dies with "RuntimeError: CUDA error: unknown error". The Stage 11
#   training code itself is healthy (deterministic, leak-free, stable memory —
#   see scripts/diagnose_stage11_cuda.py).
#
# MECHANISM:
#   run_stage11.py writes artifacts/federated/stage11/stage11_latest.pt after
#   every completed round. An interrupted round replays deterministically
#   (fixed seed, per-round sampler epoch, per-client re-seeded RNG), so an
#   automatic "--resume" relaunch loses at most the interrupted round — the
#   same mechanism that let Stage 10B survive its round-9 crash episode.
#
# BEHAVIOR:
#   * refuses to start if a Stage 11 runner is already alive (double-launch
#     guard, same as launch_stage11.ps1),
#   * if artifacts/federated/stage11/stage11_result.json exists -> the
#     experiment is already complete; exits 0 without touching anything,
#   * per-attempt relaunch policy:
#       - stage11_latest.pt exists      -> python scripts/run_stage11.py --resume
#       - best/final exist without it   -> STOP (unexpected state; inspect
#                                          artifacts/federated/stage11 manually;
#                                          never auto-discards outputs)
#       - otherwise                     -> fresh start (crash before the first
#                                          completed round)
#   * does NOT retry integrity/fingerprint failures (a runner log containing
#     FAIL:/VIOLATION: is not transient),
#   * preserves each failed attempt's log as stage11_attempt<N>_log.txt/_err.txt,
#   * sleeps -RetryDelaySeconds between attempts (driver/process cleanup).
#
# USAGE (this script blocks until Stage 11 completes or gives up):
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\launch_stage11_resilient.ps1
#   optional: -MaxAttempts 10 -RetryDelaySeconds 90

param(
    [int]$MaxAttempts = 10,
    [int]$RetryDelaySeconds = 90
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$stageDir = Join-Path $root "artifacts\federated\stage11"
$latest = Join-Path $stageDir "stage11_latest.pt"
$best   = Join-Path $stageDir "stage11_best.pt"
$final  = Join-Path $stageDir "stage11_final_round10.pt"
$result = Join-Path $stageDir "stage11_result.json"

if (Test-Path $result) {
    Write-Output "ALREADY_COMPLETE: $result exists; nothing to do."
    exit 0
}

# Double-launch guard (same convention as launch_stage11.ps1).
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'run_stage11' }
if ($existing) {
    Write-Output ("REFUSED: Stage 11 already running, PID(s): " +
        (($existing | ForEach-Object { $_.ProcessId }) -join ", "))
    exit 1
}

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {

    if ($attempt -gt 1) {
        Move-Item -Force -ErrorAction SilentlyContinue `
            (Join-Path $root "stage11_log.txt") `
            (Join-Path $root ("stage11_attempt" + ($attempt - 1) + "_log.txt"))
        Move-Item -Force -ErrorAction SilentlyContinue `
            (Join-Path $root "stage11_err.txt") `
            (Join-Path $root ("stage11_attempt" + ($attempt - 1) + "_err.txt"))
        Start-Sleep -Seconds $RetryDelaySeconds
    }

    if (Test-Path $latest) {
        $argList = @("scripts/run_stage11.py", "--resume")
        $mode = "resume"
    } elseif ((Test-Path $best) -or (Test-Path $final)) {
        Write-Output ("STOP: stage11_best/final exist but stage11_latest.pt does not; " +
            "inspect $stageDir manually. Nothing was discarded.")
        exit 1
    } else {
        $argList = @("scripts/run_stage11.py")
        $mode = "fresh"
    }

    Write-Output ("ATTEMPT {0}/{1} ({2}): python scripts/run_stage11.py{3} [{4}]" -f `
        $attempt, $MaxAttempts, $mode, `
        ($(if ($mode -eq "resume") { " --resume" } else { "" })), `
        (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

    $p = Start-Process -FilePath "python" `
        -ArgumentList $argList `
        -WorkingDirectory $root `
        -WindowStyle Hidden `
        -RedirectStandardOutput "$root\stage11_log.txt" `
        -RedirectStandardError "$root\stage11_err.txt" `
        -PassThru

    Set-Content -Path "$root\stage11_pid.txt" -Value $p.Id
    $p.WaitForExit()
    $code = $p.ExitCode
    Write-Output ("ATTEMPT {0} exited with code {1} [{2}]" -f `
        $attempt, $code, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

    if ($code -eq 0) {
        if (Test-Path $result) {
            Write-Output "SUCCESS: Stage 11 completed; stage11_result.json written."
        } else {
            Write-Output "STOP: runner exited 0 but stage11_result.json is missing; inspect stage11_log.txt."
        }
        exit 0
    }

    # Integrity/fingerprint failures are not transient: never auto-retry them.
    $logText = ""
    if (Test-Path "$root\stage11_log.txt") {
        $logText = Get-Content "$root\stage11_log.txt" -Raw
    }
    if ($logText -match "VIOLATION:" -or $logText -match "FAIL:") {
        Write-Output "STOP: runner reported an integrity/fingerprint failure; NOT retrying. See stage11_log.txt."
        exit 1
    }

    Write-Output ("Relaunching in {0} s (driver/process cleanup)..." -f $RetryDelaySeconds)
}

Write-Output ("GAVE_UP: no successful completion after {0} attempts. " +
    "Failed-attempt logs preserved as stage11_attempt*_log.txt." -f $MaxAttempts)
exit 1
