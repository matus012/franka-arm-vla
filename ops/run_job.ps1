# Run one laptop GPU job (113): serialised by ops\GPU_LOCK, hard-killed at 2 h, refused once the laptop total reaches
# 20 h; samples VRAM + free RAM every 2 s and appends a row to ops\gpu_log.csv. Copied from 112 ops\run_job.ps1.
# Usage: powershell -File ops\run_job.ps1 -Job s0a1_grasp -LogFile logs\jobs\s0a1_grasp.log -PyArgs "scripts\s0a1_grasp.py --n 300"
#        -Venv sim (default, .venv) | policy (policy\.venv)
# The job's stdout/stderr go to -LogFile / -LogFile.err. NaN-guard events are read from the last "NAN_GUARD_EVENTS=<n>" line.
# PERUN jobs get rows too (job name perun_<jobid>), written by hand from job-log timestamps.
param(
    [Parameter(Mandatory = $true)][string]$Job,
    [Parameter(Mandatory = $true)][string]$LogFile,
    [Parameter(Mandatory = $true)][string]$PyArgs,
    [ValidateSet("sim", "policy")][string]$Venv = "sim",
    [double]$MaxMinutes = 120
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
. "$root\scripts\uv_env.ps1"
New-Item -ItemType Directory -Force (Split-Path -Parent $LogFile) | Out-Null
$csv = "$root\ops\gpu_log.csv"
if (-not (Test-Path $csv)) { "job,start,end,minutes,peak_vram_mib,min_free_ram_mib,nan_guard_events,exit_code" | Out-File -Encoding utf8 $csv }
if ($MaxMinutes -gt 120) { throw "MaxMinutes > 120: the brief caps a laptop run at 2 h" }

# Budget: laptop total <= 20 h (laptop rows only; perun_* rows are PERUN time).
$used = 0.0
Import-Csv $csv | Where-Object { $_.job -notlike "perun_*" -and $_.minutes } | ForEach-Object { $used += [double]$_.minutes }
if ($used / 60.0 -ge 20.0) { throw "laptop budget exhausted: $([math]::Round($used/60,2)) h of 20 h used" }
$left = 20.0 * 60 - $used
if ($MaxMinutes -gt $left) { $MaxMinutes = $left }

# One GPU job at a time: exclusive lock file, wait until free (a lock older than 3 h is stale and removed).
$lock = "$root\ops\GPU_LOCK"
while ($true) {
    try {
        $fs = [System.IO.File]::Open($lock, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
        $w = New-Object System.IO.StreamWriter($fs); $w.WriteLine("$Job $(Get-Date -Format s) pid $PID"); $w.Close()
        break
    } catch {
        if ((Test-Path $lock) -and ([DateTime]::UtcNow - (Get-Item $lock).LastWriteTimeUtc).TotalHours -gt 3) { Remove-Item $lock -Force; continue }
        Write-Output "[run_job] GPU busy ($(Get-Content $lock -ErrorAction SilentlyContinue)); waiting"
        Start-Sleep -Seconds 20
    }
}

try {
    $start = Get-Date  # csv timestamps stay local time; durations are UTC (the 2026-10-25 DST change gave -53.59 min)
    $t0 = [DateTime]::UtcNow
    $exe = if ($Venv -eq "policy") { "$root\policy\.venv\Scripts\python.exe" } else { "$root\.venv\Scripts\python.exe" }
    $rest = "-u $PyArgs"
    $errLog = "$LogFile.err"
    $p = Start-Process -FilePath $exe -ArgumentList $rest -NoNewWindow -PassThru -RedirectStandardOutput $LogFile -RedirectStandardError $errLog
    $null = $p.Handle  # cache the handle so ExitCode is populated
    $peakVram = 0; $minFree = [int]::MaxValue; $abort = ""
    while (-not $p.HasExited) {
        $v = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        $f = [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
        if ($v -gt $peakVram) { $peakVram = $v }
        if ($f -lt $minFree) { $minFree = $f }
        if ($f -lt 1024) { $abort = "RAM_ABORT: free RAM $f MiB < 1024 MiB" }  # 112 consultant rule
        if ((([DateTime]::UtcNow) - $t0).TotalMinutes -gt $MaxMinutes) { $abort = "TIME_ABORT: run passed $MaxMinutes min" }
        if ($abort) {
            & taskkill /PID $p.Id /T /F | Out-Null
            Write-Output "[run_job] $abort, job killed"
            break
        }
        Start-Sleep -Seconds 2
    }
    $p.WaitForExit()
    $code = if ($abort -like "RAM*") { -99 } elseif ($abort) { -98 } else { $p.ExitCode }
    $end = Get-Date
    $mins = [math]::Round(([DateTime]::UtcNow - $t0).TotalMinutes, 2)
    $nan = ""
    $hit = Select-String -Path $LogFile, $errLog -Pattern "NAN_GUARD_EVENTS=(\d+)" -ErrorAction SilentlyContinue | Select-Object -Last 1
    if ($hit) { $nan = $hit.Matches[0].Groups[1].Value }
    "$Job,$($start.ToString('s')),$($end.ToString('s')),$mins,$peakVram,$minFree,$nan,$code" | Out-File -Append -Encoding utf8 $csv
    $tot = [math]::Round(($used + $mins) / 60.0, 2)
    Write-Output "[run_job] $Job exit=$code minutes=$mins peakVRAM=${peakVram}MiB minFreeRAM=${minFree}MiB nan=$nan laptop_total_h=$tot"
} finally {
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
}
exit $code
