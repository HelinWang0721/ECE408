<#
run_all_windows.ps1 -- build and run every MP and every Project convolution
implementation on a local NVIDIA GPU under Windows, and print a PASS/FAIL
summary plus the batch-5000 "Op Time" of each Project implementation.

All build products, logs and the summary go to $OutDir (default D:\Claude\ece408-build).
Nothing is written anywhere else.

Requirements
  * CUDA Toolkit >= 12.8 (Blackwell / RTX PRO 6000 = compute capability 12.0 = sm_120)
  * Visual Studio 2022 (or Build Tools) with "Desktop development with C++"
    -- nvcc on Windows needs cl.exe. The script finds it with vswhere.

Usage (from the repo root, e.g. D:\Claude\ECE408):
  powershell -ExecutionPolicy Bypass -File tools\run_all_windows.ps1
  powershell -ExecutionPolicy Bypass -File tools\run_all_windows.ps1 -Arch sm_120 -Batch 10000
#>
param(
    [string]$OutDir = "D:\Claude\ece408-build",
    [string]$Arch = "sm_120",
    [int]$Batch = 5000,
    [switch]$SkipMPs,
    [switch]$SkipProject
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Summary = Join-Path $OutDir "summary.txt"
"ECE408 run $(Get-Date -Format s)  arch=$Arch  batch=$Batch" | Set-Content $Summary

function Log($msg) { Write-Host $msg; Add-Content $Summary $msg }

# ---------------------------------------------------------------- toolchain
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) { throw "nvcc not found. Install CUDA Toolkit >= 12.8 and reopen the terminal." }
$ver = (& nvcc --version | Select-String "release (\d+)\.(\d+)").Matches[0]
Log ("nvcc release {0}.{1}" -f $ver.Groups[1].Value, $ver.Groups[2].Value)
if ([int]$ver.Groups[1].Value -lt 12 -or ([int]$ver.Groups[1].Value -eq 12 -and [int]$ver.Groups[2].Value -lt 8)) {
    Log "WARNING: sm_120 (Blackwell) needs CUDA 12.8 or newer."
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "cl.exe not found and vswhere missing: install Visual Studio 2022 Build Tools (C++)." }
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
    # import the MSVC environment into this PowerShell session
    cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
        if ($_ -match "^(.*?)=(.*)$") { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
    }
}
& nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | ForEach-Object { Log "GPU: $_" }

# /utf-8: CUDA headers contain non-ASCII characters; without it MSVC warns (C4819)
# on non-UTF-8 code pages such as Chinese Windows (936).
$NvccFlags = @("-std=c++17", "-O3", "-arch=$Arch", "-Xcompiler", "/EHsc,/utf-8", "-Wno-deprecated-gpu-targets")
$NvccExe = (Get-Command nvcc).Source

# Run a native program with stdout+stderr going to $LogFile; returns the exit code.
# (Windows PowerShell 5.1 turns any stderr output of `& exe` into a terminating
#  NativeCommandError under ErrorActionPreference=Stop, e.g. compiler warnings.)
function Invoke-Logged([string]$Exe, [string[]]$ArgList, [string]$LogFile, [string]$WorkDir = $Root, [switch]$Append) {
    $quoted = $ArgList | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }
    $out = "$LogFile.stdout.tmp"; $err = "$LogFile.stderr.tmp"
    $sp = @{ FilePath = $Exe; WorkingDirectory = $WorkDir; NoNewWindow = $true; Wait = $true; PassThru = $true
             RedirectStandardOutput = $out; RedirectStandardError = $err }
    if ($quoted) { $sp.ArgumentList = ($quoted -join ' ') }   # empty -ArgumentList is an error in PS 5.1
    $p = Start-Process @sp
    $text = @(Get-Content $out) + @(Get-Content $err)
    if ($Append) { Add-Content -Path $LogFile -Value $text } else { Set-Content -Path $LogFile -Value $text }
    Remove-Item $out, $err -ErrorAction SilentlyContinue
    return $p.ExitCode
}

# ---------------------------------------------------------------- MPs
if (-not $SkipMPs) {
    Log "`n===== MPs ====="
    foreach ($mp in "MP0","MP1","MP2","MP3","MP4","MP5","MP6","MP7","MP8") {
        $exe = Join-Path $OutDir "$mp.exe"
        $buildLog = Join-Path $OutDir "$mp.build.log"
        $rc = Invoke-Logged $NvccExe ($NvccFlags + @("-I", "$Root\tools\include", "$Root\$mp\template.cu", "-o", $exe)) $buildLog
        if ($rc -ne 0) { Log ("{0,-5} BUILD FAILED (see {1})" -f $mp, $buildLog); continue }
        if ($mp -eq "MP0") {
            $null = Invoke-Logged $exe @() (Join-Path $OutDir "MP0.log"); Log ("{0,-5} ran (device query, see MP0.log)" -f $mp); continue
        }
        # replay the dataset command lines from the MP's run_datasets script
        $script = Get-Content "$Root\$mp\run_datasets" -Raw
        $ids = ([regex]"for i in ([0-9 ]+)").Match($script).Groups[1].Value.Trim() -split "\s+"
        $line = ($script -split "`n" | Where-Object { $_ -match "^\s*\./template" })[0].Trim()
        $pass = 0; $fail = 0
        foreach ($i in $ids) {
            $argLine = $line.Replace('${i}', $i) -replace '^\./template\s*', ''
            $runLog = Join-Path $OutDir "$mp.data$i.log"
            $null = Invoke-Logged $exe ($argLine -split "\s+") $runLog "$Root\$mp"
            if (Select-String -Quiet -Path $runLog -Pattern "WB_RESULT PASS") { $pass++ } else { $fail++; Log "  $mp dataset $i FAILED -> $runLog" }
        }
        Log ("{0,-5} pass={1} fail={2}" -f $mp, $pass, $fail)
    }
}

# ---------------------------------------------------------------- Project
if (-not $SkipProject) {
    Log "`n===== Project convolution (correctness + batch $Batch Op Time) ====="
    $ops = @("base","op1","op2","op3","op4","op5","new-forward")
    for ($id = 0; $id -lt $ops.Count; $id++) {
        $op = $ops[$id]
        $tol = if ($op -eq "op3") { "2e-2" } else { "1e-3" }
        $exe = Join-Path $OutDir "test_$op.exe"
        $buildLog = Join-Path $OutDir "test_$op.build.log"
        $rc = Invoke-Logged $NvccExe ($NvccFlags + @("-I", "$Root\Project\custom", "-DOP_ID=$id", "-DTOL=$tol", "$Root\Project\test\test_ops.cu", "-o", $exe)) $buildLog
        if ($rc -ne 0) { Log ("{0,-12} BUILD FAILED (see {1})" -f $op, $buildLog); continue }
        $runLog = Join-Path $OutDir "test_$op.log"
        $null = Invoke-Logged $exe @() $runLog
        $null = Invoke-Logged $exe @("bench", "$Batch", "3") $runLog -Append
        $ok = -not (Select-String -Quiet -Path $runLog -Pattern "RESULT FAIL|CUDA error")
        # last repetition = warmed-up timing
        $times = Select-String -Path $runLog -Pattern "^(layer\d)\s+B=$Batch .*Op Time\s+([0-9.]+) ms" | Select-Object -Last 2
        $t = ($times | ForEach-Object { "{0}={1}ms" -f $_.Matches[0].Groups[1].Value, $_.Matches[0].Groups[2].Value }) -join "  "
        $sum = ($times | ForEach-Object { [double]$_.Matches[0].Groups[2].Value } | Measure-Object -Sum).Sum
        Log ("{0,-12} {1}  {2}  sum={3:N3}ms  (log: {4})" -f $op, $(if ($ok) { "PASS" } else { "FAIL" }), $t, $sum, $runLog)
    }
}
Log "`nSummary written to $Summary"
