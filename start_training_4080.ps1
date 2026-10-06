param(
    [string]$CodeDirectory = 'D:\pyj\PhaseNet++\_训练代码_NPZ_HDF5_20261006\PhaseNet++\_训练代码_NPZ_HDF5_20261006',
    [string]$PythonExecutable = 'C:\Users\X1\anaconda3\envs\dl\python.exe',
    [string]$TrainHdf5 = 'D:\pyj\70w条diting数据_dist500_无信噪比筛选\diting_train.hdf5',
    [string]$ValidationHdf5 = 'D:\pyj\70w条diting数据_dist500_无信噪比筛选\diting_val.hdf5',
    [string]$OutputRoot = 'D:\pyj\output',
    [string]$RunDirectory = '',
    [ValidateRange(1, 65536)][int]$BatchSize = 128,
    [ValidateRange(2, 100000)][int]$Epochs = 100,
    [ValidateRange(0, 128)][int]$Workers = 8,
    [switch]$CheckOnly,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath (Join-Path $CodeDirectory 'train.py') -PathType Leaf)) {
    $CodeDirectory = $CodeDirectory.Replace('\_', '_')
}
$trainScript = Join-Path $CodeDirectory 'train.py'
$environmentCheck = Join-Path $CodeDirectory 'scripts\check_environment.py'
foreach ($path in @($PythonExecutable, $trainScript, $environmentCheck, $TrainHdf5, $ValidationHdf5)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file missing: $path"
    }
}
if ([string]::IsNullOrWhiteSpace($RunDirectory)) {
    $RunDirectory = Join-Path $OutputRoot ('phasenet_70w_bs' + $BatchSize + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
if (Test-Path -LiteralPath $RunDirectory) {
    throw "Output directory already exists. Choose a new directory: $RunDirectory"
}
$trainingArguments = @(
    $trainScript,
    '--model', 'phasenet', '--backbone', 'unet', '--format', 'h5',
    '--hdf5-file', $TrainHdf5, '--test-hdf5-file', $ValidationHdf5,
    '--device', 'cuda', '--sampling-rate', '100',
    '--batch-size', "$BatchSize", '--workers', "$Workers", '--epochs', "$Epochs",
    '--opt', 'adamw', '--lr', '0.001', '--weight-decay', '0.0001',
    '--lr-scheduler', 'cosineannealinglr', '--lr-warmup-epochs', '1',
    '--lr-warmup-method', 'linear', '--lr-warmup-decay', '0.01', '--lr-min', '0',
    '--disable-snr-filter', '--disable-plot', '--early-stopping-patience', '0',
    '--validate-every', '2',
    '--compile',
    '--output-dir', $RunDirectory
)
Write-Host "Code: $CodeDirectory"
Write-Host "Python: $PythonExecutable"
Write-Host "Train: $TrainHdf5"
Write-Host "Validation: $ValidationHdf5"
Write-Host "Output: $RunDirectory"
if ($DryRun) {
    $trainingArguments
    return
}
& $PythonExecutable $environmentCheck --smoke --device cuda
if ($LASTEXITCODE -ne 0) {
    throw "Environment/GPU check failed: $LASTEXITCODE"
}
if ($CheckOnly) { return }
Push-Location -LiteralPath $CodeDirectory
try {
    & $PythonExecutable @trainingArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Training exited with code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
