param(
    [Parameter(Mandatory=$true)][string]$DataDirectory,
    [string]$PythonExecutable = 'python',
    [string]$OutputDirectory = '',
    [int]$BatchSize = 8,
    [int]$Epochs = 100,
    [int]$Workers = 4,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$launcher = Join-Path (Split-Path -Parent $PSScriptRoot) 'run_training.py'
$arguments = @($launcher, '--format', 'h5',
    '--train-list', (Join-Path $DataDirectory 'train.txt'),
    '--val-list', (Join-Path $DataDirectory 'val.txt'),
    '--train-hdf5', (Join-Path $DataDirectory 'train.hdf5'),
    '--val-hdf5', (Join-Path $DataDirectory 'val.hdf5'),
    '--batch-size', "$BatchSize", '--epochs', "$Epochs", '--workers', "$Workers")
if ($OutputDirectory) { $arguments += @('--output-dir', $OutputDirectory) }
if ($DryRun) { $arguments += '--dry-run' }
& $PythonExecutable @arguments
if ($LASTEXITCODE -ne 0) { throw "Training launcher failed: $LASTEXITCODE" }
