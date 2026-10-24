# Public-data container check. Never prints credentials or parses .env values.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
    $runDir = Join-Path $repo "runs/docker-$stamp"
    if (Test-Path $runDir) { throw 'Run directory already exists; retry in a second.' }
    $inputDir = Join-Path $runDir 'input'
    $artifactDir = Join-Path $runDir 'artifacts'
    $outputDir = Join-Path $runDir 'output'
    New-Item -ItemType Directory -Path $inputDir, $artifactDir, $outputDir | Out-Null
    & python src/prepare_inputs.py --train data/public_train.csv --test data/public_test.csv --out-dir $inputDir
    if ($LASTEXITCODE -ne 0) { throw "Input preparation failed ($LASTEXITCODE)" }

    $envArgs = @('-e', 'GEMINI_API_KEY')
    $envPath = Join-Path $repo '.env'
    if (Test-Path -LiteralPath $envPath -PathType Leaf) {
        $envArgs = @('--env-file', $envPath)
    }
    & docker build -t rulejudge .
    if ($LASTEXITCODE -ne 0) { throw "Docker build failed ($LASTEXITCODE)" }
    & docker run --rm @envArgs `
        --mount "type=bind,source=$inputDir,target=/input,readonly" `
        --mount "type=bind,source=$artifactDir,target=/artifacts" rulejudge adapt
    if ($LASTEXITCODE -ne 0) { throw "Docker adapt failed ($LASTEXITCODE)" }
    & docker run --rm @envArgs `
        --mount "type=bind,source=$inputDir,target=/input,readonly" `
        --mount "type=bind,source=$artifactDir,target=/artifacts,readonly" `
        --mount "type=bind,source=$outputDir,target=/output" rulejudge eval
    if ($LASTEXITCODE -ne 0) { throw "Docker eval failed ($LASTEXITCODE)" }
    & python src/evaluate.py --test data/public_test.csv --predictions (Join-Path $outputDir 'predictions.csv') --out (Join-Path $runDir 'score.json')
    if ($LASTEXITCODE -ne 0) { throw "Scoring failed ($LASTEXITCODE)" }
    Write-Host "Public Docker check complete: $runDir"
}
finally {
    Pop-Location
}
