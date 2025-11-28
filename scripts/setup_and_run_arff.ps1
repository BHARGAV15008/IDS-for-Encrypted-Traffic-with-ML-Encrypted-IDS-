param(
    [string]$A1 = "01_Data/Scenario A1-ARFF",
    [string]$A2 = "01_Data/Scenario A2-ARFF",
    [string]$B  = "01_Data/Scenario B-ARFF",
    [string]$Req = "requirements-minimal.txt"
)

Write-Host "Creating virtual environment..." -ForegroundColor Cyan
python -m venv env
& .\env\Scripts\Activate.ps1
python -m pip install --upgrade pip

if (Test-Path $Req) {
    Write-Host "Installing dependencies from $Req ..." -ForegroundColor Cyan
    pip install -r $Req
} else {
    Write-Host "Installing dependencies from requirements.txt ..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host "Running ARFF combined training..." -ForegroundColor Cyan
python 04_Source_Code\cli\p22.py arff-combine-train --a1 $A1 --a2 $A2 --b $B


