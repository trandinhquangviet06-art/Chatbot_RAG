# Script zip project de upload len Colab/Kaggle
# Bo qua: venv, venv312, .git, models (file GGUF nang), __pycache__

$sourceDir = "d:\Chatbot_RAG"
$outputZip = "d:\Chatbot_RAG_upload.zip"
$excludeDirs = @("venv", "venv312", ".git", "models", "__pycache__", ".agents")

Write-Host "Dang zip project..." -ForegroundColor Cyan

# Xoa zip cu neu co
if (Test-Path $outputZip) { Remove-Item $outputZip -Force }

# Lay danh sach item can zip (bo qua exclude)
$items = Get-ChildItem $sourceDir |
    Where-Object { $_.Name -notin $excludeDirs }

# Tao zip tu tung item
$items | ForEach-Object {
    Compress-Archive -Path $_.FullName -DestinationPath $outputZip -Update
}

$size = (Get-Item $outputZip).Length / 1MB
Write-Host "Da tao: $outputZip" -ForegroundColor Green
Write-Host "Kich thuoc: $([math]::Round($size, 1)) MB" -ForegroundColor Green
Write-Host "Upload file nay len Google Drive > MyDrive >" -ForegroundColor Yellow
