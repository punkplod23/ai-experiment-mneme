EXE = .\target\release\leibniz_node.exe
VAULT_D = D:\mneme_data\inference_vault

build:
	cargo build --release

start: build
	powershell -ExecutionPolicy Bypass -Command "Stop-Process -Name leibniz_node -ErrorAction SilentlyContinue; & '$(EXE)' start"

init:
	powershell -ExecutionPolicy Bypass -Command "Get-ChildItem original_books/*.txt | ForEach-Object { Write-Host 'Ingesting: ' $$_.Name -ForegroundColor Cyan; & '$(EXE)' ingest $$_.FullName }"

chat:
	$(EXE) chat

diag:
	$(EXE) diagnostics

clean:
	powershell -ExecutionPolicy Bypass -Command "Stop-Process -Name leibniz_node -ErrorAction SilentlyContinue; if (Test-Path '$(VAULT_D)') { Remove-Item -Recurse -Force '$(VAULT_D)' }"