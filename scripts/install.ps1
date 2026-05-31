# Установщик TranscribeRU для Windows: ставит Python/ffmpeg/git через winget,
# клонирует проект, создаёт venv, ставит зависимости и кладёт ярлык на стол.
# Запуск: двойной клик по install.bat, либо в PowerShell:
#   iwr -useb https://raw.githubusercontent.com/tony-kin-dev/TranscribeRU/main/scripts/install.ps1 | iex

$RepoUrl   = if ($env:REPO_URL) { $env:REPO_URL } else { 'https://github.com/tony-kin-dev/TranscribeRU.git' }
# Папка должна быть БЕЗ кириллицы: libtorch/sentencepiece (C++) не открывают
# не-ASCII пути. LOCALAPPDATA у русскоязычных сам бывает с кириллицей, поэтому
# дефолт — фиксированная ASCII-папка в корне диска.
$Dest      = if ($env:TRANSCRIBE_RU_DIR) { $env:TRANSCRIBE_RU_DIR } else { 'C:\TranscribeRU' }
$CacheDir  = if ($env:GIGAAM_CACHE_DIR)  { $env:GIGAAM_CACHE_DIR }  else { 'C:\gigaam_cache' }

function Say($m)  { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "`n$m" -ForegroundColor Red; Read-Host 'Нажмите Enter для выхода'; exit 1 }

# 1. winget ---------------------------------------------------------------
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
  Fail @"
Не найден winget (App Installer) — без него автоустановка невозможна.
Установите App Installer из Microsoft Store: https://aka.ms/getwinget
Затем запустите этот скрипт снова.
"@
}

# 2. Системные зависимости (winget сам не падает, если уже стоит) ----------
Say 'Устанавливаю Python 3.12, ffmpeg и git через winget...'
winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
winget install -e --id Gyan.FFmpeg        --silent --accept-package-agreements --accept-source-agreements
winget install -e --id Git.Git            --silent --accept-package-agreements --accept-source-agreements

# Подтянуть свежий PATH в текущую сессию (winget его не обновляет на лету)
$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' +
            [Environment]::GetEnvironmentVariable('Path','User')

# 3. Найти git и Python 3.12 (не полагаясь на устаревший PATH сессии) ------
$git = (Get-Command git -ErrorAction SilentlyContinue).Source
if (-not $git) { $git = "$env:ProgramFiles\Git\cmd\git.exe" }
if (-not (Test-Path $git)) { Fail "git не найден. Установите вручную: https://git-scm.com/downloads" }

$pyExe = $null; $pyArgs = @()
$cands = @(
  "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
  "$env:ProgramFiles\Python312\python.exe"
)
foreach ($c in $cands) { if (Test-Path $c) { $pyExe = $c; break } }
if (-not $pyExe -and (Get-Command py -ErrorAction SilentlyContinue)) { $pyExe = 'py'; $pyArgs = @('-3.12') }
if (-not $pyExe) { Fail "Python 3.12 не найден. Скачайте: https://www.python.org/downloads/" }

# 4. Код проекта ----------------------------------------------------------
if (Test-Path (Join-Path $Dest '.git')) {
  Say "Обновляю существующую установку в $Dest..."
  & $git -C $Dest pull --ff-only
} else {
  Say "Скачиваю проект в $Dest..."
  New-Item -ItemType Directory -Force -Path (Split-Path $Dest) | Out-Null
  & $git clone $RepoUrl $Dest
  if ($LASTEXITCODE -ne 0) { Fail "Не удалось клонировать $RepoUrl" }
}

# 5. Виртуальное окружение и зависимости ----------------------------------
Say 'Создаю окружение и ставлю зависимости (первый раз — несколько минут)...'
& $pyExe @pyArgs -m venv "$Dest\.venv"
$venvPy = "$Dest\.venv\Scripts\python.exe"
& $venvPy -m pip install --quiet --upgrade pip
# БЕЗ -e: editable-режим пишет .pth в ANSI, и Python в UTF-8 его игнорирует
# на путях с кириллицей → ModuleNotFoundError.
& $venvPy -m pip install $Dest
if ($LASTEXITCODE -ne 0) { Fail "Не удалось установить зависимости. Проверьте интернет и запустите снова." }

# 6. Кэш модели в ASCII-папке + предзагрузка с ретраями -------------------
Say 'Готовлю папку кэша модели и докачиваю веса...'
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
# чтобы приложение всегда брало ASCII-кэш — пишем переменную окружения (User)
[Environment]::SetEnvironmentVariable('GIGAAM_CACHE_DIR', $CacheDir, 'User')
$env:GIGAAM_CACHE_DIR = $CacheDir

# Предзагружаем веса дефолтной модели — urllib в gigaam без ретраев и таймаута
# часто рвётся на ~430 МБ. Качаем с ретраями; пропускаем, если уже есть.
$ProgressPreference = 'SilentlyContinue'
$cdn = 'https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM'
$files = @('v3_e2e_rnnt.ckpt', 'v3_e2e_rnnt_tokenizer.model')
foreach ($f in $files) {
  $out = Join-Path $CacheDir $f
  if (Test-Path $out) { continue }
  try {
    Invoke-WebRequest -Uri "$cdn/$f" -OutFile $out -MaximumRetryCount 3 -RetryIntervalSec 5
  } catch {
    Write-Host "Не удалось предзагрузить $f — модель скачается при первом запуске." -ForegroundColor Yellow
  }
}

# 7. Ярлык на Рабочем столе -----------------------------------------------
Say 'Создаю ярлык на Рабочем столе...'
$desktop = [Environment]::GetFolderPath('Desktop')
$lnk = Join-Path $desktop 'TranscribeRU.lnk'
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($lnk)
$sc.TargetPath       = "$Dest\.venv\Scripts\pythonw.exe"   # pythonw — без чёрного окна консоли
$sc.Arguments        = '-m transcribe_ru.gui'
$sc.WorkingDirectory = $Dest
$sc.Save()

Write-Host "`nГотово! Ярлык TranscribeRU на Рабочем столе — двойной клик запускает окно." -ForegroundColor Green
if ($Host.Name -eq 'ConsoleHost') { Read-Host 'Нажмите Enter для выхода' }
