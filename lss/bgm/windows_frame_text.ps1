param(
  [Parameter(Mandatory=$true)][string]$ImagePath,
  [Parameter(Mandatory=$true)][string]$ExpectedSha256,
  [Parameter(Mandatory=$true)][string]$Language
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch { }
$result = [ordered]@{
  status = 'UNAVAILABLE'; reason = 'runtime_failure'; source_sha256 = $null
  powershell_version = $PSVersionTable.PSVersion.ToString(); language = $Language
  width = $null; height = $null; recognized_text = ''; lines = @(); raw_lines = @()
  phase = 'startup'; error_type = $null; error_message = $null
}
$stream = $null; $bitmap = $null
try {
  if ($PSVersionTable.PSVersion.Major -ne 5 -or $PSVersionTable.PSVersion.Minor -ne 1) {
    $result.reason = 'unsupported_powershell_version'
    throw 'unsupported runtime'
  }
  $result.phase = 'source_hash'
  $resolved = [IO.Path]::GetFullPath($ImagePath)
  $actual = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash.ToLowerInvariant()
  $result.source_sha256 = $actual
  if ($actual -ne $ExpectedSha256.ToLowerInvariant()) {
    $result.reason = 'source_hash_mismatch'
    throw 'source integrity mismatch'
  }
  $result.phase = 'runtime_setup'
  Add-Type -AssemblyName System.Runtime.WindowsRuntime
  $null = [Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
  $null = [Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]
  $null = [Windows.Graphics.Imaging.BitmapDecoder,Windows.Foundation,ContentType=WindowsRuntime]
  $null = [Windows.Globalization.Language,Windows.Globalization,ContentType=WindowsRuntime]
  function Await-WinRT($operation, [Type]$type) {
    $method = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
      $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and
      $_.GetGenericArguments().Count -eq 1 -and $_.GetParameters().Count -eq 1 -and
      $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
    } | Select-Object -First 1
    if ($null -eq $method) { throw 'WinRT async bridge unavailable' }
    $task = $method.MakeGenericMethod($type).Invoke($null, @($operation))
    if (-not $task.Wait(35000)) { throw 'WinRT async operation timed out' }
    $task.GetAwaiter().GetResult()
  }
  $result.phase = 'language'
  $winrtLanguage = New-Object Windows.Globalization.Language $Language
  $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($winrtLanguage)
  if ($null -eq $engine) {
    $result.reason = 'language_unavailable'
    throw 'requested OCR language unavailable'
  }
  $result.phase = 'decode'
  $file = Await-WinRT ([Windows.Storage.StorageFile]::GetFileFromPathAsync($resolved)) ([Windows.Storage.StorageFile])
  $stream = Await-WinRT ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
  $decoder = Await-WinRT ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
  $bitmap = Await-WinRT ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
  $result.width = [uint32]$bitmap.PixelWidth
  $result.height = [uint32]$bitmap.PixelHeight
  $result.phase = 'recognize'
  $recognized = Await-WinRT ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
  $result.recognized_text = [string]$recognized.Text
  $result.lines = @($recognized.Lines | ForEach-Object { [string]$_.Text })
  $result.raw_lines = @($recognized.Lines | ForEach-Object {
    $line = $_
    [ordered]@{
      text = [string]$line.Text
      words = @($line.Words | ForEach-Object {
        [ordered]@{ text = [string]$_.Text; box = [ordered]@{
          x = [double]$_.BoundingRect.X; y = [double]$_.BoundingRect.Y
          width = [double]$_.BoundingRect.Width; height = [double]$_.BoundingRect.Height
        } }
      })
    }
  })
  $result.status = 'OK'; $result.reason = $null
} catch {
  if ($result.reason -eq 'runtime_failure') { $result.reason = 'runtime_failure' }
  $result.error_type = $_.Exception.GetType().Name
  $message = [string]$_.Exception.Message
  if ($resolved) { $message = $message.Replace($resolved, '[image]') }
  if ($message.Length -gt 500) { $message = $message.Substring(0, 500) }
  $result.error_message = $message
} finally {
  if ($null -ne $bitmap) { try { $bitmap.Dispose() } catch { } }
  if ($null -ne $stream) { try { $stream.Dispose() } catch { } }
}
$result | ConvertTo-Json -Depth 12 -Compress
