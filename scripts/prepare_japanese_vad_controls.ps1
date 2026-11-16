$ErrorActionPreference = 'Stop'
$vadControlRoot = Join-Path $PSScriptRoot '../.cache/local-ai/native-cpu-vad-20261028/japanese'
if (Test-Path -LiteralPath $vadControlRoot) { throw 'Existing control directory is never replaced' }
$vadControlRoot = [IO.Path]::GetFullPath($vadControlRoot)
[void][IO.Directory]::CreateDirectory($vadControlRoot)
$vadControlVoice = New-Object -ComObject SAPI.SpVoice
$vadControlHaruka = @($vadControlVoice.GetVoices() | Where-Object { $_.GetDescription() -eq 'Microsoft Haruka Desktop - Japanese' })
if ($vadControlHaruka.Count -ne 1) { throw 'Exact installed Japanese voice is unavailable' }
$vadControlVoice.Voice = $vadControlHaruka[0]
$vadControlRows = @(
    @{ name = 'short'; text = 'ごめん' },
    @{ name = 'sentence'; text = 'これは日本語の音声確認です。音楽の上に言葉が重なる場合も確かめます。' }
)
try {
    foreach ($vadControlRow in $vadControlRows) {
        $vadControlStream = New-Object -ComObject SAPI.SpFileStream
        try {
            $vadControlStream.Format.Type = 22
            $vadControlStream.Open((Join-Path $vadControlRoot ($vadControlRow.name + '.source.wav')), 3, $false)
            $vadControlVoice.AudioOutputStream = $vadControlStream
            [void]$vadControlVoice.Speak($vadControlRow.text, 0)
        } finally {
            $vadControlStream.Close()
            [void][Runtime.InteropServices.Marshal]::ReleaseComObject($vadControlStream)
        }
    }
    @{ voice = 'Microsoft Haruka Desktop - Japanese'; controls = $vadControlRows; output = 'file_only' } |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $vadControlRoot 'fixture.json') -Encoding UTF8
} finally {
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($vadControlVoice)
}
