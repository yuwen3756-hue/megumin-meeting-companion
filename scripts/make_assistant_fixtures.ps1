$ErrorActionPreference = 'Stop'
$fixtureRoot = Join-Path (Split-Path $PSScriptRoot -Parent) 'artifacts\assistant-fixtures'
New-Item -ItemType Directory -Path $fixtureRoot -Force | Out-Null
$voice = New-Object -ComObject SAPI.SpVoice
$fixtures = @(
    @{ file='question_david.wav'; voice='David'; language='en'; text='What is the next milestone for the prototype?' },
    @{ file='question_huihui.wav'; voice='Huihui'; language='zh'; text='这个项目的下一个里程碑是什么？' }
)
foreach ($item in $fixtures) {
    $token = $voice.GetVoices() | Where-Object { $_.GetDescription() -match $item.voice } | Select-Object -First 1
    if (-not $token) { throw "Installed synthetic voice not found: $($item.voice)" }
    $voice.Voice = $token
    $stream = New-Object -ComObject SAPI.SpFileStream
    $stream.Format.Type = 22
    $stream.Open((Join-Path $fixtureRoot $item.file), 3)
    $voice.AudioOutputStream = $stream
    [void]$voice.Speak($item.text)
    $stream.Close()
}
$fixtures | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $fixtureRoot 'sources.json') -Encoding utf8
Write-Output 'Synthetic questions written with installed SAPI voices; no capture or playback.'
