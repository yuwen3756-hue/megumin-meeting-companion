$ErrorActionPreference = 'Stop'
$fixtureRoot = Join-Path (Split-Path $PSScriptRoot -Parent) 'artifacts\fixtures'
New-Item -ItemType Directory -Path $fixtureRoot -Force | Out-Null
$voice = New-Object -ComObject SAPI.SpVoice
$fixtures = @(
    @{ file='english_david.wav'; voice='David'; language='en'; text='Welcome to the planning meeting. We need to review the prototype and check the delivery schedule. I will send the test report tomorrow. The next meeting is on Friday.' },
    @{ file='english_zira.wav'; voice='Zira'; language='en'; text='Thank you for the update. I will review the design and write down the open questions. We agreed to test the microphone before the next meeting. The budget still needs discussion.' },
    @{ file='mandarin_huihui.wav'; voice='Huihui'; language='zh'; text='大家好，我们开始今天的项目会议。首先检查软件的录音功能。我会在明天发送测试报告。下次会议在星期五。预算还需要讨论。' }
)
foreach ($item in $fixtures) {
    $token = $voice.GetVoices() | Where-Object { $_.GetDescription() -match $item.voice } | Select-Object -First 1
    if (-not $token) { throw "Installed synthetic voice not found: $($item.voice)" }
    $voice.Voice = $token
    $stream = New-Object -ComObject SAPI.SpFileStream
    $stream.Format.Type = 22 # 22,050 Hz, 16-bit mono; resampled by benchmark.
    $stream.Open((Join-Path $fixtureRoot $item.file), 3)
    $voice.AudioOutputStream = $stream
    [void]$voice.Speak($item.text)
    $stream.Close()
}
$fixtures | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $fixtureRoot 'sources.json') -Encoding utf8
Write-Output "Created synthetic speech using installed Microsoft SAPI voices. No microphone or playback capture."
