參考︰https://github.com/lovemefan/SenseVoice-python

原作者建立了一個基於SenseVoice ONNX 的python 庫，我修改他的代碼，以fastapi建立個api

**SenseVoice簡介︰**

SenseVoice是具有音頻理解能力的音頻基礎模型， 包括語音識別（ASR）、語種識別（LID）、語音情感識別（SER）和聲學事件分類（AEC）或聲學事件檢測（AED）。
當前SenseVoice-small支持中、粵、英、日、韓語的多語言語音識別，情感識別和事件檢測能力，具有極低的推理延遲。 本項目提供python版的SenseVoice模型所需的onnx環境安裝的與推理方式。

**實現以下目的︰**

- 在樹莓派/香橙派等邊緣裝置, 安裝運行，以全天候形式架設api server 
- 透過自由切換 fp16/int8 的SenseVoiceSmall 模型
- 自由切換不同輸出的文字格式(FUNASR 官方的詳細標注格式，顏文字emoji, 包含timecode 或者純文字版本)
- 把每個轉譯文字的紀錄都備份成文字檔，方便下一步資料整理
- 支持轉換為繁體
- 加設key, 免被外來人濫用

**個人取捨︰**

我安裝在香橙派, 主要是用來給手機whatsapp 粵語語音轉文字STT 用, 這模型輕量，對粵語友好；

其他語言，可以有更優解，請移玉步︰

普通話可以用Paraformer(更高準確率), 英文及其他語言直接用whisper (Groq-Whisper 也支持一定使用量的免費額度, 但香港要設換VPN才連上)

對於粵語，這SenseVoiceSmall 的響應時間和推理時間都短，大約是whisper base 的速度，但質量有whisper medium 或以上的效果

對於輸出結果有一定精度需求，建議用whisper large V3 (推介︰faster Whisper v3 版本；Turbo v3 對非歐洲語言，優化不好；尤其粵語、越南)

**支持平台︰**

x86 / arm64(已測試)

linux(已測試)/window/mac/ android:Termux

CPU(已測試)/GPU (Nvidia 安裝onnxruntime-gpu, AMD/Intel GPU:onnxruntime-directml)

**安裝︰**
(使用conda, 或 python venv，隨便你，如果有多種途，還是建議你把環境分開)
```shell
git clone https://github.com/kiron111/SenseVoiceAPI.git
cd SenseVoiceAPI
pip install requirements.txt
```
另外，必須安裝ffmpeg, 使其 on path (windows 在系統>>環境參數>>路徑 設置) (linux 好像安裝了就是自動設置好)

**運行 fastapi server︰**
```shell
uvicorn main:app --host 0.0.0.0 --port 9528
```

**API 參數設置︰**
```
SENSE_VOICE_KEY 鑰匙 ︳必填參數 ︳STR ︳隨便設置，但必須在keys.csv 的第一列中，第一行除外)
device 推理設備 ︳選填參數 ︳Int ︳默認 -1 :CPU； 0 :GPU_0；1:GPU_1, GPU需安裝 pip 庫 onnxruntime-gpu/ onnxruntime-directml)
num_threads 多線程數量 ︳選填參數 ︳Int ︳默認 4 (在RK3588 上測試了，再增加綫程數量，反而更慢)
language ︳選填參數 ︳STR ︳默認'auto', 其他選項 "zh": 漢語, "en": 英文, "yue": 粵語, "ja": 日文, "ko":韓文
use_int8 ︳選填參數 ︳Bool ︳默認 False, 使用fp16精度；True: 選int8 精度，速度更快；但可能有準確度損失
use_itn ︳選填參數 ︳Bool ︳默認 True, 使用itn 模型，斷句和加上標點；建議開啟，否則閱讀有問題；False: 不用
replace_tag︳選填參數 ︳STR ︳默認 None, 近乎最詳盡的格式，有標明語言和情緒及其他事件，其他選項 emoji : 標示顏文字； empty: 純文字)
s2t ︳選填參數 ︳Bool ︳默認 False: 不轉換；True: 轉換成繁體輸出
timecode ︳選填參數 ︳Bool ︳默認 True: 有類似timecode的時間標注； False: 不輸出timecode
```

**CURL 例子︰**
```shell
curl -X POST "http://192.168.3.15:9528/transcribe" -F "file=@"D:\python_project\SenseVoiceAPI\TVB.mp3"" -
F "SENSE_VOICE_KEY=sv-1234567"
```
![none.png](https://github.com/kiron111/SenseVoiceAPI/blob/main/screenshots/none.png)

```shell
curl -X POST "http://192.168.3.15:9528/transcribe" -F "file=@"D:\python_project\SenseVoiceAPI\TVB.mp3"" -F "SENSE_VOICE_KEY=sv-1234567" -F "replace_tag=emoji" -F "s2t=True"
```
![emoji_s2t](https://github.com/kiron111/SenseVoiceAPI/blob/main/screenshots/emoji_s2t.png)

```shell
curl -X POST "http://192.168.3.15:9528/transcribe" -F "file=@"D:\python_project\SenseVoiceAPI\TVB.mp3"" -F "SENSE_VOICE_KEY=sv-1234567" -F "replace_tag=empty" -F "s2t=True" -F "timecode=False"
```
![empty_s2t_no_timecode](https://github.com/kiron111/SenseVoiceAPI/blob/main/screenshots/empty_s2t_no_timecode.png)
