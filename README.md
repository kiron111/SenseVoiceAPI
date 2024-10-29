參考︰https://github.com/lovemefan/SenseVoice-python
原作者建立了一個基於SenseVoice ONNX 的python 庫，我修改他的代碼，以fastapi建立個api
實現以下目的︰
- 在樹莓派/香橙派等邊緣裝置, 安裝運行，以全天候形式架設api server 
- 透過自由切換 fp16/int8 的SenseVoiceSmall 模型
- 自由切換不同輸出的文字格式(FUNASR 官方的詳細標注格式，顏文字emoji, 包含timecode 或者純文字版本)
- 把每個轉譯文字的紀錄都備份成文字檔，方便下一步資料整理
- 支持轉換為繁體
- 加設key, 免被外來人濫用

我安裝在香橙派, 主要是用來給手機whatsapp 粵語語音轉文字STT 用, 這模型輕量，對粵語友好；
其他語言，可以有更優解，請移玉步︰
普通話可以用Paraformer(更高準確率), 英文及其他語言直接用whisper (Groq-Whisper 也支持一定使用量的免費額度)
