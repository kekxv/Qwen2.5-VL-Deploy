# Qwen2.5-VL-Deploy

> macos 5000 端口被占用，所以使用 8080 端口。

安装：
```shell
pip install -r requirements.txt
```

启动：
```shell
python app.huggingface.py
# 或者
python app.modelscope.py
```

启动后接口测试：

```shell
curl -s -X POST -H "Content-Type: application/json" \
-d '{
    "images": [
        "图片，支持多个"
    ],
    "prompt": "描述一下图片"
}' \
http://127.0.0.1:5000/infer | jq -r
```
