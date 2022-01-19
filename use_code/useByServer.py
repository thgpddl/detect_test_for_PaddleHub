import requests
import json
import base64

if __name__ == "__main__":
    img_path="1.jpeg"
    with open(img_path,'rb') as f:
        base64=base64.b64encode(f.read())   # 得到”bytes“类型的base64数据
        image_base64 = base64.decode('utf-8') # 为了能够json请求，再次转为”str“

    # ”image64“一定是自定义Module中@serving函数的参数同名
    # 当有多个参数需要传递时，在@serving函数中定义名字对应的入参即可
    data = {"image64": image_base64}
    # 指定预测方法为detect_test并发送post请求，content-type类型应指定json方式
    url = "http://127.0.0.1:5959/predict/detect_test"
    # 指定post请求的headers为application/json方式
    headers = {"Content-Type": "application/json"}

    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json())
