# detect_test_for_PaddleHub
detect_test_for_PaddleHub

针对PaddleHub自定义Module的教程项目，官方文档：[关于PaddleHub](https://paddlehub.readthedocs.io/zh_CN/release-v2.1/index.html)

我的一些博客：

[PaddleHub：自定义Module](https://blog.csdn.net/qq_40243750/article/details/122577091)

[PaddleHub：自定义Module的Serveing](https://blog.csdn.net/qq_40243750/article/details/122580366)

# 1. 安装
安装requirements.txt安装所需要的库

# 2. 安装该自定义项目到本地Hub中
进入detect_test_for_PaddleHub:

![image](https://user-images.githubusercontent.com/48787805/150083704-ac9ef358-3af6-4bc1-a907-8817bfab0a25.png)

执行安装命令：`hub install detect_test`

![image](https://user-images.githubusercontent.com/48787805/150083740-fdaa44a2-aef5-44c7-97ab-ac7db5f38795.png)

# 3. 在python代码中使用该Module
你可以执行位于/use_code/useByModule.py的脚本，它很简单
成功后会返回：
![image](https://user-images.githubusercontent.com/48787805/150084284-a678b6fa-2e57-424a-9048-929ac8bf19af.png)


# 4. 使用服务器请求使用
## 4.1 开启服务器
通过命令`hub serving start --modules detect_test --port 5959`来开启服务器
![image](https://user-images.githubusercontent.com/48787805/150084032-0831f821-7b24-4f72-9b62-ee22634dff5a.png)

## 4.2 客户端请求

执行位于/use_code/useByServer.py的脚本即可
执行成功后，会返回：
![image](https://user-images.githubusercontent.com/48787805/150084484-3fd5c1b4-d583-48a6-aeb7-05ba58d11047.png)





