# 多数据集检测、解释核验和离线行为实验复跑说明

在项目根目录运行。原始评论、账号逐行结果、第三方源码和虚拟环境均不提交。结果报告见 results/c_nov12/。此次是行为方法在固定数据上的实验；统一评估脚本、官方监督方法及图方法的可比性须由 E/D 联合确认。

1. 使用杨的8c26c202交接版本 features.zip，5张CSV及scaler按该分支results/feature_manifests校验。安装 numpy、pandas、scipy、scikit-learn；版本见检测报告。
2. 解释模块来源 https://github.com/tommasoamico/DCFO 。下载作者代码，下面 DCFO_AUTHOR_DIR 指向含 modules/ 的目录。prepare_adapter.py 仅在本地提取数值模块、修复导入和未使用依赖；不从本仓库分发作者源码。来源文件指纹见 author_source.json。
3. FraudSquad 上游 https://github.com/RafaelLiuZH/FraudSquad ，固定提交 cb72bd86baef829db4cfd063702d9c882405ea49 。下载 data/Amazon/ 的真实zip及三份生成评论csv，路径传给 C_AMAZON_DIR。validate.py 按官方 Git blob 哈希验证，读取假评与真实评论的相同user ID，独立聚合后首先验证是否能重建杨的表。
4. PowerShell 命令（前三个路径改为本机路径）：

~~~powershell
$env:C_FEATURE_ZIP='D:/path/features.zip'
$env:DCFO_AUTHOR_DIR='D:/path/DCFO-main'
$env:C_AMAZON_DIR='D:/path/FraudSquad/data/Amazon'
$env:DCFO_ADAPTER_DIR=(Join-Path (Get-Location) 'data/processed/c_author_adapter')
$env:C_RUN_DIR=(Join-Path (Get-Location) 'data/processed/c_nov12')
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
python src/models/multidata/prepare_adapter.py
python src/models/multidata/detect.py
python src/models/multidata/explain.py
python src/models/multidata/check_outputs.py
python src/models/multidata/validate.py
~~~

detect 使用已交付 scaler 对应的 _z 列，未重新 fit scaler。explain 每模型版本单独选前20；混合拟合LOF，以该版本未被接管账号分数的95%分位作为门槛。解释限于评论数>=3且爆发性可计算的人群，和检测性能表的人群不同。没有沿用10月YelpChi阈值。3个子进程并行，每账号45秒墙钟限制，包含解释器启动与独立复算；不同机器超时数可能不同。只把通过独立复算的连续向量用于解释对照；所有失败/超时保留分母。账号级分数和中间产物写入 C_RUN_DIR。提交只包含聚合结果。

复现含 YelpNYC 的完整检测表时，在上述命令前设置环境变量 C_NYC_DIR，指向杨补发的文件夹（含 feature_table_yelpnyc.csv、feature_table_yelpnyc.json、scaler_yelpnyc.json）。detect.py 默认依次处理全部六张表。只补跑纽约时，用独立的 C_RUN_DIR 执行 python src/models/multidata/detect.py yelpnyc，再按 dataset、method 合并结果。输入 CSV 按原始字节校验；scaler 清单使用 Windows 换行，交接版可还原换行后校验。
