# 目录结构
```
morphology
├── morphology
│   └── arithmetics
│       ├── common  # 公用文件夹
│       │   ├── folder_process.py  # 文件操作，主要用于生成指定目录结构
│       │   ├── stats.py
│       │   └── util.py
│       └── gmm.py  # 算法
├── results  # 检测结果文件夹
│   └── gmm_default  # 默认参数下的检测结果，名称自定义
│       ├── gmm_default  # 默认参数下的检测结果
│       │   ├── baseline 
│       │   ...
│       └── morghology  # 默认参数下使用形态学处理后的检测结果
│           ├── baseline
│           ...
├── results_stats  # 存储评估指标文件夹
│   └── gmm_default.csv
└── results_time  # 存储形态学处理耗时文件夹
    └── gmm_default.csv
```
# 评估指标文件说明
```
category_video	# 名称，由类别名和序列名组成

Recall	        # 7中评价指标
Precision	
Specificity	
FPR	
FNR	
PWC	
FMeasure
	
resolution     # 分辨率
total_frames   # 总帧数
valid_frame    # 评估帧数
start_frame    # 评估起始帧号
end_frame      # 评估结束帧号
```

