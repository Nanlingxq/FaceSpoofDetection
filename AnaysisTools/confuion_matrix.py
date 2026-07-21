import numpy as np

def print_confusion_matrix(all_preds, all_targets, num_classes):
    """
    计算并以格式化的表格形式打印混淆矩阵
    :param all_preds: 所有的预测标签列表 (List 或 Numpy Array)
    :param all_targets: 所有的真实标签列表 (List 或 Numpy Array)
    :param num_classes: 类别总数
    """
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 初始化混淆矩阵为 0
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # 填充混淆矩阵
    for t, p in zip(all_targets, all_preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    print("\n" + "="*45)
    print(f"{'Confusion Matrix':^45}")
    print("-" * 45)
    
    # 打印表头 (True / Pred)
    header = f"{'T / P':>8}"
    for i in range(num_classes):
        header += f"{'Class '+str(i):>8}"
    print(header)
    
    # 打印每一行
    for i in range(num_classes):
        row_str = f"{'Class '+str(i):>8}"
        for j in range(num_classes):
            row_str += f"{cm[i, j]:>8}"
        print(row_str)
    print("="*45)
    
    return cm