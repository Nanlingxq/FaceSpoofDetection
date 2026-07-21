import cv2

def get_available_resolutions():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    



    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return []
    
    resolutions = []
    
    # 常见分辨率列表（可根据需要扩展）
    test_resolutions = [
        (3840, 2160), (2560, 1440), (1920, 1080), (1280, 720),
        (1280, 960), (1280, 800), (1024, 768), (800, 600),
        (640, 480), (640, 360), (480, 360), (320, 240)
    ]
    
    for width, height in test_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if (actual_width, actual_height) not in resolutions:
            resolutions.append((actual_width, actual_height))
    
    cap.release()
    return resolutions

if __name__ == "__main__":
    print("正在检测摄像头支持的分辨率...")
    resolutions = get_available_resolutions()
    
    if resolutions:
        print("摄像头支持的分辨率:")
        for w, h in sorted(resolutions, key=lambda x: x[0]*x[1], reverse=True):
            print(f"  {w} x {h}")
    else:
        print("未检测到支持的分辨率")
