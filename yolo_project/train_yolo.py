

if __name__=="__main__":
    from ultralytics import YOLO
    import cv2
    from common.show.show_ui import show_imgs
    # 加载模型
    model = YOLO(r"A:\05-Codes\Gradation\DATA\yolov8n-seg.pt")

    # 预测图片
    img = cv2.imread(r"A:\05-Codes\Gradation\Gradation14\demo.png")
    results = model(img)

    # ---------- 关键步骤：获取绘制后的图像 ----------
    # results[0] 表示第一张图片（因为results是列表），plot() 返回绘制好的BGR图像数组
    annotated_img = results[0].plot()
    show_imgs(img, annotated_img)
    # 此时 annotated_img 是一个 numpy.ndarray，可以直接使用 OpenCV 处理