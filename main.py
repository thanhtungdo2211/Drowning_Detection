from detect_main import *

if __name__ == '__main__':
    # Khởi tạo class grid_tracker. class này có 2 phần đó là detect và recognize
    # Source là path video hoặc nguồn để detect
    # NUM_TRACK_FRAME là một biến rất quan trọng và nên được điều chỉnh phù hợp với từng fps
    # kết quả tốt nhất đạt được là : NUM_TRACK_FRAME = 5 cho video 8.5FPS và NUM_TRACK_FRAME = 15 cho video 25FPS

    grid_tracker = GridTracker(source='./LD_2_rain.mp4', NUM_TRACK_FRAME=5, weights=['best-total-data-v3.pt'])

    # result_video = cv2.VideoWriter('./V6(conf_thres=0.5)_demo.mp4', 
    #     cv2.VideoWriter_fourcc(*'mp4v'),
    #     10, (2560, 1440)) 

    # khởi tạo để stream video
    video_capture = grid_tracker.dataset

    # Bắt đầu đọc từng frame ảnh cho từng nguồn
    for _, img, im0s, _ in video_capture:
        # img là 2 ảnh giống nhau. 
        # tuy nhiên ảnh im0s là ảnh sẽ được đưa qua các bước preprocess để đưa vào bộ detection
        # Ảnh img là ảnh chỉ để vẽ khung và show ra màn hình

        # Sau mỗi frame thì class grid_tracker cần update thời gian để có thể thực hiện thuật toán recognize 1 cách chuẩn xác
        grid_tracker.get_time()
        
        # Đây là bước detect cánh tay. đầu vào sẽ là 2 ảnh: ảnh im0s là ảnh sẽ được xử lý để đưa vào yolov7. 
        # Ảnh img sẽ là ảnh được đóng bouding box nếu muốn draw
        # output sẽ là ảnh im0 chính là ảnh img sau khi được vẽ bouding box
        im0 = grid_tracker.detect_img(img, im0s)

        # Đây là bước recognize sự kiện đuối nước. 
        # ở đây sẽ chỉ có ảnh đầu vào là im0. và sẽ return lại im0 sau khi đã đóng khung grid cell bị đuối nước
        im0 = grid_tracker.recognize_img(im0)

        # Bởi vì ảnh có độ phân giải khá to nên cần resize lại. 
        # Như ví dụ là 50% kích thước thật của ảnh. 
        # Lưu ý nếu show ảnh quá to sẽ làm giảm FPS của chương trình

        print(im0.shape)
        # result_video.write(im0)

        im0 = resize_img(im0, 20)   
        
        cv2.imshow('yolo', im0)
        cv2.waitKey(1)